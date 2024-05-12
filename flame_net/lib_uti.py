
#import h5py
#import sklearn.metrics
#from scipy.ndimage import gaussian_filter

#import h5py
#import scipy.io

import torch
import torch.nn as nn
import numpy as np
#import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

#-------------
#import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
#----------------


import operator
from functools import reduce

from timeit import default_timer
###########################
import time
import pickle

from flame_net.libSiva import libSiva, CSolverSiva , round_num_to_txt
from flame_net.libData import libData
from flame_net.libcfdData import libcfdData

from flame_net.PFNO_Nd import PFNO_Nd, p_rescale_nu


#from flame_net.DeepONet_1d import DeepONet_1d
#from flame_net.FourierOp_Nd import FourierOp_Nd
#from flame_net.FourierOp2_Nd import FourierOp2_Nd
#from flame_net.FourierLiftOp_Nd import FourierLiftOp_Nd

from flame_net.ConvPDE_Nd import ConvPDE_Nd

#################################################
#
# lib Utilities
#
#################################################
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensorboard_fig2d_monitor(va_Pair_pick, vaPDEPara_Pair_pick, model, device , nInputStep=1, nOutputStep=1 ):
    fig, axs = plt.subplots(1, va_Pair_pick.shape[0], figsize=(12, 5))
    model.eval()
    with torch.no_grad():
        if vaPDEPara_Pair_pick is None:
            va_pred = model(  va_Pair_pick[..., 0:nInputStep].to(device), None                             ).detach().to('cpu')
        else:
            va_pred = model(  va_Pair_pick[..., 0:nInputStep].to(device), vaPDEPara_Pair_pick.to(device)   ).detach().to('cpu')

        #if nDIM == 2:    va_pred = torch.tanh(va_pred)

    for ax, pred in zip( axs, va_pred[...,-1:]) :
        ax.imshow(pred.squeeze())
    fig.tight_layout()
    return fig

def tensorboard_fig1d_monitor( va_Pair_pick, vaPDEPara_Pair_pick, model, device, nInputStep=1, nOutputStep=1 ):
    # only handle 1-D
    nSize = va_Pair_pick.shape[0]
    #fig, axs = plt.subplots( nSize,1, figsize=(15, 15) )
    fig, ax = plt.subplots( 1,1, figsize=(9, 8) )
    model.eval()
    with torch.no_grad():
        if vaPDEPara_Pair_pick is None:
            va_pred = model( va_Pair_pick[:,:,0:nInputStep].to(device), vaPDEPara_Pair_pick            ).detach().to('cpu')
        else:
            va_pred = model( va_Pair_pick[:,:,0:nInputStep].to(device), vaPDEPara_Pair_pick.to(device) ).detach().to('cpu')
    line1color = 'kbrgc'
    line2color = 'kbrgc'
    # if nSize >1:
    for i in range(nSize):
        #axs[i].plot( va_Pair_pick[icde,:,1]/20 , 'c')

        #axs[i].plot( va_Pair_pick[i,:,1]-va_Pair_pick[i,:,0],'k--')
        #axs[i].plot( va_pred[i,:,0] - va_Pair_pick[i,:,0], 'r-' )

        ax.plot( va_Pair_pick[i,:,nInputStep]-va_Pair_pick[i,:,nInputStep-1], line1color[i]+'--')
        ax.plot( va_pred[i,:,0]             - va_Pair_pick[i,:,nInputStep-1], line2color[i]+'-' )
    # else:
    #     i = 0
    #     #axs.plot( va_Pair_pick[i,:,1]/20 , 'c')
    #     axs.plot( va_Pair_pick[i,:,1]-va_Pair_pick[i,:,0],'k--')
    #     axs.plot( va_pred[i,:,0] - va_Pair_pick[i,:,1], 'r-' )
    fig.tight_layout()
    return fig


#---------------------
class my_cfd_DataSet(torch.utils.data.Dataset):
    def __init__( self , list_y, list_para, T_out=8, list_nStepSkip_cfd=1, T_in=1):
        # list_y    : [ np.array[nTime, Nx, Ny]    , .... ]
        # list_para : [ np.array[num_PDEParameters], .... ]

        self.T_in = T_in
        self.T_out = T_out
        self.list_nStepSkip_cfd = list_nStepSkip_cfd

        self.list_para = []

        self.list_y = []
        for y in list_y:
            self.list_y.append(  torch.tensor( y, dtype=torch.float ).movedim(0,-1)  )

        for p in list_para:
            self.list_para.append(  torch.tensor( p , dtype=torch.float ) )

        self.Len_y      = np.zeros( len(list_y) , dtype = int)
        self.Len_cumsum = np.zeros( len(list_y) , dtype = int)
        cumsum = 0
        for i, y in enumerate(self.list_y):
            self.Len_y[i] = (y.shape[-1] - self.T_in - self.T_out +1)//self.list_nStepSkip_cfd[i]
            cumsum += self.Len_y[i]
            self.Len_cumsum[i] =  cumsum

    def __getitem__(self, item):
        m = np.argmax( item - self.Len_cumsum < 0)
        j = ( item - (self.Len_cumsum[m]-self.Len_y[m]) ) *self.list_nStepSkip_cfd[m]
        #print('m,j=',m,j)
        y_item = self.list_y[m][..., j : j + self.T_in+self.T_out ]
        p_item = self.list_para[m]
        return y_item, p_item

    def __len__(self):
        return self.Len_cumsum[-1]

#
# dataset = my_cfd_DataSet([ np.random.rand(12,3,4) ,np.random.rand(15,3,4) ],[torch.ones(1) ,torch.ones(1)*2 ], 5, 1 )
# dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size = 6, shuffle=True)
# dataiter = iter(dataloader)
# data = dataiter.next()
# for i , (y, p) in enumerate(dataloader):
#     #print( i,' : ', y, ', p :', p)
#     print( i,' : ', y.shape, ', p :', p.shape)
# #dataset[4]

#---------------
class Cdata_sys:
    def __init__(self,sys_name='MS_RK4', list_para=[0.025,0.05,0.2], list_cfdfilename=None, method_default_siva_data_gen=1, num_PDEParameters = 1):

        assert( sys_name  in ['MS_1storder', 'MS_RK4',  'KS_RK4', 'MKS_RK4', 'cfd'] )
        assert type(list_para)==list

        self.sys_name = sys_name
        self.list_para         = list_para
        self.list_cfdfilename = list_cfdfilename
        self.method_default_siva_data_gen=method_default_siva_data_gen
        self.num_PDEParameters = num_PDEParameters

        if type( list_para[0] ) is list:    assert( num_PDEParameters == len( list_para[0] )  )


    def get_num_PDEParameters(self):
        # if 'MS' in self.sys_name or 'KS' in self.sys_name:
        #     if len(self.list_para)==1:     return 0
        #     else:                          return 1
        # elif 'cfd' in self.sys_name:
        #     if len( self.list_para)==1:    return 0
        #     else:                          return 2
        return self.num_PDEParameters

    def para_name( self ):
        if 'MS' in self.sys_name:            return 'nu'
        elif 'KS_RK4' == self.sys_name:      return 'Lpi'
        elif 'MKS_RK4' == self.sys_name:     return 'Lpi_rho'
        elif 'cfd' in self.sys_name:         return 'cfdfilename'
        else:                                raise ValueError('para_name')

# print('count_learnable_params=', str( count_learnable_params(model) ) )
def count_learnable_params(model):
    c = 0
    for p in model.parameters():
        c += reduce(operator.mul, list(p.size()))
    return c


class lib_Model:
    @staticmethod
    def set_default_params( data_sys, nDIM ):

        assert type(data_sys) == Cdata_sys

        params = {'model_name_prefix':'',
                  'data_channel': 1,
                  'method_TimeAdv':'simple',
                  'method_outputTanh':None,
                  'parallel_run': False,
                  'fourier:modes_fourier':32,
                  'fourier:width':20,
                  'fourier:depth':4,
                  'fourier:method_Attention': 0,
                  'fourier:method_WeightSharing': 1,
                  'fourier:method_SkipConnection': 1,
                  'fourier:brelu_last': 1,
                  'fourier:method_BatchNorm': 0,
                  'fourier:PDEPara_mode_level': None,  # could also be 3, [3,6]
                  'PDEPara_fc_class':'',
                  'PDEPara_ReScaling': None,
                  'fourier:method_ParaEmbedding':True,
                  'fourier:PDEPara_AcrossDepth':True,
                  'PDEPara_OutValueRange':0.2,
                  'fourier:method_SteadySolution':False,
                  'fourier:option_RealVersion':False,
                  'Use_2d_DCT':False,
                  'option_nOutputStep':1,
                  'onet:type_branch':'conv',
                  'onet:P': 30,
                  'onet:fc_layers_branch':[100,100,100,100],
                  'onet:fc_layers_trunk':[100,100,100,100],
                  'onet:trunk_featurepair': 1,
                  'onet:type_trunk': 'simple',
                  'onet:method_nonlinear_act':'tanh',
                  'onet:method_skipconnection':False,
                  'conv:en1_channels':[ [16],[32,32],[64,64],[128],[128],[64],[32]]  ,    # en1_channels=[2,2,2,2,2],[1,1,1,1,1],[4,4,4,4,4],[8,8,8,8,8],[8,16,32,64,64]
                  'conv:de1_channels': None,
                  'conv:out_channel':1,
                  'conv:method_nonlinear':'all',
                  'conv:method_types_conv':'conv_all',
                  #'conv:method_OP':'',
                  'conv:method_skip':'full',
                  'conv:bUpSampleOrConvTranspose':'upsample',
                  'conv:method_pool':'Max',
                  #'conv:method_conv':'',
                  'conv:method_BatchNorm':0,
                  'conv:PDEPara_depth': 4,
                  'conv:PDEPara_PathNum': 1,
                  'conv:method_ParaEmbedding':False
                  #'bExternalConstraint':False
                  #,'yB_1DNormalization':None
        }

        #---------------------------------------
        params['T_in'] = 1
        params['T_out'] =20
        params['T_d_out'] =1
        params['num_PDEParameters'] = data_sys.get_num_PDEParameters()

        params['nDIM']=nDIM

        #------------------------------------
        params['data:yB_estimate']=np.array([-0.7, 1.3]) * np.pi
        params['data:AspectRatio_set'] = 1
        params['data:ThicknessScale'] = 1
        params['data:dir_save_training_data']= './data/'
        params['data:nStep'] = 1
        params['data:nStepSkip'] = 1
        params['Nx'] = 128
        #---------
        params['train:weight_decay'] = 1e-4
        params['train:data_norm_rms'] = 1
        params['train:checkpoint_dir'] = './checkpoints'
        params['train:checkpoint_resume'] = None
        params['train:batch_size'] = 2000
        params['train:learning_rate'] = 0.0025
        params['train:scheduler_step'] = 100
        params['train:scheduler_gamma'] = 0.5
        params['train:epochs'] = 1000
        params['train:epochs_per_save'] = 100
        params['optimizer_method']=torch.optim.Adam
        params['train:gradient_clip'] = None
        #if nDIM == 1:
            #params['yB_1DNormalization'] =  np.array([-0.7,1.3])*np.pi

        if nDIM == 2:
            params['fourier:modes_fourier'] = [32,32]


        return params



    @staticmethod
    def update_dependent_params( data_sys, params ):

        if params['nDIM']==1:
            if 'MS' in data_sys.sys_name:
                if data_sys.list_para == [0.02]:
                    params['train:batch_size'] = 1000
                else:
                    params['train:batch_size'] = 500

        if 'cfd' in data_sys.sys_name:
            params['train:batch_size'] = 50

        return params


    @staticmethod
    def build_model(model_name_detail,params):
        if 'fno' in model_name_detail.casefold():
            model = PFNO_Nd( params['nDIM'],
                             params['fourier:modes_fourier'],
                             params['fourier:width'],  #   params['method_TimeAdv'],
                             params['T_in'],
                             params['fourier:depth'],
                             params['num_PDEParameters'],
                             params['data_channel'],  # params['fourier:method_Attention'],
                             params['fourier:method_WeightSharing'],
                             params['fourier:method_SkipConnection'],
                             params['fourier:method_BatchNorm'],
                             params['fourier:brelu_last'],
                             params['fourier:PDEPara_mode_level'],
                             params['PDEPara_fc_class'],
                             params['PDEPara_ReScaling'],
                             params['fourier:PDEPara_AcrossDepth'],
                             params['PDEPara_OutValueRange'],
                             params['fourier:method_ParaEmbedding'] ,
                             params['fourier:option_RealVersion'],
                             params['method_outputTanh'],
                             params['option_nOutputStep'],
                             params['fourier:method_SteadySolution'],
                             params['Use_2d_DCT'] ).cuda()
        # elif 'fourier2' in model_name_detail.casefold():
        #     model = FourierOp2_Nd(params['nDIM'],
        #                          params['fourier:modes_fourier'],
        #                          params['fourier:width'],
        #                          params['method_TimeAdv'],
        #                          params['T_in'],
        #                          params['fourier:depth'],
        #                          params['num_PDEParameters'],
        #                          params['data_channel'],
        #                          params['fourier:method_Attention'],
        #                          params['fourier:method_WeightSharing'],
        #                          params['fourier:method_SkipConnection'],
        #                          params['fourier:method_BatchNorm'],
        #                          params['fourier:brelu_last'] ).cuda()
        # elif 'fourier' in model_name_detail.casefold():
        #     if 'lift' in model_name_detail.casefold():
        #         model = FourierLiftOp_Nd(params['nDIM'],
        #                              params['fourier:modes_fourier'],
        #                              params['fourier:width'],
        #                              params['method_TimeAdv'],
        #                              params['T_in'],
        #                              params['fourier:depth'],
        #                              params['num_PDEParameters'],
        #                              params['data_channel'],
        #                              params['fourier:method_Attention'],
        #                              params['fourier:method_WeightSharing'],
        #                              params['fourier:method_SkipConnection'],
        #                              params['fourier:method_BatchNorm'],
        #                              params['fourier:brelu_last']    ).cuda()
        #     else:
        #         model = FourierOp_Nd(params['nDIM'],
        #                              params['fourier:modes_fourier'],
        #                              params['fourier:width'],
        #                              params['method_TimeAdv'],
        #                              params['T_in'],
        #                              params['fourier:depth'],
        #                              params['num_PDEParameters'],
        #                              params['data_channel'],
        #                              params['fourier:method_Attention'],
        #                              params['fourier:method_WeightSharing'],
        #                              params['fourier:method_SkipConnection'],
        #                              params['fourier:method_BatchNorm'],
        #                              params['fourier:brelu_last'],
        #                              params['fourier:PDEPara_mode_level'] ,
        #                              params['fourier:PDEPara_AcrossDepth'] ).cuda()
        # elif 'onet' in model_name_detail.casefold():
        #     assert params['nDIM']==1
        #     model = DeepONet_1d(params['Nx'],
        #                         params['onet:type_branch'],
        #                         params['data_channel'],
        #                         params['onet:P'],
        #                         params['onet:trunk_featurepair'],
        #                         params['onet:type_trunk'],
        #                         params['num_PDEParameters'],
        #                         params['onet:method_nonlinear_act'],
        #                         params['onet:method_skipconnection'],
        #                         params['T_in']-1,
        #                         params['train:data_norm_rms'],
        #                         params['onet:fc_layers_branch'],
        #                         params['onet:fc_layers_trunk']
        #                         #,params['yB_1DNormalization']
        #                         ).cuda()


        elif 'conv' in model_name_detail.casefold():

            model = ConvPDE_Nd(params['nDIM'],params['Nx'],
                               params['data_channel'],
                               params['conv:out_channel'],
                               params['conv:en1_channels'],
                               params['conv:de1_channels'],
                               params['conv:method_nonlinear'],
                               params['conv:method_types_conv'],
                               #params['conv:method_OP'],
                               params['conv:method_skip'],
                               params['conv:bUpSampleOrConvTranspose'],
                               params['conv:method_pool'],
                               #params['conv:method_conv'],
                               params['num_PDEParameters'],
                               params['conv:method_BatchNorm'],
                               params['conv:PDEPara_depth'],
                               params['conv:PDEPara_PathNum'],
                               params['conv:method_ParaEmbedding']
                               #params['bExternalConstraint']
                               #,params['yB_1DNormalization']
                               ).cuda()

        print('count_learnable_params=', str( count_learnable_params(model) ) )
        return model

    @staticmethod
    def get_model_name_detail(model_name, data_sys, params):

        model_name_detail = params['model_name_prefix']

        if params['num_PDEParameters']==1:             model_name_detail += 'p'
        elif params['num_PDEParameters']==2:           model_name_detail += 'p2'

        if 'onet' in model_name.casefold():
            model_name_detail += 'ONet'
        elif 'conv' in model_name.casefold():
            model_name_detail += 'Conv'
        elif 'fno' in model_name.casefold():
            model_name_detail += 'FNO'
        # elif 'fourier' in model_name.casefold():
        #     if 'fourier2' in model_name.casefold():
        #         model_name_detail += 'Fourier2'
        #     elif 'lift' in model_name.casefold():
        #         model_name_detail += 'FourierLift'
        #     else:
        #         model_name_detail += 'Fourier'

        nDIM = params['nDIM']
        if nDIM == 1:
            model_name_detail += '_'
        elif nDIM ==2:
            model_name_detail += '2D_'
            if params['Use_2d_DCT'] == True:
                model_name_detail += 'dct_'

        if params['method_outputTanh'] is not None:
            model_name_detail += 'tanh_'

        if params['data:ThicknessScale'] != 1:
            model_name_detail += 'tks{}_'.format ( params['data:ThicknessScale'] )

        if 'fourier' in model_name.casefold() or 'fno' in model_name.casefold() :
            if nDIM ==1:
               model_name_detail  += 'm'+ str( params['fourier:modes_fourier'] )+'w'+str( params['fourier:width'])
            elif nDIM ==2:
               model_name_detail  += 'm' + str(params['fourier:modes_fourier'][0]) + '_' + str(params['fourier:modes_fourier'][1]) + 'w' + str(params['fourier:width'])

        if 'MS' in data_sys.sys_name or 'KS' in data_sys.sys_name :
            para_str_ = data_sys.para_name()
            for idx, each_para in enumerate( data_sys.list_para):
                if idx == 0 or idx == len(data_sys.list_para)-1 :
                    if 'MKS_' in data_sys.sys_name: # Now having two parameters
                        para_str_ +=  "{:d}_{:g}_".format( each_para[0], each_para[1] )
                    else:
                        para_str_ +=  round_num_to_txt(each_para) + '_'
            model_name_detail += para_str_[:-1]

        elif 'cfd' in data_sys.sys_name :   #len( para_cfdNS) > 0:
            cfdstr_ = 'cfd'
            for idx, filename in enumerate( data_sys.list_cfdfilename):  # data_sys.list_para
                if idx == 0 or idx == len(data_sys.list_cfdfilename)-1 :
                   cfdstr_ += filename + '_'
            model_name_detail += cfdstr_[:-1]


        if params['method_TimeAdv']=='gru':    model_name_detail +=  '_gru'
        if params['T_in'] >=2:                 model_name_detail +=  '_Tin' + str(params['T_in'])
        if params['data:nStep'] >=2:           model_name_detail +=  '_nStep' + str(params['data:nStep'])

        #if params['num_PDEParameters']>=1:     model_name_detail +=  '_nPara'+ str(params['num_PDEParameters'])


        if params['num_PDEParameters']>=1:

            if 'conv' in model_name.casefold():
                if params['conv:PDEPara_depth'] is not None:
                    if params['conv:method_ParaEmbedding'] > 0:
                        model_name_detail += 'E'
                model_name_detail += 'd{}'.format( params['conv:PDEPara_depth'] )

            # elif 'fourier' in model_name.casefold() and 'fourier2' not in model_name.casefold()               :
            #     if params['fourier:PDEPara_AcrossDepth']:   model_name_detail += 'D{}'.format( params['fourier:PDEPara_mode_level'] )
            #     else:                                       model_name_detail += 'd{}'.format( params['fourier:PDEPara_mode_level'] )

            elif 'fno' in model_name.casefold():
                if params['fourier:PDEPara_mode_level'] is not None:
                    if params['fourier:method_ParaEmbedding'] >  0:
                        model_name_detail += 'E'

                    if params['fourier:PDEPara_AcrossDepth']:  model_name_detail += 'D'
                    else:                                      model_name_detail += 'd'

                    model_name_detail += params['PDEPara_fc_class']  # the default is empty

                    array_PDEPara_mode_level = np.array( params['fourier:PDEPara_mode_level'] )

                    for va in array_PDEPara_mode_level:
                        model_name_detail += '{}'.format( va )

                    if params['PDEPara_OutValueRange'] != 0.2:
                        model_name_detail += 'ovr{}_'.format( params['PDEPara_OutValueRange']  )

                    #if params['fourier:method_ParaEmbedding']==False:    model_name_detail += 'D{}'.format( params['fourier:PDEPara_mode_level'] )
                    #else:                                                model_name_detail += 'ED{}'.format( params['fourier:PDEPara_mode_level'] )

        if params['data_channel']>=2:          model_name_detail += '_dchan'+ str(params['data_channel'])

        if 'fourier' in model_name.casefold() or 'fno' in model_name.casefold() :
            if params['fourier:method_Attention']==1:      model_name_detail +=  '_att'
            if params['fourier:method_WeightSharing']==1:  model_name_detail +=  '_share'
            if params['fourier:method_WeightSharing']==2:  model_name_detail +=  '_share2'
            if params['fourier:method_SkipConnection']==1: model_name_detail +=  '_skip'

            if   params['fourier:method_BatchNorm']< 0:   model_name_detail +=  '_bn'
            elif params['fourier:method_BatchNorm']> 0 :   model_name_detail +=  '_ln'

            if params['fourier:brelu_last']==0:            model_name_detail += '_noLastRelu'

            if params['fourier:method_SteadySolution']==True:   model_name_detail +=  '_SS'

        elif 'conv' in model_name.casefold():
            if params['conv:method_skip'] != 'full':                 model_name_detail += '_skip'+ params['conv:method_skip']
            #if  params['conv:en1_channels'] != [ [16],[32,32],[64,64],[128],[128],[64],[32] ] :
            #    mystr = 'e'
            #    for li in params['conv:en1_channels'] :
            #        mystr += '_'
            #        for l in li:
            #            mystr += str( int(np.log2(l))  )
            #    model_name_detail += mystr
            #if  params['conv:de1_channels'] is not None:   # [[16],[32],[64],[64],[64]]:
            #    mystr = 'd'
            #    for li in params['conv:de1_channels'] :
            #        mystr += '_'
            #        for l in li:
            #            mystr += str( int(np.log2(l))  )
            #    model_name_detail += mystr

            if params['conv:method_types_conv'] != 'conv_all': model_name_detail += ( '_' + params['conv:method_types_conv'] )
            if params['conv:method_nonlinear'] != 'all':       model_name_detail += ('_nonlinear' + params['conv:method_nonlinear'])
            if params['conv:PDEPara_PathNum'] >1:             model_name_detail += ('_pathnum{}'.format( params['conv:PDEPara_PathNum'] ) )
            if params['conv:method_BatchNorm']==-1:          model_name_detail += '_bn'
            elif params['conv:method_BatchNorm']>0:          model_name_detail += '_ln'

        # elif 'onet' in model_name.casefold():
        #     model_name_detail += '_branch' + params['onet:type_branch']
        #     #if params['onet:type_branch'] !='conv':
        #     #    model_name_detail += '_branchfc'
        #     if params['onet:fc_layers_branch'] != [100,100,100,100]:
        #         model_name_detail +='Br'+''.join( [str(e)+'_' for e in params['onet:fc_layers_branch']  ]  )
        #     if params['onet:fc_layers_trunk'] != [100,100,100,100]:
        #         model_name_detail +='Tr'+''.join( [str(e)+'_' for e in params['onet:fc_layers_trunk']  ]  )
        #
        #     if params['onet:P'] != 30:
        #         model_name_detail += '_P'+str(params['onet:P'])
        #
        #     if params['onet:trunk_featurepair'] !=1:
        #         model_name_detail += '_feature' + str(params['onet:trunk_featurepair'])
        #
        #     if params['onet:type_trunk'] != 'simple':
        #         model_name_detail += '_trunkfancy'
        #
        #     if params['onet:method_skipconnection'] :
        #         model_name_detail += '_skipconn'
        #     if params['train:data_norm_rms'] != 1 :
        #         model_name_detail += '_Norm'

        if params['option_nOutputStep'] > 1:      model_name_detail += '_O{}'.format(params['T_out'])
        elif params['option_nOutputStep']==1:     model_name_detail += '_o{}'.format(params['T_out'])

        print(model_name_detail)

        return model_name_detail

class lib_DataGen:
    @staticmethod
    def print_help():
        print('----- params for DataGen -----')
        print('nDIM,T_in,T_out,Nx,nStep,nStepSkip,data_channel, data_sys.sys_name,data_sys.para_name(), data_sys.list_para)   #, data_sys.list_cfdfilename')
        print('------------------------------')

    @staticmethod
    def DataGen(data_sys,params) :

        lib_DataGen.print_help()
        t1 = default_timer()

        if 'MS' in data_sys.sys_name or 'KS' in data_sys.sys_name:
            sequence_disp, sequence_disp_test, sequence_para,sequence_para_test = \
               lib_DataGen.DataGen_siva( data_sys, params['T_in'],params['T_out']*params['T_d_out'],nDIM=params['nDIM'],Nx=params['Nx'],
                                         yB_estimate=params['data:yB_estimate'],AspectRatio_set=params['data:AspectRatio_set'],
                                         nStep=params['data:nStep'],nStepSkip=params['data:nStepSkip'],dir_save_training_data=params['data:dir_save_training_data'] ) #,
                                         #method_default_siva_data_gen=data_sys.method_default_siva_data_gen)

            train_disp, test_disp, train_PDEpara,test_PDEpara = \
                lib_DataGen.np_array_To_torch_tensor(sequence_disp, sequence_disp_test,sequence_para,sequence_para_test,data_sys,params)

            dataset_train, dataset_test =  torch.utils.data.TensorDataset( train_disp, train_PDEpara, ), torch.utils.data.TensorDataset( test_disp, test_PDEpara, )

            #return train_disp, test_disp, train_PDEpara, test_PDEpara

        elif 'cfd' in  data_sys.sys_name:
            dataset_train, dataset_test =  \
               lib_DataGen.DataGen_cfd( params['T_in'],params['T_out'],nDIM=params['nDIM'],Nx=params['Nx'],
                                       yB_estimate=params['data:yB_estimate'],AspectRatio_set=params['data:AspectRatio_set'], ThicknessScale=params['data:ThicknessScale'],
                                       data_channel=params['data_channel'],
                                       nStep=params['data:nStep'],nStepSkip=params['data:nStepSkip'],
                                       list_picklefilename=data_sys.list_cfdfilename,
                                       list_para          =data_sys.list_para )

        # if params['nDIM']==2:
        #     sequence_disp       = np.tanh(sequence_disp)
        #     sequence_disp_test  = np.tanh(sequence_disp_test)
        #     print('np.tanh is applied')

        t2 = default_timer()
        print('preprocessing finished, time used:', t2 - t1)
        return dataset_train, dataset_test


    @staticmethod
    def np_array_To_torch_tensor(sequence_disp, sequence_disp_test,sequence_para,sequence_para_test,data_sys,params):

        print('sequence_disp.shape, sequence_disp_test.shape,sequence_para.shape,sequence_para_test.shape' )
        print( sequence_disp.shape, sequence_disp_test.shape,sequence_para.shape,sequence_para_test.shape)

        nDIM =  params['nDIM']
        data_channel = params['data_channel']
        if  nDIM==1 and ('cfd' in data_sys.sys_name) :
            sequence_disp       = np.moveaxis(sequence_disp,      1, -2)
            sequence_disp_test  = np.moveaxis(sequence_disp_test, 1, -2)
            #(2965, 2048, 11, 3)
            s = sequence_disp.shape
            train_disp = torch.tensor(sequence_disp.reshape(s[0], s[1], s[2] * s[3]), dtype=torch.float)
            train_PDEpara = torch.tensor(sequence_para, dtype=torch.float)

            s = sequence_disp_test.shape
            test_disp = torch.tensor(sequence_disp_test.reshape(s[0], s[1], s[2] * s[3]), dtype=torch.float)
            test_PDEpara = torch.tensor(sequence_para_test, dtype=torch.float)
        else:
            sequence_disp       = np.moveaxis(sequence_disp,      1, -1)
            sequence_disp_test  = np.moveaxis(sequence_disp_test, 1, -1)
            #(20000, 128, 21) in 1D ,  or , (20000, 128, 128, 21)  in 2D
            train_disp = torch.repeat_interleave( torch.tensor(sequence_disp,dtype=torch.float), data_channel, dim=-1 )
            train_PDEpara = torch.tensor(sequence_para, dtype=torch.float)
            test_disp = torch.repeat_interleave(torch.tensor(sequence_disp_test, dtype=torch.float), data_channel, dim=-1)
            test_PDEpara = torch.tensor(sequence_para_test, dtype=torch.float)


        print('train_disp.shape, test_disp.shape, train_PDEpara.shape,test_PDEpara.shape')
        print(train_disp.shape, test_disp.shape, train_PDEpara.shape, test_PDEpara.shape)

        return train_disp, test_disp, train_PDEpara, test_PDEpara


    @staticmethod
    def DataGen_siva(data_sys, T_in,T_out, nDIM=1, Nx=128,
                      yB_estimate=np.array([-0.7, 1.3])*np.pi, AspectRatio_set=1,
                      nStep=1, nStepSkip=1,
                      dir_save_training_data = './data/') : #,method_default_siva_data_gen=1):

        ### skip the following check
        # if 'MS_RK4' == data_sys.sys_name:
        #     #if not all( item in [0.01, 0.02, 0.07, 0.125, 0.4, 0.7, 0.025, 0.05, 0.075, 0.1, 0.15 ] for item in data_sys.list_para ) :
        #     if not all( item in [0.01, 0.02,  0.07, 0.125, 0.4, 0.7, 0.025, 0.035, 0.05, 0.07,  0.1, 0.15 ] for item in data_sys.list_para ) :
        #         raise ValueError('DataGen_Siva, data_sys.list_para did not found for ' + data_sys.para_name() )
        # elif 'KS_RK4' == data_sys.sys_name:
        #     if not all( item in [6, 9, 12, 18, 24] for item in data_sys.list_para ) :
        #         raise ValueError('DataGen_Siva, data_sys.list_para did not found for ' + data_sys.para_name() )
        # elif 'MKS_RK4' in data_sys.sys_name:
        #     if not all( item in [0, 0.25, 0.5, 0.75, 1] for item in data_sys.list_para ) :
        #             raise ValueError('DataGen_Siva, data_sys.list_para did not found for ' + data_sys.para_name() )

        #dir_save_training_data = './data/'

        SivaEq = CSolverSiva( data_sys.sys_name, data_sys.list_para, data_sys.method_default_siva_data_gen)

        Ny, yB = libSiva.get2D_Ny_yB_from_estimate(Nx, yB_estimate, AspectRatio_set=AspectRatio_set)

        if nDIM==1:
            name_xsol= 'dsol'
        elif nDIM==2:
            name_xsol= 'ylevel'
            print( '2D: Ny_actual=', Ny, 'yB=', yB)

        list_xsol, list_para           = SivaEq.generate_or_load_DEFAULT_xsol_list('train', dir_save_training_data,
                                                                                 name_xsol=name_xsol, Nx=Nx, yB_estimate=yB,AspectRatio_set=AspectRatio_set)
        list_xsol_test, list_para_test = SivaEq.generate_or_load_DEFAULT_xsol_list('test' , dir_save_training_data,
                                                                                 name_xsol=name_xsol, Nx=Nx, yB_estimate=yB,AspectRatio_set=AspectRatio_set)
        #print('SivaEq.generate_or_load_DEFAULT_xsol_list')

        #if params['method_TimeAdv'] == 'simple':
        # sequence_disp = libData.Reorg_list_dsol( list_dsol, T_out, T_in )
        
        sequence_disp     , sequence_para      = libData.Reorg_list_xsol(list_xsol,      list_para,      T_out, T_in, nStep, nStepSkip, name_xsol=name_xsol)
        sequence_disp_test, sequence_para_test = libData.Reorg_list_xsol(list_xsol_test, list_para_test, T_out, T_in, nStep, nStepSkip, name_xsol=name_xsol)
        
        #print('libData.Reorg_list_xsol')
        #print('libData.Reorg_list_xsol')

        #else:  # params['method_TimeAdv'] == 'gru':
        #    #sequence_disp, sequence_para = libData.Reorg_list_dsol(list_dsol, list_para, seq_length, T_in)
        #    raise ValueError('Not implemented')

        return sequence_disp, sequence_disp_test, sequence_para,sequence_para_test



    def DataGen_cfd( T_in,T_out,
                     nDIM, Nx=128, yB_estimate = np.array([-0.5, 2])*np.pi,AspectRatio_set=1,ThicknessScale=1,
                     data_channel=1,
                     nStep=1,nStepSkip=1,
                     cfd_data_dir='./Data_PRE_LaminarFlame/', # '/cephyr/NOBACKUP/groups/ml_flame/siva_fourier_torch19/Data_PRE_LaminarFlame/',
                     list_picklefilename=None,
                     list_para = None ):

        #yB_estimate = np.array([-1, 2.2]) * np.pi
        if list_picklefilename is None:
            #list_picklefilename = ['L512_rho5.pkl','L512_rho8.pkl','L512_rho10.pkl']
            list_picklefilename = ['L512_rho8.pkl']


        if nDIM==1:
            varname = 'y_simple' if data_channel==1 else 'y3'

            list_y, list_p = libcfdData.load_PREdata(list_picklefilename, cfd_data_dir, Nx_target=Nx,varname=varname)

            if list_para is not None:
                list_p   =  list_para
            
            #
            sequence_disp, sequence_para = libcfdData.Reorg_list_y(list_y, list_p, T_out, T_in, nStep, nStepSkip)
            sequence_disp_test = np.copy(sequence_disp[-1:])
            sequence_para_test = np.copy(sequence_para[-1:])
            #
            dataset_train, dataset_test = torch.utils.data.TensorDataset(sequence_disp, sequence_para,), torch.utils.data.TensorDataset(sequence_disp_test, sequence_para_test,)

            return dataset_train, dataset_test
            
        elif nDIM ==2:
        
            Ny, yB = libSiva.get2D_Ny_yB_from_estimate(Nx, yB_estimate,AspectRatio_set=AspectRatio_set)

            list_y, list_p = libcfdData.load_2DPREdata(list_picklefilename, cfd_data_dir, Nx, yB, AspectRatio_set=AspectRatio_set, ThicknessScale= ThicknessScale)
            if list_para is not None:
                list_p   =  list_para

            dataset_train = my_cfd_DataSet(list_y, list_p, T_out, nStepSkip, T_in)

            y , p = dataset_train[ len(dataset_train) -1 ]
            dataset_test = torch.utils.data.TensorDataset( y.unsqueeze(0), p.unsqueeze(0),)

            return  dataset_train, dataset_test






#----------------------





#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y, c = 1 ):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms * c)
            else:
                return torch.sum(all_norms * c)

        return all_norms

    def rel(self, x, y , c = 1):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms   = torch.norm(y.reshape(num_examples,-1),                               self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms * c)
            else:
                return torch.sum(diff_norms/y_norms * c)

        return diff_norms/y_norms

    def __call__(self, x, y , c=1):
        return self.rel(x, y, c)



class lib_ModelTrain:
    @staticmethod
    def Train(dataset_train, dataset_test,   #train_disp, test_disp,train_PDEpara,test_PDEpara,
              model, model_name_detail, params ):

        print('batch_size=', params['train:batch_size'])
        #-------------
        nDIM         = params['nDIM']
        data_channel = params['data_channel']
        T_in         = params['T_in']
        T_out        = params['T_out']
        T_d_out      = params['T_d_out']
        #-------------
        if params['option_nOutputStep']>1:  assert T_out == params['option_nOutputStep']
        #----------------------------------------------------------------------------------------
        optimizer = params['optimizer_method']( model.parameters(), lr=params['train:learning_rate'], weight_decay=params['train:weight_decay'] )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['train:scheduler_step'], gamma=params['train:scheduler_gamma'])
        myloss = LpLoss(size_average=False)

        #--------------------------------------------
        list_output_info = []
        epoch0 = 0
        filename_Saved_Model =  params['train:checkpoint_dir'] + '/' + model_name_detail         # model = torch.load(filename_Saved_Model,map_location=torch.device(run_device))

        if params['train:checkpoint_resume'] is not None:
            if params['train:checkpoint_resume'] == '_best.pt':
                resumePATH = filename_Saved_Model+'_best.pt'

                snapshot = torch.load(resumePATH)
                epoch0   = snapshot['ep']
                model.load_state_dict(snapshot['model_state_dict'])
                if not params['parallel_run']:
                    optimizer.load_state_dict(snapshot['optimizer_state_dict'])
                    scheduler.load_state_dict(snapshot['scheduler_state_dict'])

                print( "Load model checkpoint '{}' (epoch {})" .format(resumePATH, epoch0 ) )

                #---- load the trainging log file --
                open_file = open(filename_Saved_Model + 'trainlog.pkl', 'rb')
                output_dict = pickle.load(open_file)
                open_file.close()
                list_output_info = output_dict['list_output_info']  # to be appended
                print('Load ' + filename_Saved_Model + 'trainlog.pkl')
                # ---------------
            else:
                model = torch.load( filename_Saved_Model+params['train:checkpoint_resume'] ) # ,map_location=torch.device(device) )



        #------------------
        ntrain = len(dataset_train)
        ntest  = len(dataset_test)
        print('ntrain=', ntrain, ' ,ntest=', ntest)

        if params['parallel_run']:

            assert params['fourier:option_RealVersion'], "please set params['fourier:option_RealVersion']=True "

            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

            local_rank  = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])

            device = local_rank


            model = model.to(local_rank)
            model = DDP(model, device_ids=[local_rank])

            train_loader =  torch.utils.data.DataLoader( dataset_train, batch_size=params['train:batch_size'], shuffle=False, sampler=torch.utils.data.distributed.DistributedSampler(dataset_train) )
            test_loader  =  torch.utils.data.DataLoader( dataset_test, batch_size=params['train:batch_size'], shuffle=False, sampler=torch.utils.data.distributed.DistributedSampler(dataset_test)  )

            print('gpu[{}] here!'.format(global_rank) )
        else:
            global_rank = 0
            local_rank = 0
            device =  torch.device('cuda')
            model = model.to(device)
            train_loader =  torch.utils.data.DataLoader( dataset_train, batch_size=params['train:batch_size'], shuffle=True)
            test_loader  =  torch.utils.data.DataLoader( dataset_test, batch_size=params['train:batch_size'], shuffle=True)

        #----
        if global_rank==0:
            writer_comment = f'{model_name_detail}_tb'
            if params['tensorboard_logdir_prefix'] is None:  writer = SummaryWriter(comment=writer_comment)
            else:                                            writer = SummaryWriter(log_dir= 'runs/'+ params['tensorboard_logdir_prefix'] + writer_comment)
        #----


        txt_SaveCtr = ''
        #-----------------
        for ep in range( epoch0, params['train:epochs']):
            model.train()
            t1 = default_timer()

            train_l2_full = 0
            test_l2_full  = 0
            # --
            for  train_a, train_p in train_loader:

                assert train_a.shape[-1] == (T_in + T_out*T_d_out)*data_channel, "train_a.shape[-1]==(T_in+T_out*T_d_out)*data_channel"

                train_a = train_a.to(device)  # train_a.shape[-1]== (T_in+T_out)*data_channel
                train_p = train_p.to(device)  # train_p.shape[-1]== (T_in+T_out)*data_channel
                current_batch_size = train_a.shape[0]

                for idx in range(T_d_out):
                    T_0 = idx*T_out*data_channel
                    x  = train_a[...,T_0                  :T_0+ T_in       *data_channel]  # x.shape[-1]== T_in*data_channel
                    yy = train_a[...,T_0+T_in*data_channel:T_0+(T_in+T_out)*data_channel]  # yy.shape[-1]== T_out*data_channel
                    p  = train_p


                    #--------------
                    if params['option_nOutputStep'] > 1:
                        if params['num_PDEParameters'] ==0:     pred = model(x)
                        else:                                   pred = model(x,p)
                    elif params['option_nOutputStep'] == 1:
                        pred = torch.zeros_like( yy , device= train_a.device)
                        for t in range(T_out):
                            y = yy[..., t*data_channel:(t+1)*data_channel]  # y.shape[-1]== 1*data_channel
                            if params['num_PDEParameters'] ==0:     im = model(x)
                            else:                                   im = model(x,p)
                            pred[...,t*data_channel:(t+1)*data_channel] = im
                            x = im
                    #--------------

                    if params['fourier:method_ParaEmbedding'] == 3:
                        l2_full = myloss( pred.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1) , p_rescale_nu(p) )
                    else:
                        l2_full = myloss( pred.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1)  )


                    train_l2_full += l2_full.item()

                    optimizer.zero_grad()
                    # loss.backward()
                    l2_full.backward()
                    if params['train:gradient_clip'] is not None:
                        if params['parallel_run']:
                            nn.utils.clip_grad_norm_(model.module.parameters(), params['train:gradient_clip'] )
                        else:
                            nn.utils.clip_grad_norm_(model.parameters(), params['train:gradient_clip'] )

                    optimizer.step()
                    print('', end='.')

            if ntest > 10:
                # validation test
                model.eval()
                with torch.no_grad():
                    for test_a, test_p in test_loader:
                        test_a = test_a.to(device)
                        test_p = test_p.to(device)
                        current_batch_size = test_a.shape[0]

                        for idx in range(T_d_out):
                            T_0 = idx * T_out * data_channel
                            x  = test_a[..., T_0                 :T_0 + T_in * data_channel]
                            yy = test_a[...,T_0+T_in*data_channel:T_0 + ( T_in + T_out) * data_channel]
                            p = test_p
                            # --------------
                            if params['option_nOutputStep'] > 1:
                                if params['num_PDEParameters'] == 0:     pred = model(x)
                                else:                                    pred = model(x, p)

                            elif params['option_nOutputStep']==1:
                                #loss = 0
                                for t in range(T_out):
                                    y = yy[..., t * data_channel:(t + 1) * data_channel]
                                    if params['num_PDEParameters'] == 0:     im = model(x)
                                    else:                                    im = model(x, p)
                                    # loss += myloss(im.reshape(current_batch_size, -1), y.reshape(current_batch_size, -1))
                                    if t == 0:                    pred = im
                                    else:                         pred = torch.cat((pred, im), -1)
                                    x = torch.cat((x[..., 1 * data_channel:], im), dim=-1)
                                    # test_l2_step += loss.item()
                            # --------------

                            test_l2_full += myloss(pred.reshape(current_batch_size, -1),  yy.reshape(current_batch_size, -1)).item()


            if params['parallel_run']:
                # In order to do the sum across devices, the variable needs to be a
                # tensor with size of at least 1. So it should not be a scalar tensor, if it is
                # you will need to put it into a 1-d tensor.
                torch.distributed.barrier()
                l2__for_print_due_to_ddp = torch.tensor([train_l2_full ,test_l2_full], dtype=torch.float).to(local_rank)
                # Then, you perform the reduction (SUM in this case) across all devices
                torch.distributed.all_reduce( l2__for_print_due_to_ddp , op=torch.distributed.ReduceOp.SUM)
                train_l2_full = l2__for_print_due_to_ddp[0].item()
                test_l2_full  = l2__for_print_due_to_ddp[1].item()


            t2 = default_timer()
            scheduler.step()
            print('')

            # -----------------------
            if ep == 0:
                output_dict = {0: 'ep', 1: 't[s]', 2: 'train_l2', 3: 'test_l2'}
                if global_rank==0:
                    for key, value in output_dict.items():     print(value, end=' ')
                    print('')

            output_info = (ep,   t2 - t1, train_l2_full / ntrain, test_l2_full / ntest )
            list_output_info.append(output_info)

            if global_rank == 0 :
                print('%d, %4.2f, %.5f, %.5f' % output_info)
                output_dict['list_output_info'] = list_output_info
                save_train_log(filename_Saved_Model, output_dict)

                # ----
                writer.add_scalars('loss', {'train_full':train_l2_full / ntrain , 'test_full':test_l2_full / ntest } , ep )
                #writer.add_scalar('time[s]', t2-t1, ep)
                if 'tensorboard_fig1d' in params:
                    fig = params['tensorboard_fig1d'](ep, device)
                    if fig is not None:
                        writer.add_figure('fig_moni',fig, ep)
                        plt.close(fig)
                writer.flush()
                # ----

            # Saving & Loading a General Checkpoint for Inference and/or Resuming Training
            ep1 = ep+1
            bForceSaveNow = False
            if global_rank == 0 :
                if os.path.exists( 'txt_SaveCtr_' + model_name_detail ):
                    file__txt_SaveCtr = open( 'txt_SaveCtr_' + model_name_detail  , "r")
                    new_txt_SaveCtr = file__txt_SaveCtr.readline()
                    file__txt_SaveCtr.close()
                    if txt_SaveCtr != new_txt_SaveCtr:
                        bForceSaveNow = True
                        txt_SaveCtr = new_txt_SaveCtr

            if global_rank == 0 and ( ep1 % params['train:epochs_per_save'] == 0 or bForceSaveNow):
                if ep1 == params['train:epochs']:  filename_SaveNow = filename_Saved_Model
                else:                              filename_SaveNow = filename_Saved_Model +'_ep{}'.format(ep1)
                print(filename_SaveNow)
                if params['parallel_run']:
                    torch.save( model.module , filename_SaveNow )
                else:
                    torch.save( model , filename_SaveNow )

            # --------------------
            #if global_rank == 0 and output_info[1]> 30 and np.argmin( np.array(list_output_info )[:,2]) == len( np.array(list_output_info)[:,2] )- 1  and ep1%5==0:
            if global_rank == 0 and np.argmin( np.array(list_output_info )[:,2]) == len( np.array(list_output_info)[:,2] )- 1  and ep1%5==0:
                filename_SaveNow = filename_Saved_Model+'_best.pt'
                print(filename_SaveNow)
                if filename_SaveNow[-3:] == '.pt':
                    if params['parallel_run']:
                        torch.save({ 'model_state_dict': model.module.state_dict(),
                                     'ep': ep1,  'loss': {'train_full':train_l2_full / ntrain, 'test_full':test_l2_full / ntest }         }, filename_SaveNow )
                    else:
                        torch.save({ 'model_state_dict': model.state_dict(),        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                                     'ep': ep1,  'loss': {'train_full':train_l2_full / ntrain, 'test_full':test_l2_full / ntest }         }, filename_SaveNow )


        # ---------------------------
        # if global_rank == 0 :
        #     print(filename_Saved_Model)
        #     torch.save(model, filename_Saved_Model)
        # retreived_list_output_info = pickle.load(open('trainlog.dump', 'rb'))

        if params['parallel_run']:
            torch.distributed.destroy_process_group()

def save_train_log(filename_Saved_Model,output_dict):
    open_file = open(filename_Saved_Model + 'trainlog.pkl', 'wb')
    pickle.dump(output_dict, open_file)
    open_file.close()

#%%

