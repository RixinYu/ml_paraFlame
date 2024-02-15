import torch
import numpy as np
import matplotlib.pyplot as plt

#from flame_net.PFNO_Nd import PFNO_Nd
#from flame_net.FourierOp_Nd import FourierOp_Nd
#from flame_net.DeepONet_1d import DeepONet_1d
#from flame_net.ConvPDE_Nd import ConvPDE_Nd

from flame_net.lib_uti import Cdata_sys, count_learnable_params, lib_Model,lib_DataGen,LpLoss,lib_ModelTrain
from timeit import default_timer



def main( parallel_run:int ) :
    #-----------------------
    #device = torch.device('cuda') # if torch.cuda.is_available() else 'cpu')
    #-----------------------
    nDIM = 1
    data_sys = Cdata_sys('MS_RK4', list_para=[0.025, 0.035, 0.05, 0.07, 0.1, 0.15], list_cfdfilename=None, num_PDEParameters=1)
    #data_sys = Cdata_sys('KS_RK4',[6, 9, 12, 18, 24],  list_cfdfilename=None, num_PDEParameters=1 )
    params = lib_Model.set_default_params(data_sys,nDIM)

    params['T_in' ] = 1
    params['T_out'] = 20
    params['data_channel'] = 1
    params['data:nStep']  = 1
    params['data:nStepSkip']=1
    params['Nx']            = 256 #512 # 128
    #-----
    params['train:batch_size'] = 1000
    params['train:learning_rate'] = 0.0025
    
    #--------------------
    # model_name = 'fno'   #'fourier2'

    model_name = 'conv'
    
    # params['fourier:method_BatchNorm' ] = 256  # 0, no-batchnorm,  -1, batch norm, >1 , layer norm
    
    if 'fno' == model_name :
        params['fourier:modes_fourier' ] =  128  # 64 # 128 # 128 # 64
        params['fourier:width' ] =  30           # 25 # 20
        params['fourier:depth' ] =  4
        params['fourier:method_WeightSharing'] =  True #
        params['fourier:method_SkipConnection'] = False
        params['fourier:method_ParaEmbedding'] = False # True #  3 # False # 2 #False # 2 or True
        params['fourier:PDEPara_mode_level'] =   [7] #   [3,5]# [7]      #[2,5] # 7 # 5 # None # 5
        params['PDEPara_fc_class'] = 'OM'
        # params['fourier:method_SteadySolution'] =  True
    elif 'conv' == model_name:
        params['conv:method_types_conv'] = 'inception_less'
        params['conv:en1_channels' ] = [ [16],[32,32],[64,64],[128],[128],[64],[32]] # [ [16],[32,32],[64,64],[128],[128],[64],[32]]
        params['conv:PDEPara_depth'] = 4                                                # 4
        params['conv:method_BatchNorm' ] = 256
        params['conv:method_ParaEmbedding']= False # True
    #-----------------
    #----------------------
    # params['PDEPara_OutValueRange']=0.3
    # params['method_outputTanh' ] = False        # True
    # params['train:epochs'] = 1000
    # params['train:checkpoint_resume'] = '_best.pt' # None     # '_best.pt'
    #----------------------
    params['train:gradient_clip'] = 50
    params['tensorboard_logdir_prefix'] = ''           # 'paraR_'
    params['model_name_prefix'] = ''                   # 'test'
    #-----------------------
    params['fourier:option_RealVersion'] = False

    params['parallel_run'] = True if parallel_run!=0 else False
    if params['parallel_run'] == True:    params['fourier:option_RealVersion'] = True

    # params['option_nOutputStep']  = params['T_out']



    #--------------------
    dataset_train, dataset_test = lib_DataGen.DataGen(data_sys,params)
    #----------------
    model_name_detail = lib_Model.get_model_name_detail(model_name,data_sys,params)
    model = lib_Model.build_model(model_name_detail,params)
    #---------------



    #%matplotlib inline
    from flame_net.lib_uti import tensorboard_fig1d_monitor
    n_list = [int(len(dataset_test)//3*0.9), int(len(dataset_test)//3*1.3), int(len(dataset_test)//3*2.1)]
    disp_peek = torch.stack( [dataset_test[n][0] for n in n_list ] )
    para_peek = torch.stack( [dataset_test[n][1] for n in n_list ] )
    def tensorboard_callback(ep, device):
        #n1,n2,n3 = 351,1050, -522
        if ep%50==0:     return tensorboard_fig1d_monitor( disp_peek, para_peek,   model, device , params['T_in'], params['option_nOutputStep'] )
        else:            return None
    #fig = tensorboard_callback()
    params['tensorboard_fig1d']= tensorboard_callback


    #--------------------
    params['train:batch_size'] =800
   

    # print( 'params=', params )
    print('')
    print('---------------------')
    for key, value in params.items():
        print(key, ":", value)
    print('---------------------')
    print('')


    #lib_ModelTrain.Train(train_disp, test_disp,train_PDEpara,test_PDEpara,model,model_name_detail,device,params )
    lib_ModelTrain.Train(dataset_train, dataset_test ,    model,model_name_detail,params )



#------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    #parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    #parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    #parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('parallel_run', default=0, type=int,  help='parallel_run(default: 0)')
    args = parser.parse_args()

    main( args.parallel_run ) # args.batch_size)

