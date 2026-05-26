
#-------------------------------------
#import torch.nn.functional as F
#from  torch.autograd.functional import vjp
#import operator
#from functools import reduce
#from functools import partial
#from utilities3 import *
#-------------------------------------
import numpy as np
import torch
import torch.nn as nn
from .MyConvNd import MyConvNd, nn_MaxPoolNd, nn_AvgPoolNd



class DeepONet_1d(nn.Module):  # (nn.Module):
    def __init__(self, N, type_branch='conv', data_channel=1, P=30, trunk_featurepair=1, type_trunk='simple',num_PDEParameters=0,
    method_nonlinear_act='tanh',method_skipconnection=False, nStepPast_Trunk=0, data_norm_rms = 1,
    fc_layers_branch=[100,100,100,100], fc_layers_trunk=[100,100,100,100]
    ): #,yB_1DNormalization=None):
        super(DeepONet_1d, self).__init__()
        nDIM = 1
        self.nDIM = nDIM

        """
        """
        self.P = P
        self.N = N  # Number of grid
        self.data_channel = data_channel
        self.type_branch = type_branch
        self.trunk_featurepair = trunk_featurepair
        self.type_trunk = type_trunk
        self.num_PDEParameters = num_PDEParameters
        self.method_skipconnection = method_skipconnection
        self.nStepPast_Trunk = nStepPast_Trunk


        self.bias = nn.Parameter(torch.zeros([1]) )

        #if yB_1DNormalization is not None:
        #    self.yB_1DNormalization = torch.tensor( yB_1DNormalization, dtype=torch.float )
        self.data_norm_rms = data_norm_rms

        nn_nonlinear_act_func = nn.Tanh() if 'tanh' in method_nonlinear_act.casefold() else  nn.ReLU()

        if 'fc' in self.type_branch.casefold():
            #self.NN_branch = nn.Sequential(
            #    nn.Linear(N * data_channel, 100), nn_nonlinear_act_func,
            #    nn.Linear(100, 100), nn_nonlinear_act_func,
            #    nn.Linear(100, 100), nn_nonlinear_act_func,
            #    nn.Linear(100, self.P)
            #)

            layers_branch = [N * data_channel] + fc_layers_branch + [self.P]
            layers = []
            for l in range( len(layers_branch)-1 ):
                layers.append( nn.Linear( layers_branch[l],layers_branch[l+1] )   )
                if l < len(layers_branch)-2:
                    layers.append( nn_nonlinear_act_func    )
            self.NN_branch = nn.Sequential(*layers)



        else: # 'conv' in self.type_branch:
            self.NN_branch = Branch_ConvNet1d(self.data_channel, self.P, self.N, bRelu=True, bNorm=False,
                                              branch_conv_type=type_branch)

        # elif self.type_branch == 'conv2':
        #    self.NN_branch =  Branch_Conv2_Net1d( self.data_channel, self.P, self.N, c0=16)
        # elif self.type_branch == 'conv3':
        #     self.NN_branch =  Branch_Conv3_Net1d( self.data_channel, self.P, self.N, c0=32)
            # self.NN_branch.apply(init_weights)

            # for p in self.N_branch.parameters():
            #    nn.init.xavier_uniform_(p)

            # self.struct_branch = [self.N, 100, 100,  P]
            # self.fc_branch = nn.ModuleList()
            # for l in range( len(self.struct_branch)-1 ):
            #    self.fc_branch.append( nn.Linear(  self.struct_branch[l], self.struct_branch[l+1] ) )

        # self.fc_trunk = nn.ModuleList()
        # self.struct_trunk = [1, 100, 100, 100,  P]
        # for l in range( len(self.struct_trunk)-1 ):
        #    self.fc_trunk.append( nn.Linear(  self.struct_trunk[l], self.struct_trunk[l+1] ) )

        self.NN_trunk = Trunk_Net1d(data_channel, self.P, trunk_featurepair=trunk_featurepair, type_trunk=type_trunk,method_nonlinear_act=method_nonlinear_act,nStepPast_Trunk=nStepPast_Trunk,fc_layers_trunk=fc_layers_trunk)

    def Branch__uN_to_bp(self, u):  # uN.shape==[...,NGrid,1] -->  bp.shape==[...,P]

        # bs, NGrid, data_channel = u.size()

        if 'fc' in self.type_branch:
            # change u_input from [bs, 512, 1] -> [bs, 512]
            u = self.NN_branch(u.view(u.shape[0], -1))
        else: #if 'conv' in self.type_branch:
            u = self.NN_branch(u.permute(0, 2, 1))

            # [bs, 512] -> ... -> [bs, P]
        # for l in range( len(self.struct_branch)-1 ):
        #    u = self.fc_branch[l](u)
        #    u = torch.tanh(u)

        return u  # u.shape ==  [bs, P]

    def Trunk__yi_to_tp(self, yi, u_nStepPast=None):  # yi.shape==[..,NGrid, 1] -->  tp.shape ==[...,NGrid,p]

        tp = self.NN_trunk(yi,u_nStepPast)

        return tp

    def forward(self, u, yi=None):


        #if self.yB_1DNormalization is not None:
        #    u = 2*( u - self.yB_1DNormalization[0] )/( self.yB_1DNormalization[1] - self.yB_1DNormalization[0] ) - 1
        u = u/self.data_norm_rms  if hasattr(self, 'data_norm_rms') else u

        #  u.shape ==[bs,NGrid,1+nStepPast_Trunk]
        #  yi.shape ==[...,NGrid,1]
        if yi is None:
            N_grid = u.shape[-2]
            yi = torch.tensor(  np.linspace(-np.pi, np.pi, N_grid,endpoint=False).reshape(-1,1), dtype=u.dtype, device=u.device )
            if self.method_skipconnection  if hasattr(self, 'method_skipconnection') else False :
                uSave = u

        bp = self.Branch__uN_to_bp(u[...,:1])  # bp.shape == [ bs,P ]

        if u.shape[-1]>1 and self.method_skipconnection if hasattr(self, 'method_skipconnection') else False :
            tp = self.Trunk__yi_to_tp(yi, u[...,1:]-u[...,:1] )  # tp.shape == [...,NGrid,P]
        else:
            tp = self.Trunk__yi_to_tp(yi, u[..., 1:])

        if tp.dim() == 3:
            Gyi = torch.einsum("bp,bnp->bn", bp, tp)
        elif tp.dim() == 2:
            Gyi = torch.einsum("bp,np->bn", bp, tp)

        Gyi =Gyi  + self.bias
        Gyi = torch.unsqueeze(Gyi, -1)


        #if self.yB_1DNormalization is not None:
        #    Gyi = (Gyi+1)/2* (self.yB_1DNormalization[1] - self.yB_1DNormalization[0]) +  self.yB_1DNormalization[0]


        if yi is None and self.method_skipconnection if hasattr(self, 'method_skipconnection') else False :
            #return uSave + Gyi
            return (uSave+Gyi)*self.data_norm_rms  if hasattr(self, 'data_norm_rms') else (uSave+Gyi)
        else:
            #return Gyi
            return  Gyi*self.data_norm_rms  if hasattr(self, 'data_norm_rms') else Gyi


#------------------------------------------------------------------------------

class Branch_ConvNet1d(nn.Module):
    def __init__(self, indata_channel, num_output, N_mesh, bRelu=True, bNorm=True, branch_conv_type='conv', method_pool='average'):
        super(Branch_ConvNet1d,self).__init__()
        nDIM = 1
        self.nDIM = nDIM
        self.bRelu = bRelu
        self.bNorm = bNorm

        self.indata_channel =  indata_channel
        self.N                 =  N_mesh
        self.num_output        =  num_output


        self.branch_conv_type = branch_conv_type
        if 'in' in branch_conv_type.casefold(): # inception
            self.branch_meshNUM  = [           128, 64, 32, 16,   8,   4,  2  ]
            self.branch_channels = [indata_channel, 16, 32, 64, 128, 128, 64   ]
            self.branch_fsizes   = [                 3,  3,  3,   3,   3,  1   ]
            self.branch_types   = [         'Conv','Inception','Conv','Inception','Inception' ,'Conv']
            print( ' branch: inception-cnn ')

        elif 'res' in branch_conv_type.casefold():
            self.branch_meshNUM  = [           128, 128, 128,  64, 32, 32, 16 , 16,  8,  4]
            self.branch_channels = [indata_channel, 16,  32,   32, 64, 64, 128, 128, 64, 32]
            self.branch_fsizes   = 3
            self.branch_types   = [        'Resid','Resid','Resid2','Resid','Resid2','Resid','Resid','Resid','Resid' ]
            print( ' branch: residual-cnn ')
        elif 'cnn2' in branch_conv_type.casefold():
            self.branch_meshNUM  = [           128, 128, 128,  64, 64, 32, 32, 16 , 16,  8,   4]
            self.branch_channels = [indata_channel, 16,  32,   32, 32, 64, 64, 128, 128, 64, 32]
            self.branch_fsizes   = 3
            self.branch_types   =  'Conv'
            print( ' branch: cnn2')
        else: #default cnn
            if self.N==128:
                self.branch_meshNUM  = [           128, 128,  64,  32, 16,    8, 4, 2]
                self.branch_channels = [indata_channel,  32,  64,  128, 128, 64, 64,64]
            elif self.N==512:
                self.branch_meshNUM  = [           512, 512, 256, 128,  64, 32, 16,  8,  4,2]
                self.branch_channels = [indata_channel, 16,  32,   64, 128, 64, 64, 64, 64,64]
            elif self.N==1024:
                self.branch_meshNUM  = [           1024, 1024, 512, 256, 128,  64, 32, 16,  8,  4 ,2]
                self.branch_channels = [indata_channel,   16,  32,  64,  128, 64, 64,  64,  64, 64,64]

            self.branch_fsizes   = 3
            self.branch_types    = 'Conv'
            print( ' branch: simple-cnn ')

        if str == type(self.branch_types):
            self.branch_types = [self.branch_types for tmp in range( len(self.branch_channels)-1 ) ]

        if int == type(self.branch_fsizes):
            self.branch_fsizes = [self.branch_fsizes for tmp in range( len(self.branch_channels)-1 ) ]

        self.method_pool = method_pool

        # ---------------------
        layers = []
        for l in range( len(self.branch_channels)-1 ):
            layers.append (  MyConvNd(nDIM, self.branch_channels[l],self.branch_channels[l+1],kernel_size=self.branch_fsizes[l], type = self.branch_types[l] ) )

            if  self.branch_meshNUM[l]//self.branch_meshNUM[l+1] == 2:
                if 'ave' in self.method_pool.casefold():
                    layers.append( nn_AvgPoolNd(nDIM)(2)  )
                else:
                    layers.append( nn_MaxPoolNd(nDIM)(2)  )


        self.block0 = nn.Sequential( *layers )
        # ---------------------

        #self.block0.apply(init_weights)

        num_afterpool =  self.branch_meshNUM[-1]*self.branch_channels[-1]
        self.fc_out = nn.Linear( num_afterpool, self.num_output  )


    def forward(self,x):
        x = self.block0(x)
        x = x.view(x.size(0),-1)   # flatten
        x = self.fc_out(x)
        return x





class Trunk_Net1d(nn.Module):
 
    def __init__(self, in_channel, out_channel, trunk_featurepair,type_trunk='simple',method_nonlinear_act='tanh',nStepPast_Trunk=0,fc_layers_trunk=[100,100,100,100] ):
        super(Trunk_Net1d,self).__init__()
        self.in_channel =  in_channel
        self.out_channel = out_channel

        self.trunk_featurepair = trunk_featurepair

        self.type_trunk  = type_trunk

        self.nStepPast_Trunk = nStepPast_Trunk

        nn_nonlinear_act_func = nn.Tanh() if 'tanh' in method_nonlinear_act.casefold() else  nn.ReLU()

        if 'simple' in self.type_trunk:

            #self.net = nn.Sequential(
            #        nn.Linear( in_channel*2*(self.trunk_featurepair)+nStepPast_Trunk, 100),  nn_nonlinear_act_func,
            #        nn.Linear( 100, 100),             nn_nonlinear_act_func,
            #        nn.Linear( 100, 100),             nn_nonlinear_act_func,
            #        nn.Linear( 100, 100),             nn_nonlinear_act_func,
            #        nn.Linear( 100, self.out_channel),     nn_nonlinear_act_func
            #       )

            layers_trunk = [in_channel*2*(self.trunk_featurepair)+nStepPast_Trunk] + fc_layers_trunk + [self.out_channel]
            layers = []
            for l in range( len(layers_trunk)-1 ):
                layers.append( nn.Linear( layers_trunk[l],layers_trunk[l+1] )   )
                layers.append( nn_nonlinear_act_func                              )
            self.net = nn.Sequential(*layers)


            self.net.apply( init_weights )

        elif 'fancy' in self.type_trunk:
            ##############################################
            self.net_combinefeature = nn.Sequential(
                nn.Linear( self.in_channel, 30),  nn_nonlinear_act_func,
                nn.Linear( 30, 30), nn_nonlinear_act_func,
                nn.Linear( 30, 30),  nn_nonlinear_act_func,
                nn.Linear( 30, self.trunk_featurepair), # nn.Tanh(),
            )
            #####################################################
            self.net_feature = nn.ModuleList()
            for l in range(self.trunk_featurepair):
                self.net_feature.append ( 
                    nn.Sequential(  nn.Linear( self.in_channel*2, 100), nn_nonlinear_act_func,
                                    nn.Linear( 100, 100),             nn_nonlinear_act_func,
                                    nn.Linear( 100, 100),            nn_nonlinear_act_func,
                                    nn.Linear( 100, 100),             #nn.Tanh(),                       
                                ) 
                )           

            self.fc_outB = nn.Sequential( nn.Linear(100,  self.out_channel), nn_nonlinear_act_func )

    def forward(self, yi, u_nStepPast = None):     #yi: [....,NGrid,1]; u_nStepPast =[ ...,NGrid, nStepPast_Trunk]

        if 'simple' in self.type_trunk:


            ################
            for l in range(self.trunk_featurepair):
                #m = 2**l
                m = l+1
                if l==0:
                    y_in_features = torch.cat( (torch.cos(m*yi), torch.sin(m*yi) ), dim =-1 )
                else:
                    y_in_features = torch.cat(   (y_in_features, torch.cos(m*yi), torch.sin(m*yi) ) , dim = -1 )
            ################
            if hasattr(self,'nStepPast_Trunk') :
                if self.nStepPast_Trunk >0:
                    if u_nStepPast.dim() == yi.dim() +1 :
                        y_in_features = torch.cat( (u_nStepPast, torch.unsqueeze(y_in_features,0).expand(u_nStepPast.shape[0],-1,-1) ), dim=-1)
                    elif u_nStepPast.dim() == yi.dim() :
                        y_in_features = torch.cat( (u_nStepPast, y_in_features), dim=-1)
                    else:
                        raise  ValueError('u_nStepPast.dim() <> yi.dim()')
            ################

            tp = self.net( y_in_features  )
            return tp

        elif 'fancy'  in self.type_trunk:
            sh = yi.shape

            weights = self.net_combinefeature(yi)   # [....,NGrid,NPairs]

            outA = torch.zeros(  *sh[:-1], 100,    device = yi.device )
            for l in range(self.trunk_featurepair):
                m = 2**l
                outA +=  weights[...,l:l+1] * self.net_feature[l](  torch.cat( (torch.cos(m*yi), torch.sin(m*yi) ), dim =-1 )  )  #[...,NGrid,100]
            
            out_B = self.fc_outB(outA)
            return out_B




def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight) #, gain=nn.init.calculate_gain('relu'))
        #m.weight.data.fill_(1.0)
        #print('init weights:', m)
    elif type(m) == nn.Conv1d:
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        #print('init weights:', m)