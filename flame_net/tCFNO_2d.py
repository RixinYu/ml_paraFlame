
#-----------------------------------------------------
#from utilities3 import *
#import operator
#from functools import reduce
#from torch.autograd import Variable
#-----------------------------------------------------


import torch
import torch.nn as nn
#from functools import partial
#import numpy as np




# --------------------------------------------------------------------------------------------------------------------------
#     Koopman theory-Inspired 'convolutional' Fourier Neural Operator(kCFNO, i.e. using convoluton along one spatial dimension and Fourier transform along the other spatial dimension),  kCFNO was used to learn the complicate DNS fractal flame 
#
#     [Yu, R., Herbert, M., Klein, M. and Hodzic, E., 2025. 'Koopman-Inspired Operator Learning for Intrinsic Flame Instabilities', presented in 1st International Symposium on AI and Fluid Mechanics (AIFLUIDs), submitted for publication in Computer & Fluids]

#
# --------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------
class Spectral_1d_Conv_2d(nn.Module):
    def __init__(self, inout_channels, modes_fourier, Ny):
        super(Spectral_1d_Conv_2d, self).__init__()
        self.modes_fourier  = modes_fourier  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.Ny = Ny

        self.inout_channels = inout_channels
        scale = 1 / (inout_channels*inout_channels)

        self.weights = nn.Parameter(scale * torch.rand( modes_fourier, Ny, inout_channels, inout_channels, dtype= torch.cfloat ) )
        return

    # ----------------------
    def forward(self, x ): # x.shape=  b,w,Nx,Ny

        batchsize = x.shape[0]
       
        dim_fft, dim_conv = -2, -1

        x_ft    = torch.fft.rfft( x, dim=dim_fft, norm="ortho")                             # 1d-rfft-along-x
        out_ft  = torch.zeros(batchsize,self.inout_channels, x.size(dim_fft)//2 + 1, x.size(dim_conv), dtype=torch.cfloat, device=x.device)
        #---------
        k0    = self.modes_fourier

        out_ft[:,:,  :k0,  :] = torch.einsum( 'bixy,xyio->boxy', x_ft[:, :,  :k0, :], self.weights )

        x   = torch.fft.irfft( out_ft, dim= dim_fft, norm='ortho')
        return x

#-------------------------------------------
class ConvFourierLayer_2d(nn.Module):
    def __init__(self, inout_channels, modes_fourier, Ny ):
        super(ConvFourierLayer_2d, self).__init__()

        self.SpectralConv = Spectral_1d_Conv_2d(inout_channels, modes_fourier, Ny)
        self.w = nn.Conv2d( inout_channels, inout_channels, 1)

        self.Conv1 = nn.Sequential( 
            nn.Conv1d( inout_channels, inout_channels, 3, padding=1, padding_mode='replicate'), nn.GELU(), 
            nn.Conv1d( inout_channels, inout_channels, 3, padding=1, padding_mode='replicate') #, nn.GELU() 
        )
        return

    def forward( self, x ):
        x = self.SpectralConv(x)+self.w(x)

        #--------------
        b,w,Nx,Ny = x.shape
        x = x.permute( [0,2,1,3] ).reshape( -1, w, Ny )
        #-------------
        x = self.Conv1(x)
        #-------------
        x = x.view( b, Nx, w, Ny ).permute( [0,2,1,3] )
            
        return x



#-------------------------------------------
#  Block containing multiple MixLayer of fixed width 
class ConvFourierBlock_2d(nn.Module):
    def __init__(self, depth, width, modes_fourier, Ny, 
                 bUseSkipConnection=False, method_WeightSharing=False, bNonlinearForLastLayer=False):
        super(ConvFourierBlock_2d, self).__init__()

        self.bNonlinearForLastLayer = bNonlinearForLastLayer
        self.bUseSkipConnection = bUseSkipConnection
        self.method_WeightSharing = method_WeightSharing
        self.depth = depth

        self.conv = nn.ModuleList()
        for j in range(self.depth):
            if j == 0 or self.method_WeightSharing==False:
                conv_j = ConvFourierLayer_2d( width, modes_fourier, Ny )
            self.conv.append(conv_j)
        return

    def forward( self, x ):
        if self.bUseSkipConnection == True:
            for j in range(self.depth):
                tmp = self.conv[j](x) 
                if j == self.depth-1 and self.bNonlinearForLastLayer == False:
                    x = x + tmp
                else:
                    x = x + nn.GELU()(tmp)  
        else:
            for j in range(self.depth):
                tmp = self.conv[j](x) 
                if j == self.depth-1 and self.bNonlinearForLastLayer == False:
                    x =  tmp
                else:
                    x =  nn.GELU()(tmp)  

        return x                 

#-------------------------------------------
class PermuteLayer_2d(torch.nn.Module):
    def __init__(self, bForward ) -> None:
        super().__init__()
        self.bForward = bForward  # True: b,(Nx,Ny),w  -> b,w,(Nx,Ny)
                                  # False: b,w,(Nx,Ny) -> b,(Nx,Ny),w    
        return
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bForward == True: 
            return input.permute( [0, 3, 1, 2] )     # b,(Nx,Ny),w --> b,w,(Nx,Ny)
        else: 
            return input.permute( [0, 2, 3, 1] )      # b,w,(Nx,Ny) -> b,(Nx,Ny),w    



###################################################################################################################################
#
#  The implemention of Koopman theory-Inspired Mixed Neural Operator (kCFNO), 
#      this implention contains some extra options for debug/test purpose.
class tCFNO_2d(nn.Module):
    def __init__(self,  modes_fourier, Ny, width, 
                 in_channel=1, kTimeStepping = 20,
                 depth_conv={'tAdv':2,'lift':3,'proj':1,'tAdv_last_nonlinear':False}, 
                 method_SkipConnection = 1,  # 0 means no-skip connection
                 method_WeightSharing = False,
                 out_channel=1): 

        super(tCFNO_2d, self).__init__()

        self.modes_fourier    = modes_fourier
        self.width            = width

        self.in_channel       = in_channel
        self.out_channel      = out_channel

        self.kTimeStepping     = kTimeStepping

        self.method_WeightSharing = method_WeightSharing
        self.method_SkipConnection = method_SkipConnection

        if method_SkipConnection == 1:       bUseSkip={'tAdv':True,'lift':True,'proj':True}
        elif method_SkipConnection == 0:     bUseSkip={'tAdv':False,'lift':False,'proj':False}
        elif method_SkipConnection == -1:    bUseSkip={'tAdv':False,'lift':True,'proj':True}


        #-------------
        self.depth_conv = depth_conv
        if 'tAdv_last_nonlinear' not in self.depth_conv: 
            self.depth_conv['tAdv_last_nonlinear'] = False  # add new default key

        self.conv_timeAdv = ConvFourierBlock_2d( depth=self.depth_conv['tAdv'], width=self.width, modes_fourier=self.modes_fourier, Ny=Ny,
                                         bUseSkipConnection = bUseSkip['tAdv'], method_WeightSharing=self.method_WeightSharing, bNonlinearForLastLayer=self.depth_conv['tAdv_last_nonlinear'] )
        #----------
        self.net_up_lift = nn.Sequential(
            nn.Linear(self.in_channel, self.width),
            PermuteLayer_2d(  bForward=True ),
            ConvFourierBlock_2d( depth=self.depth_conv['lift'], width=self.width, modes_fourier=self.modes_fourier, Ny=Ny,
                         bUseSkipConnection= bUseSkip['lift'], method_WeightSharing=self.method_WeightSharing, bNonlinearForLastLayer=True )
        )

        conv_proj = ConvFourierBlock_2d( depth=self.depth_conv['proj'], width=self.width, modes_fourier=self.modes_fourier, Ny=Ny,
                                 bUseSkipConnection= bUseSkip['proj'], method_WeightSharing=self.method_WeightSharing, bNonlinearForLastLayer=False )
        permute_layer = PermuteLayer_2d( bForward=False)

        self.net_down_proj = nn.Sequential(
            conv_proj,
            permute_layer,
            nn.Linear(self.width, 128),    nn.GELU(),  
            nn.Linear(128, self.out_channel) 
        )


        return


    def forward(self, x , p=None):

        x_ot = torch.zeros( *x.shape[:-1],self.out_channel, self.kTimeStepping, device=x.device ) # b,(Nx,Ny),out,t

        x = self.net_up_lift(x)

        for t in range(self.kTimeStepping):
            
            x = self.conv_timeAdv(x)   

            u = self.net_down_proj(x)

            x_ot[...,t] = u

        if self.kTimeStepping == 1:             x_ot = x_ot.view( *x_ot.shape[:-1] )

        return x_ot



