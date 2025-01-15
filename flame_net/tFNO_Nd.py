
#-----------------------------------------------------
#from utilities3 import *
import operator
from functools import reduce
#from torch.autograd import Variable
#-----------------------------------------------------


import torch
import torch.nn as nn
from functools import partial
import numpy as np




# --------------------------------------------------------------------------------------------------------------------------
#
#  This file of 'tFNO_Nd.py' is a 'early' implemention of (N-dimentional) 'Koopman theory-inspired Fourier Neural Operator' (kFNO), this implemention contains extra options for debug/test purpose. 
#  A second 'cleaner' implementaion of kFNO can be found in the file 'kFNO_Nd.py' which contains less options, 
#  'tFNO_Nd' can reduces to 'kFNO_Nd' under proper choices of parameters. 
#
#  [Yu, R., Herbert, M., Klein, M. and Hodzic, E., 2024. Koopman Theory-Inspired Method for Learning Time Advancement Operators in Unstable Flame Front Evolution. arXiv preprint arXiv:2412.08426.]
#
# --------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
#  First, a few utilie functions to be used in the followed main class of 'kFNO_Nd'

# -----------------------------------------------
# Complex multiplication implemented using real number
# required for multi-gpu runs due to Nvidia cuda
def compl2_einsum(op_einsum, a, b):  # a is complex
    op = partial(torch.einsum, op_einsum )
    a = torch.view_as_real(a)
    c =torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)
    return torch.view_as_complex(c)

class SpectralConv_Nd(nn.Module):
    def __init__(self, in_channels, out_channels, modes_fourier, basis_type = '', bRealVersion=False ):
        super(SpectralConv_Nd, self).__init__()
        if type(modes_fourier) == int:  self.nDIM = 1
        else:                           self.nDIM = len(modes_fourier)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_fourier = modes_fourier #Number of Fourier modes to multiply, at most floor(N/2) + 1
        scale = 1 / (in_channels * out_channels)
        self.basis_type = basis_type
        self.bRealVersion = bRealVersion

        if '+x[-1]' in self.basis_type:  self.ratio_cord = nn.Parameter( 0.5*torch.rand(in_channels, dtype=torch.float) )

        if self.bRealVersion == True:
            if self.nDIM == 3:
                self.weights1 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, 2 ))
                self.weights2 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, 2 ))
                self.weights3 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, 2 ))
                self.weights4 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, 2 ))
            elif self.nDIM == 2:
                self.weights1 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, 2 ) )
                self.weights2 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, 2 ) )
            elif self.nDIM==1:
                self.weights  = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels,  2) )
        else: # the ComplexVersion
            if self.nDIM == 3:
                self.weights1 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch.cfloat ))
                self.weights2 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch.cfloat ))
                self.weights3 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch.cfloat ))
                self.weights4 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch.cfloat ))
            elif self.nDIM == 2:
                self.weights1 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch.cfloat ) )
                self.weights2 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch.cfloat ) )
            elif self.nDIM==1:
                self.weights  = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels,  dtype=torch.cfloat) )
        return

    # ----------------------
    def forward(self, x ): # x.shape=  b,w,(Nx,Ny)

        batchsize = x.shape[0]
        if '+x[-1]' in self.basis_type:  # Add the x-coordinate in the last dimention
            N = x.shape[-1]
            xcord_extra =  torch.linspace(0,1,N).to(x.device)
            if   self.nDIM == 1: x = x+ xcord_extra.view(1,1,-1)    *self.ratio_cord.view(1,-1,1)
            elif self.nDIM == 2: x = x+ xcord_extra.view(1,1,1,-1)  *self.ratio_cord.view(1,-1,1,1)
            elif self.nDIM == 3: x = x+ xcord_extra.view(1,1,1,1,-1)*self.ratio_cord.view(1,-1,1,1,1)

        if self.bRealVersion == True:    einsum_op =  compl2_einsum
        else:                            einsum_op =  torch.einsum    # the ComplexVersion

        if self.nDIM == 3:
            # ------------------------------------
            if 'dct[1]' in self.basis_type:
                dim_dct, dim_other = -2, (-3,-1)
                x_xflip = torch.cat ( [ x, x.flip([dim_dct])[...,1:-1,:] ], dim=dim_dct )
                x_dct1  = torch.fft.fft(   x_xflip,  dim=dim_dct  , norm="ortho").real[..., :x.size(dim_dct),:] # 1d-dct-along-z
                x_ft    = torch.fft.rfftn( x_dct1, dim=dim_other, norm="ortho")                               # 1d-rfft-along-xy
                out_ft  = torch.zeros(batchsize, self.out_channels,  x.size(-3), x.size(-2),  x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            else:
                x_ft    = torch.fft.rfftn(x, dim=[-3,-2,-1], norm="ortho")
                out_ft  = torch.zeros(batchsize, self.out_channels,  x.size(-3), x.size(-2),  x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            # ----
            k0, k1, k2 = self.modes_fourier
            out_ft[:, :,   :k0,   :k1, :k2] = einsum_op( 'bixyz,xyzio->boxyz', x_ft[:,:,   :k0,   :k1, :k2], self.weights1)
            out_ft[:, :,-k0:  ,   :k1, :k2] = einsum_op( 'bixyz,xyzio->boxyz', x_ft[:,:,-k0:  ,   :k1, :k2], self.weights2)
            out_ft[:, :,   :k0,-k1:  , :k2] = einsum_op( 'bixyz,xyzio->boxyz', x_ft[:,:,   :k0,-k1:  , :k2], self.weights3)
            out_ft[:, :,-k0:  ,-k1:  , :k2] = einsum_op( 'bixyz,xyzio->boxyz', x_ft[:,:,-k0:  ,-k1:  , :k2], self.weights4)
            # ----
            if 'dct[1]' in self.basis_type:
                x_dct1 = torch.fft.irfftn( out_ft, dim= dim_other, norm='ortho')
                x      = torch.fft.ifft(  torch.cat([x_dct1, x_dct1.flip([dim_dct])[..., 1:-1,:]], dim=dim_dct), dim = dim_dct, norm="ortho" ).real[ ...,:x.size(dim_dct),:]
            else:
                x = torch.fft.irfftn(  out_ft,  dim=[-3,-2,-1], norm='ortho' )

        elif self.nDIM == 2:
            # ------------------------------------
            if 'dct[1]' in self.basis_type:
                dim_dct, dim_other = -1, -2
                x_xflip = torch.cat ( [ x, x.flip([dim_dct])[...,1:-1] ], dim=dim_dct )
                x_dct1  = torch.fft.fft(  x_xflip,  dim=dim_dct  , norm="ortho").real[..., :x.size(dim_dct)] # 1d-dct-along-y
                x_ft    = torch.fft.rfft( x_dct1, dim=dim_other, norm="ortho")                             # 1d-rfft-along-x
                out_ft  = torch.zeros(batchsize, self.out_channels,  x.size(-2)//2 + 1,  x.size(-1), dtype=torch.cfloat, device=x.device)
                #---------
                k0, k1    = self.modes_fourier
                out_ft[:,:,  :k0,    :k1]    = einsum_op( 'bixy,xyio->boxy', x_ft[:, :,  :k0,    :k1], self.weights1 )
                out_ft[:,:,  :k0, -k1:  ]    = einsum_op( 'bixy,xyio->boxy', x_ft[:, :,  :k0, -k1: ], self.weights2 )
                x_dct1 = torch.fft.irfft( out_ft, dim= dim_other, norm='ortho')
                x      = torch.fft.ifft(  torch.cat([x_dct1, x_dct1.flip([dim_dct])[..., 1:-1]], dim=dim_dct), dim = dim_dct, norm="ortho" ).real[ ...,:x.size(dim_dct)]
            else:
                x_ft    = torch.fft.rfftn(x, dim=[-2,-1], norm="ortho")
                out_ft  = torch.zeros(batchsize, self.out_channels,  x.size(-2),  x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
                #---------
                k0, k1    = self.modes_fourier
                out_ft[:,:,    :k0 , :k1]     = einsum_op( 'bixy,xyio->boxy', x_ft[:, :,  :k0, :k1], self.weights1 )
                out_ft[:,:, -k0:   , :k1]     = einsum_op( 'bixy,xyio->boxy', x_ft[:, :,-k0:, :k1], self.weights2 )
                x      = torch.fft.irfftn( out_ft, dim=[-2,-1], norm='ortho' )

        elif self.nDIM==1:
            # ------------------------------------
            x_ft      = torch.fft.rfftn (x, dim=-1, norm="ortho")
            out_ft    = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            k0        = self.modes_fourier[0]
            out_ft[:,:,:k0] = einsum_op( 'bix,xio->box', x_ft[:, :, :k0], self.weights)
            x = torch.fft.irfftn( out_ft, dim=-1, norm='ortho' )   #Return to physical space

        return x

#-------------------------------------------
class FourierLayer_Nd(nn.Module):
    def __init__(self, in_channels, out_channels, modes_fourier, basis_type = '', bRealVersion=False ):
        super(FourierLayer_Nd, self).__init__()

        self.SpectralConv = SpectralConv_Nd(in_channels, out_channels, modes_fourier, basis_type , bRealVersion )

        if type(modes_fourier) == int:  nDIM = 1
        else:                           nDIM = len(modes_fourier)

        if   nDIM == 3:  self.w = nn.Conv3d( in_channels, out_channels, 1)
        elif nDIM == 2:  self.w = nn.Conv2d( in_channels, out_channels, 1)
        elif nDIM == 1:  self.w = nn.Conv1d( in_channels, out_channels, 1)

        return

    def forward( self, x ):
        return self.SpectralConv(x)+self.w(x)





#-----------------------------------------------------------------
#
# For debug/test of super-resolution functino (can be ignored).
#
class SuperRes_Nd(nn.Module):
    def __init__(self, nDIM, M):
        super(SuperRes_Nd, self).__init__()
        self.nDIM = nDIM
        self.M    = M
    def forward(self, x , bUpRes=True ): # x.shape=  b,(Nx,Ny), 1
        if bUpRes==True:
            if self.nDIM == 2:
                batchsize, n0, n1 = x.shape[:3]
                x_ft    = torch.fft.fftn(x, dim=[1,2], norm="ortho")
                out_ft  = torch.zeros(batchsize, self.M*n0, self.M*n1, *x.shape[3:], dtype=torch.cfloat, device=x.device)
                out_ft[:,     :n0//2,     :n1//2,...] = x_ft[:,     :n0//2,      :n1//2,...]*self.M
                out_ft[:,     :n0//2,-n1//2:    ,...] = x_ft[:,     :n0//2,-n1//2:     ,...]*self.M
                out_ft[:,-n0//2:    ,     :n1//2,...] = x_ft[:,-n0//2:    ,      :n1//2,...]*self.M
                out_ft[:,-n0//2:    ,-n1//2:    ,...] = x_ft[:,-n0//2:    ,-n1//2:     ,...]*self.M
                x = torch.fft.ifftn(  out_ft,  dim=[1,2], norm='ortho' ).real
                return x
            elif self.nDIM == 1:
                batchsize, n1  = x.shape[:2]
                x_ft    = torch.fft.fftn(x, dim=[1], norm="ortho")
                out_ft  = torch.zeros(batchsize, self.M*n1,            *x.shape[2:], dtype=torch.cfloat, device=x.device)
                out_ft[:,      :n1//2,...] = x_ft[:,      :n1//2, ...]*np.sqrt(self.M)
                out_ft[:,-n1//2:     ,...] = x_ft[:,-n1//2:     , ...]*np.sqrt(self.M)
                x = torch.fft.ifftn(  out_ft, dim=[1], norm='ortho' ).real
                return x
        elif bUpRes == False: # DownSample
            if self.nDIM == 2:
                return x[:,::self.M,::self.M,...]
            elif self.nDIM == 1:
                return x[:,::self.M         ,...]



#-------------------------------------------
#  Block containing multiple FourierLayer of fixed width 
class FourierBlock_Nd(nn.Module):
    def __init__(self, depth, width, modes_fourier, basis_type = '',
                 bUseSkipConnection=False, method_WeightSharing=False, bNonlinearForLastLayer=False, 
                 bRealVersion=False ):
        super(FourierBlock_Nd, self).__init__()

        self.bNonlinearForLastLayer = bNonlinearForLastLayer
        self.bUseSkipConnection = bUseSkipConnection
        self.method_WeightSharing = method_WeightSharing
        self.depth = depth

        self.conv = nn.ModuleList()
        for j in range(self.depth):
            if j == 0 or self.method_WeightSharing==False:
                conv_j = FourierLayer_Nd( width, width, modes_fourier, basis_type, bRealVersion )
            self.conv.append(conv_j)
        return

    def forward( self, x ):
        # for j in range(self.depth):
        #     tmp = self.conv[j](x) 
        #     if j == self.depth-1 and self.bNonlinearForLastLayer == False:
        #         x = x*self.bUseSkipConnection + tmp
        #     else:
        #         x = x*self.bUseSkipConnection + nn.GELU()(tmp)  

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
class PermuteLayer_Nd(torch.nn.Module):
    def __init__(self, nDIM, bForward ) -> None:
        super().__init__()
        self.nDIM = nDIM
        self.bForward = bForward  # True: b,(Nx,Ny),w  -> b,w,(Nx,Ny)
                                   # False: b,w,(Nx,Ny) -> b,(Nx,Ny),w    
        return
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bForward == True: 
            if self.nDIM == 1:       return input.permute( [0, 2, 1] )                
            elif self.nDIM == 2 :    return input.permute( [0, 3, 1, 2] )     # b,(Nx,Ny),w --> b,w,(Nx,Ny)
            elif self.nDIM == 3 :    return input.permute( [0, 4, 1, 2, 3] )  # b,(Nx,Ny),t,w --> b,w,(Nx,Ny),t
        else: 
            if self.nDIM == 1:       return input.permute( [0, 2, 1] )                
            elif self.nDIM == 2 :    return input.permute( [0, 2, 3, 1] )      # b,w,(Nx,Ny) -> b,(Nx,Ny),w    
            elif self.nDIM == 3 :    return input.permute( [0, 2, 3, 4, 1] )   # b,w,(Nx,Ny),t ->b,(Nx,Ny),t,w


#-------------------------------------------
class ReversibleNet(nn.Module):
    def __init__(self, in_channel=1,  width=30, modes_fourier=32, depth=2, bRealVersion=False ):
        super(ReversibleNet, self).__init__()
        
        self.modes_fourier   = modes_fourier
        self.width = width
        self.depth = depth
        
        if type(modes_fourier) == int:  nDIM = 1
        else:                           nDIM = len(modes_fourier)

        self.F_rev = nn.ModuleList()
        self.F_rev.append(  nn.Sequential(  nn.Linear( in_channel, width), nn.GELU(), 
                                            PermuteLayer_Nd( nDIM, bForward=True ), 
                                            FourierBlock_Nd(depth=self.depth, width=self.width,modes_fourier=self.modes_fourier, basis_type = '', 
                                                            bUseSkipConnection=False, method_WeightSharing=False,  bNonlinearForLastLayer=True,
                                                            bRealVersion=bRealVersion),
                                            PermuteLayer_Nd( nDIM ,bForward=False),
                                            nn.Linear( width, in_channel)    )        )

        self.F_rev.append(  nn.Sequential(  nn.Linear( in_channel, 128), nn.GELU(), 
                                            nn.Linear( 128, in_channel )          )         )
        #-------------------------------
        self.G_rev = nn.ModuleList()
        self.G_rev.append(  nn.Sequential(  nn.Linear( in_channel, width), nn.GELU(),  
                                            PermuteLayer_Nd( nDIM, bForward = True), 
                                            FourierBlock_Nd(depth=self.depth, width=self.width, modes_fourier=self.modes_fourier, basis_type = '', 
                                                            bUseSkipConnection=False, method_WeightSharing=False,  bNonlinearForLastLayer=True,
                                                            bRealVersion=bRealVersion),
                                            PermuteLayer_Nd( nDIM , bForward=False),
                                            nn.Linear( width, in_channel)     )      )
        
        self.G_rev.append(  nn.Sequential(  nn.Linear( in_channel,  128), nn.GELU(),  
                                            nn.Linear( 128, width-2*in_channel)    )      )
        
        self.G_rev.append(                  nn.Linear( width-2*in_channel, in_channel)         )
        #----------------------------

        self.permute_forward   = PermuteLayer_Nd( nDIM, bForward=True )
        self.permute_backward = PermuteLayer_Nd( nDIM, bForward=False )

    def forward(self, x, bUp = True):
        if bUp == True: # Up-lift
            a0 = x[..., :]
            a1 = x[..., :]
            b1 = a1 + self.F_rev[0]( a0 )
            b0 = a0 + self.G_rev[0]( b1 )
            
            a0 = b0
            a1 = b1
            b1 = a1 + self.F_rev[1]( a0 )
            bb =      self.G_rev[1]( b1 )

            return  self.permute_forward( torch.cat(  ( a0, b1, bb),  dim = -1 ) )
            
        else:   # Down-projection
            
            x = self.permute_backward( x )

            a0 = x [...,  :1]
            b1 = x [..., 1:2]
            bb = x [..., 2: ]

            b0 = a0 + self.G_rev[2] ( bb )

            # ---------
            a0 = b0 - self.G_rev[2]( self.G_rev[1] (b1) )
            a1 = b1 - self.F_rev[1] (a0)
            b0 = a0
            b1 = a1
            
            a0 = b0 - self.G_rev[0] (b1)
            a1 = b1 - self.F_rev[0] (a0)
            b0 = a0
            b1 = a1
            return (b0+b1)/2


###################################################################################################################################
#
#  A 'early' implemention of Koopman theory-Insired Fourier Neural Operator (kFNO), this implention contains a lot extra options for debug/test purpose.
#
class tFNO_Nd(nn.Module):
    def __init__(self, nDIM, modes_fourier, width, 
                 bReversible_Uplift_Downproj=False,
                 FourierTimeDIM = False,
                 in_channel=1, kTimeStepping = 20,
                 depth_conv={'tAdv':2,'lift':3,'proj':1,'rev':2}, 
                 method_SkipConnection = 1,  # 0 means no-skip connection
                 method_WeightSharing = False,
                 basis_type= '',
                 option_RealVersion = False,   # default using the complex version
                 out_channel=1):  # lNorm = None,    # layer norm

        super(tFNO_Nd, self).__init__()

        assert nDIM == len(modes_fourier) , "tFNO_ND: please set nDIM == len(modes_fourier)"

        self.option_RealVersion = option_RealVersion
        self.nDIM = nDIM

        #if method_outputTanh is not None:        self.method_outputTanh = method_outputTanh

        self.modes_fourier    = modes_fourier

        self.width            = width

        self.in_channel       = in_channel
        self.out_channel      = out_channel

        self.kTimeStepping     = kTimeStepping

        self.FourierTimeDIM   = FourierTimeDIM

        self.basis_type       = basis_type
        self.method_WeightSharing = method_WeightSharing
        self.method_SkipConnection = method_SkipConnection

        if method_SkipConnection == 1:       bUseSkip={'tAdv':True,'lift':True,'proj':True}
        elif method_SkipConnection == 0:     bUseSkip={'tAdv':False,'lift':False,'proj':False}
        elif method_SkipConnection == -1:    bUseSkip={'tAdv':False,'lift':True,'proj':True}


        self.depth_conv = depth_conv

        # self.depth_conv_lift = depth_conv_lift
        # self.depth_conv_proj = depth_conv_proj
        # self.linearKoopmanAdv = linearKoopmanAdv
        # self.depth_conv_tAdv = depth_conv_tAdv if self.linearKoopmanAdv==False else 1

        #-------------------------
        # self.conv_tAdv = nn.ModuleList()
        # for j in range(self.depth_conv_tAdv):
        #     conv = FourierLayer_Nd( self.width, self.width, self.modes_fourier, self.basis_type, self.option_RealVersion )
        #     self.conv_tAdv.append( conv )

        self.conv_timeAdv = FourierBlock_Nd( depth=self.depth_conv['tAdv'], width=self.width, modes_fourier=self.modes_fourier, basis_type=self.basis_type,
                                             bUseSkipConnection = bUseSkip['tAdv'], method_WeightSharing=self.method_WeightSharing, bNonlinearForLastLayer=False, 
                                             bRealVersion=self.option_RealVersion)
        #----------

        self.bReversible_Uplift_Downproj = bReversible_Uplift_Downproj

        if self.bReversible_Uplift_Downproj == True:  
            assert(in_channel == out_channel), "tFNO_Nd: in_channel should be equal to out_channel for reversible net"
            self.net_reversible = ReversibleNet( in_channel, width, modes_fourier, depth=depth_conv['rev'], bRealVersion=option_RealVersion )
        else:
            #self.fc_in  = nn.Linear(self.in_channel, self.width)
            #self.fc_out = nn.Sequential( nn.Linear(self.width, 128),    nn.ReLU(),  nn.Linear(128, self.out_channel) )

            self.net_up_lift = nn.Sequential(
                nn.Linear(self.in_channel, self.width),
                PermuteLayer_Nd( self.nDIM , bForward=True ),
                FourierBlock_Nd( depth=self.depth_conv['lift'], width=self.width, modes_fourier=self.modes_fourier, basis_type=self.basis_type,
                                 bUseSkipConnection= bUseSkip['lift'], method_WeightSharing=self.method_WeightSharing, bNonlinearForLastLayer=True, 
                                 bRealVersion=self.option_RealVersion)
            )

 
            # self.conv_lift = nn.ModuleList()
            # for j in range( self.depth_conv_lift ):
            #     if j == 0 or self.method_WeightSharing==False:
            #         conv = FourierLayer_Nd( self.width, self.width, self.modes_fourier, self.basis_type, self.option_RealVersion )
            #     self.conv_lift.append(conv)

            # #-------------------------
            # self.conv_proj = nn.ModuleList()
            # for j in range( self.depth_conv_proj ):
            #     if j == 0 or self.method_WeightSharing==False:
            #         if self.FourierTimeDIM == True: # --- Fourier transform being extended to the time dimention --
            #             conv = FourierLayer_Nd( self.width, self.width, self.modes_fourier+[self.kTimeStepping//2+1], self.basis_type+'+x[-1]', self.option_RealVersion )
            #         else:
            #             conv = FourierLayer_Nd( self.width, self.width, self.modes_fourier, self.basis_type, self.option_RealVersion )
            #     self.conv_proj.append(conv)

            if self.FourierTimeDIM == True: 
                conv_proj = FourierBlock_Nd( depth=self.depth_conv['proj'], width=self.width, modes_fourier=self.modes_fourier+[self.kTimeStepping//2+1], basis_type=self.basis_type+'+x[-1]',
                                             bUseSkipConnection= bUseSkip['proj'] , method_WeightSharing=self.method_WeightSharing, bNonlinearForLastLayer=False, 
                                             bRealVersion=self.option_RealVersion)
                permute_layer = PermuteLayer_Nd( self.nDIM+1 ,  bForward=False)
            else: 
                conv_proj = FourierBlock_Nd( depth=self.depth_conv['proj'], width=self.width, modes_fourier=self.modes_fourier, basis_type=self.basis_type,
                                             bUseSkipConnection= bUseSkip['proj'], method_WeightSharing=self.method_WeightSharing, bNonlinearForLastLayer=False, 
                                             bRealVersion=self.option_RealVersion)
                permute_layer = PermuteLayer_Nd( self.nDIM , bForward=False)

            self.net_down_proj = nn.Sequential(
                conv_proj,
                permute_layer,
                nn.Linear(self.width, 128),    nn.GELU(),  
                nn.Linear(128, self.out_channel) 
            )


        # -----
        if 'supr' in self.basis_type :
            if 'supr2' in self.basis_type   : M =2
            elif 'supr3' in self.basis_type : M =3
            elif 'supr4' in self.basis_type : M =4
            self.net_SuperRes =  SuperRes_Nd( self.nDIM, M)
        return

    # def TimeAdvance( self, x ):
    #     if self.linearKoopmanAdv==True:  # koopman
    #         x = x*self.method_SkipConnection + self.conv_tAdv[0](x) 
    #     else:                      # nonlinear extention
    #         for j in range(self.depth_conv_tAdv):
    #             tmp = self.conv_tAdv[j](x) 
    #             if j< self.depth_conv_tAdv-1:   x = x*self.method_SkipConnection + nn.GELU()(tmp)  
    #             else:                           x = x*self.method_SkipConnection + tmp
    #     return x


    def forward(self, x , p=None):

        if self.FourierTimeDIM==True:

            if 'supr' in self.basis_type:    x = self.net_SuperRes( x, True )   # b,(Nx,Ny),in -> b,(2Nx,2Ny),in

            # #--------------
            # x = self.fc_in(x)                              # b,(Nx,Ny),in -> b,(Nx,Ny),w
            # if   self.nDIM==1:  x = x.permute(0, 2, 1)
            # elif self.nDIM==2:  x = x.permute(0, 3, 1, 2)  # b,(Nx,Ny),w --> b,w,(Nx,Ny)
            # for j in range(self.depth_conv_lift):
            #     x = x*self.method_SkipConnection + nn.GELU()( self.conv_lift[j](x) )
            # #--------------

            x = self.net_up_lift(x)

            x_wt = torch.zeros( *x.shape, self.kTimeStepping, device=x.device ) # b,w,(Nx,Ny),t

            for t in range(self.kTimeStepping):
                
                x = self.conv_timeAdv(x)   #x = self.TimeAdvance(x)
                x_wt[...,t] = x


            # #--------------
            # # --- Fourier transform being extended to the time dimention --
            # for j in range(self.depth_conv_proj):
            #     tmp = self.conv_proj[j](x_wt)   
            #     if j<self.depth_conv_proj-1:    x_wt = x_wt*self.method_SkipConnection + nn.GELU()(tmp)  
            #     else:                           x_wt = x_wt*self.method_SkipConnection + tmp
            # if   self.nDIM==1:  x_wt = x_wt.permute(0, 2, 3, 1 )   # b,w,(Nx),t ->b,(Nx),t,w
            # elif self.nDIM==2:  x_wt = x_wt.permute(0, 2, 3, 4, 1) # b,w,(Nx,Ny),t ->b,(Nx,Ny),t,w
            # x_ot = self.fc_out(x_wt)     # b,(Nx,Ny),t,w   -> b,(Nx,Ny),t,out
            # #--------------
            
            x_ot = self.net_down_proj(x_wt)
            x_ot = x_ot.transpose(-1,-2) # b,(Nx,Ny),t,out -> b,(Nx,Ny),out,t

            if 'supr' in self.basis_type:     x = self.net_SuperRes( x, False )  # b,(2Nx,2Ny),out,t -> b,(Nx,Ny),out,t

        elif self.FourierTimeDIM==False:

            x_ot = torch.zeros( *x.shape[:-1],self.out_channel, self.kTimeStepping, device=x.device ) # b,(Nx,Ny),out,t

            if 'supr' in self.basis_type:      x =  self.net_SuperRes(x,True)     # b,(Nx,Ny),in -> b,(2Nx,2Ny),in


            if self.bReversible_Uplift_Downproj == True:
                x = self.net_reversible(x, bUp = True)
            else:
                # #--------------
                # x = self.fc_in(x)                             # b,(Nx,Ny),in -> b,(Nx,Ny),w
                # if   self.nDIM==1:  x = x.permute(0, 2, 1)
                # elif self.nDIM==2:  x = x.permute(0, 3, 1, 2) # b,(Nx,Ny),w -> b,w,(Nx,Ny)
                # for j in range(self.depth_conv_lift):
                #     x = x*self.method_SkipConnection + nn.GELU()( self.conv_lift[j](x) )
                # #--------------

                x = self.net_up_lift(x)

            for t in range(self.kTimeStepping):
                
                x = self.conv_timeAdv(x)   # x = self.TimeAdvance(x)

                if self.bReversible_Uplift_Downproj == True:
                    u = self.net_reversible(x, bUp = False)
                else:
                    # #--------------
                    # for j in range(self.depth_conv_proj):
                    #     tmp = self.conv_proj[j](u)  
                    #     if j<self.depth_conv_proj-1:      u = u*self.method_SkipConnection + nn.GELU()(tmp)
                    #     else:                             u = u*self.method_SkipConnection + tmp
                    # if self.nDIM == 1:  u = u.permute(0, 2, 1 )
                    # elif self.nDIM==2:  u = u.permute(0, 2, 3, 1 )  # b,w,(Nx,Ny) -> b,(Nx,Ny),w
                    # u = self.fc_out(u)                              # b,(Nx,Ny),w -> b,(Nx,Ny),out
                    # #--------------

                    u = self.net_down_proj(x)


                if 'supr' in self.basis_type:   u =  self.net_SuperRes(u,False)    # b,(2Nx,2Ny),out->  b,(Nx,Ny),out

                x_ot[...,t] = u

        if self.kTimeStepping == 1:             x_ot = x_ot.view( *x_ot.shape[:-1] )

        return x_ot




#------------------------------------------------------------------------------
#
#  The following are debuging/testing codes not disussed in the paper of
#  [Yu, R., Herbert, M., Klein, M. and Hodzic, E., 2024. Koopman Theory-Inspired Method for Learning Time Advancement Operators in Unstable Flame Front Evolution. arXiv preprint arXiv:2412.08426.]
#
#



#
# def mm_ConvNd(nDIM,i,o,nFilter=3,padding=1):
#     if nDIM==1:          return nn.Conv1d(i,o,nFilter,padding_mode='circular',padding=padding )
#     elif nDIM==2:        return nn.Conv2d(i,o,nFilter,padding_mode='circular',padding=padding)
#     else:                raise ValueError('mm_ConvNd: nDIM='+str(nDIM) )

def mm_MaxPoolNd(nDIM):
    if nDIM==1:        return nn.MaxPool1d(kernel_size=2, stride=2)
    elif nDIM==2:      return nn.MaxPool2d(kernel_size=2, stride=2)
    else:              raise ValueError('mm_MaxPoolNd: nDIM='+str(nDIM) )

class ResN(nn.Module):
    def __init__(self,nDIM,i, o ,bRelu):
        super(ResN, self).__init__()
        self.net = mm_ConvNd(nDIM,i,o,3)
        self.bRelu = bRelu
    def forward(self,x):
        if self.bRelu: return x + nn.GELU()(self.net(x))
        else:          return x +           self.net(x)

class CNull(nn.Module):
    def __init__(self):     super(CNull, self).__init__()
    def forward(self,x):    return x

class kUFNO_Nd(nn.Module):
    def __init__(self, nDIM=1,in_channel=1, out_channel=1, kTimeStepping =20 ):
        super(kUFNO_Nd, self).__init__()
        self.nDIM = nDIM
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kTimeStepping = kTimeStepping
        #----------
        PoolNd = mm_MaxPoolNd(nDIM)
        ConvNd = lambda i,o, modes_fourier: FourierLayer_Nd(i,o,modes_fourier)

        #----------
        en_l = [ [       ConvNd(in_channel,8,[32]), nn.ReLU()], #64
                 [PoolNd,ConvNd(8,16,[16]), nn.ReLU()],         #32
                 [PoolNd,ConvNd(16,32,[8]), nn.ReLU()],         #16
                 [PoolNd,ConvNd(32,64,[4]), nn.ReLU()],              #8
                 [PoolNd                             ] ]             #4

        de_l = [ [ConvNd(64,64,[2]), nn.ReLU(), nn.Upsample(scale_factor=2)] ,
                 [ConvNd(2*64,32,[4]),nn.ReLU(),nn.Upsample(scale_factor=2)],
                 [ConvNd(2*32,16,[8]), nn.ReLU(), nn.Upsample(scale_factor=2)], #8
                 [ConvNd(2*16,8,[16]),  nn.ReLU(),  nn.Upsample(scale_factor=2)],
                 [ConvNd(2*8,out_channel,[32] )  ]  ]#2

        tAdv_l =[[ConvNd(8,8,[32]), nn.ReLU(), ConvNd(8,8,[32])  ] , #256
                 [ConvNd(16,16,[16]),nn.ReLU(),ConvNd(16,16,[16])], #128
                 [ConvNd(32,32,[8]),nn.ReLU(), ConvNd(32,32,[8]) ],  #64
                 [ConvNd(64,64,[4]),nn.ReLU(), ConvNd(64,64,[4]) ],    #4
                 [ConvNd(64,64,[2]),nn.ReLU(), ConvNd(64,64,[2]) ] ] #2

        self.en_conv= nn.ModuleList()
        self.de_conv= nn.ModuleList()
        self.TimeAdv_conv= nn.ModuleList()
        for l in en_l:            self.en_conv.append( nn.Sequential( *l ) )
        for l in de_l:            self.de_conv.append( nn.Sequential( *l ) )
        for l in tAdv_l:          self.TimeAdv_conv.append( nn.Sequential( *l ) )
        return

    def encoder(self,x):
        x_all = []
        for l, conv in enumerate( self.en_conv ):
            x_en = conv(x)
            x_all.append(x_en)
            x = x_en
        return x_all

    def decoder(self, x_all ):
        #x_all = x_all[::-1]  #reverse
        x = x_all[-1]
        for l, conv in enumerate(self.de_conv):
            if l==0:
                x = conv( x )
            else:  #  and  l-1<len(x_all) :

                #print('l',l, 'x.shape', x.shape,'x_all[-1].shape', x_all[-1].shape, 'x_all[-1-l].shape', x_all[-1-l].shape, 'conv', conv)

                x = torch.cat( (x, x_all[-1-l]), dim=-1-self.nDIM )

                #print('l',l, 'x.shape', x.shape,'x_all[-1].shape', x_all[-1].shape, 'x_all[-1-l].shape', x_all[-1-l].shape, 'conv', conv)

                x = conv( x )
        return x

    def TimeAdv(self, x_all ):
        for j, (x, adv_conv) in enumerate( zip(x_all, self.TimeAdv_conv) ):
            if j>= -99: #  len(x_all)-2: #-99
                x_all[j] = adv_conv(x)
        return x_all

    def forward(self, x , p=None):
        '''
            x.shape = b, c, Nx, Ny
            p.shape = b, num_PDEParameters (if not None)
        '''
        batchsize= x.shape[0]

        x_h = torch.zeros( *x.shape[:-1], self.out_channel, self.kTimeStepping, device=x.device ) # b,(Nx,Ny),out,t

        if self.nDIM==1:    x = x.permute(0, 2, 1)
        elif self.nDIM==2:  x = x.permute(0, 3, 1, 2)

        x_all = self.encoder( x )

        for i in range(self.kTimeStepping):

            if i>=0:
                x_all = self.TimeAdv(x_all)

            x = self.decoder(x_all)

            if self.nDIM==1:    x = x.permute(0, 2, 1)
            elif self.nDIM==2:  x = x.permute(0, 2, 3, 1)

            if self.kTimeStepping == 1:
                return x

            x_h[...,i] = x
        return x_h
