
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
#  This file of 'kFNO_Nd.py' is a 'cleaner' implemention of (N-dimentional) 'Koopman theory-inspired Fourier Neural Operator' (kFNO) 
#  A second (early) implementaion of kFNO can be found in the file 'tFNO_Nd.py' which contains more options (for debug/test purpose), 
#  'tFNO_Nd' can reduces to 'kFNO_Nd' under proper choices of parameters. 
#
#  [Yu, R., Herbert, M., Klein, M. and Hodzic, E., 2024. Koopman Theory-Inspired Method for Learning Time Advancement Operators in Unstable Flame Front Evolution. arXiv preprint arXiv:2412.08426.]
#
# --------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
#  First, a few utilie functions to be used in the followed main class of 'kFNO_Nd'

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


###################################################
#  A 'cleaner' implemention of Koopman theory-Insired Fourier Neural Operator (kFNO)
#-----------------------------------------------------------
class kFNO_Nd(nn.Module):
    def __init__(self, nDIM, modes_fourier, width,
                 linearKoopmanAdv=False, FourierTimeDIM = False,
                 in_channel=1, kTimeStepping = 20,
                 method_WeightSharing = False,
                 method_SkipConnection = 1,  # 0 means no-skip connection
                 option_RealVersion = False,   # default using the complex version
                 method_outputTanh    =  None,
                 basis_type= '',
                 depth_proj=4, depth_tAdv=2,out_channel=1):  # lNorm = None,    # layer norm

        super(kFNO_Nd, self).__init__()

        assert nDIM == len(modes_fourier) , "kFNO_ND: please set nDIM == len(modes_fourier)"

        self.option_RealVersion = option_RealVersion
        self.nDIM = nDIM

        if method_outputTanh is not None:        self.method_outputTanh = method_outputTanh

        self.modes_fourier    = modes_fourier

        self.width            = width
        self.in_channel       = in_channel
        self.out_channel      = out_channel
        self.kTimeStepping     = kTimeStepping
        self.FourierTimeDIM   = FourierTimeDIM
        self.basis_type       = basis_type
        self.method_WeightSharing = method_WeightSharing
        self.method_SkipConnection = method_SkipConnection

        self.depth_proj = depth_proj

        self.linearKoopmanAdv = linearKoopmanAdv
        self.depth_tAdv = depth_tAdv if self.linearKoopmanAdv==False else 1

        #----------
        self.fc_in  = nn.Linear(self.in_channel, self.width)
        #self.fc_in  = nn.Sequential(nn.Linear(self.in_channel, self.width),  nn.ReLU(),  nn.Linear(self.width, self.width) )
        self.fc_out = nn.Sequential( nn.Linear(self.width, 128),    nn.ReLU(),  nn.Linear(128, self.out_channel) )
        #---------

        #-------------------------
        self.conv0 = FourierLayer_Nd( self.width, self.width, self.modes_fourier, self.basis_type, self.option_RealVersion )

        #-------------------------
        self.conv_tAdv = nn.ModuleList()
        for j in range(self.depth_tAdv):
            conv = FourierLayer_Nd( self.width, self.width, self.modes_fourier, self.basis_type, self.option_RealVersion )
            self.conv_tAdv.append( conv )

        #-------------------------
        self.conv_proj = nn.ModuleList()
        for j in range( self.depth_proj ):
            if j == 0 or self.method_WeightSharing==False:
                if self.FourierTimeDIM == True: # --- Fourier transform being extended to the time dimention --
                    conv =  FourierLayer_Nd( self.width, self.width, self.modes_fourier+[self.kTimeStepping//2+1], self.basis_type+'+x[-1]', self.option_RealVersion )
                else:
                    conv = FourierLayer_Nd( self.width, self.width, self.modes_fourier, self.basis_type, self.option_RealVersion )

            self.conv_proj.append(conv)

        # -----
        if 'supr' in self.basis_type :
            if 'supr2' in self.basis_type   : M =2
            elif 'supr3' in self.basis_type : M =3
            elif 'supr4' in self.basis_type : M =4
            self.net_SuperRes =  SuperRes_Nd( self.nDIM, M)
        return

    def TimeAdvance( self, x ):

        if self.linearKoopmanAdv==True:  # koopman
            x = x*self.method_SkipConnection + self.conv_tAdv[0](x) 

        else:                      # nonlinear extention
            for j in range(self.depth_tAdv):
                
                tmp = self.conv_tAdv[j](x) 

                if j< self.depth_tAdv-1:
                    x = x*self.method_SkipConnection + nn.GELU()(tmp)  
                else:
                    x = x*self.method_SkipConnection + tmp
        return x

    def forward(self, x , p=None):
        if not hasattr(self,'method_SkipConnection'): 
            self.method_SkipConnection = True             # default value
        
        if self.FourierTimeDIM==True:

            if 'supr' in self.basis_type:    x = self.net_SuperRes( x, True )   # b,(Nx,Ny),in -> b,(2Nx,2Ny),in

            x = self.fc_in(x)                              # b,(Nx,Ny),in -> b,(Nx,Ny),w
            if   self.nDIM==1:  x = x.permute(0, 2, 1)
            elif self.nDIM==2:  x = x.permute(0, 3, 1, 2) # b,(Nx,Ny),w --> b,w,(Nx,Ny)
            #--------------
            x = x*self.method_SkipConnection + nn.GELU()( self.conv0(x) )
            #--------------

            x_o = torch.zeros( *x.shape, self.kTimeStepping, device=x.device ) # b,w,(Nx,Ny),t
            for t in range(self.kTimeStepping):
                if t > 0 :
                    x = self.TimeAdvance(x)
                x_o[...,t] = x

            # --- Fourier transform being extended to the time dimention --
            for j in range(self.depth_proj):
                
                tmp = self.conv_proj[j](x_o)   

                if j<self.depth_proj-1:
                    x_o = x_o*self.method_SkipConnection + nn.GELU()(tmp)  
                else: 
                    x_o = x_o*self.method_SkipConnection + tmp

            #----------------------
            if   self.nDIM==1:  x_o = x_o.permute(0, 2, 3, 1 )   # b,w,(Nx),t ->b,(Nx),t,w
            elif self.nDIM==2:  x_o = x_o.permute(0, 2, 3, 4, 1) # b,w,(Nx,Ny),t ->b,(Nx,Ny),t,w

            x_o = self.fc_out(x_o)    # b,(Nx,Ny),t,w   -> b,(Nx,Ny),t,out
            x_o = x_o.transpose(-1,-2) # b,(Nx,Ny),t,out -> b,(Nx,Ny),out,t

            if 'supr' in self.basis_type:     x = self.net_SuperRes( x, False )  # b,(2Nx,2Ny),out,t -> b,(Nx,Ny),out,t

        elif self.FourierTimeDIM==False:

            x_o = torch.zeros( *x.shape[:-1],self.out_channel, self.kTimeStepping, device=x.device ) # b,(Nx,Ny),out,t

            if 'supr' in self.basis_type:      x =  self.net_SuperRes(x,True)     # b,(Nx,Ny),in -> b,(2Nx,2Ny),in

            x = self.fc_in(x)                              # b,(Nx,Ny),in -> b,(Nx,Ny),w
            if   self.nDIM==1:  x = x.permute(0, 2, 1)
            elif self.nDIM==2:  x = x.permute(0, 3, 1, 2) # b,(Nx,Ny),w -> b,w,(Nx,Ny)

            x = x*self.method_SkipConnection + nn.GELU()( self.conv0(x) )

            for t in range(self.kTimeStepping):
                if t > 0 :
                    x = self.TimeAdvance(x)

                u = x
                for j in range(self.depth_proj):
                    
                    tmp = self.conv_proj[j](u)  
                    
                    if j<self.depth_proj-1:
                        u = u*self.method_SkipConnection + nn.GELU()(tmp)
                    else: 
                        u = u*self.method_SkipConnection + tmp

                if self.nDIM == 1:  u = u.permute(0, 2, 1 )
                elif self.nDIM==2:  u = u.permute(0, 2, 3, 1 )  # b,w,(Nx,Ny) -> b,(Nx,Ny),w
                u = self.fc_out(u)                              # b,(Nx,Ny),w -> b,(Nx,Ny),out

                if 'supr' in self.basis_type:   u =  self.net_SuperRes(u,False)    # b,(2Nx,2Ny),out->  b,(Nx,Ny),out

                x_o[...,t] = u

        if self.kTimeStepping == 1:             x_o = x_o.view( *x_o.shape[:-1] )
        if hasattr(self,'method_outputTanh'):   x_o = torch.tanh( x_o )



        return x_o


#------------------------------------------------------------------------------
#
#  The following are debuging/testing codes not disussed in the paper of
#  [Yu, R., Herbert, M., Klein, M. and Hodzic, E., 2024. Koopman Theory-Inspired Method for Learning Time Advancement Operators in Unstable Flame Front Evolution. arXiv preprint arXiv:2412.08426.]
#
#



#--------------------------------------------------------------------
#########################################################################
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
