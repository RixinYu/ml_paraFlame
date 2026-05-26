
#-----------------------------------------------------
#from utilities3 import *
import operator
from functools import reduce
#from torch.autograd import Variable
#-----------------------------------------------------


import torch
import torch.nn as nn
from functools import partial


from flame_net.RevtFNO_Nd import SpectralConv_MatrixExp_Nd


# --------------------------------------------------------------------------------------------------------------------------
#
#  Class 'tFNO_Nd' is the implemention of (N-dimentional) 'Koopman theory-inspired Fourier Neural Operator' (kFNO), 
#
#     [Yu, R., Herbert, M., Klein, M. and Hodzic, E., 2024. 'Koopman Theory-Inspired Method for Learning Time Advancement Operators in Unstable Flame Front Evolution', arXiv:2412.08426. accepted for publication in Physics of Fluids]
#     [Yu, R., Herbert, M., Klein, M. and Hodzic, E., 2025. 'Koopman-Inspired Operator Learning for Intrinsic Flame Instabilities', presented in 1st International Symposium on AI and Fluid Mechanics (AIFLUIDs), submitted for publication in Computer & Fluids]
#  

# ---------------------------------------------------------------------------------
#  First, a few utilie functions to be used in the followed main class of 'tFNO_Nd'

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

        torch_cfloat = torch.complex128 if torch.get_default_dtype() == torch.float64 else torch.complex64


        if type(modes_fourier) == int:  self.nDIM = 1
        else:                           self.nDIM = len(modes_fourier)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_fourier = modes_fourier #Number of Fourier modes to multiply, at most floor(N/2) + 1

        scale = 1 / (in_channels * out_channels)

        self.basis_type = basis_type
        self.bRealVersion = bRealVersion

        if '+x[-1]' in self.basis_type:  self.ratio_cord = nn.Parameter( 0.5*torch.rand(in_channels) )

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
                self.weights1 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch_cfloat ))
                self.weights2 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch_cfloat ))
                self.weights3 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch_cfloat ))
                self.weights4 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch_cfloat ))
            elif self.nDIM == 2:
                self.weights1 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch_cfloat ) )
                self.weights2 = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels, dtype= torch_cfloat ) )
            elif self.nDIM==1:
                self.weights  = nn.Parameter(scale * torch.rand( *modes_fourier, in_channels, out_channels,  dtype=torch_cfloat) )
        return

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f" {self.in_channels}, {self.out_channels}"
                f"  m:{self.modes_fourier}"
                f"  basis_type:{self.basis_type}")

    
    def forward(self, x ): # x.shape=  b,w,(Nx,Ny)

        torch_cfloat = torch.complex128 if torch.get_default_dtype() == torch.float64 else torch.complex64

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
                
                # x_xflip = torch.cat ( [ x, x.flip([dim_dct])[...,1:-1,:] ], dim=dim_dct )
                # x_dct1  = torch.fft.fft(   x_xflip,  dim=dim_dct  , norm="ortho").real[..., :x.size(dim_dct),:] # 1d-dct-along-z
                x_dct1  = torch.fft.hfft( x , dim=dim_dct , norm="ortho")[...,:x.size(dim_dct),:] # 1d-dct-along-z, implemented using hermitian-fft

                x_ft    = torch.fft.rfftn( x_dct1, dim=dim_other, norm="ortho")                               # 1d-rfft-along-xy
                out_ft  = torch.zeros(batchsize, self.out_channels,  x.size(-3), x.size(-2),  x.size(-1)//2 + 1, dtype=torch_cfloat, device=x.device)
            else:
                x_ft    = torch.fft.rfftn(x, dim=[-3,-2,-1], norm="ortho")
                out_ft  = torch.zeros(batchsize, self.out_channels,  x.size(-3), x.size(-2),  x.size(-1)//2 + 1, dtype=torch_cfloat, device=x.device)
            # ----
            k0, k1, k2 = self.modes_fourier
            out_ft[:, :,   :k0,   :k1, :k2] = einsum_op( 'bixyz,xyzio->boxyz', x_ft[:,:,   :k0,   :k1, :k2], self.weights1)
            out_ft[:, :,-k0:  ,   :k1, :k2] = einsum_op( 'bixyz,xyzio->boxyz', x_ft[:,:,-k0:  ,   :k1, :k2], self.weights2)
            out_ft[:, :,   :k0,-k1:  , :k2] = einsum_op( 'bixyz,xyzio->boxyz', x_ft[:,:,   :k0,-k1:  , :k2], self.weights3)
            out_ft[:, :,-k0:  ,-k1:  , :k2] = einsum_op( 'bixyz,xyzio->boxyz', x_ft[:,:,-k0:  ,-k1:  , :k2], self.weights4)
            # ----
            if 'dct[1]' in self.basis_type:
                x_dct1 = torch.fft.irfftn( out_ft, dim= dim_other, norm='ortho')
                
                # x      = torch.fft.ifft(  torch.cat([x_dct1, x_dct1.flip([dim_dct])[..., 1:-1,:]], dim=dim_dct), dim = dim_dct, norm="ortho" ).real[ ...,:x.size(dim_dct),:]
                x      = torch.fft.irfft(  x_dct1, dim = dim_dct, norm="ortho" )[ ...,:x.size(dim_dct),:]  # 1d-inverse-dct, implemented using irfft 


            else:
                x = torch.fft.irfftn(  out_ft,  dim=[-3,-2,-1], norm='ortho' )

        elif self.nDIM == 2:
            # ------------------------------------
            if 'dct[1]' in self.basis_type:
                dim_dct, dim_other = -1, -2
                
                # x_xflip = torch.cat ( [ x, x.flip([dim_dct])[...,1:-1] ], dim=dim_dct )
                # x_dct1  = torch.fft.fft(  x_xflip,  dim=dim_dct  , norm="ortho").real[..., :x.size(dim_dct)] # 1d-dct-along-y
                x_dct1  = torch.fft.hfft( x , dim=dim_dct , norm="ortho")[...,:x.size(dim_dct)] # 1d-dct-along-y, implemented using hermitian-fft

                x_ft    = torch.fft.rfft( x_dct1, dim=dim_other, norm="ortho")                             # 1d-rfft-along-x
                out_ft  = torch.zeros(batchsize, self.out_channels,  x.size(-2)//2 + 1,  x.size(-1), dtype=torch_cfloat, device=x.device)
                #---------
                k0, k1    = self.modes_fourier
                out_ft[:,:,  :k0,    :k1]    = einsum_op( 'bixy,xyio->boxy', x_ft[:, :,  :k0,    :k1], self.weights1 )
                out_ft[:,:,  :k0, -k1:  ]    = einsum_op( 'bixy,xyio->boxy', x_ft[:, :,  :k0, -k1: ], self.weights2 )
                
                x_dct1 = torch.fft.irfft( out_ft, dim= dim_other, norm='ortho')
                # x      = torch.fft.ifft(  torch.cat([x_dct1, x_dct1.flip([dim_dct])[..., 1:-1]], dim=dim_dct), dim = dim_dct, norm="ortho" ).real[ ...,:x.size(dim_dct)]
                x      = torch.fft.irfft(  x_dct1, dim = dim_dct, norm="ortho" )[ ...,:x.size(dim_dct)] # 1d-inverse-dct, implemented using irfft                 

            else:
                x_ft    = torch.fft.rfftn(x, dim=[-2,-1], norm="ortho")
                out_ft  = torch.zeros(batchsize, self.out_channels,  x.size(-2),  x.size(-1)//2 + 1, dtype=torch_cfloat, device=x.device)
                #---------
                k0, k1    = self.modes_fourier
                out_ft[:,:,    :k0 , :k1]     = einsum_op( 'bixy,xyio->boxy', x_ft[:, :,  :k0, :k1], self.weights1 )
                out_ft[:,:, -k0:   , :k1]     = einsum_op( 'bixy,xyio->boxy', x_ft[:, :,-k0:,  :k1], self.weights2 )
                x      = torch.fft.irfftn( out_ft, dim=[-2,-1], norm='ortho' )

        elif self.nDIM==1:
            # ------------------------------------
            x_ft      = torch.fft.rfftn (x, dim=-1, norm="ortho")
            out_ft    = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1, dtype=torch_cfloat, device=x.device)
            k0        = self.modes_fourier[0]


            #print('x_ft.dtype=', x_ft.dtype,   ' self.weights.dtype=', self.weights.dtype, 'torch_cfloat=', torch_cfloat, 'torch.get_default_dtype()=', torch.get_default_dtype() )

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



###################################################################################################################################
#
#  The implemention of Koopman theory-Insired Fourier Neural Operator (kFNO), 
#
#  Example of usage: 
#  net = tFNO_Nd(nDIM=1, modes_fourier=[30], width=32, FourierTimeDIM=False, in_channel=1, kTimeStepping=20,depth_conv={'tAdv':1,'lift':3,'proj':1,'rev':2,'tAdv_last_nonlinear':False}, method_SkipConnection = 1)
#
class tFNO_Nd(nn.Module):
    def __init__(self, nDIM, modes_fourier, width, 
                 # bReversible_Uplift_Downproj=False, # deprecated, this feature is moved to 'RevtFNO_Nd' (IS-FNO) instead 
                 FourierTimeDIM = False,
                 in_channel=1, kTimeStepping = 20,
                 depth_conv={'tAdv':2,'lift':3,'proj':1,'rev':2,'tAdv_last_nonlinear':False}, 
                 method_SkipConnection = 1,  # 0 means no-skip connection
                 method_WeightSharing = False,
                 basis_type= '',
                 option_RealVersion = False,   # default using the complex version
                 out_channel=1):  

        super(tFNO_Nd, self).__init__()

        assert nDIM == len(modes_fourier) , "tFNO_ND: please set nDIM == len(modes_fourier)"

        self.option_RealVersion = option_RealVersion
        self.nDIM = nDIM

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

        #----------------
        if 'tAdv_last_nonlinear' not in self.depth_conv:   
            self.depth_conv['tAdv_last_nonlinear'] = False  # add new default key

        if self.depth_conv['tAdv'] == 0:
            self.conv_timeAdv = nn.Identity()

        elif self.depth_conv['tAdv'] == 1 and 'exp' in self.depth_conv['tAdv_basis']:

            self.conv_timeAdv =  SpectralConv_MatrixExp_Nd( self.width, modes_fourier=modes_fourier, basis_type= self.depth_conv['tAdv_basis'] )

        else:
            self.conv_timeAdv = FourierBlock_Nd( depth=self.depth_conv['tAdv'], width=self.width, modes_fourier=self.modes_fourier, basis_type=self.basis_type,
                                                bUseSkipConnection = bUseSkip['tAdv'], method_WeightSharing=self.method_WeightSharing, bNonlinearForLastLayer=self.depth_conv['tAdv_last_nonlinear'], 
                                                bRealVersion=self.option_RealVersion)

        # ----------------------------------------------------------------
        self.net_up_lift = nn.Sequential(
            nn.Linear(self.in_channel, self.width),
            PermuteLayer_Nd( self.nDIM , bForward=True ),
            FourierBlock_Nd( depth=self.depth_conv['lift'], width=self.width, modes_fourier=self.modes_fourier, basis_type=self.basis_type,
                                bUseSkipConnection= bUseSkip['lift'], method_WeightSharing=self.method_WeightSharing, bNonlinearForLastLayer=True, 
                                bRealVersion=self.option_RealVersion)
        )

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

        return



    def forward(self, x , p=None):

        if self.FourierTimeDIM==True:

            x = self.net_up_lift(x)

            x_wt = torch.zeros( *x.shape, self.kTimeStepping, device=x.device ) # b,w,(Nx,Ny),t

            for t in range(self.kTimeStepping):
                
                x = self.conv_timeAdv(x)   #x = self.TimeAdvance(x)
                x_wt[...,t] = x

            
            x_ot = self.net_down_proj(x_wt)
            x_ot = x_ot.transpose(-1,-2) # b,(Nx,Ny),t,out -> b,(Nx,Ny),out,t


        elif self.FourierTimeDIM==False:
            #-----
            t, b, nxny, ch = self.kTimeStepping, x.shape[0], x.shape[1:-1], x.shape[-1]
            #-----

            x_ot = torch.zeros( *x.shape[:-1],self.out_channel, self.kTimeStepping, device=x.device ) # b,(Nx,Ny),out,t

            x = self.net_up_lift(x)


            if isinstance(self.conv_timeAdv, SpectralConv_MatrixExp_Nd):
                #---------------
                x__tbcn  = self.conv_timeAdv(x,  kStep = torch.arange(t).to(x.device)+1  ).reshape( t*b,   -1, *nxny )  
                x        = self.net_down_proj(x__tbcn).reshape( t,b,*nxny,    ch  )
                if   self.nDIM == 1: x = x.permute(1,2,3,0)       # err_u , err_scat_coef #err_x_ot
                elif self.nDIM == 2: x = x.permute(1,2,3,4,0)
                if self.kTimeStepping == 1:   x = x.view( *x.shape[:-1] )
                return x
                #---------------

            else:

                for t in range(self.kTimeStepping):
                    x = self.conv_timeAdv(x)   # x = self.TimeAdvance(x)

                    u = self.net_down_proj(x)
                    x_ot[...,t] = u


        if self.kTimeStepping == 1:             x_ot = x_ot.view( *x_ot.shape[:-1] )

        return x_ot


