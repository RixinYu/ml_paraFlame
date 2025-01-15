
#-----------------------------------------------------
#from utilities3 import *
import operator
from functools import reduce
#from torch.autograd import Variable
#-----------------------------------------------------


import torch
import torch.nn as nn
from functools import partial
from flame_net.MyConvNd import Fc_PDEpara,  my_NormNd
import numpy as np

def p_rescale_nu(p):
    return (1 - (0.025/p)**1.5 )+0.2


#-----------------------------------------------------
#  This file implements the Paramertical Fourier Neural Operator (pFNO) for solving PDEs
#
#    [Yu, R., & Hodzic, E. (2024). Parametric learning of time-advancement operators for unstable flame evolution. Physics of Fluids, 36(4).]
#    [Yu, R., Hodzic, E., & Nogenmyr, K. J. (2024). Learning Flame Evolution Operator under Hybrid Darrieus Landau and Diffusive Thermal Instability. Energies, 17(13), 3097.]    
#-----------------------------------------------------



################################################################
# require torch.__version__ > 1.12   # allow einsum for complex number
################################################################
#Complex multiplication
# (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
def compl2_einsum(op_einsum, a, b):
    op = partial(torch.einsum, op_einsum )
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

def compl3_einsum(op_einsum, a, b, s):  # s is real
    op3 = partial(torch.einsum, op_einsum )
    return torch.stack([
        op3(a[..., 0], b[..., 0],s) - op3(a[..., 1], b[..., 1],s),
        op3(a[..., 1], b[..., 0],s) + op3(a[..., 0], b[..., 1],s)
    ], dim=-1)

class LinearNet_Parameters(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_Parameters=0):
        super(LinearNet_Parameters, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_Parameters = num_Parameters

        self.net = nn.Linear(num_inputs, num_outputs, bias = True)
        if self.num_Parameters > 0:
            self.net_Parameters= nn.Linear( num_Parameters, num_outputs, bias=False )

    # shape: [b,256,401,T_in], [b,1,1,num_Parameters]
    def forward(self, x):
        if type(x) == tuple:
            x, paras = x
        x = self.net(x)
        if self.num_Parameters > 0:
            x = x + self.net_Parameters( paras )
        return x

#def torch.einsum( op_einsum, a_compl, b ):
#    return torch.einsum( op_einsum, a_compl, b )

#-----------------------------------------------------------------

class SpectralConv_Nd(nn.Module):
    def __init__(self, in_channels, out_channels, modes_fourier , Use_2d_DCT = False ):
        super(SpectralConv_Nd, self).__init__()
        if type(modes_fourier) == int:           self.nDIM = 1
        else:                                    self.nDIM = 2  # assert len(modes_fourier)==2

        """
        Nd Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_fourier = modes_fourier #Number of Fourier modes to multiply, at most floor(N/2) + 1
        scale = 1 / (in_channels * out_channels)

        if Use_2d_DCT == True:
            self.Use_2d_DCT = True

        if self.nDIM == 2:
            self.weights_dim0 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, dtype= torch.cfloat ) )
            self.weights_dim1 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, dtype= torch.cfloat ) )
        elif self.nDIM==1:
            self.weights = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_fourier, dtype=torch.cfloat) )
        return

    def forward(self, x ):
        '''
            x.shape = b, c, Nx, Ny
        '''
        batchsize = x.shape[0]
        if self.nDIM == 2:
            if hasattr(self, 'Use_2d_DCT'):
                dim_dct = -2
                dim_other = -1
                x__cat__xflip = torch.cat ( [ x, x.flip([dim_dct])[...,1:-1,:] ], dim=dim_dct )
                x__1d_dct = torch.fft.fft( x__cat__xflip , dim=dim_dct,  norm="ortho").real[...,:x.size(dim_dct),:]   # 1d-dct-along-y
                x_ft  = torch.fft.rfft ( x__1d_dct , dim= dim_other , norm="ortho")                     # 1d-rfft-along-x
                out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),   x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            else:
                x_ft   = torch.fft.rfft2(x, norm="ortho")
                out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),  x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        elif self.nDIM==1:
            x_ft   = torch.fft.rfft (x, norm="ortho")
            out_ft = torch.zeros(batchsize, self.out_channels,               x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        if self.nDIM==2:
            k0, k1 = self.modes_fourier
            out_ft[:,:, :k0 , :k1] = torch.einsum( 'bixy,ioxy->boxy', x_ft[:, :, :k0, :k1], self.weights_dim0 )
            out_ft[:,:, -k0:, :k1] = torch.einsum( 'bixy,ioxy->boxy', x_ft[:, :,-k0:, :k1], self.weights_dim1 )
        elif self.nDIM==1:
            k0 = self.modes_fourier
            out_ft[:,:,:k0] = torch.einsum( 'bix,iox->box', x_ft[:, :, :k0], self.weights)

        #Return to physical space
        if   self.nDIM == 2:
            if hasattr(self, 'Use_2d_DCT'):
                x__1d_dct = torch.fft.irfft( out_ft, dim= dim_other, norm='ortho')
                x         = torch.fft.ifft(  torch.cat([x__1d_dct, x__1d_dct.flip([dim_dct])[..., 1:-1,:]], dim=dim_dct), dim = dim_dct, norm="ortho" ).real[ ...,:x.size(dim_dct),:]
            else:
                x = torch.fft.irfft2( out_ft, (x.size(-2), x.size(-1)), norm='ortho' )
        elif self.nDIM == 1:
            x = torch.fft.irfft(  out_ft,                x.size(-1), norm='ortho' )
        return x

#-----------------------------------------------------------
class SpectralConv_Nd_ParaModeScaling_RealVersion(nn.Module):
    def __init__(self, in_channels, out_channels, modes_fourier, PDEPara_mode_level = None , Use_2d_DCT = False ):
        super(SpectralConv_Nd_ParaModeScaling_RealVersion, self).__init__()

        if type(modes_fourier) == int:   self.nDIM = 1
        else:                            self.nDIM = 2

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_fourier = modes_fourier       #Number of Fourier modes to multiply, at most floor(N/2) + 1
        scale = 1 / (in_channels * out_channels)

        if Use_2d_DCT == True:
            self.Use_2d_DCT = True

        if self.nDIM == 2:
            self.weights_dim0 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, 2 ) )
            self.weights_dim1 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, 2 ) )
        elif self.nDIM==1:
            self.weights = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_fourier, 2) )

        self.PDEPara_mode_level = PDEPara_mode_level

        if self.PDEPara_mode_level is not None:

            if self.nDIM == 2:

                # kappa_base =( 2* np.array(self.modes_fourier[0]) / 2**self.PDEPara_mode_level ).astype(int)
                # modes_pde_extra = kappa_base * 2**(self.PDEPara_mode_level-1)
                # self.weights_extra_dim0 = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_pde_extra, modes_pde_extra , 2 ) )
                # self.weights_extra_dim1 = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_pde_extra, modes_pde_extra , 2 ) )

                if len(self.PDEPara_mode_level) == 1: # type(self.PDEPara_mode_level) == int:
                    self.weights_extra_dim0 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier , 2 ) )
                    self.weights_extra_dim1 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier , 2 ) )
                else:
                    self.weights_extra_dim0 = nn.Parameter(scale * torch.rand( in_channels, out_channels, 2**(self.PDEPara_mode_level[1]), 2**(self.PDEPara_mode_level[1]), 2 ) )
                    self.weights_extra_dim1 = nn.Parameter(scale * torch.rand( in_channels, out_channels, 2**(self.PDEPara_mode_level[1]), 2**(self.PDEPara_mode_level[1]) , 2 ) )

            elif self.nDIM==1:
                # kappa_base =( 2* np.array(self.modes_fourier) / 2**self.PDEPara_mode_level ).astype(int)
                # modes_pde_extra = kappa_base * 2**(self.PDEPara_mode_level-1)
                # self.weights_extra      = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_pde_extra,       2 ) )

                if len(self.PDEPara_mode_level) == 1 : # type(self.PDEPara_mode_level) == int:
                    self.weights_extra      = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_fourier                 ,   2 ) )
                else:
                    self.weights_extra      = nn.Parameter(scale * torch.rand( in_channels, out_channels, 2**(self.PDEPara_mode_level[1]) ,   2 ) )
        return


    def forward(self, x, fourierweight_scaling=None, HighFreqScaling=None):
        #   x.shape = b, c, Nx, Ny
        batchsize = x.shape[0]

        if self.nDIM == 2:
            if hasattr(self, 'Use_2d_DCT'):
                dim_dct = -2
                dim_other = -1
                x__cat__xflip = torch.cat ( [ x, x.flip([dim_dct])[...,1:-1,:] ], dim=dim_dct )
                x__1d_dct = torch.fft.fft( x__cat__xflip , dim=dim_dct,  norm="ortho").real[...,:x.size(dim_dct),:]   # 1d-dct-along-y
                x_ft  = torch.fft.rfft ( x__1d_dct , dim= dim_other , norm="ortho")                      # 1d-rfft-along-x
                out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),   x.size(-1)//2 + 1, 2, device=x.device) # torch.float instead-of torch.cfloat
            else:
                x_ft   = torch.fft.rfft2(x, norm="ortho")
                out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),  x.size(-1)//2 + 1, 2, device=x.device)

            k0, k1 = self.modes_fourier
            out_ft[:,:,:k0 ,:k1] = compl2_einsum( 'bixy,ioxy->boxy', torch.view_as_real(x_ft)[:, :,:k0, :k1], self.weights_dim0 )
            out_ft[:,:,-k0:,:k1] = compl2_einsum( 'bixy,ioxy->boxy', torch.view_as_real(x_ft)[:, :,-k0:,:k1], self.weights_dim1 )

        elif self.nDIM==1:
            x_ft   = torch.fft.rfft (x, norm="ortho")
            out_ft = torch.zeros(batchsize, self.out_channels,               x.size(-1)//2 + 1, 2, device=x.device)

            k0 = self.modes_fourier
            out_ft[:,:,:k0] = compl2_einsum( 'bix,iox->box', torch.view_as_real(x_ft)[:, :, :k0], self.weights  )

        #---------

        if self.PDEPara_mode_level is not None:
            if type( self.PDEPara_mode_level ) == int:  array_PDEPara_mode_level = np.array( [self.PDEPara_mode_level]  )
            else:                                       array_PDEPara_mode_level = np.array(  self.PDEPara_mode_level  )

            if self.nDIM == 2:
                if array_PDEPara_mode_level.size == 1:   modes_pde_extra = self.modes_fourier
                else:                                    modes_pde_extra = self.nDIM * ( 2**(array_PDEPara_mode_level[1]),  )

                m0,m1 = modes_pde_extra
                kappa_base = ( 2* np.array( modes_pde_extra[0] ) / 2**(array_PDEPara_mode_level[0]) ).astype(int)

                # ---------
                actual_fourierweight_scaling = torch.zeros( batchsize, *modes_pde_extra,  device=x.device )
                actual_fourierweight_scaling[:,   :kappa_base, :kappa_base] = fourierweight_scaling[:, 0].view(-1,1,1)
                for depth_i in range(1, array_PDEPara_mode_level[0] ):
                    l0_i = kappa_base *2**(depth_i-1) ;     l1_i = 2* l0_i
                    actual_fourierweight_scaling[:,    0:l0_i, l0_i:l1_i ] = fourierweight_scaling[:, 3*depth_i-2 ].view(-1,1,1)
                    actual_fourierweight_scaling[:, l0_i:l1_i,    0:l0_i ] = fourierweight_scaling[:, 3*depth_i-1 ].view(-1,1,1)
                    actual_fourierweight_scaling[:, l0_i:l1_i, l0_i:l1_i ] = fourierweight_scaling[:, 3*depth_i   ].view(-1,1,1)
                # ---------
                if HighFreqScaling is None:
                    out_ft[:,:,:m0 ,:m1] = out_ft[:,:,:m0 ,:m1] + compl3_einsum( 'bixy,ioxy,bxy->boxy', torch.view_as_real(x_ft)[:, :,:m0, :m1], self.weights_extra_dim0, actual_fourierweight_scaling )
                    out_ft[:,:,-m0:,:m1] = out_ft[:,:,-m0:,:m1] + compl3_einsum( 'bixy,ioxy,bxy->boxy', torch.view_as_real(x_ft)[:, :,-m0:,:m1], self.weights_extra_dim1, actual_fourierweight_scaling )
                else:
                    out_ft__clone = torch.clone( out_ft )
                    out_ft[:,:,:m0 ,:m1] = out_ft__clone[:,:,:m0 ,:m1]*HighFreqScaling + (1-HighFreqScaling)*compl3_einsum( 'bixy,ioxy,bxy->boxy', torch.view_as_real(x_ft)[:, :,:m0, :m1], self.weights_extra_dim0, actual_fourierweight_scaling )
                    out_ft[:,:,-m0:,:m1] = out_ft__clone[:,:,-m0:,:m1]*HighFreqScaling + (1-HighFreqScaling)*compl3_einsum( 'bixy,ioxy,bxy->boxy', torch.view_as_real(x_ft)[:, :,-m0:,:m1], self.weights_extra_dim1, actual_fourierweight_scaling )
                    for i_level_highFreq in range(  int(np.log2(k0//m0))):
                        l0_i = m0 * 2**(i_level_highFreq);     l1_i = 2* l0_i
                        scale_ratio = HighFreqScaling**(i_level_highFreq + 2)
                        out_ft[:,:,     : l1_i, l0_i:l1_i] = out_ft__clone[:,:,     :l1_i, l0_i:l1_i]  * scale_ratio
                        out_ft[:,:, l0_i: l1_i,     :l0_i] = out_ft__clone[:,:, l0_i:l0_i,     :l0_i]  * scale_ratio
                        out_ft[:,:,-l1_i:     , l0_i:l1_i] = out_ft__clone[:,:,-l1_i:     , l0_i:l1_i] * scale_ratio
                        out_ft[:,:,-l1_i:-l0_i,     :l0_i] = out_ft__clone[:,:,-l1_i:-l0_i,     :l0_i] * scale_ratio

            elif self.nDIM == 1:
                if array_PDEPara_mode_level.size == 1:   modes_pde_extra = self.modes_fourier
                else:                                    modes_pde_extra = 2**(array_PDEPara_mode_level[1])

                m0 = modes_pde_extra
                kappa_base = ( 2* np.array( modes_pde_extra ) / 2**(array_PDEPara_mode_level[0]) ).astype(int)

                # -----
                actual_fourierweight_scaling = torch.zeros( batchsize, modes_pde_extra       ,  device=x.device )
                actual_fourierweight_scaling[:, :kappa_base ] = fourierweight_scaling[:,0].view(-1,1)
                for depth_i in range(1, array_PDEPara_mode_level[0] ):
                    l0_i = kappa_base * 2**(depth_i-1);      l1_i = 2*l0_i
                    actual_fourierweight_scaling[:, l0_i:l1_i ] = fourierweight_scaling[:,depth_i].view(-1,1)
                # -----
                if HighFreqScaling is None:
                    out_ft[:,:,:m0] = out_ft[:,:,:m0] + compl3_einsum( 'bix,iox,bx->box', torch.view_as_real(x_ft)[:,:,:m0], self.weights_extra, actual_fourierweight_scaling )
                else:
                    out_ft__clone = torch.clone( out_ft )
                    out_ft[:,:,:m0] = out_ft__clone[:,:,:m0]*HighFreqScaling + (1-HighFreqScaling)*compl3_einsum( 'bix,iox,bx->box', torch.view_as_real(x_ft)[:,:,:m0], self.weights_extra, actual_fourierweight_scaling )
                    for i_level_highFreq in range(  int(np.log2(k0//m0)) ):
                        l0_i = m0 * 2**(i_level_highFreq) ;     l1_i = 2* l0_i
                        out_ft[:,:,l0_i:l1_i] = out_ft__clone[:,:,l0_i:l1_i] * HighFreqScaling**(i_level_highFreq + 2)

        #Return to physical space
        if   self.nDIM == 2:
            if hasattr(self, 'Use_2d_DCT'):
                x__1d_dct = torch.fft.irfft( torch.view_as_complex(out_ft), dim= dim_other, norm='ortho')
                x         = torch.fft.ifft(  torch.cat([x__1d_dct, x__1d_dct.flip([dim_dct])[..., 1:-1,:]], dim=dim_dct), dim = dim_dct, norm="ortho" ).real[ ...,:x.size(dim_dct),:]
            else:
                x = torch.fft.irfft2( torch.view_as_complex(out_ft), (x.size(-2), x.size(-1)),norm='ortho' )
        elif self.nDIM == 1:
            x = torch.fft.irfft(  torch.view_as_complex(out_ft),              x.size(-1), norm='ortho' )
        return x

#------------------------------------------------
class SpectralConv_Nd_ParaModeScaling(nn.Module):
    def __init__(self, in_channels, out_channels, modes_fourier , PDEPara_mode_level = None , Use_2d_DCT = False ):
        super(SpectralConv_Nd_ParaModeScaling, self).__init__()

        if type(modes_fourier) == int:   self.nDIM = 1
        else:                            self.nDIM = 2

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes_fourier = modes_fourier #Number of Fourier modes to multiply, at most floor(N/2) + 1
        scale = 1 / (in_channels * out_channels)

        if Use_2d_DCT == True:
            self.Use_2d_DCT = True

        if self.nDIM == 2:
            self.weights_dim0 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, dtype=torch.cfloat ) )
            self.weights_dim1 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, dtype=torch.cfloat ) )
        elif self.nDIM==1:
            self.weights = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_fourier, dtype=torch.cfloat) )

        self.PDEPara_mode_level = PDEPara_mode_level

        if self.PDEPara_mode_level is not None:
            if self.nDIM == 2:

                # kappa_base =( 2* np.array(self.modes_fourier[0]) / 2**self.PDEPara_mode_level ).astype(int)
                # modes_pde_extra = kappa_base * 2**(self.PDEPara_mode_level-1)
                # self.weights_extra_dim0 = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_pde_extra, modes_pde_extra , dtype=torch.cfloat ) )
                # self.weights_extra_dim1 = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_pde_extra, modes_pde_extra , dtype=torch.cfloat ) )

                if len(self.PDEPara_mode_level) == 1  : # type(self.PDEPara_mode_level) == int  :
                    self.weights_extra_dim0 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, dtype=torch.cfloat ) )
                    self.weights_extra_dim1 = nn.Parameter(scale * torch.rand( in_channels, out_channels, *modes_fourier, dtype=torch.cfloat ) )
                else:
                    self.weights_extra_dim0 = nn.Parameter(scale * torch.rand( in_channels, out_channels, 2**(self.PDEPara_mode_level[1]), 2**(self.PDEPara_mode_level[1]), dtype=torch.cfloat ) )
                    self.weights_extra_dim1 = nn.Parameter(scale * torch.rand( in_channels, out_channels, 2**(self.PDEPara_mode_level[1]), 2**(self.PDEPara_mode_level[1]), dtype=torch.cfloat ) )

            elif self.nDIM==1:

                # kappa_base =( 2* np.array(self.modes_fourier) / 2**self.PDEPara_mode_level ).astype(int)
                # modes_pde_extra = kappa_base * 2**(self.PDEPara_mode_level-1)
                # self.weights_extra      = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_pde_extra,                   dtype=torch.cfloat ) )

                if len(self.PDEPara_mode_level) == 1 :  #  type(self.PDEPara_mode_level) == int:
                    self.weights_extra      = nn.Parameter(scale * torch.rand( in_channels, out_channels, modes_fourier,    dtype=torch.cfloat ) )
                else:
                    self.weights_extra      = nn.Parameter(scale * torch.rand( in_channels, out_channels, 2**(self.PDEPara_mode_level[1]),    dtype=torch.cfloat ) )

        return


    def forward(self, x, fourierweight_scaling=None,HighFreqScaling=None):
        #   x.shape = b, c, Nx, Ny
        batchsize = x.shape[0]

        #-------------
        if self.nDIM == 2:

            if hasattr(self, 'Use_2d_DCT'):
                dim_dct = -2
                dim_other = -1
                x__cat__xflip = torch.cat ( [ x, x.flip([dim_dct])[...,1:-1,:] ], dim=dim_dct )
                x__1d_dct = torch.fft.fft( x__cat__xflip , dim=dim_dct,  norm="ortho").real[...,:x.size(dim_dct),:]   # 1d-dct-along-y
                x_ft  = torch.fft.rfft ( x__1d_dct , dim= dim_other , norm="ortho")                      # 1d-rfft-along-x
                out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),   x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
            else:
                x_ft   = torch.fft.rfft2(x, norm="ortho")
                out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),  x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

            k0, k1 = self.modes_fourier
            out_ft[:,:,:k0 ,:k1] = torch.einsum( 'bixy,ioxy->boxy', x_ft[:, :,:k0, :k1], self.weights_dim0 )
            out_ft[:,:,-k0:,:k1] = torch.einsum( 'bixy,ioxy->boxy', x_ft[:, :,-k0:,:k1], self.weights_dim1 )

        elif self.nDIM==1:
            x_ft   = torch.fft.rfft (x, norm="ortho")
            out_ft = torch.zeros(batchsize, self.out_channels,               x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

            k0 = self.modes_fourier
            out_ft[:,:,:k0] = torch.einsum( 'bix,iox->box', x_ft[:, :, :k0], self.weights  )
        #---------

        if self.PDEPara_mode_level is not None:
            if type( self.PDEPara_mode_level ) is int:  array_PDEPara_mode_level = np.array( [self.PDEPara_mode_level]  )
            else:                                       array_PDEPara_mode_level = np.array(  self.PDEPara_mode_level  )

            if self.nDIM == 2:
                if array_PDEPara_mode_level.size == 1:   modes_pde_extra = self.modes_fourier
                else:                                    modes_pde_extra = self.nDIM * ( 2**(array_PDEPara_mode_level[1]),  )

                m0,m1 = modes_pde_extra
                kappa_base = ( 2* np.array( modes_pde_extra[0] ) / 2**(array_PDEPara_mode_level[0]) ).astype(int)

                # ------
                actual_fourierweight_scaling = torch.zeros( batchsize, *modes_pde_extra, dtype = torch.cfloat, device=x.device )
                actual_fourierweight_scaling[:,   :kappa_base, :kappa_base] = fourierweight_scaling[:, 0].view(-1,1,1)
                for depth_i in range(1, array_PDEPara_mode_level[0] ):
                    l0_i = kappa_base *2**(depth_i-1) ;     l1_i = 2* l0_i
                    actual_fourierweight_scaling[:,    0:l0_i, l0_i:l1_i ] = fourierweight_scaling[:, 3*depth_i-2 ].view(-1,1,1)
                    actual_fourierweight_scaling[:, l0_i:l1_i,    0:l0_i ] = fourierweight_scaling[:, 3*depth_i-1 ].view(-1,1,1)
                    actual_fourierweight_scaling[:, l0_i:l1_i, l0_i:l1_i ] = fourierweight_scaling[:, 3*depth_i   ].view(-1,1,1)
                # ------
                if HighFreqScaling is None:
                    out_ft[:,:,:m0 ,:m1] = out_ft[:,:,:m0 ,:m1] + torch.einsum( 'bixy,ioxy,bxy->boxy', x_ft[:, :,:m0, :m1], self.weights_extra_dim0, actual_fourierweight_scaling )
                    out_ft[:,:,-m0:,:m1] = out_ft[:,:,-m0:,:m1] + torch.einsum( 'bixy,ioxy,bxy->boxy', x_ft[:, :,-m0:,:m1], self.weights_extra_dim1, actual_fourierweight_scaling )
                else:
                    out_ft__clone = torch.clone( out_ft )
                    out_ft[:,:,:m0 ,:m1] = out_ft__clone[:,:,:m0 ,:m1]*HighFreqScaling + (1-HighFreqScaling)*torch.einsum( 'bixy,ioxy,bxy->boxy', x_ft[:, :,:m0, :m1], self.weights_extra_dim0, actual_fourierweight_scaling )
                    out_ft[:,:,-m0:,:m1] = out_ft__clone[:,:,-m0:,:m1]*HighFreqScaling + (1-HighFreqScaling)*torch.einsum( 'bixy,ioxy,bxy->boxy', x_ft[:, :,-m0:,:m1], self.weights_extra_dim1, actual_fourierweight_scaling )
                    for i_level_highFreq in range(  int(np.log2(k0//m0))):
                        l0_i = m0 * 2**(i_level_highFreq);     l1_i = 2* l0_i
                        scale_ratio = HighFreqScaling**(i_level_highFreq + 2)
                        out_ft[:,:,     : l1_i, l0_i:l1_i] = out_ft__clone[:,:,     :l1_i, l0_i:l1_i]  * scale_ratio
                        out_ft[:,:, l0_i: l1_i,     :l0_i] = out_ft__clone[:,:, l0_i:l0_i,     :l0_i]  * scale_ratio
                        out_ft[:,:,-l1_i:     , l0_i:l1_i] = out_ft__clone[:,:,-l1_i:     , l0_i:l1_i] * scale_ratio
                        out_ft[:,:,-l1_i:-l0_i,     :l0_i] = out_ft__clone[:,:,-l1_i:-l0_i,     :l0_i] * scale_ratio

            elif self.nDIM == 1:
                if array_PDEPara_mode_level.size == 1:   modes_pde_extra = self.modes_fourier
                else:                                    modes_pde_extra = 2**(array_PDEPara_mode_level[1])

                m0 = modes_pde_extra
                kappa_base = ( 2* np.array( modes_pde_extra ) / 2**(array_PDEPara_mode_level[0]) ).astype(int)

                # -----
                actual_fourierweight_scaling = torch.zeros( batchsize, modes_pde_extra , dtype = torch.cfloat, device=x.device )
                actual_fourierweight_scaling[:, :kappa_base ] = fourierweight_scaling[:,0].view(-1,1)
                for depth_i in range(1, array_PDEPara_mode_level[0] ):
                    l0_i = kappa_base * 2**(depth_i-1);      l1_i = 2*l0_i
                    actual_fourierweight_scaling[:, l0_i:l1_i ] = fourierweight_scaling[:,depth_i].view(-1,1)
                # -----
                if HighFreqScaling is None:
                    out_ft[:,:,:m0] = out_ft[:,:,:m0] + torch.einsum( 'bix,iox,bx->box', x_ft[:, :, :m0], self.weights_extra, actual_fourierweight_scaling )
                else:
                    out_ft__clone = torch.clone( out_ft )
                    out_ft[:,:,:m0] = out_ft__clone[:,:,:m0]*HighFreqScaling  + (1-HighFreqScaling)*torch.einsum( 'bix,iox,bx->box', x_ft[:, :, :m0], self.weights_extra, actual_fourierweight_scaling )
                    for i_level_highFreq in range(  int(np.log2(k0//m0)) ):
                        l0_i = m0 * 2**(i_level_highFreq) ;     l1_i = 2* l0_i
                        out_ft[:,:,l0_i:l1_i] = out_ft__clone[:,:,l0_i:l1_i] * HighFreqScaling**(i_level_highFreq + 2)


        #Return to physical space
        if self.nDIM ==2:
            if hasattr(self, 'Use_2d_DCT'):
                x__1d_dct = torch.fft.irfft( out_ft, dim= dim_other, norm='ortho')
                x         = torch.fft.ifft(  torch.cat([x__1d_dct, x__1d_dct.flip([dim_dct])[..., 1:-1,:]], dim=dim_dct), dim = dim_dct, norm="ortho" ).real[ ...,:x.size(dim_dct),:]
            else:
                x= torch.fft.irfft2( out_ft, (x.size(-2), x.size(-1)), norm='ortho' )
        elif self.nDIM==1:
            x= torch.fft.irfft( out_ft,                x.size(-1), norm='ortho' )
        return x



# #------------------------------------
# class SteadySolution_F_EpsPara(nn.Module):
#     def __init__(self, num_PDEParameters ):
#         super(SteadySolution_F_EpsPara, self).__init__()
#         self.num_PDEParameters = num_PDEParameters
#         self.theta = nn.Parameter(  torch.rand( 1, dtype=torch.float )-10.0 )
#         if self.num_PDEParameters >=1:
#             self.net1 = nn.Sequential(nn.Linear( self.num_PDEParameters, 50), #nn.Sigmoid(),
#                                       nn.Linear( 50, 1 , bias=False)        )
#
#             self.net2 = nn.Sequential(nn.Linear( self.num_PDEParameters, 50), nn.Sigmoid(),
#                                       nn.Linear( 50, 50 ),                    nn.Sigmoid(),
#                                       nn.Linear( 50, 1 ),                     nn.Sigmoid()       )
#
#         return
#
#     def forward(self, eps_postive, paras=None ):
#         # --------- Learning fix point ------------
#         #   f( eps, paras ) = 1 - [ 1- Tanh( eps *  r_theta_para  ) ] * net2(paras)
#         #                      where r_theta_para = 5/(  1E-3 + sigmoid( beta + net1(paras) )  )
#         #                            net1(paras) =          w1* paras + beta2
#         #                            net2(paras) = sigmoid( w2* paras + beta2)
#         if paras is None:
#             n1 = 0
#             n2 = 1
#         else:
#             n1 = self.net1( paras )
#             n2 = self.net2( paras )
#         r = 5/( 0.001 + torch.sigmoid( self.theta + n1  )   )
#         return 0.5+ 0.5 * ( 1- (1- torch.tanh(eps_postive * r) ) * n2 )
# #------------------------------------


class PFNO_Nd(nn.Module):
    def __init__(self, nDIM, modes_fourier, width,
                 T_in=1, depth=4,
                 num_PDEParameters=0,
                 data_channel=1,
                 method_WeightSharing=0,
                 method_SkipConnection=0,
                 method_BatchNorm = 0,    # can be -1 (standard_batch_norm),  0 (no norm) or N_grid (e.g. 256,  for layer_norm)
                 brelu_last=1,
                 PDEPara_mode_level = None,
                 PDEPara_fc_class =  '',
                 PDEPara_ReScaling= None,
                 PDEPara_AcrossDepth = True ,
                 PDEPara_OutValueRange = 0.2,
                 method_ParaEmbedding = True,
                 option_RealVersion = False,   # default using the complex version
                 method_outputTanh    =  None,
                 Use_2d_DCT = False ):
        super(PFNO_Nd, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 1 location (u(t-10, x), ..., u(t-1, x),  x)
        input shape: (batchsize, x=64, y=64, c=11)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.option_RealVersion = option_RealVersion
        self.nDIM = nDIM
        if nDIM ==2:
            assert len(modes_fourier)== nDIM
        if method_outputTanh is not None:
            self.method_outputTanh = method_outputTanh

        self.modes_fourier     = modes_fourier
        self.width             = width
        self.depth             = depth
        self.num_PDEParameters = num_PDEParameters
        self.T_in              = T_in
        self.data_channel      = data_channel
        #self.method_TimeAdv    = method_TimeAdv
        #self.method_Attention  = method_Attention
        self.method_WeightSharing = method_WeightSharing
        self.method_SkipConnection= method_SkipConnection
        self.method_BatchNorm     = method_BatchNorm
        self.brelu_last           = brelu_last

        if PDEPara_mode_level is None and method_ParaEmbedding==False and num_PDEParameters >0:
            assert False, "wrong setup calling pFNO"

        self.PDEPara_mode_level = PDEPara_mode_level
        self.method_ParaEmbedding = method_ParaEmbedding


        self.PDEPara_AcrossDepth = PDEPara_AcrossDepth
        #self.PDEPara_OutValueRange = PDEPara_OutValueRange

        self.PDEPara_ReScaling = PDEPara_ReScaling
        #self.PDEPara_variable_mode = False

        #
        num_level_unrepeated = self.depth if self.method_WeightSharing==0 else 1

        if Use_2d_DCT == True:
            self.Use_2d_DCT = True
        #----------
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        for l in range( num_level_unrepeated ):
            # if self.num_PDEParameters >= 1 and self.PDEPara_mode_level is not None:
            #     self.conv.append( SpectralConv_Nd_ParaModeScaling(self.width, self.width, self.modes_fourier, PDEPara_mode_level = self.PDEPara_mode_level  ) )
            # else:
            #     self.conv.append( SpectralConv_Nd(self.width, self.width, self.modes_fourier) )

            actual_modes_fourier = self.modes_fourier
            #-----
            if self.option_RealVersion == False:
                self.conv.append( SpectralConv_Nd_ParaModeScaling(            self.width, self.width, actual_modes_fourier, PDEPara_mode_level = self.PDEPara_mode_level ,Use_2d_DCT=Use_2d_DCT ) )
            else:
                self.conv.append( SpectralConv_Nd_ParaModeScaling_RealVersion(self.width, self.width, actual_modes_fourier, PDEPara_mode_level = self.PDEPara_mode_level ,Use_2d_DCT=Use_2d_DCT ) )
            #-----

            if self.nDIM == 2:                self.w.append( nn.Conv2d(self.width, self.width, 1)  )
            elif self.nDIM==1:                self.w.append( nn.Conv1d(self.width, self.width, 1)  )

        if self.method_BatchNorm is True:
            self.bn = nn.ModuleList()
            for l in range( num_level_unrepeated ):
                self.bn.append(  my_NormNd(self.nDIM    , self.width, self.method_BatchNorm )  )

        if self.num_PDEParameters >= 1:
            if self.PDEPara_mode_level is not None: # learning  PDEs with multi parameters

                if type( self.PDEPara_mode_level ) == int:  array_PDEPara_mode_level = np.array( [self.PDEPara_mode_level]  )
                else:                                       array_PDEPara_mode_level = np.array(  self.PDEPara_mode_level  )

                #  -----
                size_output = 1 + ( array_PDEPara_mode_level[0]-1 )*( 2**self.nDIM -1 )

                nDepth__for__fc_PDEPara = 1
                if self.PDEPara_AcrossDepth == True:
                      size_output *= self.depth
                      nDepth__for__fc_PDEPara = self.depth

                self.fc_PDEPara = Fc_PDEpara( self.num_PDEParameters, size_output, nDepth=nDepth__for__fc_PDEPara , ClassStyle = PDEPara_fc_class, OutValueRange=PDEPara_OutValueRange)

                if array_PDEPara_mode_level.size == 2:
                    if self.nDIM == 1:
                        if 2**(array_PDEPara_mode_level[1]) < self.modes_fourier:
                            self.fc_PDEPara_HighFreqScaling = Fc_PDEpara( self.num_PDEParameters, 1, ClassStyle=PDEPara_fc_class, OutValueRange=0)
                    elif self.nDIM == 2:
                        if 2**(array_PDEPara_mode_level[1]) < self.modes_fourier[0]:
                            self.fc_PDEPara_HighFreqScaling = Fc_PDEpara( self.num_PDEParameters, 1, ClassStyle=PDEPara_fc_class, OutValueRange=0)

        num_InputChannel = 1
        if self.num_PDEParameters>=1:
            if self.method_ParaEmbedding == 0:
                self.fc_in = nn.Linear( num_InputChannel * self.data_channel                         , self.width)
            elif self.method_ParaEmbedding == 1:
                self.fc_in = nn.Linear( num_InputChannel * self.data_channel + self.num_PDEParameters, self.width)
            elif self.method_ParaEmbedding == 2 :
                self.fc_in = LinearNet_Parameters( num_InputChannel * self.data_channel, self.width, self.num_PDEParameters)
            elif self.method_ParaEmbedding == 3 :
                # self.fc_p  = nn.Sequential( nn.linear(self.num_PDEParameters, 25), nn.ReLU(),
                #                             nn.Linear(25, 25), nn.ReLU(),
                #                             nn.Linear(25, self.num_PDEParameters) )
                self.fc_in = LinearNet_Parameters( num_InputChannel * self.data_channel, self.width, self.num_PDEParameters)
                self.fc_out_p =  nn.Linear(self.num_PDEParameters, 128, bias=False )
        else:
            self.fc_in = nn.Linear( num_InputChannel*self.data_channel, self.width)


        # input channel is T_in: the solution of the previous T_in timesteps # + 1 location (u(t-10, x), ..., u(t-1, x),  x)
        self.fc_out0 = nn.Linear(self.width, 128)
        self.fc_out1 = nn.Linear(128, 1*self.data_channel )

        return

    def update_fourierweight_scaling(self, PDEparas ):

        PDEparas = PDEparas.view(PDEparas.shape[0], self.num_PDEParameters )

        #---- to be simplified ----
        # if hasattr(self, 'PDEPara_OutValueRange'):
        #     if self.PDEPara_OutValueRange != 0.2:
        #         fourierweight_scaling = self.fc_PDEPara( PDEparas , self.PDEPara_OutValueRange)
        #     else:
        #         fourierweight_scaling = self.fc_PDEPara( PDEparas )
        # else:
        #     fourierweight_scaling = self.fc_PDEPara( PDEparas )
        fourierweight_scaling = self.fc_PDEPara( PDEparas )

        batchsize = fourierweight_scaling.shape[0]

        # if hasattr(self,'PDEPara_AcrossDepth'):
        #     if self.PDEPara_AcrossDepth == True:
        if type( self.PDEPara_mode_level ) == int:  array_PDEPara_mode_level = np.array( [self.PDEPara_mode_level]  )
        else:                                       array_PDEPara_mode_level = np.array(  self.PDEPara_mode_level  )

        #if self.fc_PDEPara.fc_net[-2].out_features == self.depth * ( 1 + ( array_PDEPara_mode_level[0] -1)* (2**self.nDIM-1) ) :
        if fourierweight_scaling.shape[-1] == self.depth * ( 1 + ( array_PDEPara_mode_level[0] -1)* (2**self.nDIM-1) ) :
            fourierweight_scaling = fourierweight_scaling.view(batchsize,-1,  self.depth)   # now the dim is increased to 3

        if hasattr(self, 'fc_PDEPara_HighFreqScaling'):
            HighFreqScaling = self.fc_PDEPara_HighFreqScaling(PDEparas)
            if self.nDIM == 1: 
               HighFreqScaling = HighFreqScaling.view(batchsize, 1, 1)   #  batch,nx,ch
            elif self.nDIM == 2: 
               HighFreqScaling = HighFreqScaling.view(batchsize, 1,1, 1)
        else:
            HighFreqScaling = None

        return fourierweight_scaling , HighFreqScaling
        # #-----------------------------
        # if self.nDIM == 2:
        #     actual_fourierweight_scaling = torch.zeros( batchsize, self.modes_pde_extra, self.modes_pde_extra, dtype = torch.cfloat, device=PDEparas.device )
        #     actual_fourierweight_scaling[:,    0:8, 0:8 ] = fourierweight_scaling[:, 0].view(-1,1,1)
        #     for depth_i in range(1, self.PDEPara_mode_level):
        #         l0_i = 8*2**(depth_i-1) ;     l1_i = 2* l0_i
        #         actual_fourierweight_scaling[:,    0:l0_i, l0_i:l1_i ] = fourierweight_scaling[:, 3*depth_i-2 ].view(-1,1,1)
        #         actual_fourierweight_scaling[:, l0_i:l1_i,    0:l0_i ] = fourierweight_scaling[:, 3*depth_i-1 ].view(-1,1,1)
        #         actual_fourierweight_scaling[:, l0_i:l1_i, l0_i:l1_i ] = fourierweight_scaling[:, 3*depth_i   ].view(-1,1,1)
        # elif self.nDIM == 1:
        #     actual_fourierweight_scaling = torch.zeros( batchsize, self.modes_pde_extra                  , dtype = torch.cfloat, device=PDEparas.device )
        #     actual_fourierweight_scaling[:, 0:8 ] = fourierweight_scaling[:,0].view(-1,1)
        #     for depth_i in range(1, self.PDEPara_mode_level):
        #         l0_i = 8*2**(depth_i-1);      l1_i = 2*l0_i
        #         actual_fourierweight_scaling[:, l0_i:l1_i ] = fourierweight_scaling[:,depth_i].view(-1,1)
        #     # ---------
        # return actual_fourierweight_scaling


    def depth_advance_fixed_width(self,x, fourierweight_scaling, HighFreqScaling= None):

        for l in range(self.depth):
            l_actual = max( 0,  l-(self.depth-1)+(self.method_WeightSharing-1)  )  if self.method_WeightSharing>0 else l

            if  self.num_PDEParameters >= 1 and self.PDEPara_mode_level is not None:

                if type( self.PDEPara_mode_level ) == int:  array_PDEPara_mode_level = np.array( [self.PDEPara_mode_level]  )
                else:                                       array_PDEPara_mode_level = np.array(  self.PDEPara_mode_level  )

                #-----
                fourierweight_scaling__each_l = fourierweight_scaling

                # if hasattr(self,'PDEPara_AcrossDepth'):
                #     if self.PDEPara_AcrossDepth == True:

                #if self.fc_PDEPara.fc_net[-2].out_features == self.depth * ( 1 + ( array_PDEPara_mode_level[0] -1)* (2**self.nDIM-1) ) :
                if  fourierweight_scaling.ndim == 3 :
                    fourierweight_scaling__each_l = fourierweight_scaling[...,l]
                #-----
                x_12 = self.conv[l_actual]( x, fourierweight_scaling__each_l, HighFreqScaling  )
            else:
                x_12 = self.conv[l_actual]( x )   # call the 'original' SpectralConv_Nd_fast

            x_12 += self.w[l_actual](x)

            if self.method_BatchNorm is True: # apply batch normlization before nonlinear func
                x_12 = self.bn[l_actual](x_12)

            #if not (l==self.depth-1 and self.method_WeightSharing ==2) :
            if l<self.depth-1  or  ( l==self.depth-1 and self.brelu_last==1) :
                x_12 =  torch.relu(x_12)

            x = x + x_12 if self.method_SkipConnection==1 else x_12

        return x


    def forward(self, xx,  paras = None ):

        fourierweight_scaling = None
        HighFreqScaling=None



        if self.num_PDEParameters >= 1:
            assert( paras is not None )

            #if self.num_PDEParameters == 1 and paras.dim() == 1:  # reshape to [batch,num_PDEParameters]
            #    paras = paras.unsqueeze(1)

            if hasattr(self, 'PDEPara_ReScaling'):
                if self.PDEPara_ReScaling is not None:
                    paras = (paras - self.PDEPara_ReScaling[0]) / (self.PDEPara_ReScaling[1]-self.PDEPara_ReScaling[0])

            paras = paras.view( paras.shape[0], self.num_PDEParameters )

            if hasattr( self, 'method_ParaEmbedding' ):
                if self.method_ParaEmbedding==3:
                    paras = p_rescale_nu(paras)


            #---------------------------------
            method_ParaEmbedding = False
            if hasattr(self, 'method_ParaEmbedding' ):
                method_ParaEmbedding = self.method_ParaEmbedding  # could be 1 or 2
            else:  #  for compatablity with older code
                if self.PDEPara_mode_level is None:
                    method_ParaEmbedding = True
            #-----------------------------------

            if method_ParaEmbedding == 1:
                if self.nDIM == 1:      x = torch.cat( ( xx , paras.unsqueeze(1)             .repeat(1,xx.shape[1],            1) ) , dim=-1 )
                elif self.nDIM == 2:    x = torch.cat( ( xx , paras.unsqueeze(1).unsqueeze(1).repeat(1,xx.shape[1],xx.shape[2],1) ) , dim=-1 )
            elif method_ParaEmbedding > 1:
                if self.T_in > 1:
                    xx = xx.unsqueeze(-1)
                    if self.nDIM == 1:      x =  ( xx , paras.unsqueeze(1).unsqueeze(1)              )
                    elif self.nDIM == 2:    x =  ( xx , paras.unsqueeze(1).unsqueeze(1).unsqueeze(1) )
                elif self.T_in == 1:
                    if self.nDIM == 1:      x =  ( xx , paras.unsqueeze(1)              )
                    elif self.nDIM == 2:    x =  ( xx , paras.unsqueeze(1).unsqueeze(1) )
            else:
                x = xx

            if self.PDEPara_mode_level is not None:
                fourierweight_scaling, HighFreqScaling = self.update_fourierweight_scaling(paras )

        else:
            assert( paras is None )
            x = xx

        if self.T_in == 1:
            if self.nDIM == 1:                  x = self.fc_in(x).permute(0, 2, 1)
            elif self.nDIM==2:
                if hasattr( self,'Use_2d_DCT'): x = self.fc_in(x).permute(0, 3, 2, 1)
                else:                           x = self.fc_in(x).permute(0, 3, 1, 2)  # default
        elif self.T_in   > 1:
            if self.nDIM == 1:                  x = self.fc_in(x).permute(0, 3, 1, 2 )
            elif self.nDIM==2:
                if hasattr( self,'Use_2d_DCT'): x = self.fc_in(x).permute(0, 4, 1, 3, 2)
                else:                           x = self.fc_in(x).permute(0, 4, 1, 2, 3)  # default

        #
        x=self.depth_advance_fixed_width(x,fourierweight_scaling, HighFreqScaling )
        #

        if self.nDIM == 1:                  x = x.permute(0, 2, 1 )
        elif self.nDIM==2:
            if hasattr( self,'Use_2d_DCT'): x = x.permute(0, 3, 2, 1 )
            else:                           x = x.permute(0, 2, 3, 1 )  # default


        #
        #u = self.fc_out1( torch.relu( self.fc_out0(x) ) )
        #
        u = self.fc_out0(x)
        if hasattr(self, 'fc_out_p'):
            if self.nDIM == 1:    u = u + self.fc_out_p(paras).unsqueeze(1)
            elif self.nDIM==2:    u = u + self.fc_out_p(paras).unsqueeze(1).unsqueeze(1)
        u = torch.relu( u )
        u = self.fc_out1(u)


        if hasattr(self,'method_outputTanh'):
            u = torch.tanh( u )
        return u






