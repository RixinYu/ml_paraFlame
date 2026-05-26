#

import torch
import torch.nn as nn
from flame_net.tFNO_Nd import PermuteLayer_Nd, FourierBlock_Nd  



#--------------------------------------------------------------------------------------------------------------
#  Class 'RevtFNO_Nd' is the implementation of inverse scattering inspired Fourier Neural Operator (IS-FNO)
#
#     [Yu R, "An Inverse Scattering Inspired Fourier Neural Operator for Time-Dependent PDE Learning",	arXiv:2512.19439, accepted for publication in Journal of Computational Physics, 2026]
#
#  This 'IS-FNO' implementation extends on the 'Koopman theory-inspired Fourier Neural Operator' (kFNO), which was implemented in 'tFNO_Nd.py' .
#
#     [Yu, R., Herbert, M., Klein, M. and Hodzic, E., 2024. 'Koopman Theory-Inspired Method for Learning Time Advancement Operators in Unstable Flame Front Evolution', arXiv:2412.08426. accepted for publication in Physics of Fluids]
#
# ==========================================================================
class SpectralConv_MatrixExp_Nd(nn.Module):
    def __init__(self, in_out_channels, modes_fourier, basis_type = '' ):  # Example of basis_type: "k" or "pure_roll_k^3" 
        super(SpectralConv_MatrixExp_Nd, self).__init__()

        torch_cfloat = torch.complex128 if torch.get_default_dtype() == torch.float64 else torch.complex64

        if type(modes_fourier) == int:  self.nDIM = 1
        else:                           self.nDIM = len(modes_fourier)

        self.in_out_channels  = in_out_channels
        self.modes_fourier = modes_fourier #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.basis_type = basis_type

        # if 'raw' in self.basis_type:
        #     scale = 1 / (in_out_channels**2)  
        #     c0 = 0
        # else:

        scale = 1 /5000/ (in_out_channels**2)  
        c0 = 0.5
            
        if self.nDIM == 2:

            if 'nonl' in self.basis_type: # noninear version
                if 'rev' in self.basis_type:
                    self.weights_F = nn.Parameter( scale * ( torch.rand( 2, 2*modes_fourier[0], modes_fourier[1], in_out_channels//2, in_out_channels-in_out_channels//2, dtype= torch_cfloat) - c0) )  
                    self.weights_G = nn.Parameter( scale * ( torch.rand( 2, 2*modes_fourier[0], modes_fourier[1], in_out_channels-in_out_channels//2, in_out_channels//2, dtype= torch_cfloat) - c0) )
                else:
                    self.weights = nn.Parameter( scale * ( torch.rand( 2, 2*modes_fourier[0], modes_fourier[1], in_out_channels, in_out_channels, dtype= torch_cfloat) - c0) ) 
            else: # linear version
                self.weights = nn.Parameter( scale * ( torch.rand(    2*modes_fourier[0], modes_fourier[1], in_out_channels, in_out_channels, dtype= torch_cfloat) - c0) ) 
 
        elif self.nDIM==1:
            
            if 'nonl' in self.basis_type: 
                if 'rev' in self.basis_type:
                    self.weights_F = nn.Parameter( scale * ( torch.rand( 2, modes_fourier[0], in_out_channels//2, in_out_channels - in_out_channels//2, dtype= torch_cfloat)-c0) ) 
                    self.weights_G = nn.Parameter( scale * ( torch.rand( 2, modes_fourier[0], in_out_channels - in_out_channels//2, in_out_channels//2, dtype= torch_cfloat)-c0) ) 
                else:
                    self.weights = nn.Parameter( scale * ( torch.rand( 2, modes_fourier[0], in_out_channels, in_out_channels, dtype= torch_cfloat)-c0) ) 

            else: # linear version
                
                if 'roll' in self.basis_type:
                    self.weights_roll =  nn.Parameter( scale * ( torch.rand( in_out_channels )-c0) )
                    if 'pure_roll' in self.basis_type: return
                
                # dtype= torch.float if 'im' in self.basis_type else torch_cfloat # 'im' for Imaginary 
                if 'k' in self.basis_type:
                    self.k_power = 3 if 'k^3' in self.basis_type  else  nn.Parameter( torch.rand( 1 )  ) # note, this is real number, not complex !
                    self.weights_k = nn.Parameter( scale * ( torch.rand(               1 , in_out_channels, in_out_channels, dtype= torch_cfloat )-c0) ) 
                else:            
                    self.weights   = nn.Parameter( scale * ( torch.rand( modes_fourier[0], in_out_channels, in_out_channels, dtype= torch_cfloat )-c0) ) 
        return


    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f" {self.in_out_channels},"
                f"  m:{self.modes_fourier}"
                f"  basis_type:{self.basis_type}")

    # 
    # x.shape =  b,w,(Nx,Ny)  ;  kStep = torch.tensor([1,2,...20]) , it must be an array
    def forward(self, x, kStep  ) : 

        torch_cfloat = torch.complex128 if torch.get_default_dtype() == torch.float64 else torch.complex64

        einsum_op =  torch.einsum   
 
        if self.nDIM == 2: 

            k0  , k1  = self.modes_fourier
            N_x , N_y = x.shape[-2:]

            if 'nonl' in self.basis_type:  # 2-dimentional nonlinear version
                if 'rev' not in self.basis_type:
                    if 'raw' in self.basis_type:  # raw version, no matrix exponential, occasionaly needed for stable training
                        d_weights = self.weights       
                    else:
                        d_weights = torch.linalg.matrix_exp(self.weights )  - torch.eye( self.weights.shape[-1], device=x.device )  # remove the identity matrix

                #-----------------
                level_nonlinear = 1 
                c = self.basis_type.find('nonl')+4   
                if c < len(self.basis_type):    
                    if self.basis_type[c].isdigit(): # Example: 'nonl3' or 'nonl4' means level_nonlinear = 3 or 4
                        level_nonlinear = int(self.basis_type[c])
                #----------------

                x_out = torch.zeros( kStep.shape[0], *x.shape, dtype=x.dtype, device=x.device ) # b,w,(Nx,Ny),t

                for i, _ in enumerate(kStep):
                    for _ in range(level_nonlinear):
                        if 'rev' in self.basis_type:
                            x_F, x_G = x[:,:self.in_out_channels//2,:,:], x[:,self.in_out_channels//2:,:,:]

                            #x_G = x_G + self.linear_net( x_F.permute( [0,2,3,1] ) ).permute( [0,3,1,2] )
                            x_F_ft = torch.fft.rfftn( x_F, dim=[-2,-1], norm='ortho' )
                            a_ft= torch.zeros( *x_F_ft.shape[:2], 2*k0 , k1, dtype=torch_cfloat, device=x.device ) # b,w,(Nx,Ny),t
                            a_ft[ :, :,   :k0, :k1] = x_F_ft[:, :,   :k0, :k1]
                            a_ft[ :, :,-k0:  , :k1] = x_F_ft[:, :,-k0:  , :k1]
                            d_ft = einsum_op( 'bixy,sxyio->sboxy', a_ft,  self.weights_F )  
                            d = torch.fft.irfftn( d_ft, s = [N_x,N_y], dim = [-2,-1], norm='ortho' )
                            x_G = x_G + d[0,...] + d[1,...]**2

                            
                            x_G_ft = torch.fft.rfftn( x_G, dim=[-2,-1], norm='ortho' )
                            a_ft= torch.zeros( *x_G_ft.shape[:2], 2*k0 , k1, dtype=torch_cfloat, device=x.device ) # b,w,(Nx,Ny),t
                            a_ft[ :, :,   :k0, :k1] = x_G_ft[:, :,   :k0, :k1]
                            a_ft[ :, :,-k0:  , :k1] = x_G_ft[:, :,-k0:  , :k1]
                            d_ft = einsum_op( 'bixy,sxyio->sboxy', a_ft,  self.weights_G )  
                            d = torch.fft.irfftn( d_ft, s = [N_x,N_y], dim = [-2,-1], norm='ortho' )
                            x_F = x_F + d[0,...] + d[1,...]**2

                            x = torch.cat( [x_F, x_G], dim=1 )
                        else:
                            x_ft = torch.fft.rfftn( x, dim=[-2,-1], norm='ortho' )
                            a_ft= torch.zeros( *x_ft.shape[:2], 2*k0 , k1, dtype=torch_cfloat, device=x.device ) # b,w,(Nx,Ny),t
                            a_ft[ :, :,   :k0, :k1] = x_ft[:, :,   :k0, :k1]
                            a_ft[ :, :,-k0:  , :k1] = x_ft[:, :,-k0:  , :k1]

                            d_ft = einsum_op( 'bixy,sxyio->sboxy', a_ft,  d_weights )  
                            d = torch.fft.irfftn( d_ft, s = [N_x,N_y], dim = [-2,-1], norm='ortho' )

                            if 'noskip' in self.basis_type:
                                x =     d[0,...]  + d[1,...]**2
                            else:
                                x = x + d[0,...]  + d[1,...]**2

                    
                    x_out[i,...] = x
                return x_out
            
            else:  # 2-dimentional linear version

                weights = torch.linalg.matrix_exp( self.weights ) 

                x_ft   = torch.fft.rfftn(x, dim=[-2,-1], norm="ortho")
                out_ft  = x_ft.unsqueeze(0).repeat( kStep.shape[0], 1, 1, 1, 1) 

                a_ft = torch.zeros( *x_ft.shape[:2], 2*k0 , k1, dtype=torch_cfloat, device=x.device ) # b,w,(Nx,Ny),t
                a_ft[:, :,   :k0, :] = x_ft[:, :,   :k0, :k1]
                a_ft[:, :,-k0:  , :] = x_ft[:, :,-k0:  , :k1]

                for i, _ in enumerate(kStep):
                    a_ft = einsum_op( 'bixy,xyio->boxy', a_ft,  weights )  
                    out_ft[i,:,:,   :k0, :k1] = a_ft[:,:,   :k0, :]
                    out_ft[i,:,:,-k0:  , :k1] = a_ft[:,:,-k0:  , :]

                x_out  = torch.fft.irfftn( out_ft, s =[N_x,N_y], dim=[-2,-1], norm='ortho' )
                return x_out

        elif self.nDIM == 1:

            k0  = self.modes_fourier[0]
            N_x = x.shape[-1]
            
            if 'nonl' in self.basis_type:  # 1-dimetional nonlinear version
                if 'rev' not in self.basis_type:
                    if 'hraw' in self.basis_type:  
                        d_weights  =  self.weights.clone()
                        d_weights [0,...] = torch.linalg.matrix_exp( self.weights[0,...] ) - torch.eye( self.weights.shape[-1], device=x.device )  # remove the identity matrix
                    elif 'raw' in self.basis_type: # raw version, no matrix exponential, occasionaly needed for stable training
                        d_weights  = self.weights
                    else: 
                        d_weights  = torch.linalg.matrix_exp( self.weights ) - torch.eye( self.weights.shape[-1], device=x.device )  # remove the identity matrix

                #-----------------
                level_nonlinear = 1 
                c = self.basis_type.find('nonl')+4   
                if c < len(self.basis_type):    
                    if self.basis_type[c].isdigit(): # Example: 'nonl3' or 'nonl4' means level_nonlinear = 3 or 4
                        level_nonlinear = int(self.basis_type[c])
                #----------------

                x_out = torch.zeros( kStep.shape[0], *x.shape, dtype=x.dtype, device=x.device ) # b,w,(Nx,Ny),t

                for i, _ in enumerate(kStep):
                    for _ in range(level_nonlinear):
                        if 'rev' in self.basis_type:
                            x_F, x_G = x[:,:self.in_out_channels//2,:], x[:,self.in_out_channels//2:,:]

                            x_F_ft  = torch.fft.rfftn( x_F, dim=-1, norm='ortho' )[:,:,:k0]
                            d_ft = einsum_op( 'bix,sxio->sbox', x_F_ft,  self.weights_F )
                            d = torch.fft.irfftn( d_ft, s = N_x, dim=-1, norm='ortho' )
                            x_G = x_G + d[0,...]  + d[1,...]**2

                            #x_G = x_G + self.linear_net( x_F.permute( [0,2,1] ) ).permute( [0,2,1] )

                            x_G_ft  = torch.fft.rfftn( x_G, dim=-1, norm='ortho' )[:,:,:k0]
                            d_ft = einsum_op( 'bix,sxio->sbox', x_G_ft,  self.weights_G )
                            d = torch.fft.irfftn( d_ft, s = N_x, dim=-1, norm='ortho' )
                            x_F = x_F + d[0,...]  + d[1,...]**2
                            #-------------------
                            x = torch.cat( [x_F, x_G], dim=1 )
                        else:
                            x_ft = torch.fft.rfftn( x, dim=-1, norm='ortho' )
                            x_ft = x_ft[:,:,:k0]   
                            d_ft = einsum_op( 'bix,sxio->sbox', x_ft,  d_weights )   
                            d = torch.fft.irfftn( d_ft, s=N_x, dim=-1, norm='ortho' )

                            if 'noskip' in self.basis_type:
                                x =     d[0,...]  + d[1,...]**2
                            else:
                                x = x + d[0,...]  + d[1,...]**2

                    x_out[i,:,:,:] = x
                return x_out
            
            else:  # 1-dimentional linear version using matrix exponential

                x_ft = torch.fft.rfftn (x, dim=-1, norm="ortho")  # shape : 'bix'

                if 'roll' in self.basis_type:
                    N = x.size(-1)//2 +1
                    
                    weights = kStep.view(-1,1,1,1) * 1j* self.weights_roll.view(-1,1) * ( torch.arange(N)/N).to(x.device)

                    x_ft1 = x_ft * torch.exp(weights)
                    if 'pure_roll' in self.basis_type:
                        x = torch.fft.irfftn( x_ft1 , dim=-1, norm='ortho' )  # einsum_op( 'bix,t1ix->tbix', x_ft,  weights ) 
                        return x
                    out_ft  = torch.clone(x_ft1) # Trunc_HighFeq == False
                else:
                    out_ft  = x_ft.unsqueeze(0).repeat( kStep.shape[0], 1, 1, 1 ) 

                if 'k' in self.basis_type: 
                    weights = 1j*self.weights_k * ( (torch.arange(k0)/k0).to(x.device).view(-1,1,1)**self.k_power )
                else:
                    weights = self.weights 
                
                if 'mexp' in self.basis_type:  # 'mexp' means matrix exponential
                    # -- same as above, but 'matrix_exp' is applied to the whole tensor, which is not efficient
                    weights =  torch.linalg.matrix_exp(   kStep.view(-1,1,1,1) * weights )
                    out_ft[:,:,:,:k0] = einsum_op( 'bix,txio->tbox', x_ft[:,:,:k0],  weights ) 

                else:  # 'exp' means element-wise exponential

                    if 'raw' in self.basis_type:  # raw version, no matrix exponential, occasionaly needed for stable training
                        pass
                    else:
                        weights =  torch.linalg.matrix_exp( weights ) 

                    a_ft  =  x_ft[:,:,:k0]
                    for i, _ in enumerate(kStep):
                        a_ft = einsum_op( 'bix,xio->box', a_ft,  weights )   
                        out_ft[i,:,:,:k0] = a_ft


                x_out = torch.fft.irfftn( out_ft, dim=-1, norm='ortho' )
                
        return x_out                  
 

#----------------------------------


class Linear_Channel_Adapter(torch.nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        if in_channel != out_channel:  self.net = nn.Linear(in_channel, out_channel)
        return    
    def forward(self, x):
        if hasattr(self, 'net')==False:  return x  
        else:                            return self.net(x)




#----------
#  RevNet
#
class Reversible_FNO(nn.Module):
    #                                note: depth_rev can be of multiple lengths, e.g [2,2,2]
    def __init__(self, width_rev_ab=[1,30], width_middle=30, depth_rev=[2,2], modes_fourier=[32], basis_type='', method_WeightSharing=False, 
                 width_last_linear = 128,       bRealVersion=False ):
        super(Reversible_FNO, self).__init__()

        if type(modes_fourier) == int:  nDIM = 1
        else:                           nDIM = len(modes_fourier)

        assert(len(width_rev_ab) == 2 or len(width_rev_ab) == 3), f"width_rev_ab should be of length 2 or 3, but got {len(width_rev_ab)}"
        
        # width_rev_ab[0] is the width of the first reversible in/out channel, width_rev_ab[1] is the width of the second in/out channel, 
        #  and if len(width_rev_ab)==3, then width_rev_ab[2] is the width of a third in/out channel replacing the first in/out channel.
        
        self.width_rev_ab = width_rev_ab  

        self.F_rev_net  = nn.ModuleList()

        for idx , depth in enumerate( depth_rev ) :
            
            #--------------
            if idx ==0 or len(width_rev_ab)==2:
                width__0, width__1 = width_rev_ab[idx%2], width_rev_ab[(idx+1)%2]
            elif idx >= 1 and len(width_rev_ab)==3:
                width__0 = width_rev_ab[2] if idx%2 == 1 else (width_rev_ab[0]+width_rev_ab[1]) - width_rev_ab[2]
                width__1 = (width_rev_ab[0]+width_rev_ab[1]) - width__0

            #-------------
            if ( method_WeightSharing == False ) or ( method_WeightSharing == True and idx == 0 ): 
                net_middle = nn.Sequential( PermuteLayer_Nd(nDIM , bForward=True), 
                                            FourierBlock_Nd(depth=depth, width=width_middle, modes_fourier=modes_fourier, basis_type =basis_type, 
                                                            bUseSkipConnection=False, method_WeightSharing=method_WeightSharing, bNonlinearForLastLayer=True, bRealVersion=bRealVersion),
                                            PermuteLayer_Nd( nDIM , bForward=False ),
                                            nn.Linear(width_middle, width_last_linear) if width_last_linear is not None else nn.Identity(),
                                            nn.GELU()                                  if width_last_linear is not None else nn.Identity()
                                            ) 

            #-------------
            self.F_rev_net.append( nn.Sequential(Linear_Channel_Adapter(width__0, width_middle),
                                                 net_middle,
                                                 nn.Linear(width_last_linear, width__1) if width_last_linear is not None else Linear_Channel_Adapter(width_middle, width__1) 
                                                )
                                   )
        #-----
        self.permute_forward  = PermuteLayer_Nd( nDIM, bForward=True )
        self.permute_backward = PermuteLayer_Nd( nDIM, bForward=False )

        return
    

    def forward(self, x, bUp = True):

        if bUp == True:  # the upward (forward) caculation

            a, b  = x, 0

            for idx, F in  enumerate(self.F_rev_net):

                if len(self.width_rev_ab)==3  and  idx == 1  : 
                    x = torch.cat( (a, b), dim=-1 )
                    a, b = x[..., :self.width_rev_ab[2]], x[..., self.width_rev_ab[2]:]

                if idx% 2 ==0:
                    b = b + F( a )
                else:
                    a = a + F( b )
                
            x = torch.cat(  (a, b),  dim = -1 )

            return  self.permute_forward(x)

        elif bUp == False: # the backward calculation

            x = self.permute_backward( x )

            if len(self.width_rev_ab)==3: channel_a = self.width_rev_ab[2]  
            else:                         channel_a = self.width_rev_ab[0]

            a, b = x[ ..., :channel_a ],  x[ ..., channel_a: ]

            for idx, F in reversed( list(enumerate(self.F_rev_net)) ):

                if idx == 0:
                    if len(self.width_rev_ab)==3:
                        x = torch.cat( (a, b), dim=-1 )
                        a = x[..., :self.width_rev_ab[0]]
                    return a 

                if idx% 2 ==0:
                    b = b - F( a )
                else:
                    a = a - F( b )
            
            return a    
             




#--------------------------------------------------------------------------------------------------------------
#     N-dimensional implemention of Inverse Scattering inspired Fourier Neural Operator (IS-FNO)
#
#    [Yu R, "An Inverse Scattering Inspired Fourier Neural Operator for Time-Dependent PDE Learning", accepted for publication in Journal of Computational Physics, 2026]
#
#        Note, 'RevtFNO_Nd' is the internal class name for the IS-FNO. 
#   

class RevtFNO_Nd(nn.Module):
    def __init__(self, nDIM, modes_fourier, width, width_rev,
                 in_out_channel = 1, kTimeStepping = 20,
                 depth_conv={'tAdv':1, 'rev':[2,2], 'tAdv_basis':'exp' },  # 'tAdv_basis' can be be  'exp_roll', 'exp_pure_roll', 'exp_k^3', 'exp_k'
                 method_SkipConnection = 0,  # 0 means no-skip connection
                 method_WeightSharing = False,
                 basis_type= '' ) :

        super(RevtFNO_Nd, self).__init__()

        assert nDIM == len(modes_fourier) , "RevtFNO_Nd: please set nDIM == len(modes_fourier)"
        self.nDIM = nDIM

        self.width            = width
        self.in_out_channel   = in_out_channel  # In- and out- channels must be same for a reversible net
        self.kTimeStepping    = kTimeStepping
        self.basis_type       = basis_type


        self.depth_conv = depth_conv

        if 'tAdv_last_nonlinear' not in self.depth_conv: 
            self.depth_conv['tAdv_last_nonlinear'] = False  # add new default key

        width_tAdv =  in_out_channel + width_rev 

        if self.depth_conv['tAdv'] == 0:
            self.conv_timeAdv = nn.Identity()

        elif self.depth_conv['tAdv'] == 1 and 'exp' in self.depth_conv['tAdv_basis']:
        
            self.conv_timeAdv = SpectralConv_MatrixExp_Nd( width_tAdv, modes_fourier=modes_fourier, basis_type= self.depth_conv['tAdv_basis'] )
        else:
            self.conv_timeAdv = FourierBlock_Nd(depth=self.depth_conv['tAdv'], width=width_tAdv, modes_fourier=modes_fourier, basis_type= self.basis_type, 
                                                bUseSkipConnection = method_SkipConnection, method_WeightSharing=method_WeightSharing, bNonlinearForLastLayer=self.depth_conv['tAdv_last_nonlinear'])

        self.net_reversible = Reversible_FNO(width_rev_ab=[in_out_channel,width_rev], width_middle=width, depth_rev= self.depth_conv['rev'],basis_type=self.basis_type, 
                                             modes_fourier=modes_fourier,method_WeightSharing=method_WeightSharing)

        # if 'up' in self.basis_type: 
        #     if 'up2' in self.basis_type: nUpSample = 2
        #     elif 'up4' in self.basis_type: nUpSample = 4
        #     self.net_adjustResolution = AdjustResolution_Nd(nDIM, nUpSample)   
               
        return


    def forward(self, x, p=None):

        #if 'up' in self.basis_type: x = self.net_adjustResolution( x, bUpSample=True ) # upsampling by 2
        
        t, b, nxny, ch = self.kTimeStepping, x.shape[0], x.shape[1:-1], x.shape[-1]
        

        x = self.net_reversible(x, bUp = True)

        if isinstance(self.conv_timeAdv, SpectralConv_MatrixExp_Nd):

            x__tbcn  = self.conv_timeAdv(x,  kStep = torch.arange(t).to(x.device)+1  ).reshape( t*b,   -1, *nxny )  

            x        = self.net_reversible(x__tbcn,      bUp = False  ).reshape( t,b,*nxny,    ch  )

            if   self.nDIM == 1: x = x.permute(1,2,3,0)       # err_u , err_scat_coef #err_x_ot
            elif self.nDIM == 2: x = x.permute(1,2,3,4,0)
            
            if self.kTimeStepping == 1:   x = x.view( *x.shape[:-1] )

            #if 'up' in self.basis_type: x = self.net_adjustResolution( x, bUpSample=False )  # Downsample by 2

            return x

        else:

            x_ot = torch.zeros( b, *nxny, self.in_out_channel, self.kTimeStepping, device=x.device ) # b,(Nx,Ny),out,t
            for t in range(self.kTimeStepping):
                x  = self.conv_timeAdv(x)
                u  = self.net_reversible(x, bUp = False)
                x_ot[...,t] = u                                              # err_x_ot[...,idx]= err_u

            if self.kTimeStepping == 1:  x_ot = x_ot.view( *x_ot.shape[:-1] )

            # if 'up' in self.basis_type: x_ot = self.net_adjustResolution( x_ot, bUpSample=False ) # Downsample by 2
            
            return x_ot                           # sum_err/p.shape[-1]  #err_x_ot   # (x_o1t - x_ot)/




#-----------------------------------------------------------------
# class AdjustResolution_Nd(nn.Module):
#     def __init__(self, nDIM,  M_upsample) -> None:
#         super(AdjustResolution_Nd, self).__init__()
#         self.nDIM = nDIM
#         self.M_upsample  = M_upsample           # M=2 means upsampling by 2, M=1 means no change
#         return 
#     def forward(self, x, bUpSample ): # x.shape=  b,(Nx,Ny),channel
#         if self.M_upsample == 1: return x  # no change
#         else: 
#             if bUpSample==True:
#                 if self.nDIM == 2:
#                     assert False, 'not implement for 2D upsampling yet'
#                     return x
#                 elif self.nDIM == 1:
#                     x_ft = torch.fft.rfftn(x, dim=[1], norm="ortho") 
#                     x_ft *= np.sqrt(self.M_upsample)
#                     x_ft [:,-1,:] = x_ft [:,-1,:] / self.M_upsample # correct the last frequency component
#                     x = torch.fft.irfftn(  x_ft, s=self.M_upsample*x.shape[1], dim=[1], norm='ortho' )
#                     return x
#             elif bUpSample == False: # DownSample
#                 if self.nDIM == 2:    return x[:,::self.M_upsample,::self.M_upsample,...]
#                 elif self.nDIM == 1:  return x[:,::self.M_upsample         ,...]
