
import torch
import torch.nn as nn
#import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from utilities3 import *

from libSiva  import *





class SivaRgr_class(nn.Module):
    def __init__(self, N=1024, L=np.pi*2, nu_para_default= -3. ,dt_Output_para_default=-3., cRatio_para_default=0.1, max_numSubStep=400):    # , mode_operation = None):
        super(SivaRgr_class, self).__init__()

        self.nu_para        = nn.Parameter(  torch.tensor(nu_para_default        , dtype=torch.float)  )
        self.dt_Output_para = nn.Parameter(  torch.tensor(dt_Output_para_default , dtype=torch.float)  )
        self.cRatio_para    = nn.Parameter(  torch.tensor(cRatio_para_default    , dtype=torch.float)  )
        self.max_numSubStep = torch.tensor( max_numSubStep, dtype=torch.int )
        #self.nu_para        = nn.Parameter(  torch.tensor( 0.3)  )
        #self.dt_Output_para = nn.Parameter(  torch.tensor( 0.01)  )
        #self.cRatio_para    = nn.Parameter(  torch.tensor( 3.   )  )


        self.L = L
        self.N = N
 
        self.x = torch.linspace(-L/2, L/2-L/N, N)   #libSiva.get_x(N,L )
        self.k = (2*np.pi/L)*torch.arange(0,N//2+1)  # libSiva.get_k(N,L )

        self.dx = self.L/self.N 
        self.k1 = 1j*self.k
        self.k2 = self.k1**2

    def To_device(self,device):
        self.k = self.k.to(device)
        self.x = self.x.to(device)
        self.k1 = self.k1.to(device)
        self.k2 = self.k2.to(device)

    def d_to_u(self, d):
        u_hat =  1j * torch.fft.rfft( d ) * self.k
        u =  torch.fft.irfft( u_hat) 
        return u

    def u_to_d(self, u):
        d_hat = -1j * torch.fft.rfft( u )
        d_hat[...,1:] = d_hat[...,1:] / self.k[1:]
        d_hat[...,0]=0
        d =  torch.fft.irfft( d_hat ) 
        return d
        


    #--------------------------
    def nu(self):
        return torch.sigmoid(self.nu_para)*(0.5-0.01)+0.01
        #return torch.relu(self.nu_para) + 0.01  # * (0.5 - 0.01) + 0.01
    def dt_Output(self):
        #return torch.relu(self.dt_Output_para) +1E-4  #* (0.1 - 0.001) + 0.001
        return torch.sigmoid(self.dt_Output_para)*(0.1-0.001)+0.001
    def cRatio(self):
        #return torch.relu(self.cRatio_para) + 0.5
        return torch.sigmoid(self.cRatio_para)*(2- 0.5) + 0.5
    #--------------------------
    def get_numSubSteps_PerTimeStep(self):
        dt_sub_StabililtyLimit = 0.01*self.dx/( 3 )
        numSubSteps_PerTimeStep = torch.div(self.dt_Output(), dt_sub_StabililtyLimit, rounding_mode='trunc').int() + 1
        #return torch.min( numSubSteps_PerTimeStep, self.max_numSubStep )
        return  numSubSteps_PerTimeStep

    def Advance_uSlope(self, u):
        #u = u
        #----
        #int( self.dt_Output() //dt_sub_StabililtyLimit  + 1)
        #-----------------
        numSubSteps_PerTimeStep = self.get_numSubSteps_PerTimeStep()
        dt = self.dt_Output() / numSubSteps_PerTimeStep
        q = 1 - dt * self.cRatio() * (self.k + self.nu() * self.k2)
        #-----------------
        for k in range(numSubSteps_PerTimeStep):
            uhat  =torch.fft.rfft(u)
            u2hat =torch.fft.rfft(0.5*u**2)
            # Calculate the u^n+1 term

            uhat_update=(-dt*self.k1*u2hat+uhat)/q
            u=torch.fft.irfft( uhat_update.to(u.device) )
        return u

    #def Advance_d_once(self, d0):
    #    u0 = self.d_to_u(d0)
    #    u = self.Advance_uSlope(u0)
    #    d = self.u_to_d(u)
    #    return d

    def Advance_d(self, d0 , nTimes):
        u  = self.d_to_u( d0 )

        usol = []
        for idx in range(nTimes):
            u = self.Advance_uSlope(u)
            usol.append(u)

        dsol   = self.u_to_d(  torch.cat( usol, dim=1) )
        return dsol
        
    def forward(self,x, nTimes=1):

        x = x.permute(0,2,1)
        if x.shape[-1] == self.N:
            x = self.Advance_d(x,nTimes)
        else:
            x_shape = x.shape
            xIntp_hat = torch.zeros( (*x_shape[:-1], self.N//2+1), dtype=torch.cfloat ,device=x.device)
            xIntp_hat[..., :x_shape[-1]//2+1 ] = torch.fft.rfft(x)
            xIntp = torch.fft.irfft(xIntp_hat)
           
            #----------
            xIntp = self.Advance_d(xIntp,nTimes)
            #----------
            x = torch.fft.irfft( torch.fft.rfft(xIntp)[...,:x_shape[-1]//2+1] )

        x = x.permute(0,2,1)
        return x

