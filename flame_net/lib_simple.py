
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import scipy.io
import scipy.fftpack
from scipy.integrate import odeint
#from scipy.integrate import solve_ivp

#import matplotlib
#from matplotlib import animation , rc

from matplotlib import animation  #, widgets
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d
import os
import pickle
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from tqdm import trange

#-------------
#import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
#----------------




#@title my_DataSet
class my_DataSet(torch.utils.data.Dataset):
    def __init__( self , data, T_out,  T_in=1):
        self.T_in = T_in
        self.T_out = T_out
        # y.shape = [100,1000,256]
        self.data          = torch.tensor( np.moveaxis(data,-1,1) , dtype=torch.float )
        self.nTraj      = self.data.shape[0]
        self.nTimeChunk = (self.data.shape[-1] - 1 )//self.T_out
        self.len   = self.nTraj*self.nTimeChunk

    def __getitem__(self, item):
        i = item // self.nTimeChunk
        j = item %  self.nTimeChunk
        js = j*self.T_out
        return self.data[i, :,   js:js+self.T_out+1   ]

    def __len__(self):
        return self.len




#@title Standard training code
#-------------
class lib_ModelTrain:

    @staticmethod
    def Train(dataset_train, dataset_test,  model,  device= torch.device('cuda'), params  = {} ):
        params.setdefault('train:weight_decay',1e-4)
        params.setdefault('train:learning_rate',0.0025)
        params.setdefault('train:scheduler_step',100)
        params.setdefault('train:scheduler_gamma',0.5)
        params.setdefault('train:epochs',1000)
        params.setdefault('train:optimizer_method',torch.optim.Adam)
        params.setdefault('train:gradient_clip', None)
        params.setdefault('T_out',20)
        params.setdefault('TimeReverse',False)
        params.setdefault('TrainLoss', 'std')
        T_out = params['T_out']
        params.setdefault('train:batch_size', 120 *(20//T_out) )
        print('batch_size=', params['train:batch_size'])

        #---------
        print('lib_ModelTrain.Train : params')
        print(params )
        #for key, value in params.items():  print(key, ":", value)
        #--------

        #-------------
        nDIM         = 1
        #----------------------------------------------------------------------------------------
        optimizer = params['train:optimizer_method']( model.parameters(), lr=params['train:learning_rate'], weight_decay=params['train:weight_decay'] )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params['train:scheduler_step'], gamma=params['train:scheduler_gamma'])
        myloss = LpLoss(size_average=False)
        #--------------------------------------------
        epoch0 = 0
        #------------------
        ntrain = len(dataset_train)
        ntest  = len(dataset_test)
        print('ntrain=', ntrain, ' ,ntest=', ntest)


        model  = model.to(device)

        train_loader =  torch.utils.data.DataLoader( dataset_train, batch_size=params['train:batch_size'], shuffle=True)
        test_loader  =  torch.utils.data.DataLoader( dataset_test, batch_size=params['train:batch_size'], shuffle=True)

        #-----------------
        def validation_callback(model,test_loader):
            # validation test
            model.eval()
            with torch.no_grad():
                test_loss  = 0
                for train_a  in test_loader:
                    if params['TimeReverse']==False:      train_a = train_a.to(device)
                    elif params['TimeReverse']==True:     train_a = train_a.flip(-1).to(device)

                    current_batch_size = train_a.shape[0]
                    x   = train_a[..., :1]
                    yy  = train_a[...,1: ]

                    loss = 0
                    if 'std' in params['TrainLoss'] :
                        pred = torch.zeros_like( yy , device= train_a.device)
                        for t in range(T_out):
                            y = yy[..., t:t+1]
                            im = model(x)
                            pred[...,t:t+1] = im
                            x = im
                        loss += myloss( pred.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1)  )

                    if 'koop' in params['TrainLoss'] or 'rev' in params['TrainLoss'] :
                        pred2 = torch.zeros_like( yy , device= train_a.device)
                        pred2 = model( train_a[..., :1] ) # , torch.arange(T_out)+1 )
                        loss += myloss( pred2.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1)  )

                    test_loss += loss.item() # accumulate
            ################
            return test_loss


        with tqdm(  range( epoch0, params['train:epochs'])   )  as tqdm_epoch:
            for ep in tqdm_epoch  :

                model.train()
                #t1 = default_timer()
                train_loss = 0

                with tqdm(train_loader, leave=bool(ep==params['train:epochs']-1)  )  as tqdm_train_loader:
                    #for  x, y in train_loader:
                    for idx, train_a  in  enumerate( tqdm_train_loader ):

                        if params['TimeReverse']==False:      train_a = train_a.to(device)
                        elif params['TimeReverse']==True:     train_a = train_a.flip(-1).to(device)


                        current_batch_size = train_a.shape[0]
                        x   = train_a[..., :1]
                        yy  = train_a[...,1: ]

                        loss = 0
                        if 'std' in params['TrainLoss'] :
                            pred = torch.zeros_like( yy , device= train_a.device)
                            for t in range(T_out):
                                y = yy[..., t:t+1]
                                im = model(x)
                                pred[...,t:t+1] = im
                                x = im

                            loss += myloss( pred.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1)  )

                        if 'koop' in params['TrainLoss'] :
                            pred2 = torch.zeros_like( yy , device= train_a.device)
                            pred2 = model( train_a[..., :1] ) #,torch.arange(T_out)+1 )
                            loss += myloss( pred2.reshape(current_batch_size, -1), yy.reshape(current_batch_size, -1)  )

                        if 'rev' in params['TrainLoss'] :
                            x21 = model.net_Rev_PQ( train_a[..., :T_out+1 ].permute(0,2,1).reshape(current_batch_size*(T_out+1),-1).unsqueeze(-1) , bUp = True)
                            N, C = x21.shape[1:]
                            pred20 = model.TimeAdv_recur( x21.view( current_batch_size,T_out+1, N,C)[:,0,:,:],  torch.arange(T_out)+1   )
                            loss += myloss( x21.view(current_batch_size,T_out+1, N,C)[:,1:,:,:].reshape(current_batch_size, -1),
                                           pred20.reshape(current_batch_size, -1)  )


                        # x = x.to(device)
                        # y = y.to(device)
                        # current_batch_size = x.shape[0]
                        # ypred = model(x)
                        # loss = myloss( ypred.view(current_batch_size, -1), y.view(current_batch_size, -1)  )

                        train_loss += loss.item() # accumulate

                        optimizer.zero_grad()
                        loss.backward()
                        if params['train:gradient_clip'] is not None:
                            nn.utils.clip_grad_norm_(model.parameters(), params['train:gradient_clip'] )

                        optimizer.step()



                        #print('', end='.')

                        #if idx== len(tqdm_train_loader)-1:
                        #    tqdm_train_loader.set_postfix( ep=ep, loss_train=train_loss/ntrain, loss_test= validation_callback(model,test_loader)/ntest  )
                        #else:

                        tqdm_train_loader.set_postfix( ep=ep, loss_train=train_loss/((idx+1)*current_batch_size)  )

                #t2 = default_timer()
                scheduler.step()

                test_loss = validation_callback(model,test_loader)

                tqdm_epoch.set_postfix( ep=ep, loss_train=train_loss/ntrain, loss_test= test_loss /ntest  )


            #if ep % params['train:nStepScreenOut'] ==  0:
            #    print('')
            #    print('ep', ep, '   t[s]', t2-t1, '  train_loss ', train_loss / ntrain , '  test_loss ' , test_loss / ntest  )
        print('Train finished, final  train_loss= ', train_loss / ntrain , '  test_loss= ' , test_loss / ntest  )

#--------------

#@title A customized loss functions (not very important through)

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


#@title libsiva
def round_num_to_txt(num):
    if num<1:    return '{:g}'.format(num)[2:]   # turn 0.02 into 02,
    else:         return '{:g}'.format(num)

from scipy.optimize import fsolve

class libSiva:
    """
       MS__ for the (Michael)-Sivashinsky equation
       KS__ for the Kuramoto-Sivashinksky equation

       u(x) : slope of displacement
       d(x) : displacement , i.e. \partial_x (u)

       _hat : fourier modes

    """
    @staticmethod
    def demo( method = 'ks' , params={}):
        if method == 'ks':
            params.setdefault('Lpi',8*3)
            params.setdefault('mu',0.1)
            params.setdefault('nu',0.1)
            params.setdefault('d_ratio',0.6)
            params.setdefault('tau',1)
            params.setdefault('N',256)
            params.setdefault('Nstep',1000)
            params.setdefault('dt',0.15)
            params.setdefault('n_dt',1)
            params.setdefault('show_ani',True)
            params.setdefault('randfft', False)
            params.setdefault('bSym', False)

            params.setdefault( 'noise_level', 0)

            for key in params:
                if key == 'd0':          print(key+": given" ,end=' , ' )
                else:                    print(key+":", params[key],end=' , ' )
            print("")

            N = params['N']
            t = np.arange( params['Nstep'] )*params['dt']

            L = 2*np.pi #*params['Lpi']

            if params.get('d0') is None:
                if params['randfft']==True: d0 = libSiva.rand_d0_FFT(N, np.random.randint(low=2, high=8) )
                else:                       d0 = libSiva.rand_d0_simple(N)
            else:
                d0 = params['d0']

            u0 = libSiva.d_to_u(d0, libSiva.get_k(N,L) )

            usol  = libSiva.RK4NumericalSolver_usol( u0, t,  params['Lpi'], params['mu'],params['nu'], params['d_ratio'], params['tau'], params['n_dt'], params['noise_level'] , bSym = params['bSym'] )

            fig, axs = plt.subplots(1,4, figsize=(16,4))
            ax = axs[0]
            im = ax.contourf(usol.transpose(1,0), 20); fig.colorbar(im, ax=ax )
            ax = axs[1]
            ax.plot(  libSiva.u_to_d( usol[-1]  )  , '-r' )
            ax.plot(  libSiva.u_to_d( usol[params['Nstep']//2] )  , '-.b' )
            ax.plot(  libSiva.u_to_d( usol[10] )  , '--k' )
            ax = axs[2]
            ax.plot(  usol[-1]  , '-r' )
            ax.plot(  usol[params['Nstep']//2 ]   , '-.b' )
            ax.plot(  usol[ 10 ]   , '--k' )


            ax = axs[3]
            ax.plot(  libSiva.d_to_u( usol[-1]                   ,libSiva.get_k(N,L)  ), '-r' )
            ax.plot(  libSiva.d_to_u( usol[params['Nstep']//2 ]  ,libSiva.get_k(N,L) ), '-.b' )
            ax.plot(  libSiva.d_to_u( usol[ 10 ]                 ,libSiva.get_k(N,L) ), '--k' )


            dsol = libSiva.usol_to_dsol(usol , libSiva.get_k(N,L) )
            return dsol

        elif method =='MS_RK4':
            params.setdefault('nu',0.025)
            params.setdefault('N',256)
            params.setdefault('Nstep',1000)
            params.setdefault('randfft',False)
            params.setdefault('show_ani',False)
            params.setdefault('n_dt',5)
            params.setdefault( 'noise_level', 0)
            if params['randfft']==True:     d0 = libSiva.rand_d0_FFT( params['N'], np.random.randint(low=2, high=8) )
            else:                           d0 = libSiva.rand_d0_simple( params['N'] , {'siva_sys_name':'MS_RK4', 'para_value':params['nu'] } )
            return libSiva.demo('ks',params = {'Lpi':1, 'mu':0, 'nu':-params['nu'], 'd_ratio':1, 'N':params['N'], 'dt':0.015, 'Nstep':params['Nstep'], 'n_dt':params['n_dt'], 'show_ani':False,'d0':d0, 'show_ani':params['show_ani'], 'noise_level':params['noise_level'] }  )

        elif method =='KS_RK4':
            params.setdefault('Lpi',25)
            params.setdefault('N',256)
            params.setdefault('Nstep',1000)
            params.setdefault('randfft',False)
            params.setdefault('show_ani',False)
            params.setdefault( 'noise_level', 0)
            if params['randfft']==True:     d0 = libSiva.rand_d0_FFT( params['N'], np.random.randint(low=2, high=8) )
            else:                           d0 = libSiva.rand_d0_simple( params['N'] , {'siva_sys_name':'KS_RK4', 'para_value':params['Lpi'] } )
            return libSiva.demo('ks',params = {'Lpi':params['Lpi'], 'mu':1, 'nu':1, 'd_ratio':0, 'N':params['N'], 'dt':0.15, 'Nstep':params['Nstep'], 'n_dt':5, 'show_ani':False,'d0':d0 , 'show_ani':params['show_ani'], 'noise_level':params['noise_level'] }  )

        elif method =='MKS_RK4':
            params.setdefault('max_amplitude',1)
            params.setdefault('dd',0.5)
            params.setdefault('Lpi',32)

            params['d_ratio'] = params['dd']* 4 * params['max_amplitude']  #[0, 0.5, 1, 2, 3,  4 ]:
            params['mu'], params['nu'] = libSiva.Coeff_d_to_mu_nu( params['d_ratio'] , params['max_amplitude'] )

            params.setdefault('N',256)
            params.setdefault('dt',0.15)
            params.setdefault('n_dt',5)
            params.setdefault('Nstep',1000)
            params.setdefault('randfft',False)
            params.setdefault('show_ani',False)
            params.setdefault( 'noise_level', 0)
            params.setdefault('bSym',False)


            libSiva.demo('any',  params )

        elif method == 'any':
            params.setdefault('Lpi',1)
            params.setdefault('mu',0)
            params.setdefault('nu',0.025)
            params.setdefault('d_ratio',1)
            params.setdefault('tau',1)
            params.setdefault('N',256)
            params.setdefault('Nstep',1000)
            params.setdefault('dt',0.015)
            params.setdefault('n_dt',5)
            params.setdefault('show_ani',False)
            params.setdefault('randfft', False)
            params.setdefault( 'noise_level', 0)
            params.setdefault('bSym',False)

            N = params['N']
            t = np.arange( params['Nstep'] )*params['dt']
            L = 2*np.pi

            u0 = 0.02*(np.random.rand(  params['N'] )-0.5)

            for key in params:
                if key == 'd0':          print(key+": given" ,end=' , ' )
                else:                    print(key+":", params[key],end=' , ' )
            print("")


            usol = libSiva.RK4NumericalSolver_usol( u0, t,  params['Lpi'], params['mu'], params['nu'], params['d_ratio'], params['tau'], params['n_dt'], params['noise_level'] , bSym = params['bSym']  )

            fig, axs = plt.subplots(1,3, figsize=(20,4))
            ax = axs[0]
            im = ax.contourf(usol.transpose(1,0),20); fig.colorbar(im, ax=ax )

            ax = axs[1]
            ax.plot(  libSiva.u_to_d( usol[-1]                  , bSym = params['bSym'] )  , '-r' )
            ax.plot(  libSiva.u_to_d( usol[params['Nstep']//2]  , bSym = params['bSym'] )  , '-.b' )
            ax.plot(  libSiva.u_to_d( usol[params['Nstep']//4]  , bSym = params['bSym'] )  , ':c' )
            ax.plot(  libSiva.u_to_d( usol[params['Nstep']//8]  , bSym = params['bSym'] )  , '--g' )

            ax = axs[2]

            k = np.arange(N//2+1)*( 2*np.pi/(L*params['Lpi'])  )
            disp_relation = params['mu']*k**4   -params['nu']*k**2- params['d_ratio']*k

            ax.plot(  -disp_relation  , '-ro' )
            ax.set_ylim( [0, np.max(-disp_relation) ])




    @staticmethod
    def RK4NumericalSolver_dsol(d0,t,Lpi=6.28, mu=1, nu=1, d_ratio=0, tau=1, n_dt=1 , noise_level = 0 , bSym = False   ):
        N = d0.shape[0]
        L = 2*np.pi  # *Lpi
        k = libSiva.get_k(N,L )
        u0  = libSiva.d_to_u( d0, k )
        usol = libSiva.RK4NumericalSolver_usol( u0 , t, Lpi, mu, nu, d_ratio,tau, n_dt, noise_level,  bSym  )
        dsol = libSiva.usol_to_dsol(usol ,k   )
        return dsol

    #---------------
    @staticmethod
    def Coeff_d_to_mu_nu(d_ratio, max_amplitude=1):
        #--------------------
        def func_SivaEq_Coeff_Relations(arg, d_ratio, max_amplitude ):
            mu, k_star = arg
            nu = mu - d_ratio
            unk = np.zeros(2)
            unk[0]= 4*mu*k_star**3 - 2*nu*k_star - d_ratio
            unk[1]= mu*k_star**4 - nu*k_star**2 - d_ratio * k_star  + max_amplitude
            return unk
        #------------------
        func = lambda arg : func_SivaEq_Coeff_Relations(arg, d_ratio, max_amplitude  )
        root = fsolve(func, [4, 0.6]) # the starting value '4' may affect the result seraching
        mu = root[0]
        nu = mu-d_ratio
        return mu, nu


    @staticmethod
    def RK4NumericalSolver_usol(u0,t,Lpi=6, mu=1,nu=1, d_ratio=0, tau=1, n_dt=1,  noise_level = 0 , bSym = False  ):
        '''
            SivashinkskyEquation for the slope u(x,t)

                u_t/tau + u*u_x + (mu)*u_xxxx + (nu)*u_xx  - (d_ratio)*Gamma(u) =0

            with peroidic condition u(x,t) = u(x+L,t) and domain size being L = 2*pi*(Lpi)

            Note:
            (i ) It reduce to Kuramato-Sivashinksky(KS) Equation when d_ratio=0 and nu<0
                   u_t + u*u_x  + (mu)*u_xxxx  + (nu)*u_xx               = 0
            (ii) It reduce to Micheal-Sivashinksky(KS) Equation through setting mu=0, nu>0 and d_ratio=1
                   u_t + u*u_x                +  (nu)*u_xx    - Gamma(u) = 0

            ---  The RK4-solution process----
            step1:
                    u = sum { uhat(k) * e^(2*pi*k * x/L )  }
            step2:
                    d/dt(u_hat) +  [ mu*(2**pi/L*k)^4  - nu*(2*pi/L*k)^2  - d_ratio*(2*pi/L*|k|) ]*u_hat = -0.5*i* (2*pi/L*k) u2_hat
            step3(integrator factor):
                    g = e^(lambda*t)     , lambda =   mu*(2*pi/L*k)^4  - nu*(2*pi/L*k)^2 - d_ratio*(2*pi/L*|k|)
            step4:
                    d/dt[ e^(lambda*t) * uhat  ] = -0.5*i* (2*pi/L*k) u2_hat * e^(lambda*t)
        '''
        u = u0

        N = u0.shape[0]

        dt = (t[1]-t[0] )/n_dt
        dt = dt*tau                # tau can be consider to rescale the output time step

        L = 2*np.pi *Lpi

        if bSym == False:
            usol = np.zeros( (t.size,  u0.size) )
            usol[0, : ] =  u
        elif bSym==True:
            usol = np.zeros( (t.size,  u0.size) )

        #u=(u-u_base)

        if (t.size > 1000):     print('RK4NumericalSolver_usol: large t.size=(', t.size, '); a single-dot is printed for 1000 steps')

        if bSym == True:
            k = np.arange(N+1) * (  2*np.pi/(2*L)  )

            uhat = scipy.fft.rfft( np.concatenate( (u, -np.flip(u,-1) ) , -1 )  )

            corr_half = np.exp( - 1j * 2*np.pi* np.arange(N+1) * 0.5/(2*N) )


            uhat=( 1j* (uhat*corr_half).imag )/ corr_half

            #uhat =   1j* uhat.imag

        elif bSym == False:

            k = np.arange(N//2+1)*( 2*np.pi/L      )
            uhat = scipy.fft.rfft( u )
        #-----
        g = -0.5j * dt * k
        E = np.exp( -dt/2 * (       mu* k**4   - nu* k**2- d_ratio*k  )  )
        E2 = E**2
        #-----

        for j  in range( t.size -1 ) :

            #t = n*dt

            for n in range(n_dt):

                a = g* scipy.fft.rfft( scipy.fft.irfft(    uhat       )**2 )
                b = g* scipy.fft.rfft( scipy.fft.irfft( E*(uhat+a/2)  )**2 )  #  4th order
                c = g* scipy.fft.rfft( scipy.fft.irfft( E* uhat+b/2   )**2 ) #  Runge-Kutta
                d = g* scipy.fft.rfft( scipy.fft.irfft( E2*uhat+E*c   )**2 )

                uhat = E2 * uhat + (E2 * a + 2*E*(b+c) + d)/6

                if bSym == True:
                    #uhat =   1j* uhat.imag
                    uhat =( 1j* (uhat*corr_half).imag )/ corr_half


            if noise_level != 0:
                #--------
                d_hat = uhat
                d_hat[1:] = -1j * uhat[1:] / k[1:] ;          d_hat[0]=0
                dd =  scipy.fft.irfft( d_hat )
                #--------
                if bSym == True:
                    eps = np.random.normal(size = 2*N)
                else:
                    eps = np.random.normal(size =   N)


                dd += noise_level*eps

                #----
                uhat = 1j * scipy.fft.rfft( dd ) * k

            #-----------

            if bSym == True:
                usol[j+1, :] = scipy.fft.irfft( uhat  ) [...,:N]

            elif bSym == False:
                usol[j+1, :] = scipy.fft.irfft(uhat)

            if( j%1000==0):  print( '', end='*')

        return usol



    @staticmethod
    def anim_dsol_and_curvsol(t,x,dsol,curvsol,nskip=10, n0=0, interval=1):
        #from IPython.display import HTML
        #get_ipython().run_line_magic('matplotlib', 'inline')

        def update_2animation_flame(num, data1,data2, x, t, nskip, n0):
            line1.set_data(x, data1[n0+num*nskip,:]       )
            line2.set_data( *curvsol[n0+num*nskip,:,:].T  )
            ax.set_title(" n = %d" % (n0+num*nskip))

        fig, ax = plt.subplots(1,1, figsize=(15, 10) )
        line1, = ax.plot(x, dsol[0,:]      ,'ko', linewidth=1)
        line2, = ax.plot( *curvsol[0,:,:].T , 'r-s', linewidth=1)
        ax.set_ylim(-3, 2)

        frames = (dsol.shape[0]-1 - n0)//nskip -1 ;
        ani = animation.FuncAnimation(fig, update_2animation_flame, frames= frames, fargs=(dsol,curvsol,x,t,nskip,n0), interval=interval,blit=False)
        #Note: below is the part which makes it work on Colab
        #rc('animation', html='jshtml')
        return ani


    @staticmethod
    def anim_2_dsol(t,x,dsol,dsol2,nskip=10, n0=0, interval=1):
        #from IPython.display import HTML
        #get_ipython().run_line_magic('matplotlib', 'inline')

        def update_2animation_flame(num, data1,data2, x, t, nskip, n0):
            line1.set_data(x, data1[n0+num*nskip,:])
            line2.set_data(x, data2[n0+num*nskip,:])
            ax.set_title(" n = %d" % (n0+num*nskip))

        fig, ax = plt.subplots(1,1, figsize=(15, 10) )
        line1, = ax.plot(x, dsol[0,:], 'k--', linewidth=1)
        line2, = ax.plot(x, dsol2[0,:], 'r-', linewidth=1)
        ax.set_ylim(-1, 2)

        frames = (dsol.shape[0]-1 - n0)//nskip -1 ;
        ani = animation.FuncAnimation(fig, update_2animation_flame, frames= frames, fargs=(dsol,dsol2,x,t,nskip,n0), interval=interval,blit=False)
        #Note: below is the part which makes it work on Colab
        #rc('animation', html='jshtml')
        return ani

    @staticmethod
    def anim_N_dsol(t,x,dsol_list,nskip=10, n0=0, interval=1,ylim_plot=None):
        def update_Nanimation_flame(num, data_list, x, t, nskip, n0):
            for idx, data in enumerate( data_list):
                line_list[idx].set_data(x, data[n0+num*nskip,:])
            ax.set_title(" n = %d" % (n0+num*nskip))

        fig, ax = plt.subplots(1,1, figsize=(15, 10) )
        line_list =[]
        list_linestyle = ['r-', 'g--', 'b-','c-.','k-']
        for idx, dsol in enumerate( dsol_list):
            line, = ax.plot(x, dsol[0,:], list_linestyle[idx], linewidth=1)
            line_list.append(line)

        if ylim_plot==None:
            ax.set_ylim(-3, 5)
        else:
            ax.set_ylim(*ylim_plot)
        ax.axes.set_aspect('equal')

        frames = (dsol_list[0].shape[0]-1 - n0)//nskip -1 ;
        ani = animation.FuncAnimation(fig, update_Nanimation_flame, frames= frames, fargs=(dsol_list,x,t,nskip,n0), interval=interval,blit=False)
        return ani

    @staticmethod
    def anim_N_dsol_Simple(dsol_list,nskip=10, n0=0, interval=1):

        numT = dsol_list[0].shape[0]
        t = np.arange(numT)

        N= dsol_list[0].shape[-1]
        x  = libSiva.get_x(N)

        return libSiva.anim_N_dsol(t,x,dsol_list,nskip=10, n0=0, interval=1)


    @staticmethod
    def anim_dsol_FFT(t,x,dsol,nskip=10, n0=0, interval=1, kNum_Trunc = 20 ):
        N = dsol[0,:].size
        k = list( range(0, N//2+1))
        N_2 = N//2
        k_2 = list ( range ( N_2//2) )

        def Truc_d_fft(d_fft, kNum_Trunc=20):
            truc_d_fft= np.zeros_like(d_fft)
            truc_d_fft[:kNum_Trunc] = d_fft[:kNum_Trunc]
            truc_d_fft[-kNum_Trunc+1:] =d_fft[-kNum_Trunc+1:]
            return truc_d_fft

        def mono_angle( angle ):
            angle_monotic = angle
            return angle_monotic

        def recon_d_using_logfft(abs__log_d_fft, angle ):
            d_fft = np.zeros(N,dtype=complex)
            trun_abs__log_d_fft = abs__log_d_fft
            trun_angle = angle

            exp_log_d_fft =  np.exp( trun_abs__log_d_fft + 1j * trun_angle )
            d_fft[:N//2+1]  =  exp_log_d_fft
            d_fft[-N//2+1:] =  np.conj( np.flip( exp_log_d_fft[1:N//2]  ) )
            return d_fft

        def  Calc_various_va ( dsol_j ):
            d_fft = scipy.fftpack.fft( dsol_j)
            log_d_fft     = np.log(d_fft)[:N//2+1]
            angle_monotic = mono_angle ( log_d_fft.imag )
            logfft_log_d_fft     = np.log( scipy.fftpack.fft( log_d_fft.real )  ) [:N_2//2]
            logfft_angle_monotic = np.log( scipy.fftpack.fft( angle_monotic ) )   [:N_2//2]
            return d_fft, log_d_fft, angle_monotic,logfft_log_d_fft,logfft_angle_monotic

        def update_animation_flame_FFT(num, data,x, t, nskip, n0):
            ax[0].set_title(" n = %d" % (n0+num*nskip))

            d = data[n0+num*nskip,:]
            d_fft, log_d_fft, angle_monotic,logfft_log_d_fft,logfft_angle_monotic = Calc_various_va( d )

            line0_0.set_data(x, data[n0+num*nskip,:])
            line0_1.set_data(x, scipy.fftpack.ifft(  Truc_d_fft(d_fft, kNum_Trunc) ).real )

            dm_fft = scipy.fftpack.fft(  np.tanh( d*3 ) )
            dm = scipy.fftpack.ifft(  Truc_d_fft( dm_fft, kNum_Trunc) ).real
            line0_2.set_data(x, np.arctanh(dm)/3    )

            line1_0.set_data( k, log_d_fft.real  )
            line1_1.set_data( k, angle_monotic )
            ax1_twin.set_ylim( -np.pi,  np.pi)

        d = dsol[0,:]
        d_fft, log_d_fft, angle_monotic,logfft_log_d_fft,logfft_angle_monotic = Calc_various_va( d )
        d3_fft = scipy.fftpack.fft( np.power(dsol[0,:], 3) )

        fig, ax = plt.subplots(2,1, figsize=(7, 10) )
        fig.tight_layout()


        line0_0, = ax[0].plot(x, dsol[0,:], 'k-', linewidth=1) ;       ax[0].set_ylim(-1, 2)
        line0_1, = ax[0].plot(x, scipy.fftpack.ifft( Truc_d_fft(d_fft, kNum_Trunc) ).real , 'r-.', linewidth=1)

        line0_2, = ax[0].plot(x, np.power(  scipy.fftpack.ifft(  Truc_d_fft( d3_fft, kNum_Trunc) ).real  , 1./3 ) , 'b--' )

        line1_0, = ax[1].plot(k , log_d_fft.real , 'r-s', linewidth=1);         ax[1].set_ylim(-8, 5)

        ax1_twin = ax[1].twinx()
        line1_1, = ax1_twin.plot (k ,  angle_monotic  , 'b--o', linewidth=1) ;
        ax1_twin.set_ylim( angle_monotic[0], angle_monotic[-1])

        frames = (dsol.shape[0]-1 - n0)//nskip -1 ;
        ani = animation.FuncAnimation(fig, update_animation_flame_FFT, frames= frames, fargs=(dsol,x,t,nskip,n0), interval=interval,blit=False)
        return ani

    @staticmethod
    def plot_dsol(x,dsol,plt_list,):
        fig,ax = plt.subplots()
        for i, style in plt_list:
            ax.plot(x,dsol[i,:],style)   # ; hold on
        ax.grid()
        plt.show()

    @staticmethod
    def rand_d0_simple( N , sys_info = None ):

        scale = 0.03 if N>=512 else 0.1
        if sys_info is not None:
            if 'MS' in sys_info['siva_sys_name']:  # 'MS_1stEuler' or 'MS_RK4'
                if sys_info['para_value'] < 0.03501:  scale = 0.03 #/ (512//N)


        print('rand_d0_simple, scale=' , scale)
        d0= scale *np.random.random_sample(N)-scale/2
        return d0 - np.average(d0)

    @staticmethod
    def rand_d0_FFT(N,k_trun=8, sys_info = None): # sys_info is no use

        scale=0.3

        theta = np.random.random(N//2+1)*2*np.pi
        d_hat = scale *N/2* np.random.random(N//2+1)* ( np.cos(theta) +  1j *np.sin(theta)  )
        d_hat = ( 0.3* (np.random.rand(1)[0]+1)/2*(1-0.3) ) *d_hat
        d_hat[0] = 0
        d_hat[k_trun:]=0
        d0    = scipy.fft.irfft(d_hat) #.real
        return d0



    @staticmethod
    def get_k(N,L=2*np.pi):
        return (2*np.pi/L)*np.arange(0,N//2+1)

    @staticmethod
    def get_x(N, L=2*np.pi):
        x = np.linspace(-L/2, L/2, N, endpoint=False)
        return x

    """ displacment to slope"""
    @staticmethod
    def d_to_u(d, k ):
        u_hat = 1j * scipy.fft.rfft(    d )   * k
        u =  scipy.fft.irfft( u_hat ) #.real
        return u

    """ slope to displacment """
    @staticmethod
    def u_to_d(u, k=None, bSym =False ) :

        if k is None :
            if  bSym==True:
                N = u.shape[-1]
                k = libSiva.get_k( 2*N, np.pi *2 )  # default
                #corr_half = np.exp( - 1j * 2*np.pi* k * 0.5/(2*N) )
                #-----------
                uhat =  scipy.fft.rfft( np.concatenate( (u, -np.flip(u,-1)) , -1)  )

                #uhat = uhat*corr_half
                #uhat = 1j* uhat.imag

                d_hat = uhat
                d_hat[1:] = -1j * uhat[1:] / k[1:]
                d_hat[0]=0

                #d_hat = d_hat.real
                #d_hat = d_hat/corr_half

                d =  scipy.fft.irfft( d_hat ) #.real
                d = d[:N]

            elif  bSym==False:
                k = libSiva.get_k(u.shape[-1], np.pi *2 )  # default
                u_hat =  scipy.fft.rfft(u)
                d_hat = u_hat
                d_hat[1:] = -1j * u_hat[1:] / k[1:]
                d_hat[0]=0
                d =  scipy.fft.irfft( d_hat ) #.real
        else:
                u_hat =  scipy.fft.rfft(u)
                d_hat = u_hat
                d_hat[1:] = -1j * u_hat[1:] / k[1:]
                d_hat[0]=0
                d =  scipy.fft.irfft( d_hat ) #.real

        return d

    @staticmethod
    def usol_to_dsol(usol, k=None, bSym=False):
        #if k is None:    k = libSiva.get_k(usol.shape[-1], np.pi *2 )  # default
        dsol = np.zeros_like(usol)
        for j in range( usol.shape[0] ):
            dsol[j,:] = libSiva.u_to_d( usol[j,:], k=None, bSym=bSym)
        return dsol

    @staticmethod
    def dsol_to_usol(dsol, k=None):
        if k is None:    k = libSiva.get_k(dsol.shape[-1], np.pi *2 )  # default
        usol = np.zeros_like(dsol)
        for j in range( dsol.shape[0] ):
            usol[j,:] = libSiva.d_to_u( dsol[j,:], k)
        return usol


    @staticmethod
    def get2D_Ny_yB_from_estimate(Nx_target, yB_estimate, AspectRatio_set=1 , L=2*np.pi ):
        dx = L/ Nx_target
        dy = dx* AspectRatio_set
        Ny_actual = int( (yB_estimate[1] - yB_estimate[0] + 1E-10) / dy )
        yB = np.copy(yB_estimate)
        yB[1] = yB[0] + (Ny_actual) * dy
        return Ny_actual, yB






#@title Auxialllary utitity code (No need for carefully reading through)
import torch
import torch.nn as nn

#------------------------------------------------------------------------------
class MyConvNd(nn.Module):  # keep strid ==1
    def __init__(self, nDIM, in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular',
                 bias=True, bRelu=True, bNorm=False, type='Conv'):

        super(MyConvNd, self).__init__()
        self.nDIM = nDIM

        self.type = type
        if bNorm ==True:   bias = False  # when either 'batch_norm'  or layer_norm are on, bias becomes redundent

        if 'r' == type.casefold()[0]:  # e.g 'Residual' , 'Residual1', 'Resid3'
            numRepeat = int(type[-1]) if type[-1].isdigit() else 1  # the repeat time is given by the last character (digit) of the given type
            self.net = ResidualBlockNd(nDIM, numRepeat, in_channels, out_channels, kernel_size, stride, padding,
                                       padding_mode, bias, bRelu, bNorm)
        else:

            layers = []
            # ----------------------------------------------------------------------
            if 'c' in type.casefold()[0]: # standard CNN
                # default parameter setting for learning flame stability
                if kernel_size == 1:
                    padding = 0
                elif kernel_size == 3:
                    padding = 1
                    padding_mode = 'circular'
                layers.append(nn_ConvNd(nDIM)(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                              padding_mode=padding_mode, bias=bias))
            elif 't' == type.casefold()[0]:
                layers.append(
                    nn_ConvTransposeNd(nDIM)(in_channels, out_channels, kernel_size, stride=stride, bias=bias))
            elif 'i' in type.casefold()[0]:
                layers.append(InceptionND_v3(nDIM, in_channels, out_channels))
            else:
                raise ValueError(type + ' is not found: MyConv1d')
            # ----------------------------------------------------------------------

            if bNorm :   layers.append( my_NormNd(nDIM,out_channels,bNorm)  )
            if bRelu == True:     layers.append(nn.ReLU())

            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#------------------------------


def nn_ConvNd(nDIM):
    if nDIM==1:         return nn.Conv1d
    elif nDIM==2:       return nn.Conv2d
    else:               raise ValueError('nn_ConvNd: nDIM='+str(nDIM) )
def nn_ConvTransposeNd(nDIM):
    if nDIM==1:          return nn.ConvTranspose1d
    elif nDIM==2:        return nn.ConvTranspose2d
    else:                raise ValueError('nn_ConvTransposeNd: nDIM='+str(nDIM) )
def nn_MaxPoolNd(nDIM):
    if nDIM==1:       return nn.MaxPool1d
    elif nDIM==2:     return nn.MaxPool2d
    else:             raise ValueError('nn_MaxPoolNd: nDIM='+str(nDIM) )
def nn_AvgPoolNd(nDIM):
    if nDIM == 1:        return nn.AvgPool1d
    elif nDIM == 2:      return nn.AvgPool2d
    else:                raise ValueError('nn_AvgPoolNd: nDIM=' + str(nDIM))

def my_NormNd(nDIM, out_channel, bNorm):
    if nDIM == 1:
        if bNorm == -1:   return nn.BatchNorm1d(out_channel)
        else:           #  elif bNorm>0:
            return nn.LayerNorm( [out_channel, bNorm] )
    elif nDIM == 2:
        if bNorm == -1:   return nn.BatchNorm2d(out_channel)
        else:
            return nn.LayerNorm( [out_channel, bNorm[0], bNorm[1] ] )

    raise ValueError('my_NormNd: nDIM={}, out_channel={}, bNorm={}', nDIM, out_channel,bNorm)

# -------------------
class InceptionND_v3(nn.Module):
    def __init__(self, nDIM, in_fts, out_fts):
        super(InceptionND_v3, self).__init__()
        self.nDIM = nDIM

        # nn_ConvNd = nn.Conv1d if nDIM==1 else nn.Conv2d
        if type(out_fts) is not list:
            out_fts = [out_fts // 4, out_fts // 4, out_fts // 4, out_fts // 4]
        ###################################
        ### 1x1 conv + 3x3  conv + 3x3 conv
        ###################################
        self.branch1 = nn.Sequential(
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[0], kernel_size=1, stride=1),
            nn_ConvNd(nDIM)(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=3, stride=1, padding=1,
                            padding_mode='circular'),
            nn_ConvNd(nDIM)(in_channels=out_fts[0], out_channels=out_fts[0], kernel_size=3, stride=1, padding=1,
                            padding_mode='circular')
        )
        ###################################
        ### 1x1 conv  + 3x3 conv
        ###################################
        self.branch2 = nn.Sequential(
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[1], kernel_size=1, stride=1),
            nn_ConvNd(nDIM)(in_channels=out_fts[1], out_channels=out_fts[1], kernel_size=3, stride=1, padding=1,
                            padding_mode='circular'),
        )
        ###################################
        ###  3x3 MAX POOL  +  1x1 CONV
        ###################################
        self.branch3 = nn.Sequential(
            nn_MaxPoolNd(nDIM)(kernel_size=3, stride=1, padding=1),
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[2], kernel_size=1, stride=1)
        )
        ###################################
        ###  1x1 CONV
        ###################################
        self.branch4 = nn.Sequential(
            nn_ConvNd(nDIM)(in_channels=in_fts, out_channels=out_fts[3], kernel_size=1, stride=1)
        )

    def forward(self, input):
        o1 = self.branch1(input)
        o2 = self.branch2(input)
        o3 = self.branch3(input)
        o4 = self.branch4(input)
        x = torch.cat([o1, o2, o3, o4], dim=-1 - self.nDIM)
        return x


# ---------------------------
class ResidualBlockNd(nn.Module):
    def __init__(self, nDIM, numRepeat, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 padding_mode='circular', bias=True, bRelu=True, bNorm=0 ):
        super(ResidualBlockNd, self).__init__()
        self.nDIM = nDIM
        self.numRepeat = numRepeat

        self.bRelu = bRelu
        self.bNorm = bNorm

        layers = []
        layers.append(      nn_ConvNd(nDIM)(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode,  bias=bias))

        if bNorm == True:     layers.append( my_NormNd(nDIM,out_channels,bNorm) )

        if bRelu == True:     layers.append(nn.ReLU())

        self.cnn1 = nn.Sequential(*layers)

        # self.cnn1 =nn.Sequential(
        #    nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding, padding_mode=padding_mode, bias=bias),
        #    nn.ReLU(),
        #    nn.BatchNorm1d(out_channels),
        # )
        # self.cnn1.apply(init_weights)

        layers = []
        layers.append(  nn_ConvNd(nDIM)(out_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode,   bias=bias))
        if bNorm:     layers.append( my_NormNd(nDIM,out_channels,bNorm)  )

        self.cnn2 = nn.Sequential(*layers)
        # self.cnn2 = nn.Sequential(
        #    nn.Conv1d(out_channels,out_channels,kernel_size,     1,padding,padding_mode=padding_mode, bias=bias),
        #    nn.BatchNorm1d(out_channels)
        # )

        # if stride != 1 or in_channels != out_channels:
        #    self.shortcut = nn.Sequential(
        #        nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=stride,bias=bias),
        #        #nn.BatchNorm1d(out_channels)
        #    )

        if stride != 1 or in_channels != out_channels:

            layers = []
            layers.append(nn_ConvNd(nDIM)(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                          padding_mode=padding_mode, bias=bias))
            if bNorm:     layers.append(  my_NormNd(nDIM,out_channels,bNorm)  )

            self.shortcut = nn.Sequential(*layers)

            # self.shortcut = nn.Sequential(
            #        nn.Conv1d(in_channels,out_channels,kernel_size=3,padding=1, padding_mode=padding_mode, stride=stride,bias=bias),
            #        nn.BatchNorm1d(out_channels)
            #      )

        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):

        for dummy in range(self.numRepeat):
            residual = x
            x = self.cnn1(x)
            x = self.cnn2(x)
            x += self.shortcut(residual)
            if self.bRelu:
                x = nn.ReLU()(x)

        return x

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

#----------------------------------------