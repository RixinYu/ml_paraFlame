

import numpy as np
import scipy.io
import scipy.fftpack


import matplotlib.pyplot as plt
import os
import pickle
import warnings


#-----------------------------
#@title libBurgers for 1D or 2D Burgers equation 
class libBurgers:
    """
       Burgers equaiton
       u(x) : solution
       u_hat(k) : fourier modes
    """
    @staticmethod
    def demo(  params={'N':256,'Nstep':1000,'Lpi':1,'rand_init':'rand_FFT', 'draw':True, 'nDIM':1} ):
        N = params['N']
        Lpi =params['Lpi']
        Nstep = params['Nstep']
        nDIM = params['nDIM']

        dt_Output, NumOutPut_dt = libBurgers.get_OutputTimeStep(N,Lpi)

        t = np.arange(Nstep  )*dt_Output

        u0    = libBurgers.rand_u0( N , Lpi, params['rand_init'] , nDIM)
        usol  = libBurgers.RK4NumericalSolver_usol( u0, t, NumOutPut_dt, Lpi, nDIM)

        if nDIM==1 and params['draw']:
            libBurgers.plot_1d(usol)

        return usol
    
    @staticmethod
    def get_OutputTimeStep(N=256,Lpi=1):
        dt_Output =  20*0.08*(Lpi/N) 
        NumOutPut_dt = 20
        # dt = dt_Output /NumOutPut_dt
        # print('LibBurges: N=',N, ' Lpi=',Lpi,' dt_Output=', dt_Output, 'NumOutPut_dt=', NumOutPut_dt,'dt=', dt)
        return dt_Output, NumOutPut_dt

    @staticmethod
    def plot_1d(usol,dN=100):
        fig, axs = plt.subplots(2,1, figsize=(15,6))
        ax = axs[0]
        ax.contourf(usol.T)

        N  = usol.shape[1]
        Nstep = usol.shape[0]
        #----------
        y = np.linspace(-1,1,N)
        m_list = np.arange(0,Nstep,dN)
        #---------
        ax       = axs[1]
        ax1_diff =axs[1].twinx()
        for m in  m_list:
            ax.plot( usol[m]                      +  6* m/dN  , y , '-')
            ax1_diff.plot( (usol[m+1]-usol[m])*10  + 6* m/dN  , y , '--' )
        
        return

    @staticmethod
    def RK4NumericalSolver_usol(u0,t,n_dt=1,Lpi=1, nDIM=1,    nu=0.002 ) : 
        '''
            ===================================================================================
            The one-dimensional(1D) Burgers-equation:  u(x,t)
                u_t  + 0.5 *(u^2)_x - nu* u_xx = 0
            with peroidic condition u(x,t) = u(x+L,t) and domain size being L = 2*pi*(Lpi/np.pi)

            ---  The RK4-solution process----
            step1:
                    u = sum { uhat(k) * e^(i*2*pi*k * x/L )  }
            step2:
                    d/dt(u_hat) +  [ nu* (2*pi/L*k)^2 ]*u_hat = -0.5*i* (2*pi/L*k)* u2_hat
            step3(integrator factor method):
                    E = e^(lambda*t)     , lambda = nu*(2*pi/L*k)^2
            step4:
                    d/dt[ e^(lambda*t) * uhat  ] = -0.5*i* (2*pi/L*k)* u2_hat * e^(lambda*t)
            
            ===================================================================================
            The two-dimensional(2D) KDV-equation:  u(x,y,t) , 2d Bugeres equation is not implemented yet !
                ??? (u_t  + 3 *(u^2)_x +  u_xxx)_x  +  u_yy = 0
            with peroidic condition  over square domain of length "Lx = Ly = 2*pi*(Lpi/np.pi) "

            ---  The RK4-solution process----
            step1:
                    ???  u = sum { uhat(k) * e^(i*2*pi*k*x/L )  }
            step2:
                    ???   (i*k_x)d/dt(u_hat) + [k_x^4 - sigma_y*k_y^2]*u_hat = 3 k_x^2 *u2_hat
                    ???  d/dt(u_hat) + i*[-k_x^3 + sigma_y*k_y^2/k_x]*u_hat = -3*i*k_x * u2_hat
            step3(integrator factor method):
                    ???  E = e^(lambda*t)   ,  lambda = i*[-k_x^3 + sigma_y*k_y^2/k_x]
            step4:
                    ???   d/dt[ e^(lambda*t) * uhat  ] = -3*i*kx*u2_hat * e^(lambda*t)
        '''

        uhat = scipy.fft.rfftn( u0 )

        usol = np.zeros( (t.size,  *u0.shape) )
        usol[0, ... ] =  u0

        dt = (t[1]-t[0] )/n_dt

        if (t.size > 1000):     print('RK4NumericalSolver_usol: large t.size=(', t.size, '); a single-dot is printed for 1000 steps')

        L = 2*np.pi * (Lpi/np.pi)

        if nDIM == 1:

            N = u0.shape[0]
            k = np.arange(N//2+1)*( 2*np.pi/L      )

            #-----
            g = -0.5*1j * dt * k
            E = np.exp( -dt/2 * ( nu *   k**2  ) )
            E2 = E**2
            #-----

        elif nDIM == 2:

            raise ValueError('Not implemented yet, the following is for 2D-KDV, not 2D-Burgers.')

            N_x, N_y = u0.shape
            
            #--------------
            L_x , L_y = L , L
            k_x = np.fft.fftfreq(N_x, d= 1./N_x/(2*np.pi/L_x ) ).reshape(-1,1)     # now  kx.shape=(nx,1)
            k_x[0,0]=1E-9
            k_y = np.fft.fftfreq(N_y, d=1./N_y/(2*np.pi/L_y ) ) 
            k_y = k_y[:N_y//2 +1]                              

            #--------------
            g = -3*1j * dt * k_x
            E = np.exp( -dt/2 *  1j * ( - k_x**3 + sigma_y*k_y**2 / k_x )    )
            E2 = E**2
            #------        


        for j  in range( t.size -1 ) :

            for _ in range(n_dt):
                a = g* scipy.fft.rfftn( scipy.fft.irfftn(    uhat       )**2 )
                b = g* scipy.fft.rfftn( scipy.fft.irfftn( E*(uhat+a/2)  )**2 )  #  4th order
                c = g* scipy.fft.rfftn( scipy.fft.irfftn( E* uhat+b/2   )**2 ) #  Runge-Kutta
                d = g* scipy.fft.rfftn( scipy.fft.irfftn( E2*uhat+E*c   )**2 )
                uhat = E2 * uhat + (E2 * a + 2*E*(b+c) + d)/6

            usol[j+1, ...] = scipy.fft.irfftn(uhat)

            if( j%1000==0):  print( '', end='.')

        return usol



    @staticmethod
    def rand_u0( N=256, Lpi = 1, rand_init = 'rand_FFT', nDIM=1 ): # 
        scale=1
        if rand_init == 'rand_FFT':
            if nDIM ==1: 
                theta = np.random.random(N//2+1)*2*np.pi
                u_hat = scale *N/2* np.random.random(N//2+1)* ( np.cos(theta) +  1j *np.sin(theta)  )
                u_hat[0]=0
                u_hat[5:]=0
                u0    = 0.4* scipy.fft.irfft(u_hat)  #.real

            elif nDIM ==2:

                raise ValueError('Not implemented yet, the following is for 2D-KDV, not 2D-Burgers')

                Nx,Ny = N,N
                uhat = np.zeros((Nx, Ny//2+1),dtype=np.complex128)
                nnn_x , nnn_y = 10, 10
                c = Nx*Ny /15
                theta = np.random.random( (nnn_x,nnn_y) )*2*np.pi
                aaa = np.random.random( (nnn_x,nnn_y) )* ( np.cos(theta) +  1j *np.sin(theta)  )
                theta = np.random.random( (nnn_x,nnn_y) )*2*np.pi
                bbb = np.random.random( (nnn_x,nnn_y) )* ( np.cos(theta) +  1j *np.sin(theta)  )

                # For 2D-KDV, the initital condition must satisfy  " integration_over_x( u_yy ) = 0 "
                one_k_x = np.ones(nnn_x); one_k_x[0] =0 
                aaa *= one_k_x.reshape(-1,1)

                uhat[:nnn_x,:nnn_y]  = aaa*c
                uhat[-nnn_x:,:nnn_y] = bbb*c
                u = scipy.fft.irfft2(uhat)
                return u  

        else:
            raise ValueError('rand_u0: rand_init')
        return u0  
    
#---------------------------
#@title CSolverBurgers 
class CSolverBurgers:

    def __init__(self,sys_name='Burgers_RK4', list_para=[1], nDIM = 1) :
        if sys_name not in ['Burgers_RK4']:
            warnings.warn('do not find CSolverBurgers.sys_name')

        self.sys_name = sys_name
        self.list_para = list_para  # Burgers Equation
        self.nDIM = nDIM

        if self.list_para == [1] and self.sys_name =='Burgers_RK4':
            self.Lpi = list_para [0]

            if self.nDIM == 1: 
                self.N   = 256
                self.dt_Output, self.NumOutPut_dt = libBurgers.get_OutputTimeStep(self.N,self.Lpi)
                self.default_training_testing_infolist_generate_data =[
                            ( 350, 350+35, 501, 1, 'rand_FFT'  ),    ]
            elif self.nDIM == 2:

                raise ValueError('Not implemented yet, the following is for 2D-KDV, not 2D-Burgers')
                self.N   = 128
                self.dt_Output, self.NumOutPut_dt = libBurgers.get_OutputTimeStep(self.N,self.Lpi)
                self.default_training_testing_infolist_generate_data =[
                            ( 350, 350+35, 501, 1, 'rand_FFT'  )     ]
        else:
            raise ValueError('CSolverBurgers.list_para')
        self.__print__()

    def __print__(self):
        print('CSolverBurgers: nDIM=',self.nDIM, ' N=', self.N, ' Lpi=',self.Lpi,' dt_Output=', self.dt_Output, 'NumOutPut_dt=', self.NumOutPut_dt ) 
        return


    def generate_dsol_single(self, len_seq, each_parameter, init_op_string_or_value):
        t_Seq = np.arange(0, len_seq)*self.dt_Output

        if type(init_op_string_or_value) is str:
            d0    = libBurgers.rand_u0(self.N, self.Lpi, init_op_string_or_value, self.nDIM) 
        else:
            d0 = init_op_string_or_value        

        if self.sys_name == 'Burgers_RK4' :
            each_Lpi = each_parameter
            dsol = libBurgers.RK4NumericalSolver_usol( d0, t_Seq, n_dt=self.NumOutPut_dt, Lpi=each_Lpi, nDIM=self.nDIM )

        assert np.any( np.isnan( dsol  ) )==False, "nan in generate_dsol_single"
        return dsol


    def generate_or_load_DEFAULT_sol_list(self, dir_save_training_data = None, bForcedRegendsol = False, bForceNoSavedsol=False ):
        
        ListAll_xsol, ListAll_pde_para, ListAll_num_traj_train = self.generate_or_load_xsol_list( self.default_training_testing_infolist_generate_data, 
                                                                                            dir_save_training_data,  bForcedRegendsol, bForceNoSavedsol )

        list_xsol_train = []
        list_xsol_test = []
        list_pde_para_train = []
        list_pde_para_test = []

        for xsol, pde_para,  num_traj_training in zip(ListAll_xsol, ListAll_pde_para, ListAll_num_traj_train):
            
            if( num_traj_training >= 1):
                list_xsol_train.append( xsol[:num_traj_training] )
                list_pde_para_train.append( pde_para )
            
            if ( num_traj_training < xsol.shape[0] ):
                list_xsol_test.append( xsol[num_traj_training:] )
                list_pde_para_test.append( pde_para )   

        return list_xsol_train, list_xsol_test, list_pde_para_train, list_pde_para_test
    

    def generate_or_load_xsol_list( self, infolist_generate_data, dir_save_training_data = None, bForcedRegendsol = False, bForceNoSavedsol=False ):

        print(infolist_generate_data)

        ListAll_xsol=[]
        ListAll_pde_para =[]
        ListAll_num_traj_train = []

        for num_traj_training, num_traj, len_seq, pde_para, init_op_string in infolist_generate_data:
            print('num_traj_training=', num_traj_training, 'num_traj=', num_traj, 'leq_seq=', len_seq, ' pde_para=', pde_para, ' ', init_op_string )

            #---------------------------------------------
            if 'Burgers_RK4' == self.sys_name :
                if self.nDIM == 1:
                    Burgers_name_prefix = 'Burgers_dsol_multraj'
                elif self.nDIM == 2:
                    Burgers_name_prefix = 'Burgers2dSq_dsol_multraj'

                pkl_filename_dsol_multraj = Burgers_name_prefix + '{:d}'.format(num_traj) + 'L{:d}'.format(len_seq) + '_Lpi{:d}'.format(pde_para)+ '_N' + '{:d}'.format(self.N) + '_'+ init_op_string
                pkl_filename_dsol_multraj += '.pkl'
            #---------------------------------------------

            picklefilename = dir_save_training_data + pkl_filename_dsol_multraj

            if os.path.isfile (picklefilename) and (bForcedRegendsol==False):
                open_file = open(picklefilename, "rb")
                data_load = pickle.load(open_file)
                open_file.close()
                dsol_multraj = data_load['dsol_multraj']

                assert num_traj == data_load['num_traj']
                assert len_seq  == data_load['len_seq']
                assert self.N   == data_load['N']
                assert self.dt_Output == data_load['dt_Output']
                if self.sys_name == 'Burgers_RK4':
                    assert pde_para == data_load['Lpi']
                print('Success: load ' + picklefilename )

                #-------
            else:  # fresh generate data if the pickle file does not exist or bForceRegen==True
                #-------

                if bForcedRegendsol==False:
                    print('The file ' +  picklefilename + ' do not exist, therefore do a fresh generation')

                if   self.nDIM == 1: dtype = np.float64
                elif self.nDIM == 2: dtype = np.float32   # ! save storage space for 2d data 

                dsol_multraj = np.zeros( (num_traj, len_seq, *(self.nDIM*[self.N])  ) , dtype = dtype )

                for i in range(num_traj):
                    dsol_multraj[i,...] = self.generate_dsol_single(len_seq, pde_para, init_op_string )
                    print('', end ='.')
                print('')

                if bForceNoSavedsol == False:
                    if self.sys_name == 'Burgers_RK4' :
                        data_for_save = {'dsol_multraj':dsol_multraj, 'num_traj':num_traj,'len_seq':len_seq, 'Lpi':pde_para, 'N':self.N, 'dt_Output':self.dt_Output,'init_op_string':init_op_string}

                    open_file = open(picklefilename, "wb")
                    pickle.dump(data_for_save, open_file)
                    open_file.close()
                    print('saving ' + picklefilename )

            #-------------------------------------------
            #-------------------------------------------
            ListAll_xsol.append( dsol_multraj )
            ListAll_pde_para.append( pde_para )
            ListAll_num_traj_train.append(num_traj_training)

        return  ListAll_xsol, ListAll_pde_para, ListAll_num_traj_train
    