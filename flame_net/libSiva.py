

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
import warnings
#import torch

from flame_net.libData import libData


from scipy.optimize import fsolve



def round_num_to_txt(num):
    if num<1:    return '{:g}'.format(num)[2:]   # turn 0.02 into 02,
    else:         return '{:g}'.format(num)






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

            usol = libSiva.RK4NumericalSolver_usol( u0, t,  params['Lpi'], params['mu'],params['nu'], params['d_ratio'], params['tau'],  params['n_dt'])

            plt.contourf(usol.transpose(1,0), 20); plt.colorbar()
            #plt.imshow(usol.transpose(1,0))  #plt.colorbar()

            if params['show_ani']==True:
                k = libSiva.get_k(N,L)
                dsol = libSiva.usol_to_dsol(usol,k)
                x = libSiva.get_x(N, L )
                ani = libSiva.anim_N_dsol(t, x, [dsol], nskip=4, n0=0, interval=0.1, ylim_plot=[-L/2 , L/2])
                return ani

        elif method =='MS_RK4':
            params.setdefault('nu',0.025)
            params.setdefault('N',256)
            params.setdefault('Nstep',1000)
            params.setdefault('randfft',False)
            params.setdefault('show_ani',False)
            params.setdefault('n_dt',5)
            if params['randfft']==True:     d0 = libSiva.rand_d0_FFT( params['N'], np.random.randint(low=2, high=8) )
            else:                           d0 = libSiva.rand_d0_simple( params['N'] , {'siva_sys_name':'MS_RK4', 'para_value':params['nu'] } )
            return libSiva.demo('ks',params = {'Lpi':1, 'mu':0, 'nu':-params['nu'], 'd_ratio':1, 'N':params['N'], 'dt':0.015, 'Nstep':params['Nstep'], 'n_dt':params['n_dt'], 'show_ani':False,'d0':d0, 'show_ani':params['show_ani']}  )

        elif method =='KS_RK4':
            params.setdefault('Lpi',25)
            params.setdefault('N',256)
            params.setdefault('Nstep',1000)
            params.setdefault('randfft',False)
            params.setdefault('show_ani',False)
            if params['randfft']==True:     d0 = libSiva.rand_d0_FFT( params['N'], np.random.randint(low=2, high=8) )
            else:                           d0 = libSiva.rand_d0_simple( params['N'] , {'siva_sys_name':'KS_RK4', 'para_value':params['Lpi'] } )
            return libSiva.demo('ks',params = {'Lpi':params['Lpi'], 'mu':1, 'nu':1, 'd_ratio':0, 'N':params['N'], 'dt':0.15, 'Nstep':params['Nstep'], 'n_dt':5, 'show_ani':False,'d0':d0 , 'show_ani':params['show_ani'] }  )

        elif method =='MKS_RK4':
            params.setdefault('max_amplitude',0.25)
            params.setdefault('rho',0.5)
            params.setdefault('Lpi',30)

            params['d_ratio'] = params['rho']* (4*params['max_amplitude'])
            params['mu'], params['nu'], params['tau'] = libSiva.Coeff_Lpi_d_to_mu_nu_tau( params['Lpi'], params['d_ratio'], params['max_amplitude'] )

            params.setdefault('N',256)
            params.setdefault('dt',0.015*params['Lpi'])
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



        elif method=='MS_1stEuler':
            params.setdefault('Nstep',1000)
            params.setdefault('nu',0.025)
            params.setdefault('dt',0.015)
            params.setdefault('N',256)
            params.setdefault('randfft',False)
            params.setdefault('show_ani',False)


            for key in params:
                if key == 'd0':
                    print(key+": given", end=' , ' )
                else:
                    print(key+":", params[key],end=' , ' )
            print('')

            dt =params['dt']
            t = np.arange( params['Nstep'])*dt

            (nu, L, N) = (params['nu'], 2*np.pi, params['N'])
            (ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (0, 1, 0)
            x = libSiva.get_x(N)
            #d0 = libSiva.rand_d0_FFT(N, np.random.randint(low=2, high=8))

            if params['randfft']==True:  d0 = libSiva.rand_d0_FFT( params['N'], np.random.randint(low=2, high=8) )
            else:                        d0 = libSiva.rand_d0_simple( params['N'] , {'siva_sys_name':'MS_1stEuler', 'para_value':params['nu'] } )

            # plt.plot(x,d0)
            dsol = libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N,L, method_TimeStepping='default_uniform')

            #---
            k = libSiva.get_k(N,L)
            usol = libSiva.dsol_to_usol( dsol, k)
            plt.contourf(usol.transpose(1,0), 20); plt.colorbar()

            #plt.imshow(usol.transpose(1,0))  #plt.colorbar()
            if params['show_ani']==True:
                (ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (.003, 1, 0)
                dsol2 = libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N,L, method_TimeStepping='default_uniform')
                ani = libSiva.anim_N_dsol(t, x, [dsol,dsol2], nskip=10, n0=0, interval=1)
                return ani

        elif method=='ms2':
            dt = 0.015
            t = np.arange(2000)*dt
            (L, N) = (2 * np.pi, 1024)
            (ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (0, 1, 0)
            x = libSiva.get_x(N)
            d0 = libSiva.rand_d0_FFT(N, np.random.randint(low=2, high=8))
            nu = 0.01
            dsol = libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L, method_TimeStepping='default_uniform')
            nu = 0.2
            dsol2 = libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L, method_TimeStepping='default_uniform')
            nu = 0.07
            dsol3 = libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L, method_TimeStepping='default_uniform')
            nu = 0.04
            (ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (0.001, 1, 0)
            dsol4 = libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L, method_TimeStepping='default_uniform')
            ani = libSiva.anim_N_dsol(t, x, [dsol, dsol2, dsol3,dsol4], nskip=1, n0=0, interval=1)
            return ani

        elif method=='ms3':
            dt = 0.015
            t = np.arange(2000)*dt
            (L, N) = (2*np.pi, 512)
            (ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (0, 1, 0)
            # (ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (.003, 1, 0)
            # (nu, L, N) = (0.05, 2*np.pi, 128)
            ##(ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (.001, 1, 0)
            x = libSiva.get_x(N)
            d0 = libSiva.rand_d0_FFT(N, np.random.randint(low=2, high=8))
            # d0  = libSiva.rand_d0_simple(N)

            nu = 0.05
            dsol = libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N,
                                                L, method_TimeStepping='default_uniform')

            #t2 = 0.017490592/0.015*t
            t2 = t
            dsol2 =libSiva.MS__NumericalSolver_dsol(d0, t2, (0.05 ,1.5), ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L, method_TimeStepping='default_uniform' )

            ani =  libSiva.anim_N_dsol(t,x,[dsol, dsol2], nskip=10, n0=0, interval=1, ylim_plot = (-2,4) )
            return ani
        elif method=='ms4':
            dt = 0.015
            t = np.arange(2000)*dt
            (L, N) = (2*np.pi, 128)
            (ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (0, 1, 0)
            x = libSiva.get_x(N)
            #d0 = libSiva.rand_d0_FFT(N, np.random.randint(low=2, high=8))
            d0  = libSiva.rand_d0_simple(N)
            nu= 0.05
            dsol = libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L, method_TimeStepping='default_uniform')

            #dsol_an_ivp, t_an, bSucess = libSiva.MS__AnalyticalSolver_ivp_filteredspectral( d0,[0,0.1],nu,libSiva.get_k(N,L)   )
            dsol_an_filtsp = libSiva.MS__AnalyticalSolver_ivp_filteredspectral( d0,t,nu,1,libSiva.get_k(N,L)   )
            #dsol_an_ivp, t_an, bSucess = libSiva.MS__AnalyticalSolver_ivp_spectral( d0,t,nu,libSiva.get_k(N,L)            )  # , t_eval=t )
            dsol_an_ivp, t_an, bSucess  = libSiva.MS__AnalyticalSolver_ivp_dsol(  d0,t,nu,1, libSiva.get_k(N,L), t_eval=t)     # , method='BDF' )
            dsol_an_ivp2,t_an2,bSucess2 = libSiva.MS__AnalyticalSolver_ivp_dsol(  d0,t,nu,2, libSiva.get_k(N,L), t_eval=t)     # , method='BDF' )

            # %matplotlib
            ani =  libSiva.anim_N_dsol(t_an,x,[dsol,dsol_an_filtsp, dsol_an_ivp, dsol_an_ivp2], nskip=1, n0=0, interval=1)
            return ani

        elif method=='ms5':

            T = 10
            t = np.linspace(0, T, 1001)

            # T = 1500 ;        t = np.linspace(0,T,100001)
            # (L, N) = (2*np.pi, 128)
            # (L, N) = (2*np.pi, 512)
            (L, N) = (2 * np.pi, 1024)
            # (L, N) = (2*np.pi, 4096)

            (ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (0, 1, 0)
            # (ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (.003, 1, 0)

            # (nu, L, N) = (0.05, 2*np.pi, 128)
            ##(ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise) = (.001, 1, 0)

            x = libSiva.get_x(N)

            d0 = libSiva.rand_d0_FFT(N, np.random.randint(low=2, high=8))
            # d0  = libSiva.rand_d0_simple(N)
            # plt.plot(x,d0)

            # d0 = .35*np.sin( x *-2.353 + 0.8 )
            # d0 = .35*np.sin( x *-2.353 +.8  )
            # plt.plot(x,scipy.fftpack.fft(d0).imag)

            # nu= 0.06
            # nu= 0.05
            nu = 0.02
            # nu= 0.01
            dsol = libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N,
                                                L, method_TimeStepping='default_uniform')

            # t2 = 0.017490592/0.015*t
            # dsol2 =libSiva.MS__NumericalSolver_dsol(d0, t2, (0.051313028 ,0.8041635), ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L, method_TimeStepping='default_uniform' )

            # dsol_an_ivp, t_an, bSucess = libSiva.MS__AnalyticalSolver_ivp_filteredspectral( d0,[0,0.1],nu,libSiva.get_k(N,L)   )
            # dsol = libSiva.MS__AnalyticalSolver_ivp_filteredspectral( d0,t,nu,libSiva.get_k(N,L)   )

            # dsol_an_ivp, t_an, bSucess = libSiva.MS__AnalyticalSolver_ivp_spectral( d0,t,nu,libSiva.get_k(N,L) )  # , t_eval=t )
            # dsol_an_ivp,   t_an, bSucess = libSiva.MS__AnalyticalSolver_ivp_dsol( d0,t,nu,libSiva.get_k(N,L),1, t_eval=t)     # , method='BDF' )
            # dsol_an_ivp2, t_an2, bSucess2 = libSiva.MS__AnalyticalSolver_ivp_dsol( d0,t,nu,libSiva.get_k(N,L),2, t_eval=t)     # , method='BDF' )

            # %matplotlib
            # widget
            # ani =  libSiva.anim_N_dsol(t_an,x,[dsol_an_ivp, dsol_an_ivp2], nskip=1, n0=0, interval=1)

            ani = libSiva.anim_N_dsol(t, x, [dsol], nskip=10, n0=0, interval=1)
            # ani =  libSiva.anim_dsol_FFT(t,x,dsol, nskip=10, n0=0, interval=1, kNum_Trunc = 10)

            # ani =  libSiva.anim_N_dsol(t,x,[dsol, dsol2], nskip=10, n0=0, interval=1, ylim_plot = (-5,10) )
            ani

            curvsol = libData.dsol_to_curvsol(x,dsol)
            ani = libSiva.anim_dsol_and_curvsol(t,x,dsol,curvsol,nskip=10, n0=0, interval=1)
            ani
            return ani

            def plot_more_curvsol(curvsol):
                nPlot =  -3
                alpha_angle, mag = libData.curvsol_to_angle_magnitude(curvsol)
                plt.figure(1)
                plt.plot(curvsol[nPlot,:,0],curvsol[nPlot,:,1], '-o')
                plt.figure(2)
                plt.plot( alpha_angle[nPlot,:]/np.pi*180, '-s')
                plt.figure(3)
                plt.plot( mag[nPlot,:],'-<')
                return

            plot_more_curvsol(curvsol)

            #ActiveNoise_stepfeq = 100
            #dsol2 =libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L )

            #(nu, L, N) = (.25, 2*np.pi, 128)
            #(nu, L, N) = (.25, 2*np.pi, 128)
            #dsol2 =libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L)
            (nu, L, N) = (.7, 2*np.pi, 128)
            #(nu, L, N) = (.25, 2*np.pi, 128)
            dsol2 =libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L)

            (nu, L, N) = (.125, 2*np.pi, 128)
            #(nu, L, N) = (.25, 2*np.pi, 128)
            dsol3 =libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L)

            (nu, L, N) = (.5, 2*np.pi, 128)
            #(nu, L, N) = (.25, 2*np.pi, 128)
            dsol4 =libSiva.MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude, ActiveNoise_stepfeq, k_ActiveNoise, N, L)

            #%matplotlib
            #libSiva.plot_dsol(x,dsol, [ (0,'k'), (100,'--b') , (1000,'-r')])
            ani =  libSiva.anim_N_dsol(t,x,[dsol,dsol2,dsol3,dsol4], nskip=1, n0=0, interval=1)
            #ani


            SivaEq = CSolverSiva('nuMulti_0noise')
            training_infolist_generate_data=[
                            (10, 501, 0, 'rand_FFT_2_8' ),
                            (10, 501, 0, 'rand_simple' ) ,
                            (10, 501, 1, 'rand_FFT_2_8' ),
                            (10, 501, 1, 'rand_simple' ) ,
                            (10, 501, 2, 'rand_FFT_2_8' ),
                            (10, 501, 2, 'rand_simple' ) ,
                            (10, 501, 3, 'rand_FFT_2_8' ),
                            (10, 501, 3, 'rand_simple' )          ]

            list_dsol,list_nu = SivaEq.generate_dsol_list( training_infolist_generate_data )
            d_all , nu_all= libData.Reorg_list_dsol(list_dsol, list_nu, T_out=20, T_in=1 )


            #%matplotlib
            t = np.linspace(0,15,501)
            x  = libSiva.get_x(128)
            ani =  libSiva.anim_N_dsol(t,x, list_dsol[6], nskip=1, n0=0, interval=1)

            return ani
    @staticmethod
    def MS__NumericalSolver_dsol(d0, t, nu, ActiveNoise_Amplitude=0, ActiveNoise_stepfeq=1, k_ActiveNoise=0, N=128, L=2*np.pi, numSubSteps_PerTimeStep=0,method_TimeStepping='uniform'):
        x = libSiva.get_x(N,L)
        k = libSiva.get_k(N,L)
        u0  = libSiva.d_to_u( d0, k)
        usol = libSiva.MS__NumericalSolver_usol( u0 , t,  nu,  ActiveNoise_Amplitude,ActiveNoise_stepfeq,k_ActiveNoise, N, L, numSubSteps_PerTimeStep,method_TimeStepping=method_TimeStepping)
        dsol = libSiva.usol_to_dsol(usol,k)
        return dsol


    @staticmethod
    def get_dt_StabilityLimit(nu,N,L,method_new_or_old='new_method'):
        dx    = L/N
        #time step for PDE integration
        if method_new_or_old == 'new_method':
            if N>=128:
                CFL =  0.1
            elif N==64:
                CFL =  0.01

            cond2 = CFL*dx/3   # max(abs(u)); % Condition for the inviscid Burgers equation
            cond_dt  = cond2      # implicit scheme already handle diffusive-stablity-limit
        else: # old_method
            CFL = 0.2
            cond1 = 0.2 * (dx ** 2) / (2 * nu)  # Condition for the heat equation
            cond2 = CFL * (dx) / 1              # Condition for the inviscid Burgers equation
            cond_dt = np.min( [cond1, cond2])      # Minimum to use the most restrictive condition
        return cond_dt

    @staticmethod
    def get_numSubSteps_PerTimeStep_default(dt,nu,N,L):
        return int( dt//libSiva.get_dt_StabilityLimit(nu,N,L,'new_method') + 1 )

    @staticmethod
    def MS__NumericalSolver_usol(u0,t,nu_or_tupleofnucRatio, ActiveNoise_Amplitude=0, ActiveNoise_stepfeq=1, k_ActiveNoise=0, N=128, L=2*np.pi, numSubSteps_PerTimeStep=0, method_TimeStepping='uniform'):
        #############################
        dx = L/N
        if type(nu_or_tupleofnucRatio) == tuple:
            nu, cRatio = nu_or_tupleofnucRatio
        else:
            nu, cRatio = nu_or_tupleofnucRatio, 1



        k = libSiva.get_k(N,L)
        k1 = 1j*k
        k2 = k1**2

        if k_ActiveNoise == 0:
            k_ActiveNoise = k.size//2  #- 14

        u = u0

        usol = np.zeros( ( t.size, u0.size ) )
        usol[0, : ] = u
        ########################
        dt_Output = t[1]-t[0]
        #-----------------
        if numSubSteps_PerTimeStep<=0:
            numSubSteps_PerTimeStep = libSiva.get_numSubSteps_PerTimeStep_default(t[1]-t[0],nu,N,L )
        assert  numSubSteps_PerTimeStep>=1 , "numSubSteps_PerTimeStep>=1"


        if  'adapt' in method_TimeStepping.casefold():
            for j in range( (t.size-1)):

                CFL = 0.1
                if N==64 and nu<0.03:
                    CFL = 0.01

                dt_sub = CFL*dx/max( (max(u), 3) )
                numSubSteps_PerTimeStep_Adaptive_j = int( dt_Output//dt_sub + 1 )
                numSubSteps_PerTimeStep_Adaptive_j = np.max( (numSubSteps_PerTimeStep, numSubSteps_PerTimeStep_Adaptive_j ) )
                dt = dt_Output/numSubSteps_PerTimeStep_Adaptive_j
                if numSubSteps_PerTimeStep_Adaptive_j>=1000:
                    raise ValueError('too large numSubSteps_PerTimeStep_Adaptive_j='+str(numSubSteps_PerTimeStep_Adaptive_j) )
                #------------

                q  = 1-dt*cRatio*( np.abs(k) + nu*k2 )

                #------------
                for m in range(numSubSteps_PerTimeStep_Adaptive_j):

                    u1=scipy.fft.rfft( u        )
                    u2=scipy.fft.rfft( 0.5*u**2 )
                    # Calculate the u^n+1 term
                    u_hat=(-dt*k1*u2+u1)/q

                    if m%ActiveNoise_stepfeq == 0:
                        u_hat[k_ActiveNoise] = u_hat[k_ActiveNoise] + (2*np.random.rand(1)-1)*ActiveNoise_Amplitude

                    u=scipy.fft.irfft(u_hat) #.real

                #------------
                usol[ 1+j , : ] = u

        elif   'uni' in method_TimeStepping.casefold():
            if (t.size > 1000):
                print('MS__NumericalSolver_usol: large t.size=(', t.size, '); a single-dot is printed for 1000 steps')

                #-----------------
            dt = dt_Output /numSubSteps_PerTimeStep
            q  = 1 - dt*cRatio*(np.abs(k) +nu*k2 )
            #-----------------
            #for j in range( (t.size-1) ):
            for j in range(1, t.size):
                for i_ in range(numSubSteps_PerTimeStep):
                    u1=scipy.fft.rfft(u)
                    u2=scipy.fft.rfft(0.5*u**2)
                    # Calculate the u^n+1 term
                    u_hat=(-dt*k1*u2+u1)/q
                    #if ActiveNoise ==True:
                    #uhat[ke_nosie-11:ke_nosie] = uhat[ke_nosie-11:ke_nosie] + (2*np.random.rand(11)-1)*.01
                    if not(ActiveNoise_Amplitude==0):
                        if (j*numSubSteps_PerTimeStep+i_) % ActiveNoise_stepfeq == 0:
                            u_hat[k_ActiveNoise] = u_hat[k_ActiveNoise] + (2*np.random.rand(1)-1)*ActiveNoise_Amplitude
                    u=scipy.fft.irfft(u_hat) #.real

                #if (j+1)%numSubSteps_PerTimeStep == 0:
                usol[j, : ] = u
                if( j%1000==0):
                    print( '', end='.')
            #print('')
        return usol

    @staticmethod
    def RK4NumericalSolver_dsol(d0,t,Lpi=6.28, mu=1, nu=1, d_ratio=0, tau=1, n_dt=1):
        N = d0.shape[0]
        L = 2*np.pi  # *Lpi
        k = libSiva.get_k(N,L)
        u0  = libSiva.d_to_u( d0, k)
        usol = libSiva.RK4NumericalSolver_usol( u0 , t, Lpi, mu, nu, d_ratio, tau, n_dt)
        dsol = libSiva.usol_to_dsol(usol,k)
        return dsol

    #---------------
    # mu,nu, tau = libSiva.Coeff_Lpi_d_to_mu_nu_tau(Lpi, d_ratio, max_amplitude ):
    @staticmethod
    def Coeff_Lpi_d_to_mu_nu_tau(Lpi, d_ratio, max_amplitude=0.25):
        #-----------------------------
        def d_to_mu_nu(d_ratio, max_amplitude=0.25):
            #--------------------
            def func_SivaEq_Coeff_Relations(arg, d_ratio, max_amplitude=0.25 ):
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
        #-----------------
        mu,nu = d_to_mu_nu(d_ratio, max_amplitude )
        tau = (Lpi/10*d_ratio) + (1-d_ratio)
        return mu,nu, tau

    @staticmethod
    def RK4NumericalSolver_usol(u0,t,Lpi=6, mu=1,nu=1, d_ratio=0,  tau=1, n_dt=1 , noise_level = 0,    bSym = False  ):
        '''
            SivashinkskyEquation for the slope u(x,t)

                u_t /tau + u*u_x + (mu)*u_xxxx + (nu)*u_xx  - (d_ratio)*Gamma(u) =0

            with peroidic condition u(x,t) = u(x+L,t) and domain size being L = 2*pi*(Lpi)

            Note:
            (i ) It reduce to Kuramato-Sivashinksky(KS) Equation when d_ratio=0 and nu<0
                   u_t + u*u_x  + (mu)*u_xxxx  + (nu)*u_xx               = 0
            (ii) It reduce to Micheal-Sivashinksky(KS) Equation through setting mu=0, nu>0 and d_ratio=1
                   u_t + u*u_x                 + (nu)*u_xx    - Gamma(u) = 0

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
        dt = dt*tau              # tau can be considered to re-scale the output time step

        L = 2*np.pi *Lpi

        if bSym == False:
            usol = np.zeros( (t.size,  u0.size) )
            usol[0, : ] =  u
        elif bSym==True:
            usol = np.zeros( (t.size,  u0.size) )

        if (t.size > 1000):     print('RK4NumericalSolver_usol: large t.size=(', t.size, '); a single-dot is printed for 1000 steps')

        if bSym == True:
            k = np.arange(N+1) * (  2*np.pi/(2*L)  )

            uhat = scipy.fft.rfft( np.concatenate( (u, -np.flip(u,-1) ) , -1 )  )

            corr_half = np.exp( - 1j * 2*np.pi* np.arange(N+1)  * 0.5/(2*N) )

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
                usol[j+1, :] = scipy.fft.irfft(uhat) [...,:N]

            elif bSym == False:
                usol[j+1, :] = scipy.fft.irfft(uhat)

            if( j%1000==0):  print( '', end='.')

        return usol

    # @staticmethod
    # def RK4NumericalSolver_usol(u0,t,Lpi=6.28, mu=1,nu=1, d_ratio=0,n_dt=1 ):
    #     '''
    #         SivashinkskyEquation for the slope u(x,t)
    #
    #             u_t + u*u_x + (mu)*u_xxxx - (nu)*u_xx  - (d_ratio)*Gamma(u) =0
    #
    #         with peroidic condition u(x,t) = u(x+L,t) and domain size being L = 2*pi*(Lpi)
    #
    #         Note:
    #         (i ) It reduce to Kuramato-Sivashinksky(KS) Equation when d_ratio=0 and nu<0
    #                u_t + u*u_x  + (mu)*u_xxxx  + (-nu)*u_xx  = 0
    #         (ii) It reduce to Micheal-Sivashinksky(KS) Equation through setting mu=0, nu>0 and d_ratio=1
    #                u_t + u*u_x                  - (nu)*u_xx    - Gamma(u) =0
    #
    #         ---  The RK4-solution process----
    #         step1:
    #                 u = sum { uhat(k) * e^(2*pi*k * x/L )  }
    #         step2:
    #                 d/dt(u_hat) +  [ mu*(2**pi/L*k)^4  - nu*(2*pi/L*k)^2  - d_ratio*(2*pi/L*|k|) ]*u_hat = -0.5*i* (2*pi/L*k) u2_hat
    #         step3(integrator factor):
    #                 g = e^(lambda*t)     , lambda =   mu*(2*pi/L*k)^4  - nu*(2*pi/L*k)^2 - d_ratio*(2*pi/L*|k|)
    #         step4:
    #                 d/dt[ e^(lambda*t) * uhat  ] = -0.5*i* (2*pi/L*k) u2_hat * e^(lambda*t)
    #     '''
    #     u = u0
    #
    #     N = u0.shape[0]
    #
    #     dt = (t[1]-t[0] )/n_dt
    #
    #     usol = torch.zeros( (t.size, u0.size) )
    #     usol[0, : ] = torch.tensor(u)
    #
    #     uhat = torch.fft.rfft(torch.tensor(u))
    #
    #     if (t.size > 1000):     print('RK4NumericalSolver_usol: large t.size=(', t.size, '); a single-dot is printed for 1000 steps')
    #
    #     L = 2*np.pi *Lpi
    #     k = torch.arange(N//2+1) * (2*np.pi/L)
    #     for j  in range( t.size -1 ) :
    #         #t = n*dt
    #         for n in range(n_dt):
    #             g = -0.5j * dt * k
    #             E = torch.exp( -dt/2 * (       mu* k**4 - nu* k**2- d_ratio*k  )  )
    #             E2 = E**2
    #             a = g* torch.fft.rfft( torch.fft.irfft(   uhat       )**2 )
    #             b = g* torch.fft.rfft( torch.fft.irfft( E*(uhat+a/2) )**2)   #  4th order
    #             c = g* torch.fft.rfft( torch.fft.irfft( E*uhat+b/2   ) **2 ) #  Runge-Kutta
    #             d = g* torch.fft.rfft( torch.fft.irfft( E2*uhat+E*c  )**2 )
    #             uhat = E2 * uhat + (E2 * a + 2*E*(b+c) + d)/6
    #
    #         #-----
    #         usol[j+1, :] = torch.fft.irfft(uhat)
    #         if( j%1000==0):  print( '', end='.')
    #
    #     return usol



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

        #min_v = 999
        #max_v = -999
        #for dsol in dsol_list:
        #    min_v = np.min( min_v, np.min(dsol ) )
        #    max_v = np.max( max_v, np.max(dsol ) )
        #ax.set_ylim( min_v, max_v )

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
        #k = list( range(0, N))
        N_2 = N//2
        k_2 = list ( range ( N_2//2) )
        #kNum_Trunc = N//20
        #kNum_Trunc = 5

        def Truc_d_fft(d_fft, kNum_Trunc=20):
            truc_d_fft= np.zeros_like(d_fft)
            truc_d_fft[:kNum_Trunc] = d_fft[:kNum_Trunc]
            truc_d_fft[-kNum_Trunc+1:] =d_fft[-kNum_Trunc+1:]
            return truc_d_fft

        def mono_angle( angle ):
            angle_monotic = angle
            #for i  in range (1, angle_monotic.size) :
            #    if angle_monotic[i]<angle_monotic[i-1]:
            #        angle_monotic[i:] = 2*np.pi + angle_monotic[i:]
            return angle_monotic

        def recon_d_using_logfft(abs__log_d_fft, angle ):
            d_fft = np.zeros(N,dtype=complex)
            trun_abs__log_d_fft = abs__log_d_fft
            trun_angle = angle
            #trun_abs__log_d_fft = scipy.fftpack.ifft( Truc_d_fft( scipy.fftpack.fft(abs__log_d_fft), 35 ) )
            #trun_angle = scipy.fftpack.ifft( Truc_d_fft( scipy.fftpack.fft(angle), 20 ) )

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
            #line0_2.set_data(x, scipy.fftpack.ifft(  recon_d_using_logfft(log_d_fft.real,  angle_monotic ) ) )

            #power = 2
            #dm_fft = scipy.fftpack.fft( np.power( np.abs(d)  , power) )

            dm_fft = scipy.fftpack.fft(  np.tanh( d*3 ) )
            dm = scipy.fftpack.ifft(  Truc_d_fft( dm_fft, kNum_Trunc) ).real
            #line0_2.set_data(x, np.sign(d)* np.power( dm, 1./power )    )
            line0_2.set_data(x, np.arctanh(dm)/3    )



            line1_0.set_data( k, log_d_fft.real  )
            line1_1.set_data( k, angle_monotic )
            #ax1_twin.set_ylim( angle_monotic[0], angle_monotic[-1])
            ax1_twin.set_ylim( -np.pi,  np.pi)

            #################
            #line2_0.set_data(k_2,   logfft_log_d_fft.real      )
            #line2_1.set_data(k_2,  logfft_angle_monotic.real    )


        d = dsol[0,:]
        d_fft, log_d_fft, angle_monotic,logfft_log_d_fft,logfft_angle_monotic = Calc_various_va( d )
        d3_fft = scipy.fftpack.fft( np.power(dsol[0,:], 3) )

        fig, ax = plt.subplots(2,1, figsize=(7, 10) )
        fig.tight_layout()


        line0_0, = ax[0].plot(x, dsol[0,:], 'k-', linewidth=1) ;       ax[0].set_ylim(-1, 2)
        line0_1, = ax[0].plot(x, scipy.fftpack.ifft( Truc_d_fft(d_fft, kNum_Trunc) ).real , 'r-.', linewidth=1)

        line0_2, = ax[0].plot(x, np.power(  scipy.fftpack.ifft(  Truc_d_fft( d3_fft, kNum_Trunc) ).real  , 1./3 ) , 'b--' )

        #line0_2, = ax[0].plot(x, scipy.fftpack.ifft(  recon_d_using_logfft(log_d_fft.real, angle_monotic) ),  'b-o' )

        #log_d_fft = np.log(d_fft[1:N//2])
        line1_0, = ax[1].plot(k , log_d_fft.real , 'r-s', linewidth=1);         ax[1].set_ylim(-8, 5)
        #ax[1].set_ylim(.0001, 100)

        ax1_twin = ax[1].twinx()
        line1_1, = ax1_twin.plot (k ,  angle_monotic  , 'b--o', linewidth=1) ;
        ax1_twin.set_ylim( angle_monotic[0], angle_monotic[-1])


        #################
        #line2_0, = ax[2].plot(k_2,   logfft_log_d_fft.real  ,  'r--o', linewidth=1)
        #ax[2].set_ylim(-2,5)
        #ax2_twin = ax[2].twinx()
        #line2_1, = ax2_twin.plot(k_2,  logfft_angle_monotic.real , 'k--s', linewidth=1)
        #ax2_twin.set_ylim(4, 10)
        #################

        frames = (dsol.shape[0]-1 - n0)//nskip -1 ;
        ani = animation.FuncAnimation(fig, update_animation_flame_FFT, frames= frames, fargs=(dsol,x,t,nskip,n0), interval=interval,blit=False)
        #Note: below is the part which makes it work on Colab
        #rc('animation', html='jshtml')
        return ani

    @staticmethod
    def plot_dsol(x,dsol,plt_list,):
        #%matplotlib
        fig,ax = plt.subplots()
        for i, style in plt_list:
            ax.plot(x,dsol[i,:],style)   # ; hold on
            #ax.plot(x,dsol[0,:],'--k')   # ; hold on
            #ax.plot(x,dsol[100,:],'-.b')  #; hold on
            #ax.plot(x,dsol[300,:],'-.b')  #; hold on
            #ax.plot(x,dsol[1000,:],'-r')
            #ax.plot(x,dsol[10000,:],'+r')
        ax.grid()
        plt.show()

    @staticmethod
    #def rand_d0_simple(N, scale=0.1):
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
        #d_hat[-k_trun+1:] = np.conj( np.flip( d_hat[1:k_trun] ) )
        d0    = scipy.fft.irfft(d_hat) #.real
        return d0


    """ spectral rhs"""
    @staticmethod
    def np_complex__viewed_as_real2( complex_array ):
        #sh = complex_array.shape
        #return np.stack ( (complex_array.real, complex_array.imag ), axis = -1 ).reshape(*sh[:-2], -1 )
        return complex_array

    @staticmethod
    def np_real2__viewed_as_complex( real2_array ):
        #return real2_array[ ... , 0::2 ] + 1j *  real2_array[ ... , 1::2 ]
        return real2_array

    @staticmethod
    def MS__siva_filter_v_rhs_fft_for_ivp(t, v_hat__viewed_as_real , nu,cRatio, k  ):
        v_hat= libSiva.np_real2__viewed_as_complex( v_hat__viewed_as_real )
        alpha_k =  cRatio*(k- nu*k**2)
        u_hat = v_hat * np.exp(alpha_k*t)
        u = scipy.fft.irfft(u_hat)

        rhs_v_hat = -1j * k * scipy.fft.rfft( 0.5*u**2 ) * np.exp( - alpha_k*t)
        return libSiva.np_complex__viewed_as_real2(rhs_v_hat)

    @staticmethod
    def MS__siva_u_rhs_fft_for_ivp(t,u_hat__viewed_as_real, nu,cRatio, k):
    #    return libSiva.siva_u_rhs_fft(u_hat__viewed_as_real,t,nu,k)
    #@staticmethod
    #def siva_u_rhs_fft(u_hat__viewed_as_real,t,nu,k):
        u_hat= libSiva.np_real2__viewed_as_complex( u_hat__viewed_as_real)
        u_hat[0] =0
        u=scipy.fft.irfft(u_hat)
        #rhs_u_hat = ( np.abs(k)- nu* k**2) * u_hat  -  1j * k * scipy.fft.rfft ( 0.5*u**2 )
        rhs_u_hat = cRatio*(k - nu* k**2) * u_hat  -  1j * k * scipy.fft.rfft ( 0.5*u**2 )
        #rhs_u_hat__viewed_as_real = np.stack ( (rhs_u_hat.real, rhs_u_hat.imag ), axis = -1 ).reshape(rhs_u_hat.shape[0], -1 )

        return libSiva.np_complex__viewed_as_real2(rhs_u_hat)  #rhs_u_hat__viewed_as_real
    @staticmethod
    def MS__siva_d_rhs_for_ivp(t,d,nu,cRatio, k):
    #    return libSiva.siva_d_rhs(d,t,nu,k)
    #@staticmethod
    #def siva_d_rhs(d,t,nu,k):
        d_hat    = scipy.fft.rfft(d)
        d_hat[0] = 0
        u_hat    = 1j * k * d_hat

        u         =   scipy.fft.irfft(u_hat) #.real
        rhs_u_hat =  cRatio*(np.abs(k)-nu* k**2 ) * u_hat  -  1j* k *scipy.fft.rfft( 0.5*u**2 )

        rhs_d_hat     = rhs_u_hat
        rhs_d_hat[1:] = -1j* rhs_u_hat[1:] / k[1:]
        rhs_d_hat[0]  = 0
        rhs_d         = scipy.fft.irfft(rhs_d_hat) #.real
        return rhs_d


    @staticmethod
    def get_k(N,L=2*np.pi):
        #k = np.arange(0,N//2-1+1)
        #k = np.append(k,np.arange(-N//2,-1+1) )
        #k = (2*np.pi/L)*k
        #return k
        return (2*np.pi/L)*np.arange(0,N//2+1)

    @staticmethod
    def get_x(N, L=2*np.pi):
        #x_1 = np.linspace(-L/2, L/2, N+1)  , x   = x_1[0:N]
        #x = np.linspace(-L/2, L/2-L/N, N)
        x = np.linspace(-L/2, L/2, N, endpoint=False)
        return x

    """ displacment to slope"""
    @staticmethod
    def d_to_u(d, k):
        #u_hat =- 1j * scipy.fft.rfft( d ) * k
        u_hat = 1j * scipy.fft.rfft( d ) * k
        u =  scipy.fft.irfft( u_hat ) #.real
        return u

    """ slope to displacment """
    @staticmethod
    def u_to_d(u, k=None, bSym =False ) :
        if k is None :
            if  bSym==True:
                N = u.shape[-1]
                k = libSiva.get_k( 2*N, np.pi *2 )  # default
                #-----------
                uhat =  scipy.fft.rfft( np.concatenate( (u, -np.flip(u,-1)) , -1)  )

                d_hat = uhat
                d_hat[1:] = -1j * uhat[1:] / k[1:]
                d_hat[0]=0
                d =  scipy.fft.irfft( d_hat ) #.real
                d = d[ :N ]

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
    # @staticmethod
    # def u_to_d(u, k=None):
    #     if k is None:
    #         k = libSiva.get_k(u.shape[-1], np.pi *2 )  # default
    #
    #     u_hat =  scipy.fft.rfft(u)
    #     d_hat = u_hat
    #     d_hat[1:] = -1j * u_hat[1:] / k[1:]
    #     d_hat[0]=0
    #     d =  scipy.fft.irfft( d_hat ) #.real
    #     return d

    @staticmethod
    def usol_to_dsol(usol, k=None):
        if k is None:    k = libSiva.get_k(usol.shape[-1], np.pi *2 )  # default

        dsol = np.zeros_like(usol)
        for j in range( usol.shape[0] ):
            dsol[j,:] = libSiva.u_to_d( usol[j,:], k)
        return dsol
    @staticmethod
    def dsol_to_usol(dsol, k=None):
        if k is None:    k = libSiva.get_k(dsol.shape[-1], np.pi *2 )  # default
        usol = np.zeros_like(dsol)
        for j in range( dsol.shape[0] ):
            usol[j,:] = libSiva.d_to_u( dsol[j,:], k)
        return usol

    @staticmethod
    def MS__AnalyticalSolver_ivp_spectral(d0, t, nu,cRatio, k  , method ='RK45', t_eval=None,atol=1e-6):
        #dsol = scipy.integrate.odeint(libSiva.siva_d_rhs, d0, t, args=(nu, k), rtol=rtol )  # integrate PDE in spectral space using ode45

        d_hat0 = scipy.fft.rfft(d0)  # d_hat0[0] = 0
        u_hat0  = 1j * k * d_hat0


        #uhat_sol_viewed_as_real2 = scipy.integrate.odeint(libSiva.siva_u_rhs_fft, libSiva.np_complex__viewed_as_real2( u_hat0 ), t, args=(nu, k), atol=atol )  # integrate PDE in spectral space using ode45

        sol = scipy.integrate.solve_ivp( libSiva.MS__siva_u_rhs_fft_for_ivp, [t[0], t[-1]], libSiva.np_complex__viewed_as_real2( u_hat0 ) , t_eval=t_eval, args=(nu,cRatio, k) , method=method,atol=atol)

        uhat_sol_viewed_as_real2 = np.transpose(sol.y)
        usol = scipy.fft.irfft(  libSiva.np_real2__viewed_as_complex( uhat_sol_viewed_as_real2 )  )
        dsol = libSiva.usol_to_dsol( usol,k )
        return dsol, sol.t, sol.success

    @staticmethod
    def MS__AnalyticalSolver_ivp_dsol(d0, t, nu, cRatio, k, method ='RK45', t_eval=None,atol=1e-6):
        sol = scipy.integrate.solve_ivp( libSiva.MS__siva_d_rhs_for_ivp, [t[0], t[-1]], d0, t_eval=t_eval, args=(nu,cRatio, k) , method=method,atol=atol)  # integrate PDE in spectral space using ode45
        return np.transpose(sol.y), sol.t, sol.success


    @staticmethod
    def MS__AnalyticalSolver_ivp_filteredspectral(d0, t, nu,cRatio, k, method ='RK45', atol=1e-6):
        dhat0 = scipy.fft.rfft(d0)
        uhat0  = 1j * k * dhat0
        alpha_k = (k - nu * k ** 2)
        dt = t[1]-t[0]
        exp_dt_alpha_k = np.exp( dt * alpha_k)

        uhat_sol = np.zeros( (len(t), len(uhat0)), dtype= complex )
        uhat_sol[0,:] = uhat0

        uhat = uhat0
        for idx,_ in  enumerate( t ):
            if idx >=1 :
                sol = scipy.integrate.solve_ivp( libSiva.MS__siva_filter_v_rhs_fft_for_ivp, [0, dt], libSiva.np_complex__viewed_as_real2( uhat ) , t_eval=[dt], args=(nu, cRatio, k) , method=method,atol=atol)
                vhat = libSiva.np_real2__viewed_as_complex( np.transpose(sol.y)[-1,:] )
                uhat = vhat * exp_dt_alpha_k
                uhat_sol[idx,:] = uhat

        usol = scipy.fft.irfft( uhat_sol  )
        dsol = libSiva.usol_to_dsol( usol,k )
        return dsol

    @staticmethod
    def get2D_Ny_yB_from_estimate(Nx_target, yB_estimate, AspectRatio_set=1 , L=2*np.pi ):
        dx = L/ Nx_target
        dy = dx* AspectRatio_set
        Ny_actual = int( (yB_estimate[1] - yB_estimate[0] + 1E-10) / dy )
        yB = np.copy(yB_estimate)
        yB[1] = yB[0] + (Ny_actual) * dy
        return Ny_actual, yB





class CSolverSiva:
    def __init__(self,siva_sys_name, list_para, method_default_siva_data_gen=1) :
        #assert()
        if siva_sys_name not in ['MS_1stEuler', 'MS_RK4', 'KS_RK4','MKS_RK4']:
            warnings.warn('do not find CSolverSiva.siva_sys_name')


        self.L = 2*np.pi
        self.siva_sys_name = siva_sys_name
        self.list_para = list_para

        # if list_para == [0.01]  and method_default_siva_data_gen==1 :
        #     self.nu_extention_list =[0.01]
        #     self.N =  1024
        #     self.ActiveNoise_Amplitude, self.ActiveNoise_stepfeq, self.k_ActiveNoise = (0, 1, 0)
        #     self.dt_Output =.015
        #     self.NumOutPut_dt = np.zeros( len(self.nu_extention_list) ,dtype=int )
        #     for idx, each_nu in enumerate(self.nu_extention_list):
        #         self.NumOutPut_dt[idx] = libSiva.get_numSubSteps_PerTimeStep_default(self.dt_Output,each_nu,self.N,self.L)
        #
        #     self.default_training_infolist_generate_data =[
        #                 (1, 200001, 0.01, 'rand_FFT_2_8'  ),
        #                 (500, 201,  0.01, 'rand_FFT_2_8' ),
        #                 (500, 201,  0.01, 'rand_simple'  )     ]
        #     self.default_testing_infolist_generate_data =[
        #                 (1,  20001, 0.01, 'rand_FFT_2_8'  ),
        #                 (50,  201,  0.01, 'rand_FFT_2_8' ),
        #                 (50,  201,  0.01, 'rand_simple'  )     ]

        if self.list_para == [0.02]  and method_default_siva_data_gen==1 and self.siva_sys_name =='MS_1stEuler':
            self.nu_extention_list =[0.02]
            self.N =  512
            self.ActiveNoise_Amplitude, self.ActiveNoise_stepfeq, self.k_ActiveNoise = (0, 1, 0)
            self.dt_Output =.015
            self.NumOutPut_dt = np.zeros( len(self.nu_extention_list) ,dtype=int )
            for idx, each_nu in enumerate(self.nu_extention_list):
                self.NumOutPut_dt[idx] = libSiva.get_numSubSteps_PerTimeStep_default(self.dt_Output,each_nu,self.N,self.L)

            self.default_training_infolist_generate_data =[
                        (1, 200001, 0.02, 'rand_FFT_2_8'  ),
                        (500, 201,  0.02, 'rand_FFT_2_8' ),
                        (500, 201,  0.02, 'rand_simple'  )     ]
            self.default_testing_infolist_generate_data =[
                        (1,  20001, 0.02, 'rand_FFT_2_8'  ),
                        (50,  201,  0.02, 'rand_FFT_2_8' ),
                        (50,  201,  0.02, 'rand_simple'  )     ]

        # elif list_para ==  [0.05] and method_default_siva_data_gen==1:
        #     self.nu_extention_list = [0.05]
        #     self.N =   128
        #     self.ActiveNoise_Amplitude, self.ActiveNoise_stepfeq, self.k_ActiveNoise = (.001, 1, 0)
        #     self.dt_Output =.015
        #
        #
        #     self.NumOutPut_dt = np.zeros( len(self.nu_extention_list) ,dtype=int  )
        #     for idx, each_nu in enumerate(self.nu_extention_list):
        #         self.NumOutPut_dt[idx] = libSiva.get_numSubSteps_PerTimeStep_default(self.dt_Output,each_nu,self.N,self.L)
        #
        #     self.default_training_infolist_generate_data =[
        #                 (10,  10001, 0.05, 'rand_FFT_2_8'  ),
        #                 (200, 201,  0.05, 'rand_FFT_2_8' ),
        #                 (200, 201,  0.05, 'rand_simple'  )     ]
        #     self.default_testing_infolist_generate_data =[
        #                 (2,  10001, 0.05, 'rand_FFT_2_8'  ),
        #                 (40, 201,  0.05, 'rand_FFT_2_8' ),
        #                 (40, 201,  0.05, 'rand_simple'  )     ]

        elif all( item in [0.07, 0.125, 0.4, 0.7] for item in self.list_para ) and  self.siva_sys_name =='MS_1stEuler' and method_default_siva_data_gen in [1, 2, 3] :# Paper Revision Change
            self.nu_extention_list = [0.07, 0.125, 0.4 , 0.7]
            self.N = 128
            self.ActiveNoise_Amplitude, self.ActiveNoise_stepfeq, self.k_ActiveNoise = (0, 1, 0)
            self.dt_Output =.015
            self.NumOutPut_dt = np.zeros( len(self.nu_extention_list) ,dtype=int )
            for idx, each_nu in enumerate(self.nu_extention_list):
                self.NumOutPut_dt[idx] = libSiva.get_numSubSteps_PerTimeStep_default(self.dt_Output,each_nu,self.N,self.L)

            nTimeSteps = 1+method_default_siva_data_gen*500  # Paper Revision Change

            self.default_training_infolist_generate_data  =   [
                        (200, nTimeSteps, 0.07, 'rand_FFT_2_8' ),                        (200, nTimeSteps, 0.07, 'rand_simple' ) ,
                        (200, nTimeSteps, 0.125, 'rand_FFT_2_8' ) ,                        (200, nTimeSteps, 0.125, 'rand_simple' ) ,
                        (200, nTimeSteps, 0.4, 'rand_FFT_2_8' ),                        (200, nTimeSteps, 0.4, 'rand_simple' ) ,
                        (200, nTimeSteps, 0.7, 'rand_FFT_2_8' ),                        (200, nTimeSteps, 0.7, 'rand_simple' )          ]

            self.default_testing_infolist_generate_data  =   [
                        (40, nTimeSteps, 0.07, 'rand_FFT_2_8' ),                        (40, nTimeSteps, 0.07, 'rand_simple' ) ,
                        (40, nTimeSteps, 0.125, 'rand_FFT_2_8' ),                        (40, nTimeSteps, 0.125, 'rand_simple' ) ,
                        (40, nTimeSteps, 0.4, 'rand_FFT_2_8' ),                        (40, nTimeSteps, 0.4, 'rand_simple' ) ,
                        (40, nTimeSteps, 0.7, 'rand_FFT_2_8' ),                        (40, nTimeSteps, 0.7, 'rand_simple' )          ]

        # elif all( item in [0.025, 0.05, 0.075, 0.1, 0.15] for item in self.list_para ) and self.siva_sys_name =='MS_RK4':
        #     self.nu_extention_list = [0.025, 0.05, 0.075, 0.1, 0.15]
        #     self.N = 256
        #     self.ActiveNoise_Amplitude, self.ActiveNoise_stepfeq, self.k_ActiveNoise = (0, 1, 0)
        #     self.dt_Output =0.015
        #     # self.NumOutPut_dt = np.zeros( len(self.nu_extention_list) ,dtype=int )
        #     # for idx, each_nu in enumerate(self.nu_extention_list):
        #     #     self.NumOutPut_dt[idx] = libSiva.get_numSubSteps_PerTimeStep_default(self.dt_Output,each_nu,self.N,self.L)
        #     self.default_training_infolist_generate_data  =   [
        #         (1, 250001, 0.025, 'rand_simple'  ), (250, 501, 0.025, 'rand_FFT_2_8' ),    (250, 501, 0.025, 'rand_simple' ) ,
        #         (1, 250001, 0.05, 'rand_simple'  ),  (250, 501, 0.05, 'rand_FFT_2_8' ),     (250, 501, 0.05, 'rand_simple' ) ,
        #         (1, 250001, 0.075, 'rand_simple' ),  (250, 501, 0.075, 'rand_FFT_2_8' ),    (250, 501, 0.075, 'rand_simple' ) ,
        #         (1, 250001, 0.1, 'rand_simple'  ),   (250, 501, 0.1, 'rand_FFT_2_8' ),      (250, 501, 0.1, 'rand_simple' ),
        #                                              (250,1001, 0.15, 'rand_FFT_2_8' ),     (250,1001, 0.15, 'rand_simple' )      ]
        #     self.default_testing_infolist_generate_data  =   [
        #         (1, 25001, 0.025, 'rand_simple'  ), (25, 501, 0.025, 'rand_FFT_2_8' ),   (25, 501, 0.025, 'rand_simple' ) ,
        #         (1, 25001, 0.05, 'rand_simple'  ),  (25, 501, 0.05, 'rand_FFT_2_8' ),    (25, 501, 0.05, 'rand_simple' ) ,
        #         (1, 25001, 0.075, 'rand_simple'  ), (25, 501, 0.075, 'rand_FFT_2_8' ),   (25, 501, 0.075, 'rand_simple' ) ,
        #         (1, 25001, 0.1, 'rand_simple'  ),   (25, 501, 0.1, 'rand_FFT_2_8' ),     (25, 501, 0.1, 'rand_simple' ),
        #                                             (25,1001, 0.15, 'rand_FFT_2_8' ),    (25,1001, 0.15, 'rand_simple' )      ]

        elif all( item in [0.025, 0.035, 0.05, 0.07, 0.1, 0.15] for item in self.list_para ) and self.siva_sys_name =='MS_RK4':
            self.nu_extention_list = [0.025, 0.035, 0.05, 0.07, 0.1, 0.15]
            self.N = 256
            self.ActiveNoise_Amplitude, self.ActiveNoise_stepfeq, self.k_ActiveNoise = (0, 1, 0)
            self.dt_Output =0.015
            # self.NumOutPut_dt = np.zeros( len(self.nu_extention_list) ,dtype=int )
            # for idx, each_nu in enumerate(self.nu_extention_list):
            #     self.NumOutPut_dt[idx] = libSiva.get_numSubSteps_PerTimeStep_default(self.dt_Output,each_nu,self.N,self.L)
            self.default_training_infolist_generate_data  =   [
                (1, 125001, 0.025, 'rand_simple'  ),    (250, 501, 0.025, 'rand_simple' ) ,
                (1, 125001, 0.035, 'rand_simple'  ),    (250, 501, 0.035, 'rand_simple' ) ,
                (1, 125001, 0.05, 'rand_simple'  ),     (250, 501, 0.05, 'rand_simple' ) ,
                                                        (250,1001, 0.07, 'rand_simple' ) ,
                                                        (250,1001, 0.1, 'rand_simple' ),
                                                        (250,1001, 0.15, 'rand_simple' ) ]
            self.default_testing_infolist_generate_data  =   [
                (1, 12501, 0.025, 'rand_simple'  ),    (25, 501, 0.025, 'rand_simple' ) ,
                (1, 12501, 0.035, 'rand_simple'  ),    (25, 501, 0.035, 'rand_simple' ) ,
                (1, 12501, 0.05, 'rand_simple'  ),     (25, 501, 0.05, 'rand_simple' ) ,
                                                        (25,1001, 0.07, 'rand_simple' ) ,
                                                        (25,1001, 0.1, 'rand_simple' ),
                                                        (25,1001, 0.15, 'rand_simple' ) ]

        elif all( item in [6, 9, 12, 18, 24] for item in self.list_para ) and self.siva_sys_name =='KS_RK4':
            #self.nu_extention_list = [0.025, 0.05, 0.075, 0.1, 0.2]
            self.N = 256
            #self.ActiveNoise_Amplitude, self.ActiveNoise_stepfeq, self.k_ActiveNoise = (0, 1, 0)
            self.dt_Output =0.15
            # self.NumOutPut_dt = np.zeros( len(self.nu_extention_list) ,dtype=int )
            # for idx, each_nu in enumerate(self.nu_extention_list):
            #     self.NumOutPut_dt[idx] = libSiva.get_numSubSteps_PerTimeStep_default(self.dt_Output,each_nu,self.N,self.L)
            self.default_training_infolist_generate_data  =   [
                # (1, 250001, 6, 'rand_simple'  ),  (250, 501, 6, 'rand_FFT_2_8' ) ,  (250, 501, 6, 'rand_simple' ) ,
                # (1, 250001, 12, 'rand_simple' ),  (250, 501, 12, 'rand_FFT_2_8' ) , (250, 501, 12, 'rand_simple' ) ,
                # (1, 250001, 24, 'rand_simple' ),  (250, 501, 24, 'rand_FFT_2_8' ) , (250, 501, 24, 'rand_simple' )         ]
                (1, 125001, 6, 'rand_simple'  ),                                     (250, 501, 6, 'rand_simple' ) ,
                (1, 125001, 9, 'rand_simple'  ),                                     (250, 501, 9, 'rand_simple' ) ,
                (1, 125001, 12, 'rand_simple' ),                                     (250, 501, 12, 'rand_simple' ) ,
                (1, 125001, 18, 'rand_simple'  ),                                    (250, 501, 18, 'rand_simple' ) ,
                (1, 125001, 24, 'rand_simple' ),                                     (250, 501, 24, 'rand_simple' )         ]

            self.default_testing_infolist_generate_data  =   [
                # (1, 25001, 6, 'rand_simple'  ),   (25, 501, 6, 'rand_FFT_2_8' ) , (25, 501, 6, 'rand_simple' ) ,
                # (1, 25001, 12, 'rand_simple'  ),  (25, 501, 12, 'rand_FFT_2_8' ), (25, 501, 12, 'rand_simple' ) ,
                # (1, 25001, 24, 'rand_simple'  ),  (25, 501, 24, 'rand_FFT_2_8' ), (25, 501, 24, 'rand_simple' )         ]
                (1, 12501, 6, 'rand_simple'  )  ,                                   (25, 501, 6, 'rand_simple' ) ,
                (1, 12501, 9, 'rand_simple'  ) ,                                   (25, 501, 9, 'rand_simple' ) ,
                (1, 12501, 12, 'rand_simple'  ) ,                                   (25, 501, 12, 'rand_simple' ) ,
                (1, 12501, 18, 'rand_simple'  ) ,                                   (25, 501, 18, 'rand_simple' ) ,
                (1, 12501, 24, 'rand_simple'  ) ,                                   (25, 501, 24, 'rand_simple' )         ]

        elif  all( item in [[10,0],[10,0.25],[10,0.5], [10,0.75], [10,1],
                            [25,0],[25,0.25],[25,0.5], [25,0.75], [25,1],
                            [40,0],[40,0.25],[40,0.5], [40,0.75], [40,1]        ] for item in self.list_para ) \
        and self.siva_sys_name =='MKS_RK4':
            self.N = 256
            self.dt_Output =0.15
            self.default_training_infolist_generate_data  =   [
                (1, 125001, [40, 0   ], 'rand_simple' ), (250, 501, [40, 0   ], 'rand_simple' ) ,
                (1, 125001, [40, 0.25], 'rand_simple' ), (250, 501, [40, 0.25], 'rand_simple' ) ,
                (1, 125001, [40, 0.5 ], 'rand_simple' ), (250, 501, [40, 0.5 ], 'rand_simple' )  ,
                (1, 125001, [40, 0.75], 'rand_simple' ), (250, 501, [40, 0.75], 'rand_simple' ) ,
                (1, 125001, [40, 1   ], 'rand_simple' ), (250, 501, [40, 1   ], 'rand_simple' ) ,
                (1, 125001, [25, 0   ], 'rand_simple' ), (250, 501, [25, 0   ], 'rand_simple' ) ,
                (1, 125001, [25, 0.25], 'rand_simple' ), (250, 501, [25, 0.25], 'rand_simple' ) ,
                (1, 125001, [25, 0.5 ], 'rand_simple' ), (250, 501, [25, 0.5 ], 'rand_simple' )  ,
                (1, 125001, [25, 0.75], 'rand_simple' ), (250, 501, [25, 0.75], 'rand_simple' ) ,
                                                         (250,1001, [25, 1   ], 'rand_simple' ) , #(1, 125001, [25, 1   ], 'rand_simple' ),  (250, 501, [25, 1   ], 'rand_simple' ) ,
                (1, 125001, [10, 0   ], 'rand_simple' ), (250, 501, [10, 0   ], 'rand_simple' ) ,
                (1, 125001, [10, 0.25], 'rand_simple' ), (250, 501, [10, 0.25], 'rand_simple' ) ,
                (1, 125001, [10, 0.5 ], 'rand_simple' ), (250, 501, [10, 0.5 ], 'rand_simple' )  ,
                (1, 125001, [10, 0.75], 'rand_simple' ), (250, 501, [10, 0.75], 'rand_simple' ) ,
                                                         (250,1001, [10, 1   ], 'rand_simple' ) , #(1, 125001, [10, 1   ], 'rand_simple' ),  (250, 501, [10, 1   ], 'rand_simple' )
                ]
            self.default_testing_infolist_generate_data  =   [
                (1, 12501, [40, 0   ], 'rand_simple' ),  (25, 501, [40, 0   ], 'rand_simple' ) ,
                (1, 12501, [40, 0.25], 'rand_simple' ),  (25, 501, [40, 0.25], 'rand_simple' ) ,
                (1, 12501, [40, 0.5 ], 'rand_simple' ),  (25, 501, [40, 0.5 ], 'rand_simple' )  ,
                (1, 12501, [40, 0.75], 'rand_simple' ),  (25, 501, [40, 0.75], 'rand_simple' ) ,
                (1, 12501, [40, 1   ], 'rand_simple' ),  (25, 501, [40, 1   ], 'rand_simple' ),
                (1, 12501, [25, 0   ], 'rand_simple' ),  (25, 501, [25, 0   ], 'rand_simple' ) ,
                (1, 12501, [25, 0.25], 'rand_simple' ),  (25, 501, [25, 0.25], 'rand_simple' ) ,
                (1, 12501, [25, 0.5 ], 'rand_simple' ),  (25, 501, [25, 0.5 ], 'rand_simple' )  ,
                (1, 12501, [25, 0.75], 'rand_simple' ),  (25, 501, [25, 0.75], 'rand_simple' ) ,
                                                         (25,1001, [25, 1   ], 'rand_simple' ) , #(1, 12501, [25, 1   ], 'rand_simple' ),   (25, 501, [25, 1   ], 'rand_simple' ) ,
                (1, 12501, [10, 0   ], 'rand_simple' ),  (25, 501, [10, 0   ], 'rand_simple' ) ,
                (1, 12501, [10, 0.25], 'rand_simple' ),  (25, 501, [10, 0.25], 'rand_simple' ) ,
                (1, 12501, [10, 0.5 ], 'rand_simple' ),  (25, 501, [10, 0.5 ], 'rand_simple' ) ,
                (1, 12501, [10, 0.75], 'rand_simple' ),  (25, 501, [10, 0.75], 'rand_simple' ) ,
                                                         (25,1001, [10, 1   ], 'rand_simple' )   #(1, 12501, [10, 1   ], 'rand_simple' ),   (25, 501, [10, 1   ], 'rand_simple' )
                ]
        elif  all( item in [0, 0.25, 0.5, 0.75, 1 ] for item in self.list_para )   and self.siva_sys_name =='old_MKS_Lpi32_RK4':
            self.N = 256
            self.dt_Output =0.15
            self.default_training_infolist_generate_data  =  [                ]
            self.default_testing_infolist_generate_data  =   [                ]
            warnings.warn( 'CSolverSiva.siva_sys_name ==old_MKS_Lpi32_RK4')

        elif self.list_para in [  [0.07] ] and self.siva_sys_name =='MS_1stEuler' and method_default_siva_data_gen==-1: #Paper Revision Change

            self.nu_extention_list = [0.07, 0.125, 0.4 , 0.7]
            self.N = 128
            self.ActiveNoise_Amplitude, self.ActiveNoise_stepfeq, self.k_ActiveNoise = (0, 1, 0)
            self.dt_Output =.015
            self.NumOutPut_dt = np.zeros( len(self.nu_extention_list) ,dtype=int )
            for idx, each_nu in enumerate(self.nu_extention_list):
                self.NumOutPut_dt[idx] = libSiva.get_numSubSteps_PerTimeStep_default(self.dt_Output,each_nu,self.N,self.L)

            self.default_training_infolist_generate_data  =   [
                        (100, 501, 0.07, 'rand_FFT_2_8' ),
                        (100, 501, 0.07, 'rand_simple' )
            ]
            self.default_testing_infolist_generate_data  =   [
                        (20, 501, 0.07, 'rand_FFT_2_8' ),
                        (20, 501, 0.07, 'rand_simple' )
            ]

        else:
            raise ValueError('CSolverSiva.list_para')

        self.__print__()

    def __print__(self):
        print( 'siva_sys_name=',self.siva_sys_name)
        print( 'N=',            self.N,  end =" ," )
        print( 'dt_Output=' ,   self.dt_Output,  end =" ," )

        print( 'self.list_para =',    self.list_para,            end =" ," )

        #print( 'nu_extention_list =',    self.nu_extention_list,            end =" ," )

        if hasattr(self,'NumOutPut_dt'): print('NumOutPut_dt=', self.NumOutPut_dt  )

        if hasattr(self,'ActiveNoise_Amplitude'):
            print( 'ActiveNoise_Amplitude=%f,stepfeq=%d,k = %d ' %(self.ActiveNoise_Amplitude, self.ActiveNoise_stepfeq,self.k_ActiveNoise ) )


    #def get_default_init_func_d0(self, bRandFFT=True):
    #    if bRandFFT == True:
    #        return lambda: libSiva.rand_d0_FFT( self.N, np.random.randint(low=2,high=8) )
    #    else:
    #        return lambda: lambda: libSiva.rand_d0_simple(self.N )

    def get_init_func_from_txt(self, init_op_string):
        if init_op_string == 'rand_FFT_2_8':
            return lambda sys_info : libSiva.rand_d0_FFT(self.N, np.random.randint(low=2,high=8) , sys_info )
        elif init_op_string == 'rand_FFT_2_6':
            return lambda sys_info : libSiva.rand_d0_FFT(self.N, np.random.randint(low=2,high=6) , sys_info )
        elif init_op_string == 'rand_simple':
            return lambda sys_info : libSiva.rand_d0_simple(self.N,                                 sys_info )
        else:
            assert False, 'undefined init_op_string'

    def generate_dsol_single(self, len_seq, each_parameter, init_op_string_or_value):
        ###############################
        t_Seq = np.arange(0, len_seq)*self.dt_Output
        if type(init_op_string_or_value) is str:
            init_func_for_d0   = self.get_init_func_from_txt (init_op_string_or_value)
            sys_info = { 'siva_sys_name':self.siva_sys_name, 'para_value':each_parameter}
            d0       = init_func_for_d0( sys_info )
        else:
            d0 = init_op_string_or_value

        if self.siva_sys_name == 'MS_1stEuler' :
            each_nu = each_parameter
            idx_nu = np.where( np.isclose( self.nu_extention_list, each_nu) )[0]
            if  idx_nu.size ==0 : # If the given 'each_nu' was not found in the default list
                NumOutPut_dt_now = libSiva.get_numSubSteps_PerTimeStep_default(self.dt_Output,each_nu,self.N,self.L)
            else:
                NumOutPut_dt_now = (self.NumOutPut_dt[idx_nu])[0]
            dsol = libSiva.MS__NumericalSolver_dsol( d0, t_Seq, each_nu, self.ActiveNoise_Amplitude, self.ActiveNoise_stepfeq, self.k_ActiveNoise, self.N, self.L, NumOutPut_dt_now)

        elif self.siva_sys_name == 'MS_RK4' :
            each_nu = -each_parameter  # negative
            dsol = libSiva.RK4NumericalSolver_dsol(d0, t_Seq, Lpi=1, mu=0, nu=each_nu, d_ratio=1, tau=1, n_dt=5)
        elif self.siva_sys_name == 'KS_RK4' :
            each_Lpi = each_parameter
            dsol = libSiva.RK4NumericalSolver_dsol(d0, t_Seq, Lpi=each_Lpi, mu=1, nu=1, d_ratio=0, tau=1, n_dt=1)
        elif self.siva_sys_name == 'MKS_RK4' :
            each_Lpi, each_rho = each_parameter
            d_ratio = each_rho
            mu,nu, tau = libSiva.Coeff_Lpi_d_to_mu_nu_tau( each_Lpi, d_ratio)
            dsol = libSiva.RK4NumericalSolver_dsol(d0, t_Seq, Lpi=each_Lpi, mu=mu, nu=nu, d_ratio=d_ratio, tau=tau, n_dt=5)

        elif self.siva_sys_name == 'old_MKS_Lpi32_RK4': # can be deleted later
            warnings.warn(' self.siva_sys_name == old_MKS_Lpi32_RK4,  this should be deleted later ' )
            each_rho = each_parameter
            d_ratio = each_rho * 4
            Lpi = 32 # old way to force this value
            mu,nu, tau = libSiva.Coeff_Lpi_d_to_mu_nu_tau( Lpi, d_ratio , max_amplitude = 1)
            tau = 1 # old way to force this value
            dsol = libSiva.RK4NumericalSolver_dsol(d0, t_Seq, Lpi=Lpi, mu=mu, nu=nu, d_ratio=d_ratio, tau=tau, n_dt=5)

        assert np.any( np.isnan( dsol  ) )==False, "nan in generate_dsol_single"
        return dsol


    # infolist_generate_data = [ (2,  10001,  lambda: libSiva.rand_d0_FFT(N, np.random.randint(low=2,high=6) ) ),
    #                           (100, 201,   lambda: libSiva.rand_d0_FFT(N, np.random.randint(low=2,high=6) ) ),
    #                           (100, 201,   lambda: 0.1*np.random.random_sample(N)-0.05 )      ]

    def generate_or_load_DEFAULT_xsol_list(self, train_or_test,  dir_save_training_data = None,
                                           name_xsol='dsol', Nx=None, yB_estimate=np.array([-0.7,1.3])*np.pi,
                                           AspectRatio_set=1,bForcedRegendsol = False, bForceNoSavedsol=False ):
        if train_or_test == 'train':
            infolist= self.default_training_infolist_generate_data
        else:
            infolist= self.default_testing_infolist_generate_data

        actual_infolist_selected_nu = []

        for num_traj, len_seq, pdepara, init_op_string in infolist:
            if pdepara in self.list_para:
                actual_infolist_selected_nu.append( (num_traj, len_seq, pdepara, init_op_string ) )

        if len( actual_infolist_selected_nu ) == 0:
            raise ValueError('none of nu has been found: generate_or_load_DEFAULT_dsol_list')

        return self.generate_or_load_xsol_list( actual_infolist_selected_nu, dir_save_training_data,name_xsol, Nx,yB_estimate,AspectRatio_set,  bForcedRegendsol, bForceNoSavedsol)


    def generate_or_load_xsol_list( self, infolist_generate_data,  dir_save_training_data = None,
                                    name_xsol='dsol', Nx=None, yB_estimate=np.array([-0.7,1.3])*np.pi,
                                    AspectRatio_set=1,  bForcedRegendsol = False, bForceNoSavedsol=False  ):

        #if infolist_generate_data is None:
        #    infolist_generate_data = self.default_training_infolist_generate_data
        print(infolist_generate_data)


        if 'level' in name_xsol.casefold() :
            assert  len(yB_estimate)==2
            Ny, yB = libSiva.get2D_Ny_yB_from_estimate(Nx,yB_estimate,AspectRatio_set=AspectRatio_set)

        list_xsol=[]
        list_pde_para =[]

        for num_traj, len_seq, pde_para, init_op_string in infolist_generate_data:
            print('num_traj=', num_traj, ' leq_seq=', len_seq, ' pde_para=', pde_para, ' ', init_op_string )

            # idx_nu = np.where( np.isclose( nu, self.nu_extention_list) )[0]
            # assert idx_nu.size >=0 , 'idx_nu fail in generate_or_load_xsol_list'

            #####
            if 'MS_1stEuler' == self.siva_sys_name:
                pkl_filename_dsol_multraj =    'dsol_multraj' + '{:d}'.format(num_traj) + 'L{:d}'.format(len_seq) + '_nu' + round_num_to_txt(pde_para) + '_N' + '{:d}'.format(self.N) +'_dt' +  round_num_to_txt(self.dt_Output)  + '_'+ init_op_string
                if abs(self.ActiveNoise_Amplitude) >= 1E-9:   pkl_filename_dsol_multraj +=   '_Noise'
            elif 'MS_RK4' == self.siva_sys_name:
                pkl_filename_dsol_multraj = 'RK4_dsol_multraj' + '{:d}'.format(num_traj) + 'L{:d}'.format(len_seq) + '_nu' + round_num_to_txt(pde_para) + '_N' + '{:d}'.format(self.N) +'_dt' +  round_num_to_txt(self.dt_Output)  + '_'+ init_op_string
                if abs(self.ActiveNoise_Amplitude) >= 1E-9:   pkl_filename_dsol_multraj +=   '_Noise'
            elif 'KS_RK4' == self.siva_sys_name  :
                pkl_filename_dsol_multraj ='KSRK4_dsol_multraj' + '{:d}'.format(num_traj) + 'L{:d}'.format(len_seq) + '_Lpi' + round_num_to_txt(pde_para) + '_N' + '{:d}'.format(self.N) +'_dt' + round_num_to_txt(self.dt_Output)  + '_'+ init_op_string
            elif 'MKS_RK4' == self.siva_sys_name :
                pkl_filename_dsol_multraj ='MKS_dsol_multraj' + '{:d}'.format(num_traj) + 'L{:d}'.format(len_seq) + '_Lpi{:d}rho{:g}'.format(pde_para[0], pde_para[1])+ '_N' + '{:d}'.format(self.N) +'_dt' + round_num_to_txt(self.dt_Output)  + '_'+ init_op_string

            pkl_filename_dsol_multraj += '.pkl'
            #####

            picklefilename = dir_save_training_data + pkl_filename_dsol_multraj

            if os.path.isfile (picklefilename) and (bForcedRegendsol==False):
                open_file = open(picklefilename, "rb")
                data_load = pickle.load(open_file)
                open_file.close()
                dsol_multraj = data_load['dsol_multraj']

                assert num_traj == data_load['num_traj']
                assert len_seq == data_load['len_seq']
                assert self.N == data_load['N']
                assert self.dt_Output == data_load['dt_Output']
                #assert self.ActiveNoise_Amplitude == data_load['ActiveNoise_Amplitude']
                if self.siva_sys_name ==  'MS_1stEuler' or  self.siva_sys_name == 'MS_RK4' :
                    assert pde_para == data_load['nu']
                elif self.siva_sys_name == 'KS_RK4' :
                    assert pde_para == data_load['Lpi']
                elif self.siva_sys_name == 'MKS_RK4':
                    assert pde_para == data_load['Lpi_rho']
                print('Success: load ' + picklefilename )

            else:  # fresh generate data if the pickle file does not exist or bForceRegen==True
                if bForcedRegendsol==False:
                    print('The file ' +  picklefilename + ' do not exist, therefore do a fresh generation')

                dsol_multraj = np.zeros( (num_traj, len_seq, self.N) )
                for i in range(num_traj):
                    dsol_multraj[i,:,:] = self.generate_dsol_single(len_seq, pde_para, init_op_string )
                    print('', end ='.')
                print('')

                if bForceNoSavedsol == False:
                    if self.siva_sys_name ==  'MS_1stEuler' or  self.siva_sys_name == 'MS_RK4' :
                        data_for_save = {'dsol_multraj':dsol_multraj, 'num_traj':num_traj,'len_seq':len_seq,  'nu':pde_para,    'N':self.N,'dt_Output':self.dt_Output,'init_op_string':init_op_string,'ActiveNoise_Amplitude':self.ActiveNoise_Amplitude,'ActiveNoise_stepfeq':self.ActiveNoise_stepfeq,'k_ActiveNoise':self.k_ActiveNoise}
                    elif self.siva_sys_name == 'KS_RK4' :
                        data_for_save = {'dsol_multraj':dsol_multraj, 'num_traj':num_traj,'len_seq':len_seq,  'Lpi':pde_para,  'N':self.N,'dt_Output':self.dt_Output,'init_op_string':init_op_string}
                    elif self.siva_sys_name == 'MKS_RK4' :
                        data_for_save = {'dsol_multraj':dsol_multraj, 'num_traj':num_traj,'len_seq':len_seq, 'Lpi_rho':pde_para,  'N':self.N,'dt_Output':self.dt_Output,'init_op_string':init_op_string}

                    open_file = open(picklefilename, "wb")
                    pickle.dump(data_for_save, open_file)
                    open_file.close()
                    print('saving ' + picklefilename )

            #-------------------------------------------
            if 'ylevel' in name_xsol.casefold():
                #ylevelsol_multraj = np.zeros( (num_traj, len_seq, self.N , Ny) )

                assert np.mod(self.N, Nx)==0
                nSkip_Nx = self.N//Nx
                dx = np.pi*2/Nx

                yi = np.linspace(yB[0],yB[1],Ny, endpoint=False)

                ylevelsol_multraj = dsol_multraj[:,:,::nSkip_Nx].reshape(num_traj,len_seq,Nx,1) - yi.reshape(1,1,1,Ny)
                ylevelsol_multraj /= dx

            elif 'levelset' in name_xsol.casefold():
                #####
                pkl_filename_Levelsetsol_multraj = pkl_filename_dsol_multraj.replace('dsol_multraj', 'Levelsetsol_multraj_Nx'+str(Nx)+'_Ny'+str(Ny) )
                #'Levelsetsol_multraj' + '{:d}'.format(num_traj) + 'L{:d}'.format(len_seq) + '_nu' + '{:g}'.format(nu)[2:] + '_N' + '{:d}'.format(self.N) +'_dt' +  '{:g}'.format(self.dt_Output)[2:]  + '_'+ init_op_string
                #if abs(self.ActiveNoise_Amplitude) >= 1E-9:
                #    pkl_filename_Levelsetsol_multraj +=   '_Noise'
                #pkl_filename_Levelsetsol_multraj  += '.pkl'
                #####
                picklefilename = dir_save_training_data + pkl_filename_Levelsetsol_multraj
                if os.path.isfile (picklefilename) and (bForcedRegendsol==False):
                    open_file = open(picklefilename, "rb")
                    data_load = pickle.load(open_file)
                    open_file.close()
                    levelsetsol_multraj = data_load['levelsetsol_multraj']

                    assert num_traj == data_load['num_traj']
                    assert len_seq == data_load['len_seq']
                    assert pde_para == data_load['nu']
                    assert self.N == data_load['N']
                    assert self.dt_Output == data_load['dt_Output']
                    assert self.ActiveNoise_Amplitude == data_load['ActiveNoise_Amplitude']
                    #
                    assert Nx == data_load['Nx']
                    assert Ny == data_load['Ny']
                    #assert yB == data_load['yB']

                    print('Sucess: load ' + picklefilename )

                else:  # fresh generate data if the pickle file does not exist or bForceRegen==True

                    levelsetsol_multraj = np.zeros( (num_traj, len_seq, Nx , Ny) )
                    for i in range(num_traj):
                                                           #dsol_single_to_Levelsetsol(xi,dsol, Ny=128, Ny=128,yB=np.array([-0.7,1.3])*np.pi):
                        levelsetsol_multraj[i,:,:,:] = libData.dsol_single_to_Levelsetsol( libSiva.get_x(self.N),dsol_multraj[i,:,:], Nx, Ny, yB=yB )
                        print('', end ='.')
                    print('')

                    if bForceNoSavedsol == False:
                        data_for_save = {'levelsetsol_multraj':levelsetsol_multraj, 'num_traj':num_traj,'len_seq':len_seq,'nu':nu,'N':self.N,'dt_Output':self.dt_Output,'init_op_string':init_op_string,'ActiveNoise_Amplitude':self.ActiveNoise_Amplitude,'ActiveNoise_stepfeq':self.ActiveNoise_stepfeq,'k_ActiveNoise':self.k_ActiveNoise,'Nx':Nx,'Ny':Ny,'yB':yB }
                        open_file = open(picklefilename, "wb")
                        pickle.dump(data_for_save, open_file)
                        open_file.close()
                        print('saving ' + picklefilename )
            #-------------------------------------------

            #print( dsol.shape )
            #list_dsol.append( dsol_multraj )

            if 'ylevel' in name_xsol.casefold() :
                list_xsol.append( ylevelsol_multraj )
            elif 'levelset' in name_xsol.casefold() :
                list_xsol.append( levelsetsol_multraj )
            else: #'dsol'
                list_xsol.append( dsol_multraj )

            list_pde_para.append( pde_para )


        return  list_xsol, list_pde_para



    #@staticmethod
    #def LoadSavedData512(seq_length=200,N=512):
    #    # load data
    #    # Do not modify unless you want re-generate training data
    #    FileName_SaveShortFlame = "Data_Flame_short_dt40.pkl"
    #    if os.path.isfile(FileName_SaveShortFlame):
    #        with open(FileName_SaveShortFlame, 'rb') as file:
    #            short_sequence_disp = pickle.load(file)
    #        print(  FileName_SaveShortFlame, ' load: done' )
    #    else:
    #        print('Cannot find',  FileName_SaveShortFlame)
    #
    #    python_FileName_SaveLongFlame = "Data_Flame_Long_dt40.pkl"
    #    if os.path.isfile(python_FileName_SaveLongFlame):
    #        with open(python_FileName_SaveLongFlame, 'rb') as file:
    #            LongSequence_disp = pickle.load(file)
    #        print(  python_FileName_SaveLongFlame, ' load: done' )
    #    else:
    #        print('Cannot find',  python_FileName_SaveLongFlame)
    #
    #    if (seq_length + 1) != 201:
    #        n0 = short_sequence_disp.shape[0]
    #        dn = short_sequence_disp.shape[1] // (seq_length + 1)  # 101//20
    #
    #        sequence_disp = short_sequence_disp[:, :(dn * (seq_length + 1)), :].reshape(dn * n0, seq_length + 1, N)
    #    else:
    #        sequence_disp = short_sequence_disp
    #    #NN = LongSequence_disp.shape[0]//(seq_length+1)
    #    sequence_disp= np.vstack( ( (   LongSequence_disp[  1000*(seq_length+1):2000*(seq_length+1),: ].reshape(  1000, (seq_length+1), N )  ),  sequence_disp) )
    #    return sequence_disp






#%%

