
from difflib import diff_bytes
import numpy as np
import torch
#from utilities3 import *

import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML


from .libSiva import libSiva
from .libData import libData


#from matplotlib.animation import FuncAnimation, PillowWriter

#%matplotlib
class libPlot:

    # Nx, Ny, yB = 512, 819, np.array([-1, 2.1953125]) * np.pi
    @staticmethod
    def animlevel2D_realtime_model(model, Nx,Ny, yB, nStep=1,nSkipStep_plot=1, nReset=50, nrows=2, ncols=2 ,
                                   method_levelset='levelset_fast', bTrainedByTanh=True, yRef0=0 ,T_in=1,data_channel=1,bExtra_Reconstruct_G2D=False):

        yRef0_torch = torch.tensor( yRef0, dtype=torch.float32).reshape(-1, 1)

        if bTrainedByTanh == True:
            func_on_ylevel = lambda x: np.tanh(x)
            func_on_model = lambda x: x
            func_on_plot = lambda x: x
        else:  # False
            func_on_ylevel = lambda x: x
            func_on_model = lambda x: np.tanh(x)
            func_on_plot = lambda x: np.tanh(x)


        #data_channel = model.data_channel
        #T_in = model.T_in
        #data_channel = 1
        #T_in = 1



        nu_check = 0.07 #dummy

        #----------------
        x = libSiva.get_x( Nx)
        y = np.linspace( yB[0], yB[1], Ny)
        #y = np.linspace(-0.7, 1.3, Ny) * np.pi
        xx, yy = np.meshgrid(x, y, indexing='ij')

        global d_an, G2D_model  # d_model
        global cfs_model

        #d_an = np.zeros((nrows * ncols, Nx, T_in))
        # d_model = torch.zeros( (nrows*ncols, N, T_in*data_channel), dtype = torch.float)
        G2D_model = torch.zeros((nrows * ncols, Nx, Ny, T_in * data_channel), dtype=torch.float)

        fig, axs = plt.subplots(figsize=(12, 8), nrows=nrows, ncols=ncols)
        plt.subplots_adjust(left=0.05, bottom=0.05, right=.95, top=.95)

        line_an = []
        line_model = []
        list_linestyle_model = ['r--', 'b-.', 'g:']

        cfs_model = []
        if nrows * ncols == 1:
            axs = np.array([axs])
        for idx, ax in enumerate(axs.reshape(-1)):
            # ax.set_ylim(-2, 3)

            #line, = ax.plot(x, x, 'r-', linewidth=3);
            #line_an.append(line)

            # for c in range( data_channel ):
            # line_m,  = ax.plot(x, x, list_linestyle_model[c], linewidth=1); line_model.append(line_m)
            cfs_model.append(ax.contourf(xx, yy,   G2D_model[0, :, :, 0].numpy()  , 7, vmin =-1, vmax=1))
            ax.axes.set_aspect('equal')

            #ax.set_ylim(*ylim)

        def update_animation_realtimecaculate2D(num, model,yRef0 ): #, SivaEq):
            global d_an, G2D_model  # d_model,
            global cfs_model
            if num % nReset == 0:
                #for idx, _ in enumerate(d_an):
                #    if idx < len(d_an) // 2:
                #        d_an[idx, :, 0] = SivaEq.get_init_func_from_txt('rand_FFT_2_8')()
                #    else:
                #        d_an[idx, :, 0] = SivaEq.get_init_func_from_txt('rand_simple')()

                #    if T_in >= 2:
                #        d_tmp = SivaEq.generate_dsol_single(T_in, nu_check, d_an[idx, :, 0])  # d_tmp.shape = T_in,N
                #        d_an[idx, :, :] = np.moveaxis(d_tmp, 0, -1)

                for idx in range(G2D_model.shape[0]):

                    d0 = (np.random.rand(Nx) - 0.5) * 0.02
                    G2D_model[idx, :, :, 0] =  torch.tensor(  func_on_ylevel( libData.dsol_single_to_Levelsetsol(x, d0, Nx, Ny, yB, method_levelset)   ) - yRef0,  dtype=torch.float)
                    #G2D_model[idx,:,:,0] = torch.tensor( ( libData.dsol_single_to_Levelsetsol(x,d_an[idx,:,0],Ny=Ny,method_levelset=method_levelset) ),dtype = torch.float )
                # d_model = torch.repeat_interleave( torch.tensor(d_an,dtype = torch.float), data_channel, dim= -1)

            else:
                # step 1: analytical time adv
                #for idx, _ in enumerate(d_an):
                #    d2_tmp = SivaEq.generate_dsol_single(nStep + 1, nu_check, d_an[idx, :, -1])
                # d_an[idx,:]=d2_tmp[-1,:]
                # d_an[idx,:,:] = np.concatenate ( (d_an[idx,:,1:], np.moveaxis(d2_tmp, 0,-1)[:,[-1]] ), axis = 1 )
                #    d_an[idx, :, :-1] = d_an[idx, :, 1:]
                #    d_an[idx, :, -1] = d2_tmp[-1, :]
                # step 2:  time adv using the trained model
                with torch.no_grad():
                    for tmp in range( nSkipStep_plot) :
                        # im = model(d_model, torch.tensor([nu_check]) )
                        # d_model = torch.cat( (d_model[:,:,1*data_channel:], im), dim = -1)
                        # print('cacl0', G2D_model.shape)
                        #im = model(G2D_model, torch.tensor([nu_check]).repeat(G2D_model.shape[0]))
                        im = model( G2D_model )
                        if bExtra_Reconstruct_G2D==True:
                            #im = torch.clamp(im, min=-1, max=1)
                            im_np = libData.correct_tanh_ylevel(im[0,:,:,0].detach().numpy(),Nx,Ny,yB)
                            im[0,:,:,0] = torch.tensor(im_np)

                        im[:,:, :10,:]  = 1.
                        im[:,:, -10:,:] = -1.
                        im = torch.clamp(im+ yRef0_torch, min=-1., max=1.)-yRef0_torch

                        # print('cacl0', im.shape)

                        # for idx in range( im.shape[0] ):
                        #    G_beforeReInit = im[idx,:,:,0].detach().numpy()

                        #    im[idx,:,:,0] = torch.tensor( np.tanh( libData.Reinit_LevelsetG(G_beforeReInit,btanh=True) ), dtype= torch.float)

                        # print(im.shape)

                        # print('cacl1', im.shape)
                        G2D_model = torch.cat((G2D_model[:, :, :, 1 * data_channel:], im), dim=-1)
                        # print('cacl2')

            # plotting

            for idx, ax in enumerate(axs.reshape(-1)):
                # for idx,_ in enumerate(d_an):
                #line_an[idx].set_data(x, d_an[idx, :, -1])

                # for c in range( data_channel):
                # line_model[c+idx*data_channel].set_data(x, d_model[idx,:,(T_in-1)*data_channel+c].numpy() )
                for coll in cfs_model[idx].collections:  # save memerory
                    coll.remove()

                cfs_model[idx] = ax.contourf(xx, yy,
                                             func_on_plot(G2D_model[idx, :, :, (T_in - 1) * data_channel].numpy())+yRef0    , 7, vmin =-1, vmax=1)

                # print('idx',idx, end=', ')

            fig.suptitle("n = %d " % (num))

            return

        ani = animation.FuncAnimation(fig, update_animation_realtimecaculate2D, nReset, fargs=(model,yRef0), interval=10, blit=False)
        return ani



    @staticmethod
    def animlevel2D_realtimeCmp_model_analytical( SivaEq, model,nu_check=0.07, nStep=1,nSkipStep_plot=1, nReset=500,x=None, nrows=2,ncols=2, ylim=(-0.7*np.pi,1.3*np.pi) ,  method_levelset='levelset_fast',bTrainedByTanh=True , yRef0 =0):
        yRef0_torch = torch.tensor( yRef0, dtype=torch.float32).reshape(-1, 1)

        if bTrainedByTanh == True:
            func_on_ylevel =  lambda x: np.tanh(x)
            func_on_model = lambda x: x
            func_on_plot = lambda x: x
        else: # False
            func_on_ylevel =  lambda x: x
            func_on_model = lambda x: np.tanh(x)
            func_on_plot = lambda x: np.tanh(x)

        #nonlinear_fun = lambda x : x
        nonlinear_fun = np.tanh
        data_channel = 1 #model.data_channel
        T_in =1 #  model.T_in

        N = SivaEq.N

        if x == None:
            #x = 2*np.pi*np.arange(N)/N-np.pi
            x = libSiva.get_x(N)

        Ny = 128
        Nx = 128
        y = np.linspace(-0.7,1.3,Ny)*np.pi
        xx , yy = np.meshgrid(x,y, indexing='ij')

        global d_an,G2D_model #d_model
        global cfs_model

        d_an     =   np.zeros( (nrows*ncols, N, T_in             )  )
        #d_model = torch.zeros( (nrows*ncols, N, T_in*data_channel), dtype = torch.float)
        G2D_model = torch.zeros( (nrows*ncols, N, Ny, T_in*data_channel), dtype = torch.float)

        fig, axs = plt.subplots(figsize=(12,8), nrows=nrows, ncols=ncols )
        plt.subplots_adjust(left=0.05, bottom=0.05, right=.95, top=.95)

        line_an    = []
        line_model = []
        list_linestyle_model = ['r--','b-.','g:']


        cfs_model =[]
        if nrows*ncols ==1:
            axs = np.array( [axs])


        for idx, ax in enumerate( axs.reshape(-1) ):
            #ax.set_ylim(-2, 3)

            line,  = ax.plot(x, x, 'r-', linewidth=3); line_an.append(line)

            #for c in range( data_channel ):
                #line_m,  = ax.plot(x, x, list_linestyle_model[c], linewidth=1); line_model.append(line_m)
            cfs_model.append(  ax.contourf( xx,yy, G2D_model[0,:,:,0].numpy(),3) )

            ax.set_ylim(*ylim )


        def update_animation_realtimecaculate2D(num,model,SivaEq) :
            global d_an,G2D_model #d_model,
            global cfs_model
            if num % nReset == 0:
                for idx,_ in enumerate(d_an):
                    if idx < len(d_an)//2:
                        d_an[idx,:,0] = SivaEq.get_init_func_from_txt ('rand_FFT_2_8') ()
                    else:
                        d_an[idx,:,0] = SivaEq.get_init_func_from_txt ('rand_simple') ()

                    if T_in >= 2:
                        d_tmp = SivaEq.generate_dsol_single( T_in , nu_check, d_an[idx,:,0]  )  #d_tmp.shape = T_in,N
                        d_an[idx,:,:] = np.moveaxis(d_tmp, 0,-1)
                for idx in range(d_an.shape[0]):
                    G2D_model[idx,:,:,0] = torch.tensor( func_on_ylevel( libData.dsol_single_to_Levelsetsol(x,d_an[idx,:,0],Nx,Ny=Ny,method_levelset=method_levelset) ),dtype = torch.float )
                    #G2D_model[idx,:,:,0] = torch.tensor( ( libData.dsol_single_to_Levelsetsol(x,d_an[idx,:,0],Ny=Ny,method_levelset=method_levelset) ),dtype = torch.float )
                #d_model = torch.repeat_interleave( torch.tensor(d_an,dtype = torch.float), data_channel, dim= -1)

            else:
                # step 1: analytical time adv
                for tmp in range(nSkipStep_plot):
                    for idx,_ in enumerate(d_an):
                        d2_tmp = SivaEq.generate_dsol_single( nStep+1, nu_check, d_an[idx,:,-1] )
                        #d_an[idx,:]=d2_tmp[-1,:]
                        #d_an[idx,:,:] = np.concatenate ( (d_an[idx,:,1:], np.moveaxis(d2_tmp, 0,-1)[:,[-1]] ), axis = 1 )
                        d_an[idx,:,:-1] = d_an[idx,:,1:]
                        d_an[idx,:,-1 ] = d2_tmp[-1,:]
                    # step 2:  time adv using the trained model
                    with torch.no_grad():
                        #im = model(d_model, torch.tensor([nu_check]) )
                        #d_model = torch.cat( (d_model[:,:,1*data_channel:], im), dim = -1)
                        #print('cacl0', G2D_model.shape)

                        #im = model(G2D_model, torch.tensor([nu_check]).repeat(G2D_model.shape[0]) )
                        im = model(G2D_model) #, torch.tensor([nu_check]).repeat(G2D_model.shape[0]) )
                      #  im[:, :, :10, :] = 1.
                      #  im[:, :, -10:, :] = -1.
                      #  im = torch.clamp(im + yRef0_torch, min=-1., max=1.) - yRef0_torch
                        #print('cacl0', im.shape)
                        #for idx in range( im.shape[0] ):
                        #    G_beforeReInit = im[idx,:,:,0].detach().numpy()

                        #    im[idx,:,:,0] = torch.tensor( np.tanh( libData.Reinit_LevelsetG(G_beforeReInit,btanh=True) ), dtype= torch.float)

                            #print(im.shape)
                        #print('cacl1', im.shape)
                        G2D_model = torch.cat( (G2D_model[:,:,:,1*data_channel:], im ), dim = -1)
                        #print('cacl2')
            # plotting
            for idx, ax in enumerate( axs.reshape(-1) ):
            #for idx,_ in enumerate(d_an):
                line_an[idx].set_data( x, d_an[idx,:,-1]   )

                #for c in range( data_channel):
                    #line_model[c+idx*data_channel].set_data(x, d_model[idx,:,(T_in-1)*data_channel+c].numpy() )
                for coll in cfs_model[idx].collections: # save memerory
                    coll.remove()

                cfs_model[idx] = ax.contourf(xx,yy,   func_on_plot( G2D_model[idx,:,:,(T_in-1)*data_channel].numpy() ),3)
                #print('idx',idx, end=', ')
            fig.suptitle("n = %d " % (num) )
            return


        ani = animation.FuncAnimation(fig, update_animation_realtimecaculate2D, nReset, fargs=(model,SivaEq), interval=100,blit=False)
        return ani















    @staticmethod
    def anim_realtimeCmp_model_analytical( SivaEq, model,nu_check=0.07, nReset=500,x=None, nrows=2,ncols=2, ylim=(-1.5,3) ,nStep=1 ):

        data_channel = 1 # model.data_channel
        T_in =1 # model.T_in

        N = SivaEq.N

        if x == None:
            x = 2*np.pi*np.arange(N)/N-np.pi

        global d_an,d_model
        d_an     =   np.zeros( (nrows*ncols, N, T_in             )  )
        d_model = torch.zeros( (nrows*ncols, N, T_in*data_channel), dtype = torch.float)



        fig, axs = plt.subplots(figsize=(12,8), nrows=nrows, ncols=ncols )
        plt.subplots_adjust(left=0.05, bottom=0.05, right=.95, top=.95)

        line_an    = []
        line_model = []
        list_linestyle_model = ['r--','b-.','g:']
        if nrows*ncols ==1:
            axs = np.array( [axs])

        for ax in axs.reshape(-1):
            #ax.set_ylim(-2, 3)

            line,  = ax.plot(x, x, 'k-', linewidth=1); line_an.append(line)

            for c in range( data_channel ):
                line_m,  = ax.plot(x, x, list_linestyle_model[c], linewidth=1); line_model.append(line_m)

            ax.set_ylim( *ylim )


        def update_animation_realtimecaculate(num,model,SivaEq) :
            global d_an,d_model
            if num % nReset == 0:
                for idx,_ in enumerate(d_an):
                    if idx < len(d_an)//2:
                        d_an[idx,:,0] = SivaEq.get_init_func_from_txt ('rand_FFT_2_8') ()
                    else:
                        d_an[idx,:,0] = SivaEq.get_init_func_from_txt ('rand_simple') ()

                    if T_in >= 2:
                        d_tmp = SivaEq.generate_dsol_single( T_in , nu_check, d_an[idx,:,0]  )  #d_tmp.shape = T_in,N
                        d_an[idx,:,:] = np.moveaxis(d_tmp, 0,-1)

                d_model = torch.repeat_interleave( torch.tensor(d_an,dtype = torch.float), data_channel, dim= -1)
            else:
                # step 1: analytical time adv
                for _ in range(nStep):
                    for idx,_ in enumerate(d_an):
                        d2_tmp = SivaEq.generate_dsol_single( 2, nu_check, d_an[idx,:,-1] )
                        #d_an[idx,:]=d2_tmp[-1,:]
                        #d_an[idx,:,:] = np.concatenate ( (d_an[idx,:,1:], np.moveaxis(d2_tmp, 0,-1)[:,[-1]] ), axis = 1 )
                        d_an[idx,:,:-1] = d_an[idx,:,1:]
                        d_an[idx,:,-1 ] = d2_tmp[-1,:]
                # step 2:  time adv using the trained model
                with torch.no_grad():
                    im = model(d_model ) # , torch.tensor([nu_check]) )
                    d_model = torch.cat( (d_model[:,:,1*data_channel:], im), dim = -1)
            # plotting

            for idx,_ in enumerate(d_an):
                line_an[idx].set_data( x, d_an[idx,:,-1]   )

                for c in range( data_channel):
                    line_model[c+idx*data_channel].set_data(x, d_model[idx,:,(T_in-1)*data_channel+c].numpy() )

            fig.suptitle("n = %d " % (num) )

            #return

        interval = 1000 if nStep>=10 else 10
        ani = animation.FuncAnimation(fig, update_animation_realtimecaculate, nReset, fargs=(model,SivaEq), interval=interval,blit=False)
        return ani


    #################################################################################


    @staticmethod
    def anim_singleCmp_model_analytical( SivaEq, model, init_str='rand_FFT', nu=0.07, numTotalTimeStep=1000, x=None,  ylim=(-2,3) ):

        T_in        = model.T_in
        data_channel= model.data_channel

        N = SivaEq.N
        if x == None:
            x = 2*np.pi*np.arange(N)/N-np.pi

        init_str_op = 'rand_FFT_2_8' if 'FFT' in init_str else 'rand_simple'
        d0 = SivaEq.get_init_func_from_txt (init_str_op) ()

        d_an = SivaEq.generate_dsol_single( numTotalTimeStep , nu, d0 )      # d_an.shape ==numTotalTimeStep,  N
        d_an = np.moveaxis(d_an,-1,0).reshape(1,N,numTotalTimeStep)           # d_an.shape ==1, N, numTotalTimeStep
        d_an = torch.tensor( d_an, dtype=torch.float )

        if model.method_TimeAdv.casefold() == 'gru':
            assert T_in == 1

            with torch.no_grad():
                #xx = test_a[44,:,:]  # 1 cordinate + 10 time
                #xx = truth_plot[:,:, :1 ]
                #xx = xx.reshape(1,N,-1)
                xx = torch.repeat_interleave(  d_an[...,:T_in] , data_channel, dim = -1) # xx.shape = 1, N, T_in*data_channel
                xx_nohidden = xx
                h_t = torch.zeros(1,N,data_channel)
                for t in range( d_an.shape[-1] - 1 ):
                    #xx=xx.to(device)
                    im, h_t            = model(xx, torch.tensor([nu]), h_t)

                    zeroh_t            = torch.zeros(1,N,data_channel)
                    im_nohidden, zeroh = model(xx_nohidden, torch.tensor([nu]), zeroh_t)

                    if t == 0:
                        pred          =  torch.cat( ( xx[..., :]   , im         ), -1)
                        pred_nohidden =  torch.cat( ( xx[..., :]   , im_nohidden), -1)
                    else:
                        pred          =  torch.cat( ( pred         , im         ), -1)
                        pred_nohidden =  torch.cat( ( pred_nohidden, im_nohidden), -1)

                    xx = im  #  torch.cat(( xx[..., 1:], im  ), dim=-1)
                    xx_nohidden = im_nohidden

        else: # model.method_TimeAdv.casefold() == 'simple':

            with torch.no_grad():

                xx = torch.repeat_interleave(  d_an[...,:T_in] , data_channel, dim = -1) # xx.shape = 1, N, T_in*data_channel

                for t in range( d_an.shape[-1]- T_in ):
                    #xx=xx.to(device)
                    #xx_input = xx.expand(-1,-1,data_channel)
                    im = model(xx,  torch.tensor([nu]) )
                    #im = yy_output[...,0:1]

                    if t == 0:
                        pred = torch.cat( (xx,         im), -1 )
                    else:
                        pred = torch.cat( (pred,      im), -1 )

                    xx = torch.cat( (xx[..., 1*data_channel:], im  ), dim=-1)


        # cut away the inital T_in step
        d_an = d_an[...,(T_in-1):].squeeze().numpy()
        print('d_an.shape=', d_an.shape)

        pred = pred[...,(T_in-1)*data_channel:].squeeze().detach().numpy()
        print('pred.shape=', pred.shape)
        if model.method_TimeAdv.casefold() == 'gru':
            pred_nohidden = pred_nohidden.squeeze().detach().numpy() # be aware that T_in == 1
        #

        #pred = pred.to('cpu')
        #truth_plot= truth_plot.to('cpu')
        #t_OutPut = np.arange(0, 1000) # * dt_OutPut

        def update_animation_flame(num, d_an, pred, data3=None ):
            line_an.set_data(x, d_an[:,num])
            for c in range(data_channel):
                list_line_model[c].set_data(  x, pred[:, (num-1)*data_channel+c]  )

            if  data3 is not  None:
                line3.set_data(x, data3[:,num])
            #line1, = ax.plot(x[:-1], (np.diff(  data1[num+N0]  )) )
            #line2, = ax.plot(x[:-1], (np.diff(  data2[num]  )) )
            ax.set_title("n = %d" % (num))


        loop_length =  d_an.shape[-1]

        #%matplotlib
        fig, ax = plt.subplots(figsize=(8,6))
        plt.subplots_adjust(left=0.05, bottom=0.05, right=.95, top=.95)

        line_an, = ax.plot(x, d_an[:,0], 'k-', linewidth=1)
        list_line_model = []
        list_linestyle_model = ['r-','b-','g-']
        for c in range(data_channel):
            line_m, = ax.plot(x, pred[:, c ], list_linestyle_model[c], linewidth=1);    list_line_model.append(line_m)

        if model.method_TimeAdv.casefold() == 'gru':
            line3, = ax.plot(x, pred_nohidden[:,0], 'r:', linewidth=1)

        ax.set_ylim(  *ylim )


        if model.method_TimeAdv.casefold() == 'gru':
            ani = animation.FuncAnimation(fig, update_animation_flame, loop_length-1, fargs=(d_an, pred,pred_nohidden), interval=1,blit=False)
        else:
            ani = animation.FuncAnimation(fig, update_animation_flame, loop_length-1, fargs=(d_an, pred), interval=1,blit=False)

        return ani





    @staticmethod
    def anim_singleCmp_model_PRE( N, model, init_str='rand_FFT', nu=10., numTotalTimeStep=1000, x=None,  ylim=(-2,3) ):

        T_in        = model.T_in
        data_channel= model.data_channel

        if x == None:
            x = 2*np.pi*np.arange(N)/N-np.pi

        init_str_op = 'rand_FFT_2_8' if 'FFT' in init_str else 'rand_simple'

        #d0 = 0.5*np.random.rand(N)
        d0 = 1*np.sin(  np.arange(N)/N*np.pi*2 )



        d0 = torch.tensor( d0, dtype=torch.float).reshape(1,N,1)

        #SivaEq.get_init_func_from_txt (init_str_op) ()

        #d_an = SivaEq.generate_dsol_single( numTotalTimeStep , nu, d0 )      # d_an.shape ==numTotalTimeStep,  N
        #d_an = np.moveaxis(d_an,-1,0).reshape(1,N,numTotalTimeStep)           # d_an.shape ==1, N, numTotalTimeStep
        #d_an = torch.tensor( d_an, dtype=torch.float )


        # model.method_TimeAdv.casefold() == 'simple':
        with torch.no_grad():
            #xx = torch.repeat_interleave(  d0[...,:T_in] , data_channel, dim = -1) # xx.shape = 1, N, T_in*data_channel
            xx = torch.repeat_interleave(  d0 , data_channel, dim = -1) # xx.shape = 1, N, T_in*data_channel

            for t in range( numTotalTimeStep - T_in ):
                #xx=xx.to(device)
                #xx_input = xx.expand(-1,-1,data_channel)
                im = model(xx,  torch.tensor([nu]) )
                #im = yy_output[...,0:1]

                if t == 0:
                    pred = torch.cat( (xx,         im), -1 )
                else:
                    pred = torch.cat( (pred,      im), -1 )

                xx = torch.cat( (xx[..., 1*data_channel:], im  ), dim=-1)


        # cut away the inital T_in step
        #d_an = d_an[...,(T_in-1):].squeeze().numpy()
        #print('d_an.shape=', d_an.shape)

        pred = pred[...,(T_in-1)*data_channel:].squeeze().detach().numpy()
        print('pred.shape=', pred.shape)


        def update_animation_flame(num,  pred, data3=None ):
            #line_an.set_data(x, d_an[:,num])
            for c in range(data_channel):
                list_line_model[c].set_data(  x, pred[:, (num-1)*data_channel+c]  )

            if  data3 is not  None:
                line3.set_data(x, data3[:,num])
            #line1, = ax.plot(x[:-1], (np.diff(  data1[num+N0]  )) )
            #line2, = ax.plot(x[:-1], (np.diff(  data2[num]  )) )
            ax.set_title("n = %d" % (num))


        #loop_length =  d_an.shape[-1]
        loop_length= numTotalTimeStep

        #%matplotlib
        fig, ax = plt.subplots(figsize=(8,6))
        plt.subplots_adjust(left=0.05, bottom=0.05, right=.95, top=.95)

        #line_an, = ax.plot(x, d_an[:,0], 'k-', linewidth=1)
        list_line_model = []
        list_linestyle_model = ['r-','b-','g-']
        for c in range(data_channel):
            line_m, = ax.plot(x, pred[:, c ], list_linestyle_model[c], linewidth=1);    list_line_model.append(line_m)

        #if model.method_TimeAdv.casefold() == 'gru':
        #    line3, = ax.plot(x, pred_nohidden[:,0], 'r:', linewidth=1)

        ax.set_ylim(  *ylim )


        #if model.method_TimeAdv.casefold() == 'gru':
        #    ani = animation.FuncAnimation(fig, update_animation_flame, loop_length-1, fargs=(d_an, pred,pred_nohidden), interval=1,blit=False)
        #else:

        ani = animation.FuncAnimation(fig, update_animation_flame, loop_length-1, fargs=( pred,), interval=1,blit=False)

        return ani

    @staticmethod
    def Take_4RandInit_plot_Asequence(SivaEq, model, nu=0.07, numTotalTimeStep=1000, list_i_t_plot = [1, 20, 50, 80, 100,200,300,500,600,1000],figsize=(9,5) ):
        T_in = 1 #model.T_in
        data_channel = 1# model.data_channel

        N = SivaEq.N
        x = 2 * np.pi * np.arange(N) / N - np.pi

        list__init_str = ['FFT','FFT','rand','rand']
        list_d_an = []
        list_pred = []
        list_NormalizedTotalLen_an =[]
        list_NormalizedTotalLen_pred = []

        for init_str in list__init_str:

            init_str_op = 'rand_FFT_2_8' if 'FFT' in init_str else 'rand_simple'
            d0 = SivaEq.get_init_func_from_txt(init_str_op)()
            d_an = SivaEq.generate_dsol_single(numTotalTimeStep, nu, d0)  # d_an.shape ==numTotalTimeStep,  N
            d_an = np.moveaxis(d_an, -1, 0).reshape(1, N, numTotalTimeStep)  # d_an.shape ==1, N, numTotalTimeStep
            d_an = torch.tensor(d_an, dtype=torch.float)
            # -----------
            with torch.no_grad():

                xx = torch.repeat_interleave(d_an[..., :T_in], data_channel, dim=-1)  # xx.shape = 1, N, T_in*data_channel

                for t in range(d_an.shape[-1] - T_in):
                    # xx=xx.to(device)
                    # xx_input = xx.expand(-1,-1,data_channel)
                    im = model(xx ) # , torch.tensor([nu]))
                    # im = yy_output[...,0:1]

                    if t == 0:
                        pred = torch.cat((xx, im), -1)
                    else:
                        pred = torch.cat((pred, im), -1)

                    xx = torch.cat((xx[..., 1 * data_channel:], im), dim=-1)
            # -----------
            # cut away the inital T_in step
            d_an = d_an[..., (T_in - 1):].squeeze().numpy()
            print('d_an.shape=', d_an.shape)

            pred = pred[..., (T_in - 1) * data_channel:].squeeze().detach().numpy()
            print('pred.shape=', pred.shape)

            #len_an_0 ,_   = libData.dsol_to_whole_length(x, d_an.transpose() )
            #len_pred_0 ,_ = libData.dsol_to_whole_length(x, pred.transpose() )
            len_an_0    = libData.dsol_to_whole_length(x, d_an.transpose() )
            len_pred_0  = libData.dsol_to_whole_length(x, pred.transpose() )

            #######################
            list_d_an.append(d_an.transpose() )
            list_pred.append(pred.transpose() )
            list_NormalizedTotalLen_an.append(len_an_0)
            list_NormalizedTotalLen_pred.append(len_pred_0)




        #---------------------------------------------------------------------------
        #fig = plt.figure(1, figsize= [16,8])
        #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
        fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize= figsize )
        for ax, d_an,pred,len_an,len_pred in zip(axs.reshape(-1), list_d_an,list_pred,list_NormalizedTotalLen_an,list_NormalizedTotalLen_pred) :
            ax.set_yticks( [-np.pi, 0, np.pi] )
            ax.set_yticklabels(['$-\pi$', '0', '$\pi$'] , rotation='vertical', fontsize=14)
            ax.set_xticks( [ 0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi] )
            ax.set_xticklabels(['0','$\pi$','2$\pi$', '3$\pi$','4$\pi$' ] , fontsize=14 )
            ax.set_aspect('equal')

            for ii in list_i_t_plot:
                i = ii -1
                d = (i)/100
                ax.plot(  d_an[i] + d,   x,'-r' )
                ax.plot(  pred[i] + d,   x,'--k' )

        #for ax in axs.flat:
            #ax.set(xlabel='d(x,t)+t/100', ylabel='')
            #ax.set_ylabel('x', fontsize=16 )
        #    ax.set_xlabel('d(x,t)+t/100', fontsize=16)
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        #for ax in axs.flat:
        #    ax.label_outer()
        #fig.tight_layout()
        plt.subplots_adjust(left=0.065, right=0.98, top=0.99, bottom=0.13, wspace=0.05)
        fig.text(0.5, 0.005, 'd(x,t)+t/100', ha='center',fontsize=16)
        fig.text(0.001, 0.52, 'x', va='center', rotation='vertical',fontsize=16 )
        #plt.savefig('disp_2x2_07.png')


        #--------------------------------------------------------------------
        fig, axs = plt.subplots(2,2,sharex=True, sharey=True,figsize= [8,4])
        for ax, d_an,pred,len_an,len_pred in zip(axs.reshape(-1), list_d_an,list_pred,list_NormalizedTotalLen_an,list_NormalizedTotalLen_pred) :
            ax.set_yticks( [ 1, 1.5] )
            ax.set_yticklabels(['1', '1.5'], rotation='vertical',  fontsize=14 )
            ax.set_xticks( [ 0, 100, 500, 1000] )
            ax.set_xticklabels(['0','100','500', '1000' ], fontsize=14)
            #ax.set_ylim(1,1.6)

            list_i_t_plot = [1, 20, 50, 80, 100,200,300,500,600,1000]
            for ii in list_i_t_plot:
                #i = ii -1
                #d = (i)/100
                ax.plot(  len_an,'-r' )
                ax.plot(  len_pred, '--k' )
        #for ax in axs.flat:
            #ax.set(xlabel='t/0.015', ylabel='total length')
            #ax.set_xlabel('t/0.015', fontsize=16)
            #ax.set_ylabel('total length', fontsize=16)
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        #for ax in axs.flat:
            #ax.label_outer()
        #fig.tight_layout()
        plt.subplots_adjust(left=0.065, right=0.98, top=0.99, bottom=0.13, wspace=0.05)

        fig.text(0.5, 0.005, 't/0.015', ha='center',fontsize=16)
        fig.text(0.001, 0.52, 'normalized total length', va='center', rotation='vertical',fontsize=16 )

        #plt.savefig('len_2x2_07.png')

