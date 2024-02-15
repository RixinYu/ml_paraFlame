




# N=128
# Fa = torch.fft.rfft( torch.sin(  (torch.arange(N)/N*2*np.pi-np.pi)**2 ), norm='forward' )
# plt.figure()
# plt.plot(np.arange(N//2+1), Fa.real ,'r')
# plt.plot(np.arange(N//2+1), Fa.imag, 'b--')
# NN=768
# Fb = torch.fft.rfft( torch.sin(  (torch.arange(NN)/NN*2*np.pi-np.pi)**2 ), norm='forward' )
# plt.plot( np.arange(NN//2+1),Fb.real, 'rs')
# plt.plot( np.arange(NN//2+1),Fb.imag, 'bs')




#case_list = [
#            { 'dir':'L256_rho10_muc' , 'rhoR':10 ,'L':256 , 'L_y':768, 'N':1024,'N_y':3072,'dyScale':1  },
#            { 'dir':'L320_rho10_muc' , 'rhoR':10 ,'L':320 , 'L_y':960, 'N':1280,'N_y':3840,'dyScale':1  },
#            { 'dir':'L384_rho10_muc' , 'rhoR':10 ,'L':384 , 'L_y':1152, 'N':1536,'N_y':4608,'dyScale':1  },
#            { 'dir':'L448_rho10_muc' , 'rhoR':10 ,'L':448 , 'L_y':1344, 'N':1792,'N_y':5376,'dyScale':1  },
#            {'dir': 'L480_rho10_muc', 'rhoR': 10, 'L': 480, 'L_y': 1440, 'N': 1920, 'N_y': 5760, 'dyScale': 1},
#            { 'dir':'L496_rho10_muc' , 'rhoR':10 ,'L':496 , 'L_y':1488, 'N':1984,'N_y':5952,'dyScale':1  },
#            { 'dir':'L512_rho10_muc' , 'rhoR':10 ,'L':512 , 'L_y':1536, 'N':2048,'N_y':6144,'dyScale':1  },
#            { 'dir':'L1024_rho10_muc', 'rhoR':10 ,'L':1024, 'L_y':3072, 'N':2048,'N_y':9216,'dyScale':1.5  },
#            { 'dir':'L1536_rho10_muc', 'rhoR':10 ,'L':1536, 'L_y':4608, 'N':3072,'N_y':13824,'dyScale':1.5  },
#
#            { 'dir':'L256_rho8_muc' , 'rhoR':8 ,'L':256 , 'L_y':768, 'N':1024,'N_y':3072,'dyScale':1  },
#            { 'dir':'L320_rho8_muc' , 'rhoR':8 ,'L':320 , 'L_y':960, 'N':1280,'N_y':3840,'dyScale':1  },
#             #error{ 'dir':'L384_rho8'     , 'rhoR':8 ,'L':384 , 'L_y':1152, 'N':1536,'N_y':4608,'dyScale':1  },
#            { 'dir':'L512_rho8_muc' , 'rhoR':8 ,'L':512 , 'L_y':1536, 'N':2048,'N_y':6144,'dyScale':1  },
#            { 'dir':'L768_rho8_muc' , 'rhoR':8 ,'L':768 , 'L_y':2304, 'N':3072,'N_y':9216,'dyScale':1  },
#            { 'dir':'L1536_rho8_muc', 'rhoR':8 ,'L':1536, 'L_y':4608, 'N':3072,'N_y':13824,'dyScale':1.5  },
#
#            { 'dir':'L256_rho5_muc' , 'rhoR':5 ,'L':256 , 'L_y':768, 'N':1024,'N_y':3072,'dyScale':1 },
#            { 'dir':'L384_rho5_muc' , 'rhoR':5 ,'L':384 , 'L_y':1152, 'N':1536,'N_y':4608,'dyScale':1  },
#            { 'dir':'L512_rho5_muc' , 'rhoR':5 ,'L':512 , 'L_y':1536, 'N':2048,'N_y':6144,'dyScale':1  },
#            { 'dir':'L1024_rho5_muc', 'rhoR':5 ,'L':1024, 'L_y':3072, 'N':2048,'N_y':9216,'dyScale':1.5  },
#            { 'dir':'L1536_rho5_muc', 'rhoR':5 ,'L':1536, 'L_y':4608, 'N':3072,'N_y':13824,'dyScale':1.5  }
#            ]
##################

import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation

from flame_net.libSiva import libSiva



class libcfdData:
    @staticmethod
    def demo( case = 1 , params={} ):

        params.setdefault('file','L512_rho8')

        cfd_data_dir='d:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\'
        pickle_file = params['file'] + '.pkl'

        if case ==1 :
            yB_estimate = np.array([-0.4, 1.6]) * np.pi
            Nx_target = 128
            AspectRatio_set = 1
            Ny_actual, yB = libSiva.get2D_Ny_yB_from_estimate(Nx_target, yB_estimate, AspectRatio_set)
            G, xyC = libcfdData.get_levelsetG2D(cfd_data_dir+pickle_file, Nx_target,
                                                yB_estimate=yB, ipick_single=600, AspectRatio_set=AspectRatio_set)
            x = libSiva.get_x(Nx_target)
            y = np.linspace(yB_estimate[0], yB_estimate[1], Ny_actual)
            xx, yy = np.meshgrid(x, y, indexing='ij')
            plt.figure()
            plt.contourf(xx, yy, np.tanh(G[0] / AspectRatio_set))
        elif case ==2:

            params.setdefault('list_picklefilename' , ['L512_rho8.pkl','L768_rho8.pkl','L1536_rho8.pkl']  )
            libcfdData.get_yBminmax_from_PREdata(list_picklefilename = params['list_picklefilename'], cfd_data_dir=cfd_data_dir)
        elif case ==3:
            libcfdData.animate_cfd_file( params['file'])
        elif case==4:

            params.setdefault('list_picklefilename' , ['L1536_rho8.pkl' ]  )
            #params['list_picklefilename']= ['L320_rho8.pkl' ]
            params.setdefault('yB_estimate' , [-0.7, 2.5] )
            Nx=256
            yB_estimate= np.array( params['yB_estimate']  ) * np.pi   #yB_estimate=np.array([-0.7, 2.5]) * np.pi

            Ny, yB = libSiva.get2D_Ny_yB_from_estimate(Nx, yB_estimate,AspectRatio_set=1)
            list_y, list_p = libcfdData.load_2DPREdata( params['list_picklefilename'], cfd_data_dir, Nx, yB, AspectRatio_set=1, ThicknessScale= 4 )
            G = list_y[0]

            fig, ax = plt.subplots(figsize=(  6*(yB_estimate[1]-yB_estimate[0])/2/np.pi  , 6 ) )
            ax.axes.set_aspect('equal')
            #im = ax.imshow( G[515],  interpolation='none' )
            im = ax.imshow( G[0],  interpolation='none' )
            fig.tight_layout()
            def update_animation(i):
                #a = im.get_array()
                #i = i+510
                ax.set_title( str(i) )
                im.set_array( G[i] )
                return [im]
            ani = animation.FuncAnimation(fig, update_animation, len(G), interval=1,blit=False)

            return ani


    @staticmethod
    def nouse_Reorg_y(y, T_out=20,T_in=1):
        num_traj=1 # only for CFD - PRE data
        length_y, N, mm3 =  y.shape

        #num__split_seq_pierce =  (length_y-T_in)// T_out
        #sequence_disp = np.zeros( ( num__split_seq_pierce*num_traj, T_out+T_in, N , mm3 ) )
        #for i in range ( num__split_seq_pierce ):
        #    i_start    =  i*T_out
        #    #sequence_disp[ i*num_traj:(i+1)*num_traj, :, :, :] = y[ i_start:i_start+T_out+T_in, :, :  ]
        #    sequence_disp[ i*num_traj:(i+1)*num_traj, :, :, :] = y[ i_start:i_start+T_out+T_in, :, :  ]

        num__split_seq_pierce =  (length_y-T_out) - T_in  + 1   #T_in)// T_out
        sequence_disp = np.zeros( ( num__split_seq_pierce, T_out+T_in, N , mm3 ) )
        for i in range ( num__split_seq_pierce ):
            i_start    =  i #  *T_out
            #sequence_disp[ i*num_traj:(i+1)*num_traj, :, :, :] = y[ i_start:i_start+T_out+T_in, :, :  ]
            sequence_disp[ i:(i+1), :, :, :] = y[ i_start:i_start+T_out+T_in, :, :  ]
        return sequence_disp

    @staticmethod
    def Reorg_y(y, T_out=20,T_in=1, nStep=1, nStepSkip=1):
        num_traj=1 # only for CFD - PRE data
        assert nStepSkip <= nStep
        assert np.mod(nStep, nStepSkip) == 0

        #length_y, Nx, Ny,one_nosue =  y.shape
        length_y, *shape_rest =  y.shape

        #---------------
        if nStep==1:
            num__split_seq_pierce =  (length_y-T_out) - T_in  + 1   #T_in)// T_out
            # print('Reorg_y: num__split_seq_pierce=', num__split_seq_pierce)

            sequence_disp = np.zeros( ( num__split_seq_pierce, T_out+T_in, *shape_rest ) )
            for i in range ( num__split_seq_pierce ):
                i_start    =  i #  *T_out
                #sequence_disp[ i*num_traj:(i+1)*num_traj, :, :, :] = y[ i_start:i_start+T_out+T_in, :, :  ]
                #sequence_disp[ i:(i+1), ...] = y[ i_start:i_start+T_out+T_in, ...]
                sequence_disp[ i, ...] = y[ i_start:i_start+T_out+T_in, ...]
                # print('',end='.')

        else: #nStep>1
            #num__split_seq_pierce = ((1 + (length_y - 1) // nStep) - T_in) // T_out
            num__split_seq_pierce = ((     (length_y - 1) // nStep) - T_in) // T_out -1
            print( 'Reorg_y:num__split_seq_pierce', num__split_seq_pierce)

            sequence_disp = np.zeros(((nStep // nStepSkip) * num__split_seq_pierce * num_traj, T_out + T_in, *shape_rest) )
            for i in range(num__split_seq_pierce):
                i_start = i * T_out * nStep
                for k in range(0, nStep, nStepSkip):
                    k_k = k // nStepSkip
                    k_i_start = i_start + k
                    #print('Reorg_y: k=',k, 'k_i_start=', k_i_start, ',  k_k=',k_k, ' , i_start=',i_start)
                    assert  k_i_start + (T_out + T_in) * nStep < length_y , 'Reorg_y: k_i_start + (T_out + T_in) * nStep < length_y '
                    sequence_disp[ (k_k + i * nStep//nStepSkip) * num_traj:((k_k + i * nStep // nStepSkip) + 1) * num_traj, ...] = y[k_i_start:k_i_start + (T_out + T_in) * nStep:nStep, ...]

        return sequence_disp



    @staticmethod
    def load_PREdata(list_picklefilename =None, cfd_data_dir= None, Nx_target=None, varname = 'y3' ): #Nx_target, default keep the number of mesh-points the same as the one stored inside PRE file
        if list_picklefilename==None:
            list_picklefilename = [ 'L512_rho5.pkl',  'L512_rho8.pkl',   'L512_rho10.pkl'  ]
        print(list_picklefilename)

        list_y  =[]
        list_p  =[]

        for picklefilename in list_picklefilename:

            if not('.pkl' ==  picklefilename[-4:])  :
                picklefilename +='.pkl'

            picklefullfilename = picklefilename
            if cfd_data_dir is not None:
                picklefullfilename = cfd_data_dir+picklefilename
            open_file = open(picklefullfilename, "rb")

            loaded_list = pickle.load(open_file)
            open_file.close()
            ##############################################
            N_mesh = loaded_list['N']         #x = np.arange(N_mesh)
            if Nx_target is None:
                Nx_target = N_mesh
                nSkip =1                    # no skip
            else:
                assert N_mesh>= Nx_target
                assert np.mod( N_mesh,Nx_target ) == 0
                nSkip = N_mesh//Nx_target

            dyScale = loaded_list['dyScale']

            y3      = loaded_list['y3']/dyScale    #xy_curv = loaded_list['xy_curv']
            y0Ref   = np.average( y3, axis = (1,2) )
            y3      = (y3-y0Ref.reshape(-1,1,1) )/N_mesh*  2*np.pi

            y_simple    = loaded_list['y_simple'][:10*len(y3):10]/dyScale  # xy_curv = loaded_list['xy_curv']
            y0Ref_simple= np.average(y_simple, axis=(1) )
            y_simple    = (y_simple - y0Ref_simple.reshape(-1,1)) / N_mesh * 2 * np.pi
            y_simple    = np.expand_dims(y_simple,axis=-1)

            p = np.array( [loaded_list['rhoR'], loaded_list['L'] ] )  #loaded_list['rhoR']
            list_p.append(p)
            ##
            if 'y3' == varname:
                list_y.append( y3[:,::nSkip,:])
            elif 'y_simple' in varname:
                list_y.append( y_simple[:,::nSkip,:])

        return list_y, list_p


    @staticmethod
    def Reorg_list_y(list_y, list_p, T_out, T_in, nStep=1, nStepSkip=1):
        y_all = []
        p_all= []
        for idx, y in enumerate(list_y):
            y_tmp = libcfdData.Reorg_y(y,T_out,T_in,nStep,nStepSkip)
            y_all.append(  y_tmp )
            p_all.append(  np.repeat( list_p[idx].reshape(1,-1), y_tmp.shape[0], axis=0 ) )

            print('y_tmp.shape: ', y_tmp.shape, ' , ',  np.repeat( list_p[idx].reshape(1,-1), y_tmp.shape[0], axis=0 ).shape )

        y_all  = np.concatenate(  y_all, axis = 0 )
        p_all  = np.concatenate(  p_all, axis = 0)

        print('y_all.shape: ', y_all.shape)
        print('p_all.shape: ', p_all.shape)

        return y_all , p_all





    @staticmethod
    def load_2DPREdata(list_picklefilename =None, cfd_data_dir= None, Nx_target=None, yB_estimate=None,AspectRatio_set=1, ThicknessScale=1 ): #Nx_target, default keep the number of mesh-points the same as the one stored inside PRE file
        if list_picklefilename==None:
            list_picklefilename = [ 'L512_rho5.pkl',  'L512_rho8.pkl',   'L512_rho10.pkl'  ]
        print(list_picklefilename)

        list_y =[]
        list_p  =[]
        for picklefilename in list_picklefilename:

            if not('.pkl' ==  picklefilename[-4:]) :
                picklefilename +='.pkl'

            picklefullfilename = picklefilename
            if cfd_data_dir is not None:
                picklefullfilename = cfd_data_dir+picklefilename
            open_file = open(picklefullfilename, "rb")

            loaded_list = pickle.load(open_file)
            open_file.close()
            p = np.array( [loaded_list['rhoR']/10, loaded_list['L']/2048] , dtype='f')
            list_p.append(p)
            del loaded_list
            ##############################################

            ylevelG2D_all,  _ = libcfdData.get_levelsetG2D(picklefullfilename, Nx_target, yB_estimate,AspectRatio_set=AspectRatio_set)
            #list_y.append( np.expand_dims( ylevelG2D_all,axis=-1)  )

            #----------
            ylevelG2D_all = np.tanh(ylevelG2D_all / ThicknessScale )
            print(picklefullfilename + '  np.tanh is applied on ',  ylevelG2D_all.shape)
            #----------

            list_y.append( ylevelG2D_all)

            ##
            #if 'y3' == varname:
            #    list_y.append( y3[:,::nSkip,:])
            #elif 'y_simple' in varname:
            #    list_y.append( y_simple[:,::nSkip,:])

        return list_y,list_p

    #default
    # yB_estimate = np.array([-1,2.2])*np.pi
    #@staticmethod
    #def get2D_Ny_yB_from_estimate(Nx_target, yB_estimate, AspectRatio_set=1 ):
    #    dx = (2*np.pi)/ Nx_target
    #    dy = dx* AspectRatio_set
    #    Ny_actual = int( (yB_estimate[1] - yB_estimate[0] + 1E-10) / dy )
    #    yB = np.copy(yB_estimate)
    #    yB[1] = yB[0] + (Ny_actual) * dy
    #    return Ny_actual, yB

    @staticmethod
    def get_levelsetG2D(picklefilename, Nx_target, yB_estimate=None, ipick_single=None, AspectRatio_set = 1):
        # picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L1536_rho8.pkl'

        time_start = time.time()
        open_file = open(picklefilename, "rb")
        loaded_list = pickle.load(open_file)
        open_file.close()

        #
        N_mesh = loaded_list['N']
        xy_curv = loaded_list['xy_curv']
        dyScale = loaded_list['dyScale']
        y_simple = loaded_list['y_simple'][:10*len(xy_curv):10]

        ave_y0 = np.average(y_simple, axis=-1)
        #corr_ave_y0 = ave_y0 - ave_y0[0]

        assert Nx_target <= N_mesh and np.mod(N_mesh, Nx_target) ==0
        nSkip = N_mesh//Nx_target


        dx_targeted = (2 * np.pi)/Nx_target
        dy_targeted = dx_targeted * AspectRatio_set

        coeffy_for_scale_and_skip = (1./dyScale) /nSkip / AspectRatio_set


        if yB_estimate is None:
            # --------------------------
            min_y_integer = 999999
            max_y_integer = -999999
            for xy_each, avey0_each in zip(xy_curv, ave_y0):
                # min_y_integer = np.min((min_y_integer, np.min( xy_each[:, 0] )  ))
                # max_y_integer = np.max((max_y_integer, np.max(xy_each[:, 0]) )   )
                min_y_integer = np.min((min_y_integer, np.min(  (xy_each[:, 0]-avey0_each)*coeffy_for_scale_and_skip  )  ))
                max_y_integer = np.max((max_y_integer, np.max(  (xy_each[:, 0]-avey0_each)*coeffy_for_scale_and_skip  )  ))

            min_y_integer = (min_y_integer - 4).astype(int)
            max_y_integer = (max_y_integer + 4).astype(int)
            #---------------------------

            Ny_actual =  max_y_integer - min_y_integer
            yB = np.zeros(2)
            yB[0] = min_y_integer *dy_targeted
            yB[1] = max_y_integer *dy_targeted
        else:
            min_y_integer=0

            Ny_actual, yB = libSiva.get2D_Ny_yB_from_estimate(Nx_target, yB_estimate,AspectRatio_set)


        #--------------------------

        # for xy_each , y0 in zip( xy_curv, ave_y0):
        #     xy_each[:,0] =   (xy_each[:,0]-y0 )*coeffy_for_scale_and_skip
        #     if yB_estimate is not None:
        #         xy_each[:,0] -= ( yB_estimate[0]/dy_targeted   )

        #--------------------------

        print('Ny_actual=', Ny_actual, '; yB/(pi)=', yB/(np.pi) , ', AspectRatio_set=', AspectRatio_set)

        y = np.arange(min_y_integer, min_y_integer + Ny_actual)  # now the aspect ratio becomes the one we set

        #Ny_after_skip = max_y_integer - min_y_integer
        #Nx_after_skip = Nx_target
        #y = np.arange( min_y_integer,max_y_integer   )  # now the aspect ratio becomes 1

        if ipick_single is not None:
            print( 'here we are inspecting xy_curv[' + str(ipick_single) + '] ...' )
            xy_curv = [ xy_curv[ipick_single] ]
            ave_y0  = [ ave_y0[ipick_single]  ]

        ylevelG2D_all = np.zeros( ( len(xy_curv), Nx_target, Ny_actual) )

        for  (y_x, y0, ylevelG ) in  zip(xy_curv,  ave_y0,  ylevelG2D_all ) :

            # For debug
            # y_x=xy_curv[0]; ylevelG=ylevelG2D_all[0]; y0= ave_y0[0]

            # -----
            y_processed       = (y_x[:,0]-y0 )*coeffy_for_scale_and_skip
            if yB_estimate is not None:
                y_processed  -= ( yB_estimate[0]/dy_targeted   )
            #----

            for i_skip in range(Nx_target):

                # i_skip = 228  #For debug
                #i_skip = 221

                i = i_skip * nSkip # i is the index on the underlying fine mesh

                idx = np.where( y_x[:, 1] == i)[0]  # note: this is the x-coordinate, and it is integer !!!
                #------------------
                y_tmp           = y_x[idx, 0]
                y_tmp_processed = y_processed[idx]
                #------------------

                #y_tmp_unique_sorted_beforeclean = np.sort( np.unique(y_tmp) )
                #y_tmp_unique_sorted = y_tmp_unique_sorted_beforeclean[
                #  np.insert(1 + np.where( np.diff(y_tmp_unique_sorted_beforeclean) > 0.51 * coeffy_for_scale_and_skip  )[0], 0, 0)]

                #y_tmp_unique_sorted = y_tmp_unique_sorted_beforeclean
                #idx_all = np.where( np.mod(y_tmp_unique_sorted_beforeclean,1)==0 )[0]
                #y_tmp_unique_sorted = np.insert( y_tmp_unique_sorted, idx_all, y_tmp_unique_sorted[idx_all]  )

                # -----
                # j_seg = (y_tmp_unique_sorted).astype(int) - min_y_integer + 1
                # ylevelG[i_skip, :j_seg[0]] = y_tmp_unique_sorted[0] - y[:j_seg[0]]
                #
                # num_negative_power = 0
                # for ii in range(len(y_tmp_unique_sorted) - 1):
                #     num_negative_power += 1
                #     if (j_seg[ii + 1] > j_seg[ii]):  # do not treat equal
                #         sign = ((-1) ** num_negative_power)
                #         ylevelG[i_skip, j_seg[ii]:j_seg[ii + 1]] = sign * np.minimum( -(y_tmp_unique_sorted[ii] - y[j_seg[ii]:j_seg[ii + 1]]),  (y_tmp_unique_sorted[ii + 1] - y[j_seg[ii]:j_seg[ii + 1]]))
                # ylevelG[i_skip, j_seg[-1]:] = y_tmp_unique_sorted[-1] - y[j_seg[-1]:]
                # ----


                # y_tmp_unique_sorted = np.sort(  np.unique(y_tmp) )

                idxs_sorted = np.argsort( y_tmp )
                y_tmp_sorted           =  y_tmp [ idxs_sorted ]
                y_tmp_processed_sorted =  y_tmp_processed [ idxs_sorted ]

                idx_integer = np.where( np.mod(y_tmp_sorted,1)==0 )[0]
                idx_closeToInteger = np.where( np.diff(y_tmp_sorted)<0.05 )[0]
                for idx in idx_integer:
                    if idx-1 in idx_closeToInteger :
                        y_tmp_sorted[idx-1] = y_tmp_sorted[idx]
                    elif idx in idx_closeToInteger :
                        y_tmp_sorted[idx+1] = y_tmp_sorted[idx]
                #-----------
                y_tmp_unique_sorted, idxs_unique, count_repeats = np.unique( y_tmp_sorted, return_index = True,return_counts=True)
                y_tmp_processed_sorted =  y_tmp_processed_sorted [ idxs_unique ]
                #----------
                idx_integer = np.where( np.mod(y_tmp_unique_sorted,1)==0 )[0]

                #------------------------------------------
                sign_flip=np.ones_like( y_tmp_unique_sorted )
                if len( y_tmp_unique_sorted) % 2 == 0:  # even number, should be odd number
                    if len( idx_integer) >=1 :
                        ydiff = np.diff( y_tmp_unique_sorted )
                        ks = np.argmin( ydiff )
                        if ydiff[ks] < 0.5:
                           sign_flip[ks] = 0
                           #print ( 'CloseApproxmity:', idx_integer, y_tmp_unique_sorted,  count_repeats ) #idxs_unique

                        else: # Tangent scenarior
                           idx = np.argmin ( count_repeats[ idx_integer ] )
                           sign_flip[  idx_integer[ idx ] ] = 0
                           #print ( 'TangentScenarior:', idx_integer, y_tmp_unique_sorted,  count_repeats ) #idxs_unique

                        #assert False , 'len( idx_integer)'
                    else:
                        print ( idx_integer, y_tmp_unique_sorted,  count_repeats ) #idxs_unique
                        assert False , 'len( idx_integer)'


                #------------------------------------------

                j_seg = (y_tmp_processed_sorted).astype(int) - min_y_integer + 1

                ylevelG[i_skip, :j_seg[0]] = y_tmp_processed_sorted[0] - y[:j_seg[0]]

                num_negative_power = 0
                for ii in range(len(y_tmp_unique_sorted) - 1):
                    num_negative_power +=  sign_flip[ii] # 1
                    if (j_seg[ii + 1] > j_seg[ii]):  # do not treat equal
                        sign = ((-1) ** num_negative_power)

                        ylevelG[i_skip, j_seg[ii]:j_seg[ii + 1]] = sign * np.minimum( -(y_tmp_processed_sorted[ii] - y[j_seg[ii]:j_seg[ii + 1]] ),  ( y_tmp_processed_sorted[ii + 1] - y[j_seg[ii]:j_seg[ii + 1]] ) )

                ylevelG[i_skip, j_seg[-1]:] = y_tmp_processed_sorted[-1] - y[j_seg[-1]:]


            print('',end='.')

        time_end = time.time()
        print( '{:.3f}[s]'.format(time_end - time_start) )
        print('Done for ', picklefilename)

        ##############
        return ylevelG2D_all, xy_curv



    #'d:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\'
    @staticmethod
    def get_yBminmax_from_PREdata(list_picklefilename =None, cfd_data_dir= None):
        if list_picklefilename==None:
            #list_picklefilename = [ 'L512_rho5.pkl',    'L512_rho8.pkl',   'L512_rho10.pkl'    ]
            list_picklefilename = ['L1536_rho5.pkl', 'L1536_rho8.pkl', 'L1536_rho10.pkl']
        print(list_picklefilename)

        min_y = 999999
        max_y = -999999

        for picklefilename in list_picklefilename:
            if cfd_data_dir==None:
                open_file = open(picklefilename, "rb")
            else:
                open_file = open(cfd_data_dir+picklefilename, "rb")

            loaded_list = pickle.load(open_file)
            open_file.close()


            #---------------------------------------------
            N_mesh = loaded_list['N']
            dyScale = loaded_list['dyScale']
            xy_curv = loaded_list['xy_curv']
            y_simple = loaded_list['y_simple'][:10*len(xy_curv):10]

            ave_y0 = np.average(y_simple, axis=-1)
            #--------------------------
            for xy_each, avey0_each in zip(xy_curv, ave_y0):
                min_y =  np.min( (min_y, np.min( xy_each[:,0]-avey0_each ) /dyScale*(2*np.pi/N_mesh)  )  )
                max_y =  np.max( (max_y, np.max( xy_each[:,0]-avey0_each ) /dyScale*(2*np.pi/N_mesh)  )  )

        print( '[min_y,max_y]/np.pi=', [min_y/np.pi,max_y/np.pi])
        return [min_y,max_y]

    @staticmethod
    def animate_cfd_file(filename_without_pkl='L1024_rho10', dir_path='d:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\'):

        case_text = filename_without_pkl ;     case_text = case_text.replace('_',' , ') ;     case_text = case_text.replace('L','L/ \delta=') ;   case_text = case_text.replace('rho','\Theta=')

        picklefilename = dir_path + filename_without_pkl + '.pkl'
        open_file = open(picklefilename, "rb")
        loaded_list = pickle.load(open_file)
        open_file.close()

        ##############################################
        N_mesh   = loaded_list['N']
        x        = np.arange(N_mesh)
        ################################################
        dyScale  = loaded_list['dyScale']
        y3       = loaded_list['y3']/dyScale
        xy_curv  = loaded_list['xy_curv']
        y_simple = loaded_list['y_simple'][::10]/dyScale
        y0Ref    = np.average( y_simple,  axis = 1)
        ################################################
        x        =                         x /N_mesh         #  * 2*np.pi
        y3       = (y3      - y0Ref.reshape(-1,1,1))/N_mesh    #  * 2*np.pi
        y_simple = (y_simple- y0Ref.reshape(-1,1))   /N_mesh  #  * 2*np.pi

        min_y = 9999.
        max_y = -9999.
        for idx, xy_each, in enumerate( xy_curv ):
            xy_each[:,0] =( ( xy_each[:,0]/dyScale) - y0Ref[idx] ) /N_mesh  #*2*np.pi
            xy_each[:,1] =    xy_each[:,1]                         /N_mesh  #*2*np.pi

            min_y = np.min( (min_y, np.min(xy_each[:, 0]) ) )
            max_y = np.max( (max_y, np.max(xy_each[:, 0]) ) )
        ##############################################

        Ly = max_y-min_y
        #plt.rcParams['text.usetex'] = False
        fig, ax = plt.subplots(figsize=(  (6*Ly) , 6 ) )
        #plt.subplots_adjust(left=0.1, bottom=0.1, right=.9, top=.9)

        iii = 0
        line5, = ax.plot(  xy_curv[iii][:,0] ,xy_curv[iii][:,1], '.k' ) #,  linewidth=0.01)
        #idx = y3[iii,:,1]>y3[iii,:,0]+0.01/N_mesh*2*np.pi
        #line1, = ax.plot( x[idx], y3[iii,idx,1], 'ob', linewidth=1)
            #line1, = ax.plot( xx_all[iii,:,1], yy_all[iii,:,1], '-ob', linewidth=1)
        #line2, = ax.plot( x, y3[iii,:,2], '-g', linewidth=1)
        #line3, = ax.plot( x, y3[iii,:,0], '-r', linewidth=1)
        #line4, = ax.plot( x, y_simple[iii,:]-1, '-m', linewidth=2)

        #ax.set_ylim(  np.min(y3), np.max(y3) )
        #ax.set_ylim( -1*2*np.pi, 3*2*np.pi )

        #txt_object = plt.text(max_y-0.5,0.8, r'$ n= %d,(%s)$'%(0, case_text) )

        txt_object = plt.text(max_y-0.5,0.8, r'$(%s)$'%(case_text) )

        ax.set_ylim(0,1)
        ax.set_xlim(min_y,max_y)
        ax.axes.set_aspect('equal')

        fig.tight_layout()

        def update_animation(num,xycurv, y3, y_simple,case_text): # , N_mesh,):
            #num = num
            #x = np.arange(N_mesh)
            #x  =                  x  /N_mesh #*2*np.pi
            #idx = y3[num,:,1]>y3[num,:,0]+0.01/N_mesh*2*np.pi
            #line1.set_data( x[idx] ,y3[num,idx,1]  )
            #line2.set_data( x      ,y3[num,:,2]  )
            #line3.set_data( x      ,y3[num,:,0]  )
            #line4.set_data( x  ,y_simple[num,:]  )
            line5.set_data( xycurv[num][:,0] ,xycurv[num][:,1] )
            #txt_object.set_text(r'$ n= %d,(%s)$'%(num, case_text) )
            ax.set_title( " n= %d " %   (num) )
            #ax.set_title( r' $%s, n= %d$ ' %   (case_text, num) )
            #ax.set_ylim(min(data[num,:,1]), max(data[num,:,1]) )


            return line5

        ani = animation.FuncAnimation(fig, update_animation, len(xy_curv), fargs= (xy_curv, y3, y_simple,case_text), interval=1,blit=False)
        return ani
