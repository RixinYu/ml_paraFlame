

import numpy as np
import scipy.io
import scipy.fftpack
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

import matplotlib
from matplotlib import animation , rc

from matplotlib import animation  #, widgets
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import pickle


class libData:
    @staticmethod
    def Reorg_xsol(xsol, T_out=20, T_in=1, nStep=1, nStepSkip=1, name_xsol='dsol'):
        assert nStepSkip <= nStep
        assert np.mod(nStep, nStepSkip) == 0

        if 'level' in name_xsol.casefold():
            num_traj, length_sol, Nx, Ny = xsol.shape
        else:  # 'dsol'
            num_traj, length_sol, Nx = xsol.shape

        #num__split_seq_pierce = ((1 + (length_sol - 1) // nStep) - T_in) // T_out
        num__split_seq_pierce = ((     (length_sol - 1) // nStep) - T_in) // T_out -1
        print('Reorg_xsol:num__split_seq_pierce', num__split_seq_pierce)

        if 'level' in name_xsol.casefold():
            sequence_disp = np.zeros(((nStep // nStepSkip) * num__split_seq_pierce * num_traj, T_out + T_in, Nx, Ny))
        else:  # 'dsol'
            sequence_disp = np.zeros(((nStep // nStepSkip) * num__split_seq_pierce * num_traj, T_out + T_in, Nx))

        for i in range(num__split_seq_pierce):
            i_start = i * T_out * nStep
            for k in range(0, nStep, nStepSkip):
                k_k = k // nStepSkip
                k_i_start = i_start + k


                assert k_i_start + (T_out + T_in) * nStep < length_sol, 'Reorg_xsol:k_i_start + (T_out + T_in) * nStep < length_sol'

                #if 'level' in name_xsol.casefold():
                #    sequence_disp[
                #    (k_k + i * nStep // nStepSkip) * num_traj:((k_k + i * nStep // nStepSkip) + 1) * num_traj, :, :,
                #    :] = xsol[:, k_i_start:k_i_start + (T_out + T_in) * nStep:nStep, :, :]
                #else:  # 'dsol'
                #    sequence_disp[
                #    (k_k + i * nStep // nStepSkip) * num_traj:((k_k + i * nStep // nStepSkip) + 1) * num_traj, :,
                #    :] = xsol[:, k_i_start:k_i_start + (T_out + T_in) * nStep:nStep, :]
                sequence_disp[ (k_k + i * nStep // nStepSkip) * num_traj:((k_k + i * nStep // nStepSkip) + 1) * num_traj, ...] = xsol[:, k_i_start:k_i_start + (T_out + T_in) * nStep:nStep, ...]

        return sequence_disp

    @staticmethod
    def Reorg_dsol_previousversion(dsol, T_out=20, T_in=1):

        num_traj, length_dsol, N = dsol.shape

        num__split_seq_pierce = (length_dsol - T_in) // T_out

        sequence_disp = np.zeros((num__split_seq_pierce * num_traj, T_out + T_in, N))

        for i in range(num__split_seq_pierce):
            i_start = i * T_out
            sequence_disp[i * num_traj:(i + 1) * num_traj, :, :] = dsol[:, i_start:i_start + T_out + T_in, :]

        return sequence_disp

    @staticmethod
    def Reorg_list_xsol(list_xsol, list_nu, T_out, T_in, nStep=1, nStepSkip=1, name_xsol='dsol'):
        x_all = []
        nu_all = []
        for idx, xsol in enumerate(list_xsol):
            xsol_tmp = libData.Reorg_xsol(xsol, T_out, T_in, nStep, nStepSkip, name_xsol)
            x_all.append(xsol_tmp)
            nu_all.append(list_nu[idx] * np.ones(xsol_tmp.shape[0]))

        x_all = np.concatenate(x_all, axis=0)
        nu_all = np.concatenate(nu_all, axis=0)

        return x_all, nu_all

    @staticmethod
    def Reorg_curvsol(curvsol, T_out, T_in):
        num_traj, length_sol, N_mesh, curvDIM = curvsol.shape
        num__split_seq_pierce = (length_sol - T_in) // T_out
        sequence_curv = np.zeros((num__split_seq_pierce * num_traj, (T_out + T_in) * curvDIM, N_mesh))

        for i in range(num__split_seq_pierce):
            i_start = i * T_out

            # here we stack the array using the order in curvDIM as  (x -> y -> z ...)
            for j in range(curvDIM):
                sequence_curv[i * num_traj:(i + 1) * num_traj, j::curvDIM, :] = curvsol[:,
                                                                                i_start:i_start + (T_out + T_in), :, j]
        return sequence_curv

    @staticmethod
    def Reorg_list_dsol_into_curvall(xi, list_dsol, T_out, T_in):
        curv_all = []
        for dsol in list_dsol:
            curvsol = libData.dsol_to_curvsol(xi, dsol)
            curv_all.append(libData.Reorg_curvsol(curvsol, T_out, T_in))

        curv_all = np.concatenate(curv_all, axis=0)
        return curv_all

    # high accuracy fourier intepolation based on uniformed spaced x
    # @staticmethod
    # def Interp__ft(points, Num_divide_pierces):
    #    #points = np.array( [xi , yi] ).T
    #    Num_in,_ = points.shape
    #
    #    xi = points[:,0]
    #    yi = points[:,1]
    #    fourier_k_in = scipy.fft.rfft(yi)
    #
    #    n_zeropad = ( (Num_divide_pierces*Num_in)//2-1) - (Num_in//2-1)
    #
    #    fourier_k_out = np.pad( fourier_k_in, (0, n_zeropad ), mode='constant')
    #
    #    yi_intp = scipy.fft.irfft( fourier_k_out*Num_divide_pierces , Num_divide_pierces*Num_in )
    #
    #    xi_intp = np.linspace(xi[0], xi[-1], Num_divide_pierces*(Num_in-1)+1, endpoint=True )
    #    return np.array( [   xi_intp , yi_intp[:Num_divide_pierces*(Num_in-1)+1]     ] ).T

    # high accuracy fourier intepolation of yi(xi) based on an uniformed spaced xi-mesh
    @staticmethod
    def InterpY_ft_UniformFinerCut(yi, Num_divide_pierces, option='old'):
        *_, Num_in = yi.shape
        fourier_k_in = scipy.fft.rfft(yi, axis=-1)

        n_zeropad = ((Num_divide_pierces * Num_in) // 2 - 1) - (Num_in // 2 - 1)

        if yi.ndim == 3:
            fourier_k_out = np.pad(fourier_k_in, ((0, 0), (0, 0), (0, n_zeropad)), mode='constant')
        elif yi.ndim == 2:
            fourier_k_out = np.pad(fourier_k_in, ((0, 0), (0, n_zeropad)), mode='constant')
        elif yi.ndim == 1:
            fourier_k_out = np.pad(fourier_k_in, (0, n_zeropad), mode='constant')
        else:
            assert yi.ndim >= 1 and yi.ndim <= 3

        yi_intp = scipy.fft.irfft(fourier_k_out * Num_divide_pierces, Num_divide_pierces * Num_in, axis=-1)

        if 'period' in option:
            return yi_intp
        else:
            return yi_intp[..., :Num_divide_pierces * (Num_in - 1) + 1]

    @staticmethod
    def InterpX_UniformFinerCut(xi, Num_divide_pierces, option='old'):
        *_, Num_in = xi.shape
        if 'period' in option:
            dx = xi[..., 1] - xi[..., 0]
            assert dx > 0
            xi_intp = np.linspace(xi[..., 0], xi[..., -1] + dx, Num_divide_pierces * Num_in, endpoint=False, axis=-1)
        else:
            xi_intp = np.linspace(xi[..., 0], xi[..., -1], Num_divide_pierces * (Num_in - 1) + 1, endpoint=True,
                                  axis=-1)
        return xi_intp

    @staticmethod
    def Interp__ft(points, Num_divide_pierces):
        # points = np.array( [xi , yi] ).T
        *_, Num_in, _ = points.shape
        xi = points[..., 0]
        yi = points[..., 1]

        # fourier_k_in = scipy.fft.rfft(yi)
        # n_zeropad = ( (Num_divide_pierces*Num_in)//2-1) - (Num_in//2-1)
        # fourier_k_out = np.pad( fourier_k_in, (0, n_zeropad ), mode='constant')
        # yi_intp = scipy.fft.irfft( fourier_k_out*Num_divide_pierces , Num_divide_pierces*Num_in )
        # xi_intp = np.linspace(xi[0], xi[-1], Num_divide_pierces*(Num_in-1)+1, endpoint=True )
        # return np.array( [   xi_intp , yi_intp[:Num_divide_pierces*(Num_in-1)+1]     ] ).T

        xi_intp = libData.InterpX_UniformFinerCut(xi, Num_divide_pierces)
        yi_intp = libData.InterpY_ft_UniformFinerCut(yi, Num_divide_pierces)

        return np.concatenate([np.expand_dims(xi_intp, -1), np.expand_dims(yi_intp, -1)], axis=-1)

    @staticmethod
    def curvsol_to_angle_magnitude(curvsol):
        xy = np.diff(curvsol, axis=-2)
        mag = np.sqrt(xy[..., 0] ** 2 + xy[..., 1] ** 2)
        alpha_angle = np.arcsin(xy[..., 1] / mag)
        con1 = (xy[..., 0] < 0) & (xy[..., 1] > 0)
        alpha_angle[con1] = np.pi - alpha_angle[con1]
        con2 = (xy[..., 0] < 0) & (xy[..., 1] < 0)
        alpha_angle[con2] = -np.pi - alpha_angle[con2]

        return alpha_angle, mag

    @staticmethod
    def dsol_to_curvsol(xi, dsol):
        *nLen, Num_in = dsol.shape  # Num_in is the number of coarse mesh

        # step 1: a high accuracy, "refined" fourier intepolation
        Num_divide_pierces = 10
        xi_intp = libData.InterpX_UniformFinerCut(xi, Num_divide_pierces)
        xi_intp = np.tile(xi_intp, (*nLen, 1))
        yi_intp = libData.InterpY_ft_UniformFinerCut(dsol, Num_divide_pierces)
        points_ft = np.concatenate([np.expand_dims(xi_intp, -1), np.expand_dims(yi_intp, -1)], axis=-1)

        # step2: a re-parametrimization based on distace
        curvsol = libData.Interp_distance(points_ft, Num_in)
        # nLen, Num_in, curvDIM == curvsol.shape
        return curvsol

    @staticmethod
    def dsol_to_whole_length(x, dsol):
        *nLen, Num_in = dsol.shape  # Num_in is the number of coarse mesh

        points = np.concatenate([np.expand_dims(np.tile(x, (*nLen, 1)), -1), np.expand_dims(dsol, -1)], axis=-1)
        whole_length_0 = np.sum(np.sqrt(np.sum(np.diff(points, axis=-2) ** 2, axis=-1)), axis=-1)

        ## step 1: a high accuracy, "refined" fourier intepolation
        # Num_divide_pierces=20
        # xi_intp = libData.InterpX_UniformFinerCut(    x,Num_divide_pierces)
        # xi_intp = np.tile( xi_intp, (*nLen,1) )
        # yi_intp = libData.InterpY_ft_UniformFinerCut( dsol,Num_divide_pierces)
        # points_ft = np.concatenate( [np.expand_dims(xi_intp,-1), np.expand_dims(yi_intp,-1) ] , axis = -1  )
        ## step2: a re-parametrimization based on d
        # whole_length_intp = np.sum( np.sqrt( np.sum( np.diff(points_ft, axis=-2)**2, axis=-1 ))    , axis = -1)

        return whole_length_0 / (np.pi * 2)  # , whole_length_intp/(np.pi*2)

    @staticmethod
    def Interp_distance(points, Num_out):
        ######################
        # points = np.array( [xi , yi] ).T

        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=-2) ** 2, axis=-1)), axis=-1)
        alpha = np.linspace(0, 1, Num_out)

        if points.ndim == 2:
            distance = np.insert(distance, 0, 0, axis=-1) / distance[..., -1]
            # Interpolation for different methods:
            # interpolations_methods = ['slinear', 'quadratic', 'cubic']
            interpolator = interp1d(distance, points, kind='cubic', axis=-2)
            interpolated_points = interpolator(alpha)
        else:  # points.ndim>=3:
            distance = np.insert(distance, 0, 0, axis=-1) / np.expand_dims(distance[..., -1], axis=-1)
            *nLen, Num_in, _ = points.shape
            interpolated_points = np.zeros((*nLen, Num_out, 2))

            # for idx, each_distance in enumerate( distance):
            #    #interpolations_methods = ['slinear', 'quadratic', 'cubic']
            #    interpolator =  interp1d(each_distance, points[idx,...], kind='cubic', axis=-2)
            #    interpolated_points[idx,...] = interpolator(alpha)

            for idx_tuple in np.ndindex(*nLen):
                # interpolations_methods = ['slinear', 'quadratic', 'cubic']
                interpolator = interp1d(distance[idx_tuple], points[idx_tuple], kind='cubic', axis=-2)
                interpolated_points[idx_tuple] = interpolator(alpha)

                ##########################
        return interpolated_points

    @staticmethod
    def dsol_single_to_Levelsetsol(xi, dsol, Nx, Ny, yB=np.array([-0.7, 1.3]) * np.pi, method_levelset='levelset_fast'):

        # -----
        assert xi.shape[0] == dsol.shape[-1]
        assert np.mod(dsol.shape[-1], Nx) == 0
        nSkip_Nx = dsol.shape[-1] // Nx
        # -----

        Lx = np.pi * 2
        dx = Lx / Nx

        if dsol.ndim == 2:
            length_dsol = dsol.shape[0]
            Levelset_tanh_sol = np.zeros((length_dsol, Nx, Ny))
            print('dosl_to_Levelsetsol, length_dsol=', length_dsol)
            for l, each_dsol in enumerate(dsol):
                G_signed = libData.xycurvsingle_to_LevelsetG(xi[::nSkip_Nx], each_dsol[::nSkip_Nx], Nx, Ny, yB,
                                                             method_levelset=method_levelset)
                # Levelset_tanh_sol[l,:,:]= np.tanh( G_signed/dx )
                Levelset_tanh_sol[l, :, :] = G_signed / dx
                print('', end='+')
            print('')
        elif dsol.ndim == 1:
            G_signed = libData.xycurvsingle_to_LevelsetG(xi[::nSkip_Nx], dsol[::nSkip_Nx], Nx, Ny, yB,
                                                         method_levelset=method_levelset)
            # Levelset_tanh_sol = np.tanh( G_signed/dx )
            Levelset_tanh_sol = G_signed / dx

        return Levelset_tanh_sol

    @staticmethod
    def plot_levelset_from_dsol(dsol, nSkip=20, Nx=128, Ny=128, yB=np.array([-0.7, 1.3]) * np.pi,
                                method_levelset='levelset_fast'):

        # Nx=128
        # Ny=128
        # yB= np.array([-1,2])*np.pi

        # -----
        assert np.mod(dsol.shape[-1], Nx) == 0
        nSkip_Nx = dsol.shape[-1] // Nx
        # -----

        Lx = np.pi * 2
        xi = np.linspace(-0.5, 0.5, Nx, endpoint=False) * Lx

        yi = np.linspace(yB[0], yB[1], Ny, endpoint=False)
        dx = Lx / Nx

        G_Levelset_tanh_multi = np.zeros((dsol[::nSkip].shape[0], Nx, Ny))
        # all_points_ft0 = np.zeros( (dsol.shape[0], Nx,Ny ) )

        for l, each_dsol in enumerate(dsol[::nSkip]):
            G_signed = libData.xycurvsingle_to_LevelsetG(xi, each_dsol[::nSkip_Nx], Nx, Ny, yB,
                                                         method_levelset=method_levelset)
            G_Levelset_tanh_multi[l, :, :] = np.tanh(G_signed / dx)
            print('.', end='')

        # G, sign, points_ft,xi,yi = libData.xycurvsingle_to_LevelsetG(xt,yt, Nx,Ny,yB=np.array([-0.7,1.3])*np.pi )

        def update_animation_levelset(num, Gs, dd):
            # for idx, G in enumerate( Gs):
            ax.clear()
            ax.contourf(xx, yy, Gs[num], 10)
            ax.plot(xi, dd[num], '--k')
            ax.set_title(" n = %d" % (num))
            return ct

        xx, yy = np.meshgrid(xi, yi, indexing='ij')

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # line_list =[]
        # for idx, dsol in enumerate( dsol_list):
        #    line, = ax.plot(x, dsol[0,:], '--', linewidth=1)
        #    line_list.append(line)
        ct = plt.contourf(xx, yy, G_Levelset_tanh_multi[0], 10)  # , levels=[0], color='-b')
        line1, = plt.plot(xi, dsol[0, :], '--k')
        ax.set_ylim(-1, 2)
        ax.axes.set_aspect('equal')
        ani = animation.FuncAnimation(fig, update_animation_levelset, frames=G_Levelset_tanh_multi.shape[0],
                                      fargs=(G_Levelset_tanh_multi, dsol[::nSkip]), interval=10, blit=False)

        xx, yy = np.meshgrid(xi, yi, indexing='ij')
        fig = plt.figure()
        # plt.contour(xx,yy, G2*sign, levels=[0],colors="r", linestyles="solid") # ,cmap=plt.cm.bone )
        # plt.contour(xx, yy, G*sign, levels=[0],  colors="b", linestyles="solid")
        # plt.contourf(xx, yy, np.tanh( G*sign/dx), 100)
        plt.contour(xx, yy, np.tanh(G * sign / dx), levels=[0], color='-b')

        # plt.contour(xx,yy, G2*sign, levels=[0]) # ,cmap=plt.cm.bone )
        # plt.contour(xx,yy, G2*sign-G*sign, 100) # ,cmap=plt.cm.bone )
        # plt.scatter(xx,yy)
        # plt.contour(xx,yy, G*sign,  levels=[0],color='r', linewidths=(1,),)
        plt.plot(points_ft[:, 0], points_ft[:, 1], '--k')

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.contour3D(xx, yy, G*sign, 100, cmap='binary')
        ax.plot_surface(xx, yy, np.tanh(G * sign / dx), rstride=1, cstride=1)
        ax.contour(xx, yy, G * sign, levels=[0], colors="k", linestyles="solid")
        plt.plot(points_ft[:, 0], points_ft[:, 1], '--k')

        return

    @staticmethod
    def Reinit_LevelsetG(G_norm_or_Gtanh, btanh=True, Nx=128, Ny=128, yB=np.array([-0.7, 1.3]) * np.pi,
                         method_levelset='ylevelset'):
        # Nx=128
        # Ny=128
        # yB = np.array([-1,2])*np.pi
        # xt=x
        # yt=dsol[700,:]

        # dy = ( yB[1]-yB[0] ) / Ny
        # G2D = np.copy(G)

        if btanh == True:
            G2D = np.tanh(G_norm_or_Gtanh)
            G2D = np.arctanh(G2D - np.sign(G2D) * 1E-9)
        else:
            G2D = G_norm_or_Gtanh
        dy = 1
        jLoc_i = np.concatenate([np.where(G2D[i, 1:] * G2D[i, :-1] < 0) for i in range(Nx)])
        for i, j in enumerate(jLoc_i):
            if len(j) != 1:
                raise ValueError('should be len=1 but len(j)=' + str(len(j)))
            jLoc = j[0]
            ratio_corr = (G2D[i, jLoc] - G2D[i, jLoc + 1]) / dy
            G2D[i, jLoc] /= ratio_corr  # * G2D[i, jLoc]
            G2D[i, jLoc::-1] = G2D[i, jLoc] + np.arange(jLoc + 1) * dy
            G2D[i, jLoc + 1] /= ratio_corr  # * G2D[i, jLoc+1]
            G2D[i, jLoc + 1:] = G2D[i, jLoc + 1] - np.arange(Ny - jLoc - 1) * dy
        return G2D

    #-----------------------------------
    #-----------------------------------
    #-----------------------------------
    #-----------------------------------
    #-----------------------------------
    #-----------------------------------
    @staticmethod
    def correct_tanh_ylevel( tanh_ylevel, Nx, Ny, yB=np.array([-0.7, 1.3])*np.pi ):
        dy_divide_dx =((yB[1]-yB[0])/Ny) /(np.pi*2/Nx)

        tanh_ylevel_corr = np.copy(tanh_ylevel)
        jy = np.arange(Ny)
        for ix in range(Nx):
            G = tanh_ylevel[ix,:]
            idx1 = np.where(G[:-1] * G[1:] <= 0 )[0]


            yt_ix=[]
            for j0 in idx1:
                yt_now = ( -G[j0]  /( G[j0+1]-G[j0] ) * 1 ) + j0
                #yt_now = ( -np.arctanh( G[j0] ) /( np.arctanh( G[j0+1] )-np.arctanh( G[j0] ) ) * 1 ) + j0
                yt_ix.append(yt_now)
            yt_ix = np.array(yt_ix)

            for j in jy:
                tanh_ylevel_corr[ix,j] = np.min( np.abs( j - yt_ix ) )

        tanh_ylevel_corr= np.tanh( tanh_ylevel_corr*np.sign(tanh_ylevel) * dy_divide_dx )
        return tanh_ylevel_corr
    #-----------------------------------
    #-----------------------------------
    #-----------------------------------
    #-----------------------------------
    #-----------------------------------


    @staticmethod
    def xycurvsingle_to_LevelsetG(xt, yt, Nx, Ny, yB=np.array([-0.7, 1.3]) * np.pi, method_levelset='levelset_fast'):
        # Nx=128
        # Ny=128
        # yB = np.array([-1,2])*np.pi
        # xt=x
        # yt=dsol[700,:]

        yi = np.linspace(yB[0], yB[1], Ny, endpoint=False)

        if 'ylevel' in method_levelset:
            # G_signed = np.zeros((Nx,Ny) )
            # for i in range(Nx):
            #    G_signed[i,:] = yt[i]- yi[:]
            # return G_signed

            return yt.reshape(Nx, 1) - yi.reshape(1, Ny)

        elif 'levelset' in method_levelset:

            Lx = np.pi * 2  # x-domain length being peroidic
            xi = np.linspace(-0.5, 0.5, Nx, endpoint=False) * Lx
            dx = xi[1] - xi[0]
            dy = yi[1] - yi[0]

            def x_to_i(x, Nx=Nx):
                return (x / Lx + 0.5) * Nx

            def y_to_j(y, Ny=Ny, yB=yB):
                return (y - yB[0]) / (yB[1] - yB[0]) * Ny

            G_approx = np.zeros((Nx, Ny))  # sign = np.full( (Nx,Ny),False, dtype=bool) ;
            sign = np.ones((Nx, Ny), dtype=int)
            for i in range(Nx):
                for j in range(Ny):
                    sign[i, j] = 1 if yt[i] >= yi[j] else -1
                    G_approx[i, j] = np.min(np.sqrt((xi[i] - xt) ** 2 + (yi[j] - yt) ** 2))  # *sign[i,j]

            # step 1: a high accuracy, "refined" fourier intepolation of yt,xt
            Num_divide_pierces = 10
            xi_intp = libData.InterpX_UniformFinerCut(xi, Num_divide_pierces,
                                                      option='periodic')  # xi_intp = np.tile( xi_intp, (*nLen,1) )
            yi_intp = libData.InterpY_ft_UniformFinerCut(yt, Num_divide_pierces, option='periodic')

            # points_ft = np.concatenate( [np.expand_dims(xi_intp,-1), np.expand_dims(yi_intp,-1) ] , axis = -1  )

            G = G_approx
            # G =np.copy( G_approx)
            dN_BoundBox = 3
            # NUM_intp = yi_intp.shape[-1]
            # dN_intp  = 3
            # def kValid_period( k, dN_intp=dN_intp, NUM_intp=NUM_intp):
            # return np.mod(  k+ np.arange(-dN_intp,dN_intp+1)  + NUM_intp, NUM_intp )

            NUM_ft_intp = yi_intp.shape[-1]
            dN_intp = 30

            def kValid_ft_intp_period(k, dN_intp=dN_intp, NUM_ft_intp=NUM_ft_intp):
                return np.mod(k + np.arange(0, dN_intp + 1) + NUM_ft_intp, NUM_ft_intp)

            xy_Intp = np.zeros((dN_intp + 1, 2))

            for k in range(0, NUM_ft_intp, dN_intp):
                # k = NUM_intp//2
                kValid_segment = kValid_ft_intp_period(k)
                xy_Intp[:, 0] = xi_intp[kValid_segment]
                xy_Intp[:, 1] = yi_intp[kValid_segment]
                if kValid_segment[0] > kValid_segment[-1]:  # to handle peroidicity
                    xy_Intp[:, 0] = np.mod(xi_intp[kValid_segment] + Lx, Lx)

                if 'fast' in method_levelset:  # 'levelset_fast'
                    distance = np.cumsum(np.sqrt(np.sum(np.diff(xy_Intp, axis=-2) ** 2, axis=-1)), axis=-1)
                    distance = np.insert(distance, 0, 0, axis=-1) / distance[..., -1]
                    interpolator_func_singlecurv = interp1d(distance, xy_Intp, kind='linear',
                                                            axis=-2)  # ,fill_value="extrapolate")
                    alpha = np.linspace(0, 1, 100)
                    interpolated_points = interpolator_func_singlecurv(alpha)

                xMin = np.min(xy_Intp[:, 0] + Lx) - dN_BoundBox * dx;
                xMax = np.max(xy_Intp[:, 0] + Lx) + dN_BoundBox * dx
                yMin = np.min(xy_Intp[:, 1]) - dN_BoundBox * dy
                yMax = np.max(xy_Intp[:, 1]) + dN_BoundBox * dy

                iS = int(np.floor(x_to_i(xMin)))
                iE = int(np.ceil(x_to_i(xMax)))
                jS = int(np.floor(y_to_j(yMin)))
                jE = int(np.ceil(y_to_j(yMax)))

                ii = np.mod(np.arange(iS, iE), Nx)
                jj = np.arange(jS, jE)

                for i in ii:
                    for j in jj:
                        if 'fast' in method_levelset:  # 'levelset_fast'
                            cur_minG = np.min(np.sqrt(np.mod((xi[i] + Lx - interpolated_points[:, 0]), Lx) ** 2 + (
                                        yi[j] - interpolated_points[:, 1]) ** 2))
                        else:
                            cur_minG = np.min(
                                np.sqrt(np.mod((xi[i] + Lx - xy_Intp[:, 0]), Lx) ** 2 + (yi[j] - xy_Intp[:, 1]) ** 2))
                        G[i, j] = np.min((G_approx[i, j], cur_minG))

                return G * sign  # , points_ft, xi,yi

    @staticmethod
    def dsol_to_pointsftsol(xi, dsol):
        # xi = x
        *nLen, Num_in = dsol.shape  # Num_in is the number of coarse mesh

        # step 1: a high accuracy, "refined" fourier intepolation
        Num_divide_pierces = 10
        xi_intp = libData.InterpX_UniformFinerCut(xi, Num_divide_pierces, option='periodic')
        xi_intp = np.tile(xi_intp, (*nLen, 1))
        yi_intp = libData.InterpY_ft_UniformFinerCut(dsol, Num_divide_pierces, option='periodic')
        points_ft = np.concatenate([np.expand_dims(xi_intp, -1), np.expand_dims(yi_intp, -1)], axis=-1)

        # return points_ft
        # step2: a re-parametrimization based on distace
        # curvsol = libData.Interp_distance(points_ft,Num_in)
        # nLen, Num_in, curvDIM == curvsol.shape
        # return curvsol

        points_singlecurv = points_ft[-1]
        points_singlecurv = np.concatenate([points_singlecurv, np.array([[np.pi * 2, 0]]) + points_singlecurv[0, :]],
                                           axis=-2)

        distance = np.cumsum(np.sqrt(np.sum(np.diff(points_singlecurv, axis=-2) ** 2, axis=-1)), axis=-1)
        distance = np.insert(distance, 0, 0, axis=-1) / distance[..., -1]

        interpolator_func_singlecurv = interp1d(distance, points_singlecurv, kind='cubic', axis=-2,
                                                fill_value="extrapolate")

        from scipy.optimize import minimize

        def alpha_to_dS(alpha_middle):
            interpolated_middle_points = interpolator_func_singlecurv(alpha_middle)
            xy0 = points_singlecurv[0:1, :]
            interpolated_all_points = np.concatenate(
                [xy0, interpolated_middle_points, np.array([[np.pi * 2, 0]]) + xy0], axis=0)

            dS = np.sqrt(np.sum(np.diff(interpolated_all_points, axis=-2) ** 2, axis=-1))
            return dS

        def myloss(alpha_middle):  # dAlpha) : #alpha_middle ):
            # alpha_middle = np.cumsum(dAlpha)
            # loss0 = 10*(max(1, np.abs(alpha_middle[-1])) -1 )**2

            interpolated_middle_points = interpolator_func_singlecurv(alpha_middle)
            xy0 = points_singlecurv[0:1, :]
            interpolated_all_points = np.concatenate(
                [xy0, interpolated_middle_points, np.array([[np.pi * 2, 0]]) + xy0], axis=0)

            dS = np.sqrt(np.sum(np.diff(interpolated_all_points, axis=-2) ** 2, axis=-1))

            # loss1 = np.max( np.diff(dS,axis=-1)**2 )
            loss1 = np.max((dS[:-1] / dS[1:] - 1) ** 2)

            return loss1  # loss0

        alpha0 = np.linspace(0, 1, 64)
        alpha0 = alpha0[1:-1]
        # dAlpha = np.diff( alpha0 )
        # dAlpha=dAlpha[:-1]
        # bounds= []
        # for dummy in dAlpha:
        #    bounds.append( (0, float("0.08" )) )

        method = 'Nelder-Mead'
        # method ='powell'
        res = minimize(myloss, alpha0, method=method, options={'xtol': 1e-8, 'disp': True, 'maxiter': 10000})

        # res =  minimize(myloss, dAlpha, method='trust-constr', jac=rosen_der, hess=rosen_hess,
        #       constraints=[linear_constraint, nonlinear_constraint],
        #       options={'verbose': 1}, bounds=bounds)

        interpolated_middle_points = interpolator_func_singlecurv(alpha0)
        plt.plot(interpolated_middle_points[:, 0], interpolated_middle_points[:, 1], '--o')

        final_alpha = res.x
        final_interpolated_points = interpolator_func_singlecurv(final_alpha)

        plt.plot(final_interpolated_points[:, 0], final_interpolated_points[:, 1], '-s')

        print(myloss(alpha0), myloss(final_alpha))
        plt.figure(2)
        plt.plot((alpha_to_dS(alpha0)), 'ko-')
        plt.plot((alpha_to_dS(final_alpha)), 'r-')

#%%

