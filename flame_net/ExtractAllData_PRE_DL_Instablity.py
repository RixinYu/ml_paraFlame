#import torch
import numpy as np
#import scipy.io
#import h5py
#import sklearn.metrics
#import torch.nn as nn
#from scipy.ndimage import gaussian_filter

#from numpy.core.numeric import _outer_dispatcher
#import torch
#import torch.nn as nn
#import torch.nn.functional as F

#import operator
#from functools import reduce
#from functools import partial


import os
import sys
import fileinput
import re

from io import StringIO

 
import matplotlib
#from matplotlib import animation, rc
from matplotlib import animation
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d

import pickle








def Read_xycurv_from_iso_ForPorcess( file_iso ):
    fp = open(  file_iso  , "r")
    alllines = fp.readlines()
    fp.close()

    keytextToSearch = 'T= "'
    xy_curv_list = []
    TimeStep_list  = [] 
    nStep_previous_save = -1
    count = 0
    line_num = 0
    for line_header in alllines:
        if keytextToSearch in line_header:
            allnumbers_in_header1 = re.findall(r'\d+', line_header )
            nStep = int( allnumbers_in_header1[0] )

            allnumbers_in_header2 = re.findall(r'\d+', alllines[line_num+1] )
            numTotalPoints = int( allnumbers_in_header2[0] )
            assert np.mod(numTotalPoints,2)==0  # it must be even number (because 3d --> 2d marching-cube to marching square)

            #if nStep == 4885000: 
            if nStep > nStep_previous_save:  
                xyz_tecplot = np.loadtxt( StringIO( " ".join( alllines[line_num+2: line_num+2 + numTotalPoints//2] )) )
                xy_curv_list.append ( xyz_tecplot[:,:2]  ) #dump z cordinate
                TimeStep_list.append( nStep)
                nStep_previous_save = nStep
                count = count  + 1
                if count%100 == 1:
                    #print( count,  line_header )
                    print( nStep , sep=' ', end='.', flush=True  )
            else: 
                print( "Warnining: nStep={}<nStep_previous={}".format(nStep, nStep_previous_save)  )
        line_num = line_num + 1
    return xy_curv_list, TimeStep_list



#########################################################################
def BoundMinMax_Process_Iso( xy_curv_list, N_mesh) :
    y_minmaxbound_all =  np.zeros( ( len(xy_curv_list), N_mesh,2 ) )
    for k_idx, xy in enumerate( xy_curv_list):
        print("", sep=' ', end='.', flush=True)
        y_minmaxbound_all[k_idx,...] = BoundMinMax_Processing_each_xy(xy,N_mesh)
    return y_minmaxbound_all

def BoundMinMax_Processing_each_xy( xy, N_mesh) :
    y_minmaxbound=  np.zeros( (N_mesh, 2) )
    y_x = xy # note: now we call 'y' ( CFD ) as x 
    for i in range(N_mesh):
        idx= np.where( y_x[:,1] == i)[0] 
        y_tmp= y_x[idx,0]
        y_minmaxbound[i,0] = np.min( y_tmp )
        y_minmaxbound[i,1] = np.max( y_tmp )
    return y_minmaxbound
############################################ 


######################################################### 
def nouse_overcomplicated_RayProcess_Iso( xy_curv_list, N_mesh=2048, N_ray=3) :
    y_Ray3 =  np.zeros( ( len(xy_curv_list), N_mesh,N_ray ) )
    for k_idx, xy in enumerate( xy_curv_list):
        print("", sep=' ', end='.', flush=True)
        y_Ray3[k_idx,...] = nouse_overcomplicated_RayProcessing_each_xy(xy,N_mesh,N_ray)
    return y_Ray3

def nouse_overcomplicated_RayProcessing_each_xy( xy, N_mesh=2048, N_ray=3) :
    y_Ray =  np.zeros( (N_mesh, N_ray) )
    y_x = xy # note: now we call 'y' ( CFD ) as x 
    for i in range(N_mesh):
        #x_Ray[i,:] = i
        
        idx= np.where( y_x[:,1] == i)[0] 
        y_tmp= y_x[idx,0]

        ###########
        #y_tmp_unique, idx_idx_unique = np.unique( y_tmp, return_index = True) 
        #idx_unique = idx[ idx_idx_unique ]
        #idx_sorted = idx_unique[ np.argsort( y_tmp_unique ) ]
        #idx_sorted_after_diff = idx_sorted[   np.insert( 1+ np.where( np.diff(y_tmp_unique) > 0.01)[0] , 0, 0 )   ]

        #if idx_sorted_after_diff.size >= N_ray+1:
        #    idx_sorted_after_diff = np.array( [ idx_sorted_after_diff[0], idx_sorted_after_diff[1], idx_sorted_after_diff[-1] ] )
        #if idx_sorted_after_diff.size == 2:
        #    yy[i,:2] = y_x[idx_sorted_after_diff    ,0]
        #    yy[i, 2] = y_x[idx_sorted_after_diff[-1],0]
        #else:
        #    yy[i,:] = y_x[idx_sorted_after_diff,0]
        ###########

        y_tmp_unique_sorted = np.sort( np.unique( y_tmp) )
        if y_tmp_unique_sorted.size >= N_ray+1:
            middle = y_tmp_unique_sorted[1:-1]
            idx = np.argmax( np.abs( (middle- y_tmp_unique_sorted[0])*(middle- y_tmp_unique_sorted[-1]) )  ) 
            y_tmp_unique_sorted = np.array( [ y_tmp_unique_sorted[0], y_tmp_unique_sorted[1+idx], y_tmp_unique_sorted[-1] ] ) 

        assert y_tmp_unique_sorted.size <= N_ray
        N = y_tmp_unique_sorted.size

        y_Ray[i,:N] = y_tmp_unique_sorted 
        y_Ray[i,N:] = y_tmp_unique_sorted[-1]
    return y_Ray
############################################################


def Loady_from_isoSimple( fileName_isosimple,N_mesh):
    fp = open(fileName_isosimple , "r")
    alllines = fp.readlines()
    fp.close()

    keytextToSearch = 'T= "'
    
    # xy_list = []   # no need to read x == np.arrange(2048)
    y_list  = []
    nStep_list  = [] 


    nStep_previous_save = -1
    count = 0
    line_num = 0
    for line_header in alllines:
        if keytextToSearch in line_header:

            allnumbers_in_header = re.findall(r'\d+', line_header )
            nStep = int( allnumbers_in_header[0] )

            if nStep > nStep_previous_save:  
                
                xy_cfd = np.loadtxt( StringIO( " ".join( alllines[line_num+1: line_num+N_mesh+1] )) ) 

                #xy_list.append (  xy_cfd )
                if np.isnan(xy_cfd).any(): 
                    # do nothing
                    print( "Warn: nan at nStep={}".format(nStep) )
                else:
                    y_list.append(  xy_cfd[:,1] )

                    nStep_list.append( nStep)
                    
                    nStep_previous_save = nStep

                    count = count  + 1
                    if count == 1:
                        print( count,  line_header )
            else: 
                print( "Warn: nStep ={} < Prev={}".format(nStep, nStep_previous_save)  )

            #line_header_list.append (line_header )

        line_num = line_num + 1

    #xy = np.array( xy_list )
    y =  np.array( y_list )

    TimeSteps = np.array(nStep_list )
    #return xy, TimeSteps
    return y, TimeSteps

#################################################33

file_iso_simple = 'T_iso__simple.dat'
file_iso = 'T_iso__0_1.dat'
####################################################


#folder_list = 
directory_raw_data = 'u:\\WorkProjects\\03_LaminarFlameInstability'

case_list = [
            #{ 'dir':'L256_rho10_muc' , 'rhoR':10 ,'L':256 , 'L_y':768, 'N':1024,'N_y':3072,'dyScale':1  },    
            #{ 'dir':'L320_rho10_muc' , 'rhoR':10 ,'L':320 , 'L_y':960, 'N':1280,'N_y':3840,'dyScale':1  },    
            #{ 'dir':'L384_rho10_muc' , 'rhoR':10 ,'L':384 , 'L_y':1152, 'N':1536,'N_y':4608,'dyScale':1  },    
            #{ 'dir':'L448_rho10_muc' , 'rhoR':10 ,'L':448 , 'L_y':1344, 'N':1792,'N_y':5376,'dyScale':1  },
            #{'dir': 'L480_rho10_muc', 'rhoR': 10, 'L':480, 'L_y': 1440, 'N': 1920, 'N_y': 5760, 'dyScale': 1},
            #{ 'dir':'L496_rho10_muc' , 'rhoR':10 ,'L':496 , 'L_y':1488, 'N':1984,'N_y':5952,'dyScale':1  },
            #{ 'dir':'L512_rho10_muc' , 'rhoR':10 ,'L':512 , 'L_y':1536, 'N':2048,'N_y':6144,'dyScale':1  },
            #{ 'dir':'L1024_rho10_muc', 'rhoR':10 ,'L':1024, 'L_y':3072, 'N':2048,'N_y':9216,'dyScale':1.5  },
            #{ 'dir':'L1536_rho10_muc', 'rhoR':10 ,'L':1536, 'L_y':4608, 'N':3072,'N_y':13824,'dyScale':1.5  },

             #{ 'dir':'L256_rho8_muc' , 'rhoR':8 ,'L':256 , 'L_y':768, 'N':1024,'N_y':3072,'dyScale':1  },    
             #{ 'dir':'L320_rho8_muc' , 'rhoR':8 ,'L':320 , 'L_y':960, 'N':1280,'N_y':3840,'dyScale':1  },    
             #error{ 'dir':'L384_rho8'     , 'rhoR':8 ,'L':384 , 'L_y':1152, 'N':1536,'N_y':4608,'dyScale':1  },    
             #{ 'dir':'L512_rho8_muc' , 'rhoR':8 ,'L':512 , 'L_y':1536, 'N':2048,'N_y':6144,'dyScale':1  },
             #{ 'dir':'L768_rho8_muc' , 'rhoR':8 ,'L':768 , 'L_y':2304, 'N':3072,'N_y':9216,'dyScale':1  },
             #{ 'dir':'L1536_rho8_muc', 'rhoR':8 ,'L':1536, 'L_y':4608, 'N':3072,'N_y':13824,'dyScale':1.5  },

             #{ 'dir':'L256_rho5_muc' , 'rhoR':5 ,'L':256 , 'L_y':768, 'N':1024,'N_y':3072,'dyScale':1 },    
             #{ 'dir':'L384_rho5_muc' , 'rhoR':5 ,'L':384 , 'L_y':1152, 'N':1536,'N_y':4608,'dyScale':1  },    
             #{ 'dir':'L512_rho5_muc' , 'rhoR':5 ,'L':512 , 'L_y':1536, 'N':2048,'N_y':6144,'dyScale':1  },
             #{ 'dir':'L1024_rho5_muc', 'rhoR':5 ,'L':1024, 'L_y':3072, 'N':2048,'N_y':9216,'dyScale':1.5  },
             #{ 'dir':'L1536_rho5_muc', 'rhoR':5 ,'L':1536, 'L_y':4608, 'N':3072,'N_y':13824,'dyScale':1.5  },
            ]

#case =     { 'dir':'L256_rho5_muc' , 'rhoR':5 ,'L':256 , 'L_y':768, 'N':1024,'N_y':3072,'dyScale':1 }    

for case in case_list:
    folder = directory_raw_data + '\\run_rho' + str( case['rhoR'] ) + '\\' +  case['dir'] + '\\'

    xy_curv, Tstep_curv = Read_xycurv_from_iso_ForPorcess(folder+file_iso)
    y2_MinMax            = BoundMinMax_Process_Iso( xy_curv, case['N'] )
    y_simple, Tstep_simple = Loady_from_isoSimple ( folder+file_iso_simple, case['N'] ) 
    y_simple = y_simple -1  # adjust one cell-shift

    assert Tstep_simple[-1] >= Tstep_curv[-1]
    y3 = np.concatenate(  (   y2_MinMax[:,:,0:1],  np.expand_dims(y_simple[::10,:], -1),  y2_MinMax[:,:,1:2]  ), axis = -1 ) 

    fullcasedata = {'dir':case['dir'], 'rhoR':case['rhoR'], 'L': case['L'], 'L_y':case['L_y'], 'N':case['N'], 'N_y':case['N_y'], 'dyScale':case['dyScale'],
               'name': 'L'+str(case['L']) +'_rho'+ str(case['rhoR']),  
               'y_simple':y_simple, 'Tstep_simple': Tstep_simple, 
               'y3': y3, 'xy_curv':xy_curv, 'Tstep_curv': Tstep_curv}

    picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\'+fullcasedata['name'] + '.pkl'
    open_file = open(picklefilename, "wb")
    pickle.dump(fullcasedata, open_file)
    open_file.close()


#xy_curv_all, TimeStep_list = Read_iso_ForPorcess(folder +file_iso)
#y_Ray3_all = RayProcess_Iso( xy_curv_all, N_mesh=2048, N_ray=3)



####################################################################################################
#picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L512_rho10.pkl'
#picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L512_rho10.pkl'
picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L1024_rho10.pkl'
#picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L496_rho10.pkl'
#picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L320_rho10.pkl'

#N_mesh=1280
N_mesh=2048
Nx_target=2048
ylevelG2D_all , Ny, yB, xy_curv = libcfdData.get_levelsetG2D(picklefilename, Nx_target , ipick_single=-75 )


fig, ax = plt.subplots(figsize=(12, 8))
#x = np.linspace(-1,1,Nx_target,endpoint=False) *np.pi
#y = np.linspace(yB[0],yB[1],Ny,endpoint=False)
x = np.arange(0,Nx_target)
y = np.arange(0,Ny)

#*(2*np.pi)/Nx_target
xx,yy =np.meshgrid(x,y,indexing='ij')

i_pick_inspect = 0
#i_pick_inspect = -370
ax.contourf(xx, yy, np.tanh( ylevelG2D_all[i_pick_inspect,:,:] ), 3)
#ax.plot( (xy_curv[i_pick_inspect][:,1]/N_mesh-0.5)*2*np.pi, xy_curv[i_pick_inspect][:,0]/Nx_target*np.pi*2  ,'.k')
ax.axes.set_aspect('equal')

################################################################################################3

# picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L512_rho10.pkl'
# picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L1024_rho5.pkl'












#filename_without_pkl =case_text

def animate_file(filename_without_pkl='L1024_rho10', dir_path='d:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\'):

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
    txt_object = plt.text(max_y-0.5,0.8, r'$ n= %d,(%s)$'%(0, case_text) )
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
        #ax.set_title( " n= %d " %   (num) )
        #ax.set_title( r' $%s, n= %d$ ' %   (case_text, num) )
        #ax.set_ylim(min(data[num,:,1]), max(data[num,:,1]) )

        txt_object.set_text(r'$ n= %d,(%s)$'%(num, case_text) )

        return

    ani = animation.FuncAnimation(fig, update_animation, len(xy_curv), fargs= (xy_curv, y3, y_simple,case_text), interval=100,blit=False)
    return ani

    #ani = animation.FuncAnimation(fig, update_animation, 60, fargs= (xx_all,yy_all), interval=100,blit=False)

#             #error{ 'dir':'L384_rho8',
#allcasename = ['L256_rho10','L320_rho10', 'L384_rho10', 'L448_rho10', 'L480_rho10','L496_rho10', 'L512_rho10','L1024_rho10','L1536_rho10',
#      'L256_rho8','L320_rho8','L512_rho8','L768_rho8','L1536_rho8' ,
#      'L256_rho5','L384_rho5','L512_rho5','L1024_rho5','L1536_rho5']

#picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L1536_rho10.pkl'
#picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L512_rho10.pkl'
#picklefilename = 'd:\\Work\\00_MLproj\\Data_PRE_LaminarFlame\\' + 'L1024_rho10.pkl'



case_list = [
            #{ 'dir':'L256_rho10_muc' , 'rhoR':10 ,'L':256 , 'L_y':768, 'N':1024,'N_y':3072,'dyScale':1  },
            #{ 'dir':'L320_rho10_muc' , 'rhoR':10 ,'L':320 , 'L_y':960, 'N':1280,'N_y':3840,'dyScale':1  },
            #{ 'dir':'L384_rho10_muc' , 'rhoR':10 ,'L':384 , 'L_y':1152, 'N':1536,'N_y':4608,'dyScale':1  },
            #{ 'dir':'L448_rho10_muc' , 'rhoR':10 ,'L':448 , 'L_y':1344, 'N':1792,'N_y':5376,'dyScale':1  },
            #{ 'dir':'L480_rho10_muc' , 'rhoR':10 ,'L':480 , 'L_y':1440, 'N':1920,'N_y':5760,'dyScale':1  },
            #{ 'dir':'L496_rho10_muc' , 'rhoR':10 ,'L':496 , 'L_y':1488, 'N':1984,'N_y':5952,'dyScale':1  },
            #{ 'dir':'L512_rho10_muc' , 'rhoR':10 ,'L':512 , 'L_y':1536, 'N':2048,'N_y':6144,'dyScale':1  },
            #{ 'dir':'L1024_rho10_muc', 'rhoR':10 ,'L':1024, 'L_y':3072, 'N':2048,'N_y':9216,'dyScale':1.5  },
            #{ 'dir':'L1536_rho10_muc', 'rhoR':10 ,'L':1536, 'L_y':4608, 'N':3072,'N_y':13824,'dyScale':1.5  },

             #{ 'dir':'L256_rho8_muc' , 'rhoR':8 ,'L':256 , 'L_y':768, 'N':1024,'N_y':3072,'dyScale':1  },
             #{ 'dir':'L320_rho8_muc' , 'rhoR':8 ,'L':320 , 'L_y':960, 'N':1280,'N_y':3840,'dyScale':1  },
             #error{ 'dir':'L384_rho8'     , 'rhoR':8 ,'L':384 , 'L_y':1152, 'N':1536,'N_y':4608,'dyScale':1  },
             #{ 'dir':'L512_rho8_muc' , 'rhoR':8 ,'L':512 , 'L_y':1536, 'N':2048,'N_y':6144,'dyScale':1  },
             #{ 'dir':'L768_rho8_muc' , 'rhoR':8 ,'L':768 , 'L_y':2304, 'N':3072,'N_y':9216,'dyScale':1  },
             #{ 'dir':'L1536_rho8_muc', 'rhoR':8 ,'L':1536, 'L_y':4608, 'N':3072,'N_y':13824,'dyScale':1.5  },

             #{ 'dir':'L256_rho5_muc' , 'rhoR':5 ,'L':256 , 'L_y':768, 'N':1024,'N_y':3072,'dyScale':1 },
             #{ 'dir':'L384_rho5_muc' , 'rhoR':5 ,'L':384 , 'L_y':1152, 'N':1536,'N_y':4608,'dyScale':1  },
             #{ 'dir':'L512_rho5_muc' , 'rhoR':5 ,'L':512 , 'L_y':1536, 'N':2048,'N_y':6144,'dyScale':1  },
             #{ 'dir':'L1024_rho5_muc', 'rhoR':5 ,'L':1024, 'L_y':3072, 'N':2048,'N_y':9216,'dyScale':1.5  },
             #{ 'dir':'L1536_rho5_muc', 'rhoR':5 ,'L':1536, 'L_y':4608, 'N':3072,'N_y':13824,'dyScale':1.5  },
            ]

for case_dic in case_list:
    #case_dic  = case_list[-1]
    case_text = 'L' + str(case_dic['L'] ) + '_rho' + str( case_dic['rhoR'] )
    ani = animate_file(case_text)

    ani.save( case_text +'.gif',  writer='pillow' )
















































%matplotlib

#plt.figure(3)
#plt.plot( xy[4427,:,0] ,xy[4427,:,1], 'o')
#plt.plot( xy[4428,:,0] ,xy[4428,:,1], 'r')
#plt.plot( xy[4429,:,0] ,xy[4429,:,1], 'b--')
#plt.plot( xy[4920,:,0] ,xy[4920,:,1], 'r')

from matplotlib import animation, rc
import matplotlib.pyplot as plt
from IPython.display import HTML

def update_animation(num,datax,datay): 
    line1.set_data( datax[num,:] ,datay[num,:]  )
    ax.set_title("num = %d " % (num) )
    return 

fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(left=0.1, bottom=0.1, right=.9, top=.9)
line1, = ax.plot( loaded_list['x'][0,:], loaded_list['y'][0,:], 'k', linewidth=1)
ax.set_ylim( np.min(loaded_list['y']), np.max(loaded_list['y']) )

ani = animation.FuncAnimation(fig, update_animation, xy.shape[0], fargs= (loaded_list['x'],loaded_list['y']), interval=1,blit=False)












from matplotlib import animation, rc
import matplotlib.pyplot as plt
from IPython.display import HTML

def update_animation(num,datay): 
    num = num
    idx = datay[num,:,1]>datay[num,:,0]+0.01
    N_mesh = 2048
    x = np.arange(N_mesh)
    line1.set_data( x[idx] ,datay[num,idx,1]  )
    line2.set_data( x ,datay[num,:,2]  )
    line3.set_data( x ,datay[num,:,0]  )
    ax.set_title("num = %d " % (num) )
    #ax.set_ylim(min(data[num,:,1]), max(data[num,:,1]) )
    return 


%matplotlib
fig, ax = plt.subplots(figsize=(8,6))
plt.subplots_adjust(left=0.1, bottom=0.1, right=.9, top=.9)
#iii = 173
#iii = 132
iii = 590
y_Ray3_all[iii,:,1]>y_Ray3_all[iii,:,0]
idx = y_Ray3_all[iii,:,1]>y_Ray3_all[iii,:,0]+0.01
N_mesh = 2048
x = np.arange(N_mesh)
line1, = ax.plot( x[idx], y_Ray3_all[iii,idx,1], 'ob', linewidth=1)
#line1, = ax.plot( xx_all[iii,:,1], yy_all[iii,:,1], '-ob', linewidth=1)
line2, = ax.plot( x, y_Ray3_all[iii,:,2], '-g', linewidth=1)
line3, = ax.plot( x, y_Ray3_all[iii,:,0], '-r', linewidth=1)
ax.set_ylim( np.min(y_Ray3_all[:,:,:]), np.max(y_Ray3_all[:,:,:]) )


ani = animation.FuncAnimation(fig, update_animation, y_Ray3_all.shape[0], fargs= (y_Ray3_all,), interval=100,blit=False)
#ani = animation.FuncAnimation(fig, update_animation, 60, fargs= (xx_all,yy_all), interval=100,blit=False)

