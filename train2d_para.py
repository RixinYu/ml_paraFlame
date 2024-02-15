import torch
import numpy as np
import matplotlib.pyplot as plt
#from flame_net.FourierOp_Nd import FourierOp_Nd
#from flame_net.DeepONet_1d import DeepONet_1d
#from flame_net.ConvPDE_Nd import ConvPDE_Nd
from flame_net.lib_uti import Cdata_sys, count_learnable_params, lib_Model,lib_DataGen,LpLoss,lib_ModelTrain
from timeit import default_timer



def main( parallel_run:int ) :
    nDIM = 2
    data_sys = Cdata_sys('cfd', list_para=[320/2048, 512/2048, 768/2048, 1536/2048],
                                list_cfdfilename=['L320_rho8', 'L512_rho8', 'L768_rho8', 'L1536_rho8'],
                                num_PDEParameters=1)

    params = lib_Model.set_default_params(data_sys,nDIM)
    #--------------------------------
    params['data:nStep']  = 1
    params['data:nStepSkip']=[2,2,2,1]

    params['T_in' ] = 1
    params['data_channel'] = 1
    params['data:yB_estimate']=np.array([-0.9, 2.3]) * np.pi
    params['Nx']                  =  256     #512

    #-----------
    params['data:ThicknessScale']  =  2
    #-----------
    
    #model_name = 'conv'
    model_name = 'fno'
    
    if model_name == 'fno':
        #-----------------
        params['Use_2d_DCT'] = True 
        params['T_out'] = 10
        params['data:AspectRatio_set'] = (2.3+0.9)/2   # 1
        #-----------------
        params['fourier:modes_fourier' ] = [64,64]
        params['fourier:width' ] = 20
        params['fourier:depth' ] = 4
        params['fourier:method_WeightSharing'] = True
        params['fourier:PDEPara_AcrossDepth'] = True
        #----------------
        # params['fourier:method_SkipConnection'] = False
        params['fourier:method_ParaEmbedding'] =   True  # False #  True # keep this to
        params['fourier:PDEPara_mode_level'] = [6]       # 6 #None # 4

        # params['PDEPara_fc_class'] = 'OM'
        # params['fourier:method_BatchNorm']= 256 #

        params['fourier:option_RealVersion'] = False

        params['train:batch_size'] = 46 # 36
        if params['fourier:PDEPara_mode_level'] is not None:
            params['train:batch_size'] = 40 # 29 # 32     # GPU limit
        
        if params['Use_2d_DCT']:
            params['train:batch_size'] = 12 # 6

    elif model_name == 'conv':
        params['T_out'] = 10
        params['data:AspectRatio_set'] = (2.3+0.9)/2
        params['conv:method_types_conv'] = 'inception_less'
        params['conv:en1_channels' ] = [ [16,32],[32,32],[64,64],[128],[128],[64],[32]]
        params['conv:PDEPara_depth'] = 6
        #params['conv:method_BatchNorm' ] = 256

        params['train:batch_size'] = 28

    #----------------------
    params['parallel_run'] = True if parallel_run!=0 else False
    if params['parallel_run'] == True:  params['fourier:option_RealVersion'] = True

    #------------------------------
    params['train:gradient_clip'] = 50
    params['tensorboard_logdir_prefix']=''
    params['model_name_prefix'] ='' # 'std2_'


    # ----------------------------------------------
    dataset_train, dataset_test = lib_DataGen.DataGen(data_sys,params)
    #----------------------------------
    #params['method_outputTanh' ] = False # True
    #params['train:checkpoint_resume'] = None # '_best.pt'
    #-----------------------

    #%matplotlib inline
    from flame_net.lib_uti import tensorboard_fig2d_monitor
    n_list = [ 139 , 2275 ]
    disp_peek = torch.stack( [dataset_train[n][0] for n in n_list ] )
    para_peek = torch.stack( [dataset_train[n][1] for n in n_list ] ) # para_peek = None
    def tensorboard_callback(ep, device):
        if ep%50==0:  return tensorboard_fig2d_monitor( disp_peek, para_peek,   model, device)
        else:         return None
    params['tensorboard_fig1d']= tensorboard_callback
    # ----------------------


    print('')
    print('---------------------')
    for key, value in params.items():
        print(key, ":", value)
    print('---------------------')
    print('')


    #---------------------------
    model_name_detail = lib_Model.get_model_name_detail(model_name,data_sys,params)
    model = lib_Model.build_model(model_name_detail,params)
    #---------------------------


    lib_ModelTrain.Train( dataset_train, dataset_test , model,model_name_detail, params )



#------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    #parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    #parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    #parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('parallel_run', default=0, type=int,  help='parallel_run(default: 0)')
    args = parser.parse_args()

    main( args.parallel_run ) # args.batch_size)

