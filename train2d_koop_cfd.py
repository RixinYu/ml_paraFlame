import torch
import numpy as np
import matplotlib.pyplot as plt
#from flame_net.FourierOp_Nd import FourierOp_Nd
#from flame_net.DeepONet_1d import DeepONet_1d
#from flame_net.ConvPDE_Nd import ConvPDE_Nd
from flame_net.lib_uti import Cdata_sys, count_learnable_params, lib_Model,lib_DataGen,LpLoss,lib_ModelTrain
from timeit import default_timer


# parallel_run:int

def main( params_ex ) :
    nDIM = 2

    # data_sys = Cdata_sys('cfd', list_para=[320/2048, 512/2048, 768/2048, 1536/2048],
    #                             list_cfdfilename=['L320_rho8', 'L512_rho8', 'L768_rho8', 'L1536_rho8'],
    #                             num_PDEParameters=1)


    
    if 'cfd_data' not in params_ex: params_ex['cfd_data'] = 'L1536_rho8'
    
    tmp_number_str = params_ex['cfd_data'][1:].split('_rho')  #'L1536_rho768'[1:].split('_rho') , to get two number 
    data_sys = Cdata_sys('cfd', list_para        = [ int(tmp_number_str[0]) / 2048 ] , 
                                list_cfdfilename = [  params_ex['cfd_data'] ] , num_PDEParameters=0)

    #data_sys = Cdata_sys('cfd', list_para=[1536/2048], list_cfdfilename=['L1536_rho8'], num_PDEParameters=0)



    params = lib_Model.set_default_params(data_sys,nDIM)
    #--------------------------------
    params['data:nStep']  = 1
    #params['data:nStepSkip']=[2,2,2,1]
    params['data:nStepSkip']=[1]
    params['T_in' ] = 1
    params['data_channel'] = 1
    params['data:yB_estimate']=np.array([-0.896, 2.3])* np.pi   # -0.895 

    
    params['Nx']  = params_ex['Nx'] if 'Nx' in params_ex else 512
    params['Ny']  = params['Nx']

    
    model_name    = params_ex['model'] if 'model' in params_ex else 'tfno'  # 'tfno'  'tcfno'  'kfno'  'kconv'

    #model_name = 'kconv'
    #model_name = 'kfno'
    #if 'model' in params_ex: 
    #    model_name = params_ex['model'] 
    # if model_name ==  'tfno' :     params['Use_2d_DCT']  = True
    # elif model_name =='tcfno':     params['Use_2d_DCT']  = False
    
    params['Use_2d_DCT']  = True if  model_name ==  'tfno' else False

    params['T_out'] = 10
    params['kTimeStepping']         = params['T_out']
    params['train:gradient_clip']   = 50
    params['data:AspectRatio_set']  = (2.3+0.896)/2   # 1    




    if model_name ==  'tcfno':   
        params['train:batch_size'] = 10


    for key, value in params_ex.items():
        #if key == 'Nx':                    params['Nx'] = value
        if key == 'kTimeStepping':         params['kTimeStepping'] = value
        if key == 'dct':	               params['Use_2d_DCT'] = bool(value)
        if key == 'asp':	               params['data:AspectRatio_set'] = value 

        # ----
        if key == 'gamma':                  params['train:scheduler_gamma']  = float(value)
        if key == 'decay':                  params['train:weight_decay']  = float( value )
        if key =='lr':                      params['train:learning_rate'] = float(value)
        # ----

        if key == 'gradientclip':	       params['train:gradient_clip'] = value

        if key == 'batchsize':	           params['train:batch_size'] = value
        if key == 'prefix':	               params['model_name_prefix'] = value
        if key == 'seed':                  torch.manual_seed(value)

        if key == 'Ny':                    params['Ny'] =  value
        #---------------


    if model_name ==  'tfno' :   params['fourier:modes_fourier' ] = [ params['Nx']//4 , params['Nx']//4 ]  #[128,128]
    elif model_name =='tcfno':   params['fourier:modes_fourier' ] = [ params['Nx']//4 , params['Ny']    ]  #[128,512]


    #if model_name == 'tfno':
    #-----------------
    params['fourier:width' ] = 20
    params['fourier:depth_conv']  = {'tAdv': 2,'lift':3,'proj':1,'rev':2,'tAdv_last_nonlinear':True}
    params['fourier:reversible']  = False  # bRev
    params['FourierTimeDIM']      = False  # FourierTimeDIM   #  False
    params['fourier:method_WeightSharing'] = True  # False
    params['fourier:method_SkipConnection']= False # True
    params['fourier:basis_type'] = ''              # fourier_basis_type
    
    for key, value in params_ex.items():
        if key == 'width':     params['fourier:width'] = value

        if key == 'modes':     
            if model_name ==  'tfno' :      params['fourier:modes_fourier'] = [value,value]
            elif model_name ==  'tcfno':    params['fourier:modes_fourier'] = [value, params['Ny']]

        if key == 'rev':       params['fourier:reversible'] = value
        if key == 'Ftime':     params['FourierTimeDIM'] = value
        if key == 'tAdvD':	    params['fourier:depth_conv']['tAdv'] = value
        if key == 'liftD':    	params['fourier:depth_conv']['lift'] = value
        if key == 'projD':	    params['fourier:depth_conv']['proj'] = value
        if key == 'revD':	    params['fourier:depth_conv']['rev'] = value
        
        if key == 'skipC':    	params['fourier:method_SkipConnection'] = value
        if key == 'weightS':	params['fourier:method_WeightSharing'] = value
        if key == 'basis_type':	params['fourier:basis_type'] = value
        if key == 'tanh_loss':  params['tanh_loss'] = value
        #-----------------

    #if 'V100' in torch.cuda.get_device_name('cuda'): # 32 GB
    if model_name ==  'tfno' : 
        if params['Use_2d_DCT'] == True:      params['train:batch_size'] = 5  if params['fourier:modes_fourier'][0] == 128 else 6
        else:                                 params['train:batch_size'] = 7  if params['fourier:modes_fourier'][0] == 128 else 8 



    # elif model_name == 'kfno':
    #     #-----------------
    #     params['T_out'] = 10
    #     params['fourier:modes_fourier' ] =  [128,128] # [256,408] # [128,204] #
    #     params['fourier:width' ] = 20
    #     params['fourier:depth' ] = 3 # 4 (for other )
    #     params['fourier:linearKoopmanAdv'] = False  
    #     params['FourierTimeDIM']           = False
    #     params['fourier:method_WeightSharing'] = True 
    #     params['fourier:option_RealVersion'] = False
    #     params['fourier:method_SkipConnection'] = False
        
    #     # params['tanh_loss'] = False # True
    #     #---------------
    #     params['kTimeStepping']  = params['T_out']
    #     if params['Use_2d_DCT'] == True:        params['train:batch_size'] = 7  # 8 
    #     else:                                   params['train:batch_size'] =  11 # 16

    # elif model_name == 'kconv':
    #     params['T_out'] = 10
    #     params['data:AspectRatio_set'] = (2.3+0.896)/2
    #     params['conv:method_types_conv'] = 'inception_less'
    #     params['conv:en1_channels' ] = [ [16,32],[32,32],[64,64],[128],[128],[64],[32]]
    #     params['conv:PDEPara_depth'] = 6
    #     #params['conv:method_BatchNorm' ] = 256
    #     params['train:batch_size'] = 28




    #----------------------------------------------------
    # params['parallel_run'] = True if parallel_run!=0 else False
    # if params['parallel_run'] == True:  params['fourier:option_RealVersion'] = True

    #----------------------------------------------------
    params['train:epochs_per_save'] = [800,900]
    params['tensorboard_logdir_prefix']=''
    
    # if params['tanh_loss'] == True:     params['model_name_prefix'] = '' 
    # else:                               params['model_name_prefix'] = 'stdloss_' 
    


    # ----------------------------------------------
    dataset_train, dataset_test = lib_DataGen.DataGen(data_sys,params)
    #----------------------------------
    #params['method_outputTanh' ] = False # True
    #params['train:checkpoint_resume'] = None # '_best.pt'
    #-----------------------

    #%matplotlib inline
    from flame_net.lib_uti import tensorboard_fig2d_monitor
    n_list = [ 139 , 470 ]
    disp_peek = torch.stack( [dataset_train[n][0] for n in n_list ] )
    para_peek = torch.stack( [dataset_train[n][1] for n in n_list ] ) # para_peek = None
    def tensorboard_callback(ep, device):
        #if ep%50==0:  return tensorboard_fig2d_monitor( disp_peek, para_peek,   model, device)
        #else:         return None
        return None
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
    #import argparse
    #parser = argparse.ArgumentParser(description='simple distributed training job')
    #parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    #parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    #parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    #parser.add_argument('parallel_run', default=0, type=int,  help='parallel_run(default: 0)')
    #args = parser.parse_args()
    #main( args.parallel_run ) # args.batch_size)

    import argparse
    parser = argparse.ArgumentParser(description='')
    def is_int(num): 
        try: 
            int(num) ; 
            return True;  
        except ValueError:  
            return False;  
    parser.add_argument('params',  type = lambda x: {k:int(v) if is_int(v) else v for k,v in (i.split(':') for i in x.split(','))},
                        help='comma-separated field:position pairs, e.g. Date:0,Amount:2,Payee:5,Memo:9' )
    args = parser.parse_args()
    print('args.params=',args.params)


    main( args.params ) 

