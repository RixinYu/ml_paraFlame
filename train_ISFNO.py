import torch
import numpy as np
import matplotlib.pyplot as plt



from flame_net.lib_uti import Cdata_sys, count_learnable_params, lib_Model,lib_DataGen,LpLoss,lib_ModelTrain
from timeit import default_timer



def main( params_ex ) :
#def main( model_name, params_ex ) :
    #-----------------------
    #device = torch.device('cuda') # if torch.cuda.is_available() else 'cpu')
    #-----------------------

    if 'KDV' in params_ex:
        #if type( params_ex['KDV'] ) is str:
        #    list_para = [ float( params_ex['KDV'] ) ]
        list_para = [ params_ex['KDV'] ]
        data_sys = Cdata_sys('KDV_RK4',list_para, num_PDEParameters=0 )
 
    elif 'Burgers' in params_ex:
        list_para = [ params_ex['Burgers'] ]
        data_sys = Cdata_sys('Burgers_RK4',list_para, num_PDEParameters=0 )

    else:
        #
        #data_sys = Cdata_sys('MS_RK4', list_para=[0.025, 0.035, 0.05, 0.07, 0.1, 0.15], list_cfdfilename=None, num_PDEParameters=1)
        #data_sys = Cdata_sys('KS_RK4',[6, 9, 12, 18, 24],  list_cfdfilename=None, num_PDEParameters=1 )
        #list_para=[ [10,0],[10,0.25],[10,0.5], [10,0.75], [10,1],
        #            [25,0],[25,0.25],[25,0.5], [25,0.75], [25,1],
        #            [40,0],[40,0.25],[40,0.5], [40,0.75], [40,1]  ]
        #-----------------
        if type( params_ex['rho'] ) is str:
            params_ex['rho'] = float( params_ex['rho'] )
        #-----------------
        list_para = [ [ params_ex['Lpi'], params_ex['rho'] ] ]
        data_sys = Cdata_sys('MKS_RK4',list_para, num_PDEParameters=0 )

    #-----------------------
    nDIM = 1
    if 'nDIM' in params_ex: 
        nDIM = params_ex['nDIM']

    params = lib_Model.set_default_params(data_sys,nDIM)

    params['T_in' ] = 1
    params['T_out'] = 20

    params['data_channel']  = 1
    params['data:nStep']    = 1
    params['data:nStepSkip']= 1


    params['train:learning_rate'] = 0.0025
    
    #--------------------

    for key, value in params_ex.items():
        if key == 'kTimeStepping':         params['kTimeStepping'] = value
        if key == 'batchsize':	           params['train:batch_size'] = value
        if key == 'prefix':	               params['model_name_prefix'] = value
        if key == 'seed':                  torch.manual_seed(value)


    # if 'model' in params_ex: 	model_name = params_ex['model']
    # else:                     model_name = 'tFNO'



    model_name = 'RevtFNO'
    params['fourier:reversible']  = True 
    params['fourier:method_WeightSharing'] = False 
    params['fourier:method_SkipConnection'] = True

    if 'rev' in params_ex:
        if params_ex['rev'] == 0: 
            model_name = 'tFNO'
            params['fourier:reversible']  = False 
 

    #--------------------
    if nDIM == 1:
        params['train:batch_size'] = 512
        params['Nx'] = 256
        params['fourier:modes_fourier' ]   = [128] # [32]
        params['fourier:modes_fourier_rev']= [128]
        params['fourier:width']            = 30 
        params['fourier:width_rev']        = 30      # 1  # 30

    elif nDIM == 2:
        params['train:batch_size'] = 32

        params['Nx'] = 128
        params['fourier:modes_fourier' ]   = [64,64] 
        params['fourier:modes_fourier_rev']= [64,64]
        params['fourier:width']            = 20 
        params['fourier:width_rev']        = 20      # 1  # 30
    #--------------------

    params['kTimeStepping']  = params['T_out'] 
    params['fourier:depth_conv']={ 'tAdv':1,  'rev':[2,2],  'tAdv_basis':'' , 'lift':3, 'proj':1 }
    params['fourier:basis_type'] = ''                                    # fourier_basis_type
    params['train:gradient_clip'] = 10
    params['train:weight_decay']  = 1e-6
    params['train:eps']  = 1e-6
    params['train:nstep_save_best'] = 99999  # do not save _best.pt
    
    params['train:nan_loss_save'] = -1

    for key, value in params_ex.items():
        #if key == 'rev':       params['fourier:reversible'] = value
        if key == 'width':      params['fourier:width'] = value
        if key == 'width_rev':  params['fourier:width_rev'] = value
        if key == 'modes':      
            if   nDIM == 1:     params['fourier:modes_fourier'] = [value]
            elif nDIM == 2:     params['fourier:modes_fourier'] = [value, value]
        if key == 'modes_rev':  
            if   nDIM == 1:     params['fourier:modes_fourier_rev'] = [value]
            elif nDIM == 2:     params['fourier:modes_fourier_rev'] = [value, value]
        
        if key == 'tAdvD':	    params['fourier:depth_conv']['tAdv'] = value
        if key == 'tAdv_rev':   params['fourier:depth_conv']['rev'] = [2,2,2]
        if key == 'tAdv_basis': params['fourier:depth_conv']['tAdv_basis'] = value
        
        if key == 'skipC':    	params['fourier:method_SkipConnection'] = value
        if key == 'weightS':	params['fourier:method_WeightSharing'] = value

        if key == 'basis_type':	 params['fourier:basis_type'] = value
        if key == 'kTimeStepping': params['kTimeStepping']  = value
        #if key == 'scheduler':    params['train:scheduler']  = value
        if key == 'gamma':    params['train:scheduler_gamma']  = float(value)
        if key == 'decay':    params['train:weight_decay']  = float( value )
        if key == 'eps':      params['train:eps']  = float( value )
        #if key == 'scheduler_clip': params['train:scheduler_clip']  = True
        if key == 'gradient_clip':  params['train:gradient_clip']  = value
        if key =='lr':              params['train:learning_rate'] = float(value)
        if key =='adamw':           params['train:optimizer'] = torch.optim.AdamW
        if key == 'data_upsample':  params['data:upsample'] = value
        if key == 'beststep':       params['train:nstep_save_best'] = value
        if key == 'batch_size':     params['train:batch_size'] = value
        if key == 'nan_loss_save':  params['train:nan_loss_save'] = float(value)
        if key == 'resume':         params['train:checkpoint_resume'] = value
        if key == 'gradient_acc':   params['train:NUM_gradient_accumulation_STEPS'] = int(value)

        #if key =='grad_scaler':     params['train:grad_scaler'] = value
        #if key =='upsample':       params['fourier:upsample'] = value
        #if key == 'float64':        torch.set_default_dtype(torch.float64)


    #----------------------
    # params['train:epochs'] = 1000
    # params['train:checkpoint_resume'] = '_best.pt' 
    #----------------------
    
    params['train:epochs_per_save'] = [900]
    params['tensorboard_logdir_prefix'] = ''        # 'paraR_'
    
    
    #-----------------------
    params['fourier:option_RealVersion'] = False
    params['parallel_run'] = False # if parallel_run!=0 else False

    #if params['parallel_run'] == True:    params['fourier:option_RealVersion'] = True



    #--------------------
    dataset_train, dataset_test = lib_DataGen.DataGen(data_sys,params)
    #----------------
    model_name_detail = lib_Model.get_model_name_detail(model_name,data_sys,params)
    model = lib_Model.build_model(model_name_detail,params)
    #---------------



    #%matplotlib inline
    from flame_net.lib_uti import tensorboard_fig1d_monitor
    n_list = [int(len(dataset_test)//3*0.9), int(len(dataset_test)//3*1.3), int(len(dataset_test)//3*2.1)]
    disp_peek = torch.stack( [dataset_test[n][0] for n in n_list ] )
    para_peek = None   # torch.stack( [dataset_test[n][1] for n in n_list ] )
    def tensorboard_callback(ep, device):
        #n1,n2,n3 = 351,1050, -522
        if ep%50==0:     return None # tensorboard_fig1d_monitor( disp_peek, para_peek,   model, device , params['T_in'] )
        else:            return None
    #fig = tensorboard_callback()
    params['tensorboard_fig1d']= tensorboard_callback


   

    # print( 'params=', params )
    print('')
    print('---------------------')
    for key, value in params.items():
        print(key, ":", value)
    print('---------------------')
    print('')

    #
    # torch.set_default_dtype(torch.float32)
    #

    #lib_ModelTrain.Train(train_disp, test_disp,train_PDEpara,test_PDEpara,model,model_name_detail,device,params )
    lib_ModelTrain.Train(dataset_train, dataset_test ,    model,  model_name_detail,  params )



#------------------
if __name__ == "__main__":
    
    #import json
    import argparse
	
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    #parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    #parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    
    #parser.add_argument('model_name', help='model name')
    #parser.add_argument('Lpi', type= int,   help='Lpi')
    #parser.add_argument('rho', type= int,   help='rho')
    #parser.add_argument('kTimeStepping', type=int,  help='kTimeStepping')
    #parser.add_argument('FourierTimeDIM', type=int, help='FourierTimeDIM')
    #parser.add_argument('linearKoopmanAdv', type=int, help='linearKoopmanAdv')
    #parser.add_argument('fourier_basis_type', help='fourier_basis_type')
    #parser.add_argument('parallel_run', default=0, type=int,  help='parallel_run(default: 0)')
	
    #parser.add_argument('-p', '--params', type=json.loads, help='parameter dictionary')
    #parser.add_argument('params', type=json.loads )

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

    #main( args.parallel_run ) # args.batch_size)
    #main( args.model_name, args.Lpi, args.rho, args.kTimeStepping, args.FourierTimeDIM,  args.linearKoopmanAdv, args.fourier_basis_type, 0 )

    main( args.params )

