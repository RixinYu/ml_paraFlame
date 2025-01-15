import torch
import numpy as np
import matplotlib.pyplot as plt

#from flame_net.PFNO_Nd import PFNO_Nd
#from flame_net.FourierOp_Nd import FourierOp_Nd
#from flame_net.DeepONet_1d import DeepONet_1d
#from flame_net.ConvPDE_Nd import ConvPDE_Nd

from flame_net.lib_uti import Cdata_sys, count_learnable_params, lib_Model,lib_DataGen,LpLoss,lib_ModelTrain
from timeit import default_timer



def main( params_ex ) :
#def main( model_name, params_ex ) :
    #-----------------------
    #device = torch.device('cuda') # if torch.cuda.is_available() else 'cpu')
    #-----------------------
    
    nDIM = 2
    
    #data_sys = Cdata_sys('MS_RK4', list_para=[0.025, 0.035, 0.05, 0.07, 0.1, 0.15], list_cfdfilename=None, num_PDEParameters=1)
    #data_sys = Cdata_sys('KS_RK4',[6, 9, 12, 18, 24],  list_cfdfilename=None, num_PDEParameters=1 )
    
   
    #list_para=[ [10,0],[10,0.25],[10,0.5], [10,0.75], [10,1],
    #            [25,0],[25,0.25],[25,0.5], [25,0.75], [25,1],
    #            [40,0],[40,0.25],[40,0.5], [40,0.75], [40,1]  ]
    #--------------------
    
    list_para = [ [ params_ex['Lpi'], params_ex['rho'] ] ]

    data_sys = Cdata_sys('MKS_RK4',list_para, num_PDEParameters=0 )
    
    params = lib_Model.set_default_params(data_sys,nDIM)

    params['T_in' ] = 1
    params['T_out'] = 20

    params['data_channel']  = 1
    params['data:nStep']    = 1
    params['data:nStepSkip']= 1
    if nDIM == 1:   
        params['Nx']  = 256 
        params['train:batch_size'] = 500
    elif nDIM == 2:           
        params['Nx']  = 128  # 128^2
        params['train:batch_size'] = 32
    #-----
    params['train:learning_rate'] = 0.0025
    
    #--------------------
    params['kTimeStepping']  = params['T_out'] 
    params['train:gradient_clip'] = 30

    
    #model_name = 'kFNO' 
    #model_name = 'kConv'
    # params['fourier:method_BatchNorm' ] = 256  # 0, no-batchnorm,  -1, batch norm, >1 , layer norm
    
    if 'model' in params_ex: 	model_name = params_ex['model']
    else:                       model_name = 'tFNO'

    # overding above
    for key, value in params_ex.items():
        if key == 'kTimeStepping':         params['kTimeStepping'] = value
        if key == 'batchsize':	           params['train:batch_size'] = value
        if key == 'prefix':	               params['model_name_prefix'] = value
        if key == 'seed':                  torch.manual_seed(value)
        if key == 'gradientclip':	       params['train:gradient_clip'] = value


    if 'tFNO' == model_name :
        if nDIM == 1:
            params['fourier:modes_fourier' ]  = [128]
            params['fourier:width']           = 30
        else: 
            params['fourier:modes_fourier' ]  = [64,64]   
            params['fourier:width']           = 20       # reduce computational cost 

        #params['fourier:linearKoopmanAdv'] = linearKoopmanAdv # False  
        params['fourier:depth_conv']  = {'tAdv': 2,'lift':3,'proj':1,'rev':2}
        params['fourier:reversible']  = False  # bRev
        params['FourierTimeDIM']      = False  # FourierTimeDIM   #  False
        params['fourier:method_WeightSharing'] = True # False
        params['fourier:method_SkipConnection']= True          # for run 3
        params['fourier:basis_type'] = ''                      # fourier_basis_type
		
        for key, value in params_ex.items():
            if key == 'rev':       params['fourier:reversible'] = value
            if key == 'Ftime':     params['FourierTimeDIM'] = value
			
            if key == 'width':     params['fourier:width'] = value
            if key == 'modes':     params['fourier:modes_fourier'] = value
			
            if key == 'tAdvD':	    params['fourier:depth_conv']['tAdv'] = value
            if key == 'liftD':    	params['fourier:depth_conv']['lift'] = value
            if key == 'projD':	    params['fourier:depth_conv']['proj'] = value
            if key == 'revD':	    params['fourier:depth_conv']['rev'] = value
			
            if key == 'skipC':    	params['fourier:method_SkipConnection'] = value
            if key == 'weightS':	params['fourier:method_WeightSharing'] = value
            if key == 'basis_type':	 params['fourier:basis_type'] = value


    elif 'kConv' == model_name:
        params['conv:method_types_conv'] = 'inception_less'
        params['conv:method_BatchNorm' ] = params['Nx']
        if nDIM == 1:             params['conv:en1_channels' ] = [ [16,32],[32,32],[64,64],[128],[128],[64],[32]]
        elif nDIM == 2:           params['conv:en1_channels' ] = [ [16,32],[32,32],[64,64],[128],[128],[64]]




    #----------------------
    # params['method_outputTanh' ] = False        # True
    # params['train:epochs'] = 1000
    # params['train:checkpoint_resume'] = '_best.pt' # None     # '_best.pt'
    #----------------------
    
    params['tensorboard_logdir_prefix'] = ''           # 'paraR_'
    #params['model_name_prefix'] ='' 
    params['train:epochs_per_save'] = [800,900]

    
    #-----------------------
    params['fourier:option_RealVersion'] = False

    params['parallel_run'] = False # if parallel_run!=0 else False
    if params['parallel_run'] == True:    params['fourier:option_RealVersion'] = True



    #--------------------
    dataset_train, dataset_test = lib_DataGen.DataGen(data_sys,params)
    #----------------
    model_name_detail = lib_Model.get_model_name_detail(model_name,data_sys,params)
    model = lib_Model.build_model(model_name_detail,params)
    #---------------



    #----------------------
    #%matplotlib inline
    from flame_net.lib_uti import tensorboard_fig1d_monitor
    n_list = [int(len(dataset_test)//3*0.9), int(len(dataset_test)//3*1.3), int(len(dataset_test)//3*2.1)]
    disp_peek = torch.stack( [dataset_test[n][0] for n in n_list ] )
    para_peek = None   # torch.stack( [dataset_test[n][1] for n in n_list ] )
    def tensorboard_callback(ep, device):
        #n1,n2,n3 = 351,1050, -522

        #if ep%50==0:     return tensorboard_fig1d_monitor( disp_peek, para_peek,   model, device , params['T_in'] )
        #else:            return None
        return None
    params['tensorboard_fig1d']= tensorboard_callback
    
    #----------------------


    print('')
    print('---------------------')
    for key, value in params.items():
        print(key, ":", value)
    print('---------------------')
    print('')


    #lib_ModelTrain.Train(train_disp, test_disp,train_PDEpara,test_PDEpara,model,model_name_detail,device,params )
    lib_ModelTrain.Train(dataset_train, dataset_test ,   model,model_name_detail,params )



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

