#!/bin/env bash
#SBATCH -A NAISS2025-22-410 -p alvis
#SBATCH --gpus-per-node=A40:1



# ----- 2D ------------------------------

# #SBATCH -t 150:00:00   # for 1000 epochs of 2d training of RevtFNO with 'exp_nonl' basis
# #SBATCH -a 7,8

# #SBATCH -o out_draft_2d_tFNO_n_batchrun_%a
# #SBATCH -e out_draft_2d_tFNO_n_batchrun_%a
# pArr=("nDIM:2,KDV:10,modes:64,rev:0,tAdv_basis:exp_nonl,gradient_acc:4,beststep:5")  
# pArr+=("nDIM:2,Lpi:15,rho:0,rev:0,modes_rev:32,tAdv_basis:exp_nonl,gradient_acc:4,beststep:5")  
# pArr+=("nDIM:2,Lpi:15,rho:1,rev:0,modes_rev:32,tAdv_basis:exp_nonlraw,gradient_acc:4,beststep:5")  



# ---2D KDV-----
# #SBATCH -o out_draft_2dKDV_m64_batchrun_%a
# #SBATCH -e out_draft_2dKDV_m64_batchrun_%a

# pArr=("nDIM:2,KDV:10,modes:64,tAdv_basis:exp_nonl,gradient_acc:4,beststep:5") 
# pArr+=("nDIM:2,KDV:10,modes:64,tAdv_basis:exp,gradient_acc:4,beststep:5,eps:1e-5,adamw:1,prefix:run3_")  
# pArr+=("nDIM:2,KDV:10,modes:64,tAdvD:2,rev:0,gradient_acc:4,beststep:5")   
# pArr+=("nDIM:2,KDV:10,modes:64,tAdvD:2,rev:0,gradient_acc:4,beststep:5,kTimeStepping:1,decay:0.0001")  
# pArr+=("nDIM:2,KDV:10,modes:64,tAdv_basis:exp_nonl,gradient_acc:4,beststep:5,prefix:run2_") #4
# pArr+=("nDIM:2,KDV:10,modes:64,tAdv_basis:exp,gradient_acc:4,beststep:5,prefix:run2_")  #5
# pArr+=("nDIM:2,KDV:10,modes:64,tAdvD:2,rev:0,gradient_acc:4,beststep:5,prefix:run2_")   #6
# pArr+=("nDIM:2,KDV:10,modes:64,tAdv_basis:exp_nonl,gradient_acc:4,beststep:5,prefix:run3_") #7
# pArr+=("nDIM:2,KDV:10,modes:64,tAdv_basis:exp_nonlraw,gradient_acc:4,beststep:5,prefix:run3_") #8

# ---2D KDV-----
# #SBATCH -a 0
# #SBATCH -o out_draft_2dKDV_m32_batchrun_%a
# #SBATCH -e out_draft_2dKDV_m32_batchrun_%a
# pArr=("nDIM:2,KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_nonl,gradient_acc:4,eps:1e-5,adamw:1,beststep:5,nan_loss_save:0.2")  #
# pArr=("nDIM:2,KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_nonlraw,lr:5e-4,beststep:5,gradient_acc:4,nan_loss_save:0.2")  #
# pArr+=("nDIM:2,KDV:10,modes:32,modes_rev:32,tAdv_basis:exp,gradient_acc:4,beststep:5,eps=5e-6,prefix:run2_")  
# pArr+=("nDIM:2,KDV:10,modes:32,modes_rev:32,tAdvD:2,rev:0,gradient_acc:4,kTimeStepping:1,beststep:5,decay:0.0001") 
# pArr+=("nDIM:2,KDV:10,modes:32,modes_rev:32,tAdvD:2,rev:0,gradient_acc:4,beststep:5")  

# ---- 1D KDV-----
# #SBATCH -t 4:00:00   # for 1000 epochs of 2d training of RevtFNO with 'exp_nonl' basis
# #SBATCH -a 3,4
# #SBATCH -o out_draft_1DKDV_batchrun_%a
# #SBATCH -e out_draft_1DKDV_batchrun_%a
# pArr=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_nonlraw,prefix:run2_") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_k^3") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdvD:2,prefix:run2_") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdvD:2,rev:0,prefix:run2_") 



#-----1D Siva----
# #SBATCH -t 4:00:00   # for 1000 epochs of 2d training of RevtFNO with 'exp_nonl' basis
# #SBATCH -a 6
# #SBATCH -o out_draft_Lpi25_batchrun_%a
# #SBATCH -e out_draft_Lpi25_batchrun_%a

# pArr=("Lpi:25,rho:0,tAdv_basis:exp") 
# pArr+=("Lpi:25,rho:0,tAdv_basis:exp_nonl") 
# pArr+=("Lpi:25,rho:0,tAdvD:2")             
# pArr+=("Lpi:25,rho:0,tAdv_basis:exp_nonl,rev:0") 
# pArr+=("Lpi:25,rho:0,tAdv_basis:exp,rev:0") 

# pArr+=("Lpi:25,rho:1,tAdv_basis:exp") 
# pArr+=("Lpi:25,rho:1,tAdv_basis:exp_nonl,decay:1e-6,prefix:run3_") 
# pArr+=("Lpi:25,rho:1,tAdvD:2")             
# pArr+=("Lpi:25,rho:1,tAdv_basis:exp_nonl,rev:0") 
# pArr+=("Lpi:25,rho:1,tAdv_basis:exp,rev:0") 

#---------------------------
# pArr+=("Lpi:10,rho:1,tAdv_basis:exp_nonlraw")
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonlraw,lr:5e-4,decay:0.0001")
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonl2raw,lr:5e-4,decay:0.0001")


# #SBATCH -t 4:00:00   
# #SBATCH -a 8,9
# #SBATCH -o out_draft_Lpi_rho401_batchrun_%a
# #SBATCH -e out_draft_Lpi_rho401_batchrun_%a
# pArr=("Lpi:40,rho:1,tAdvD:2,decay:0.0001,batch_size:128,prefix:runDraft_")  
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonl,decay:0.0001,batch_size:128,prefix:runDraft_")  
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonlraw,decay:0.0001,batch_size:128,prefix:runDraft_")  
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp,prefix:runDraft_")  

# #SBATCH -o out_draft_tFNO_n_batchrun_%a
# #SBATCH -e out_draft_tFNO_n_batchrun_%a
# pArr=("Lpi:10,rho:1,rev:0,tAdv_basis:exp_nonl,prefix:run3_")  
# pArr+=("Lpi:10,rho:0,rev:0,tAdv_basis:exp_nonl,prefix:run3_")  
# pArr+=("Lpi:10,rho:1,rev:0,tAdv_basis:exp_nonl,prefix:run4_")  

# pArr+=("Lpi:10,rho:0,rev:0,tAdv_basis:exp_nonl,eps:1e-5,prefix:run4_")  
# pArr+=("Lpi:10,rho:0,tAdv_basis:exp_nonl,prefix:run2_")  
# pArr+=("Lpi:10,rho:0,tAdv_basis:exp_nonl,prefix:run3_")  

# pArr+=("KDV:10,modes:32,modes_rev:32,rev:0,tAdv_basis:exp_nonl,prefix:run2_")   
# pArr+=("KDV:10,modes:32,modes_rev:32,rev:0,tAdv_basis:exp_nonl,eps:1e-5,prefix:run3_")   

# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_nonl,prefix:run3_")   
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_nonl,eps:1e-5,prefix:run4_")   



# ----  1D -------

# #SBATCH -t 5:00:00
# #SBATCH -a  0-1

# #SBATCH -o out_more_MKS__batchrun_%a
# #SBATCH -e out_more_MKS__batchrun_%a

# #SBATCH -o out_siva400_batchrun_%a
# #SBATCH -e out_siva400_batchrun_%a
# pArr=("Lpi:40,rho:0,tAdvD:2,rev:0")  
# pArr+=("Lpi:40,rho:0,tAdvD:2,rev:0,kTimeStepping:1")  


# pArr=("Lpi:10,rho:1,tAdvD:2,prefix:runM_")  
# pArr+=("Lpi:10,rho:1,tAdvD:2,modes_rev:32")  
# pArr+=("Lpi:40,rho:1,tAdvD:2,prefix:runM_")  
# pArr+=("Lpi:40,rho:1,tAdvD:2,prefix:runM_,modes_rev:32")  
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonl,prefix:runM_")  
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonl,prefix:runM_,modes_rev:32")  
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp,prefix:runM_")  
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp,prefix:runM_,modes_rev:32")  



# ----- 2D ------------------------------
# #SBATCH -t 150:00:00   # for 1000 epochs of 2d training of RevtFNO with 'exp_nonl' basis
# #SBATCH -a 4,5,6,7           # 12 # 8,10 # 4,5,6,7  

# #SBATCH -o out_2d_MKSKDV__batchrun_%a
# #SBATCH -e out_2d_MKSKDV__batchrun_%a

#--------
# #SBATCH -o out_MKS__batchrun_%a
# #SBATCH -e out_MKS__batchrun_%a

# #SBATCH -o out_Rev_MKS_tAdvD2_batchrun_%a
# #SBATCH -e out_Rev_MKS_tAdvD2_batchrun_%a

# #SBATCH -o out_MKS_Lpi10_mode32_batchrun_%a
# #SBATCH -e out_MKS_Lpi10_run2_batchrun_%a
# #SBATCH -o out_2_KDVMKS_batchrun_%a
# #SBATCH -e out_2_KDVMKS_batchrun_%a
#######################

# pArr=("nDIM:2,Lpi:15,rho:0,modes_rev:32,tAdv_basis:exp_nonl,beststep:5,lr:2e-4")  # crash ,beststep:1,nan_loss_save:0.2
# pArr+=("nDIM:2,Lpi:15,rho:1,modes_rev:32,tAdv_basis:exp_nonl,beststep:5,lr:2e-4,nan_loss_save:0.2")  # crash  ,nan_loss_save:0.2,resume:_best.pt
# pArr+=("nDIM:2,Lpi:15,rho:0,modes_rev:32,tAdv_basis:exp,beststep:5")  # 2: perfect
# pArr+=("nDIM:2,Lpi:15,rho:1,modes_rev:32,tAdv_basis:exp,beststep:5")  # 3: perfect

# pArr+=("nDIM:2,KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_nonl,beststep:5,lr:1e-3,resume:_best.pt,nan_loss_save:0.2")  # 4  #,gradient_acc:1
# pArr+=("nDIM:2,KDV:10,modes:32,modes_rev:32,tAdv_basis:exp,gradient_acc:4,beststep:5")  # 5
# pArr+=("nDIM:2,KDV:10,modes:32,modes_rev:32,tAdvD:2,rev:0,gradient_acc:4,beststep:5")   # 6
# pArr+=("nDIM:2,KDV:10,modes:32,modes_rev:32,tAdvD:2,rev:0,gradient_acc:4,kTimeStepping:1,beststep:5")  # 7

# pArr+=("nDIM:2,KDV:10,modes:64,modes_rev:32,tAdv_basis:exp_nonl,gradient_acc:4,beststep:5")  # 8
# pArr+=("nDIM:2,KDV:10,modes:64,modes_rev:32,tAdv_basis:exp,gradient_acc:16,beststep:5")  # 9
# pArr+=("nDIM:2,KDV:10,modes:64,modes_rev:32,tAdvD:2,rev:0,gradient_acc:4,beststep:5")   # 10
# pArr+=("nDIM:2,KDV:10,modes:64,modes_rev:32,tAdvD:2,rev:0,gradient_acc:4,kTimeStepping:1,beststep:5")  # 11

# pArr+=("nDIM:2,KDV:10,modes:64,modes_rev:32,tAdv_basis:exp,gradient_acc:16,beststep:5,prefix:run2_,lr:1.5e-4")  # 12
# ----- 2D ------------------------------





# pArr=("Lpi:40,rho:0,modes_rev:32,tAdv_basis:exp,beststep:10")  
# pArr+=("Lpi:40,rho:1,modes_rev:32,tAdv_basis:exp_nonl,beststep:10")  
# pArr+=("Lpi:40,rho:1,modes_rev:32,tAdv_basis:exp_nonl,skipC:0,beststep:10")  


#-------------
# pArr=("Lpi:10,rho:0,tAdvD:2")  
# pArr+=("Lpi:10,rho:1,tAdvD:2")  
# pArr+=("Lpi:40,rho:0,tAdvD:2")  
# pArr+=("Lpi:40,rho:1,tAdvD:2")  

#-----
# pArr=("Lpi:10,rho:0,modes_rev:32,tAdv_basis:exp")  
# pArr+=("Lpi:10,rho:0,modes_rev:32,tAdv_basis:exp_nonl") 
# pArr+=("Lpi:10,rho:1,modes_rev:32,tAdv_basis:exp")  
# pArr+=("Lpi:10,rho:1,modes_rev:32,tAdv_basis:exp_nonl")  

# pArr+=("Lpi:10,rho:0,modes_rev:32,tAdv_basis:exp,rev:0") # wrong
# pArr+=("Lpi:10,rho:0,modes_rev:32,tAdv_basis:exp_nonl,rev:0") # wrong
# pArr+=("Lpi:10,rho:1,modes_rev:32,tAdv_basis:exp,rev:0")  # wrong
# pArr+=("Lpi:10,rho:1,modes_rev:32,tAdv_basis:exp_nonl,rev:0")   # wrong

#----------------------
# pArr=("Lpi:10,rho:0,tAdv_basis:exp,seed:99,beststep:10,decay:0.0001,prefix:run3_")  
# pArr+=("Lpi:10,rho:0,tAdv_basis:exp,rev:0,seed:99,beststep:10,decay:0.0001,prefix:run3_") 
# pArr+=("Lpi:10,rho:0,tAdv_basis:exp_nonl,seed:99,beststep:10,decay:0.0001,prefix:run3_") 
# pArr+=("Lpi:10,rho:0,tAdv_basis:exp_nonl,rev:0,seed:99,beststep:10,decay:0.0001,prefix:run3_") 

# pArr+=("Lpi:10,rho:1,tAdv_basis:exp,seed:99,beststep:10,decay:0.0001,prefix:run3_")  
# pArr+=("Lpi:10,rho:1,tAdv_basis:exp,rev:0,seed:99,beststep:10,decay:0.0001,prefix:run3_") 
# pArr+=("Lpi:10,rho:1,tAdv_basis:exp_nonl,seed:99,decay:0.0001,beststep:10,prefix:run3_")  
# pArr+=("Lpi:10,rho:1,tAdv_basis:exp_nonl,rev:0,seed:99,decay:0.0001,beststep:10,prefix:run3_")  
#--------------------------------------

# pArr=("KDV:10,tAdv_basis:exp_nonl,rev:0,adamw:1")  # ok 
# pArr+=("Lpi:10,rho:1,tAdv_basis:exp_nonl,rev:0,lr:2e-4,adamw:1") # ok, but learning rate can be larger
# pArr+=("Lpi:10,rho:1,tAdv_basis:exp_nonl,rev:0,lr:1e-3,adamw:1,prefix:run2_lr1e3_")  # ?...  try
# pArr+=("Lpi:40,rho:0,tAdv_basis:exp,lr:1e-3,adamw:1,prefix:run2_lr1e3_")          # ok
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonlraw,lr:1e-3,adamw:1,prefix:lr1e3_")       # ok 
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonlraw,rev:0,lr:5e-4,adamw:1,prefix:lr5e4_") # ok
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonlhraw,lr:5e-4,adamw:1,prefix:lr5e4_")      # ok
# ###pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonl,adamw:1,grad_scaler:1,prefix:grad_scaler_") # ok
# # --- 7
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonl,skipC:0,data_upsample:2,prefix:upsample2_") # cancled
# pArr+=("Lpi:40,rho:1,tAdvD:2,skipC:0,decay:5e-6") # this run is rerun
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonl,skipC:0,gradient_clip:50") # ok.
# pArr+=("Lpi:40,rho:1,tAdvD:2,skipC:0,prefix:run2_") # get a better result



################################
### #SBATCH -a 0-27
### #SBATCH -a 1,16,22  # run2 for mild-late-oscillation
## #SBATCH -o out_KDVMKS_batchrun_%a
## #SBATCH -e out_KDVMKS_batchrun_%a

### #SBATCH -a 0-11
### #SBATCH -o out_KDVm32_lr1e3_batchrun_%a
### #SBATCH -e out_KDVm32_lr1e3_batchrun_%a

### #SBATCH -a 3
### #SBATCH -o out_KDV_mexp_batchrun_%a
### #SBATCH -e out_KDV_mexp_batchrun_%a
################################

# pArr=("KDV:10,modes:32,modes_rev:32,beststep:5,tAdv_basis:mexp") 
# pArr+=("KDV:10,modes:32,modes_rev:32,beststep:5,tAdv_basis:mexp,tAdv_rev:ThreeLayer") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:mexp_pure_roll,prefix:run2_") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:mexp_k,prefix:run2_") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:mexp_k^3") 

# pArr+=("KDV:10,modes:32,modes_rev:32,beststep:10,tAdv_basis:mexp_k,width_rev:1") 
# pArr+=("KDV:10,modes:32,modes_rev:32,beststep:10,tAdv_basis:mexp_k^3,width_rev:1,seed:99,prefix:run2_") 
# pArr+=("KDV:10,modes:32,modes_rev:32,beststep:10,tAdv_basis:mexp,rev:0") 

# pArr+=("KDV:10,modes:32,modes_rev:32,beststep:10,tAdv_basis:exp_nonl,prefix:run2_") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdvD:2,rev:0") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdvD:2,rev:0,kTimeStepping:1") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_nonl,rev:0")   
# pArr+=("KDV:10,modes:32,modes_rev:32,beststep:10") 
# pArr+=("KDV:10,modes:32,modes_rev:32,beststep:10,tAdvD:2") 


# *************** lr=1e-4
# pArr=("KDV:10,modes:32,modes_rev:32,decay:0.0001,beststep:5,tAdv_basis:exp") 
# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,beststep:5,tAdv_basis:exp_nonl") # * mild late ossillation 
# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,beststep:5,tAdv_basis:exp,rev:0") 
# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,tAdv_basis:exp_nonl,rev:0")   
# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,tAdvD:2,rev:0") 
# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,tAdvD:2,rev:0,kTimeStepping:1") 

# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,tAdv_basis:exp_pure_roll") 
# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,tAdv_basis:exp_k") 
# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,tAdv_basis:exp_k^3") 
# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,tAdv_basis:exp_k,width_rev:1") 
# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,tAdv_basis:exp_k^3,width_rev:1") 
# pArr+=("KDV:10,modes:32,modes_rev:32,decay:0.0001,beststep:5,tAdv_basis:exp,tAdv_rev:ThreeLayer") 


#-----------------------------
# *************** lr=1e-6
# pArr=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_nonl,prefix:run2_,adamw:1") # * mild late ossillation 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp,rev:0") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_nonl,rev:0")   # ! diverged !
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdvD:2,rev:0") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdvD:2,rev:0,kTimeStepping:1") 

# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_pure_roll") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_k") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_k^3") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_k,width_rev:1") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp_k^3,width_rev:1") 
# pArr+=("KDV:10,modes:32,modes_rev:32,tAdv_basis:exp,tAdv_rev:ThreeLayer") 

#------------------------------


# pArr+=("Lpi:10,rho:0,tAdv_basis:exp") 
# pArr+=("Lpi:10,rho:0,tAdv_basis:exp,rev:0") 
# pArr+=("Lpi:10,rho:0,tAdv_basis:exp_nonl") 
# pArr+=("Lpi:10,rho:0,tAdv_basis:exp_nonl,rev:0") 

# pArr+=("Lpi:10,rho:1,tAdv_basis:exp,prefix:run2_,adamw:1")  #  mild late ossillation
# pArr+=("Lpi:10,rho:1,tAdv_basis:exp,rev:0") 
# pArr+=("Lpi:10,rho:1,tAdv_basis:exp_nonl")  
# pArr+=("Lpi:10,rho:1,tAdv_basis:exp_nonl,rev:0")     # ! diverged !

# pArr+=("Lpi:40,rho:0,tAdv_basis:exp,adamw:1")        # too large ossillation , ok later
# pArr+=("Lpi:40,rho:0,tAdv_basis:exp,rev:0")   
# pArr+=("Lpi:40,rho:0,tAdv_basis:exp_nonl,prefix:run2_,adamw:1")  # mild late ossillation 
# pArr+=("Lpi:40,rho:0,tAdv_basis:exp_nonl,rev:0") 

# pArr+=("Lpi:40,rho:1,tAdv_basis:exp") 
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp,rev:0") 
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonlraw")       # ! diverged !
# pArr+=("Lpi:40,rho:1,tAdv_basis:exp_nonlraw,rev:0") # ! diverged !



# ---- 1D Burgers (revision) -----
#SBATCH -t 3:00:00   # for 1000 epochs of 2d training of RevtFNO with 'exp_nonl' basis
#SBATCH -a 8                        # 0,2,9,10,15,16,17,21,22,23,24,25,26 
#SBATCH -o out_draft_1DBurgers_batchrun_%a
#SBATCH -e out_draft_1DBurgers_batchrun_%a
pArr=("Burgers:1,tAdv_basis:exp_nonl")                  # 0 *
pArr+=("Burgers:1,tAdv_basis:exp_nonl,rev:0") 
pArr+=("Burgers:1,tAdv_basis:exp")                      # 2 *
pArr+=("Burgers:1,tAdv_basis:exp,rev:0") 
pArr+=("Burgers:1,tAdvD:2") 
pArr+=("Burgers:1,tAdvD:2,rev:0") 
pArr+=("Burgers:1,tAdvD:2,rev:0,kTimeStepping:1")       

pArr+=("Burgers:1,tAdv_basis:exp,prefix:run2_")         #7

pArr+=("Burgers:1,tAdv_basis:exp_nonl,prefix:run2_")    # 8 <
pArr+=("Burgers:1,tAdv_basis:exp_nonl,rev:0,prefix:run2_")  # 9 *
pArr+=("Burgers:1,tAdv_basis:exp,rev:0,prefix:run2_")       # 10 *
pArr+=("Burgers:1,tAdvD:2,prefix:run2_") 
pArr+=("Burgers:1,tAdvD:2,rev:0,prefix:run2_") 
pArr+=("Burgers:1,tAdvD:2,rev:0,kTimeStepping:1,prefix:run2_") #13

pArr+=("Burgers:1,tAdv_basis:exp_nonl,prefix:run3_") 
pArr+=("Burgers:1,tAdv_basis:exp_nonl,rev:0,prefix:run3_")  # 15 *
pArr+=("Burgers:1,tAdv_basis:exp,prefix:run3_")             # 16 * 
pArr+=("Burgers:1,tAdv_basis:exp,rev:0,prefix:run3_")       # 17 *
pArr+=("Burgers:1,tAdvD:2,prefix:run3_") 
pArr+=("Burgers:1,tAdvD:2,rev:0,prefix:run3_") 
pArr+=("Burgers:1,tAdvD:2,rev:0,kTimeStepping:1,prefix:run3_") 



pArr+=("Burgers:1,tAdv_basis:exp_nonl,prefix:run1_")           # 21 *
pArr+=("Burgers:1,tAdv_basis:exp_nonl,rev:0,prefix:run1_")  
pArr+=("Burgers:1,tAdv_basis:exp,prefix:run1_") 
pArr+=("Burgers:1,tAdv_basis:exp,rev:0,prefix:run1_") 
pArr+=("Burgers:1,tAdvD:2,prefix:run1_") 
pArr+=("Burgers:1,tAdvD:2,rev:0,prefix:run1_") 
#pArr+=("Burgers:1,tAdvD:2,rev:0,kTimeStepping:1,prefix:run1_")  # no need 


module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 tensorboard/2.15.1-gfbf-2023a  h5py/3.9.0-foss-2023a matplotlib/3.7.2-gfbf-2023a Tkinter/3.11.3-GCCcore-12.3.0 

#ipython -c "%run train1D_MKS_fourier_Koop.ipynb"
#
#torchrun --standalone --nproc_per_node=4 train1d_Koop.py 1 
#

python train_ISFNO.py  ${pArr[$SLURM_ARRAY_TASK_ID]} 



#python train1d_Koop.py ${models[$SLURM_ARRAY_TASK_ID]}  ${LpiLpi[$SLURM_ARRAY_TASK_ID]} ${rhorho[$SLURM_ARRAY_TASK_ID]} ${kTStep[$SLURM_ARRAY_TASK_ID]} ${FTAdva[$SLURM_ARRAY_TASK_ID]} ${Linear[$SLURM_ARRAY_TASK_ID]} ${basist[$SLURM_ARRAY_TASK_ID]}

