#!/bin/env bash

### #SBATCH -A NAISS2024-22-378 -p alvis
#SBATCH -A NAISS2025-22-410 -p alvis


#------------------------------
###  #SBATCH -t 130:00:00   # for dct with aspec ratio 1.6 ( 404 s,  batch_size=5, m256_256, not-run )
###  #SBATCH -t 100:00:00 # for dct with aspec ratio 1.6 ( 344 s,batch_size=6, m128_256, koopman, run)
### #SBATCH -t 90:00:00  # for dct with aspec ratio 1.6 ( 285 s, batch_size=6, m128_256, standand_k1, run)
### #SBATCH --gpus-per-node=A40:1   ### A100fat:1

### #SBATCH -o out_kfno2d_Nx512_m128_m256_aspR16_noskip_k1_cfd
### #SBATCH -e out_kfno2d_Nx512_m128_m256_aspR16_noskip_k1_cfd
###--BATCH -o out_kfno2d_Nx512_m128_m256_aspR16_noskip_cfd
###--SBATCH -e out_kfno2d_Nx512_m128_m256_aspR16_noskip_cfd
###--SBATCH -o out_stdloss_kfno2dm128_512_aspR_noskip_cfd
###--SBATCH -e out_stdloss_kfno2dm128_512_aspR_noskip_cfd
#----------------------------------




#--------------------------------
#SBATCH -t 50:00:00  

### #SBATCH --gpus-per-node=V100:1 

#SBATCH --gpus-per-node=A40:1 


### #SBATCH -a 0-6  # (AI-fluid Conf 2025)
### #SBATCH -a 7-8      # (AI-fluid Conf 2025)
###  #SBATCH -a 9-10      # (AI-fluid Conf 2025)
### #SBATCH -a 11-12      # (AI-fluid Conf 2025)
## #SBATCH -a 13-14      # (AI-fluid Conf 2025)
#SBATCH -a 19      # (AI-fluid Conf 2025)

#SBATCH -o out_2d_koop_cfd_AIfluidConf2025_%a
#SBATCH -e out_2d_koop_cfd_AIfluidConf2025_%a


module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 tensorboard/2.15.1-gfbf-2023a  h5py/3.9.0-foss-2023a matplotlib/3.7.2-gfbf-2023a Tkinter/3.11.3-GCCcore-12.3.0 


#--round a.0-5
pArr=("Nx:256")   #  40 sec, 11 hour
pArr+=("dct:1")   # 176 sec, 48 hour
pArr+=("dct:0")   # 100 sec, 28 hour

pArr+=("Nx:256,dct:0")   #  less than 40 sec
pArr+=("Nx:256,dct:0,kTimeStepping:1")   #  less than 40 sec

pArr+=("Nx:256,dct:0,tanh_loss:1,prefix:tanhloss_")   #  less than 40 sec


#--round a.6-8
pArr+=("model:tcfno,Nx:256,kTimeStepping:1")   #  less than 40 sec
pArr+=("model:tcfno,Nx:256,tanh_loss:1,prefix:tanhloss_")   #  less than 40 sec
pArr+=("model:tcfno,Nx:256")   #  less than 40 sec

#--round a.9-10
pArr+=("model:tcfno,Nx:256,kTimeStepping:1,lr:0.001,prefix:lr1e3_")   #  less than 40 sec  #default 0.0025
pArr+=("model:tcfno,Nx:256,kTimeStepping:1,seed:15,prefix:seed15_")   #  less than 40 sec
#--round a.11-12
pArr+=("model:tcfno,Nx:256,kTimeStepping:1,lr:0.0001,prefix:lr1e4_")   #  less than 40 sec  #default 0.0025
pArr+=("model:tcfno,Nx:256,kTimeStepping:1,lr:0.00001,prefix:lr1e5_")   #  less than 40 sec  #default 0.0025

#--round a.13-14
pArr+=("cfd_data:L768_rho8,model:tcfno,Nx:256")   #  less than 40 sec  #default 0.0025
pArr+=("cfd_data:L768_rho8,model:tcfno,Nx:256,kTimeStepping:1,lr:0.001,prefix:lr1e3_")   #  less than 40 sec  #default 0.0025

#--round a.15-17
pArr+=("cfd_data:L1536_rho5,model:tcfno,Nx:256,lr:0.002")   #  less than 40 sec  #default 0.0025
pArr+=("cfd_data:L1536_rho5,model:tcfno,Nx:256,kTimeStepping:1,lr:0.001,prefix:lr1e3_")   #  less than 40 sec  #default 0.0025
pArr+=("cfd_data:L1536_rho5,model:tcfno,Nx:256,lr:0.001,prefix:lr1e3_")   

#--round a.18-19
pArr+=("cfd_data:L1536_rho5,model:tcfno,Nx:256,kTimeStepping:1,lr:0.001,seed:15,prefix:seed15lr1e3_")   #  less than 40 sec  #default 0.0025
pArr+=("model:tcfno,Nx:256,kTimeStepping:1,lr:0.001,seed:15,prefix:seed15lr1e3_")   #  less than 40 sec  #default 0.0025

python train2d_koop_cfd.py  ${pArr[$SLURM_ARRAY_TASK_ID]} 



#------------------------------------------------

#ipython -c "%run train1D_MKS_fourier_Koop.ipynb"
#torchrun --standalone --nproc_per_node=4 train1d_Koop.py 1 

#python train1d_Koop.py ${models[$SLURM_ARRAY_TASK_ID]}  ${LpiLpi[$SLURM_ARRAY_TASK_ID]} ${rhorho[$SLURM_ARRAY_TASK_ID]} ${kTStep[$SLURM_ARRAY_TASK_ID]} ${FTAdva[$SLURM_ARRAY_TASK_ID]} ${Linear[$SLURM_ARRAY_TASK_ID]} ${basist[$SLURM_ARRAY_TASK_ID]}

