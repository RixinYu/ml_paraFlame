#!/bin/env bash
#SBATCH -A NAISS2024-22-378 -p alvis
#SBATCH -t 12:00:00
####SBATCH --gpus-per-node=T4:1
#SBATCH --gpus-per-node=A40:1

####SBATCH -a 0-17
#####SBATCH -a 0-1
####SBATCH -a 0-16
####SBATCH -a 0-3
####SBATCH -a 0-1
###SBATCH -a 0-0
####SBATCH -o out_tFNO_koop_batchrun4_%a
#####SBATCH -e out_tFNO_koop_batchrun4_%a

#####SBATCH -a 0-2
#####SBATCH -a 0-0
####SBATCH -o out_koop_final1_%a
#####SBATCH -e out_koop_final1_%a

#SBATCH -a 0-8
#SBATCH -o out_koop_finalpaper_%a
#SBATCH -e out_koop_finalpaper_%a



module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 tensorboard/2.15.1-gfbf-2023a  h5py/3.9.0-foss-2023a matplotlib/3.7.2-gfbf-2023a Tkinter/3.11.3-GCCcore-12.3.0 

#ipython -c "%run train1D_MKS_fourier_Koop.ipynb"

#
#torchrun --standalone --nproc_per_node=4 train1d_Koop.py 1 
#

#----------------------
# pArr=("Lpi:40,rho:1,skipC:0") 
# pArr+=("Lpi:40,rho:1,rev:1,skipC:0") 
# pArr+=("Lpi:40,rho:1,rev:1,revD:0,skipC:0") 

# pArr+=("Lpi:10,rho:1") 
# pArr+=("Lpi:10,rho:1,tAdvD:1") 
# pArr+=("Lpi:10,rho:1,rev:1") 
# pArr+=("Lpi:10,rho:1,rev:1,revD:0") 
# pArr+=("Lpi:10,rho:1,rev:1,tAdvD:1") 

# pArr+=("Lpi:40,rho:0") 
# pArr+=("Lpi:40,rho:0,tAdvD:1") 
# pArr+=("Lpi:40,rho:0,rev:1") 
# pArr+=("Lpi:40,rho:0,rev:1,revD:0") 
# pArr+=("Lpi:40,rho:0,rev:1,tAdvD:1") 

# pArr+=("Lpi:10,rho:0") 
# pArr+=("Lpi:10,rho:0,tAdvD:1") 
# pArr+=("Lpi:10,rho:0,rev:1") 
# pArr+=("Lpi:10,rho:0,rev:1,revD:0") 
# pArr+=("Lpi:10,rho:0,rev:1,tAdvD:1") 

#---------------------------

# pArr=("Lpi:40,rho:1,skipC:-1") 
# pArr+=("Lpi:40,rho:1,rev:1,skipC:0") 

#---------------------------

# pArr=("Lpi:40,rho:1,skipC:0")                   # conflicted

# pArr+=("Lpi:40,rho:1,skipC:0,liftD:1,projD:3") 
# pArr+=("Lpi:10,rho:1,liftD:1,projD:3") 
# pArr+=("Lpi:40,rho:0,liftD:1,projD:3") 
# pArr+=("Lpi:10,rho:0,liftD:1,projD:3") 


# pArr+=("Lpi:40,rho:1,skipC:0,kTimeStepping:1")   # tFNO_m128w30Lpi_rho40_1_noskip_o20_kT20backup
# pArr+=("Lpi:10,rho:1,kTimeStepping:1")           # tFNO_m128w30Lpi_rho10_1_o20_kT20backup
# pArr+=("Lpi:40,rho:0,kTimeStepping:1")           # tFNO_m128w30Lpi_rho40_0_o20_kT20backup  
# pArr+=("Lpi:10,rho:0,kTimeStepping:1")           # tFNO_m128w30Lpi_rho10_0_o20_kT20backup



# Fail
# pArr+=("Lpi:40,rho:1,skipC:0,kTimeStepping:1,rev:1")  #Fail
# pArr+=("Lpi:10,rho:1,kTimeStepping:1,rev:1")  # Fail 
# pArr+=("Lpi:40,rho:0,kTimeStepping:1,rev:1")  # Fail
# pArr+=("Lpi:10,rho:0,kTimeStepping:1,rev:1")  # Fail


# pArr+=("Lpi:40,rho:1,skipC:0,kTimeStepping:1,rev:1,revD:0") #Fail
# pArr+=("Lpi:10,rho:1,kTimeStepping:1,rev:1,revD:0")   # Fail
# pArr+=("Lpi:40,rho:0,kTimeStepping:1,rev:1,revD:0")   # Fail
# pArr+=("Lpi:10,rho:0,kTimeStepping:1,rev:1,revD:0")   # Fail 

#------------------------------
#pArr=("Lpi:40,rho:1,skipC:0")                   # rerun because of conflict
#------------------------------


#------------------------------
#pArr=("Lpi:40,rho:1,skipC:0,prefix:run2_")                   # rerun because of diverge
#pArr+=("Lpi:40,rho:1,skipC:-1,prefix:run2_")                   # rerun because of diverge
#------------------------------

#pArr=("Lpi:40,rho:0,kTimeStepping:1")                   # rerun because files get damanged

# pArr=("Lpi:10,rho:1,kTimeStepping:1")                   # rerun because files get damanged
# pArr+=("Lpi:10,rho:0,kTimeStepping:1")                   # rerun because files get damanged
# pArr+=("model:kConv,Lpi:10,rho:1,seed:99,prefix:run2_")   # rerun because files get damanged

#-----
#pArr=("model:kConv,Lpi:10,rho:0,seed:99,prefix:run2_")   # rerun because files get damanged

#-----
pArr=("Lpi:10,rho:0,Ftime:1")
pArr+=("Lpi:40,rho:0,Ftime:1")
pArr+=("Lpi:10,rho:1,Ftime:1")
pArr+=("Lpi:40,rho:1,Ftime:1")
pArr+=("Lpi:40,rho:1,skipC:0,Ftime:1")
pArr+=("Lpi:40,rho:1")
pArr+=("Lpi:40,rho:1,kTimeStepping:1")
pArr+=("Lpi:40,rho:1,skipC:0,tAdvD:1")
pArr+=("Lpi:40,rho:1,tAdvD:1")

python train1d_Koop.py  ${pArr[$SLURM_ARRAY_TASK_ID]} 



#python train1d_Koop.py ${models[$SLURM_ARRAY_TASK_ID]}  ${LpiLpi[$SLURM_ARRAY_TASK_ID]} ${rhorho[$SLURM_ARRAY_TASK_ID]} ${kTStep[$SLURM_ARRAY_TASK_ID]} ${FTAdva[$SLURM_ARRAY_TASK_ID]} ${Linear[$SLURM_ARRAY_TASK_ID]} ${basist[$SLURM_ARRAY_TASK_ID]}

