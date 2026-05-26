#!/bin/env bash
#SBATCH -A NAISS2024-22-378 -p alvis
#SBATCH -t 5:00:00
#SBATCH --gpus-per-node=A40:1


#---- run 1
# #SBATCH -a 0-9
# #SBATCH -o out_koop_rho25_AIfluid_%a
# #SBATCH -e out_koop_rho25_AIfluid_%a
# pArr=("Lpi:25,rho:0")
# pArr+=("Lpi:25,rho:1")
# pArr+=("Lpi:25,rho:0.25")
# pArr+=("Lpi:25,rho:0.5")
# pArr+=("Lpi:25,rho:0.75")
# pArr+=("Lpi:25,rho:0,kTimeStepping:1")
# pArr+=("Lpi:25,rho:1,kTimeStepping:1")
# pArr+=("Lpi:25,rho:0.25,kTimeStepping:1")
# pArr+=("Lpi:25,rho:0.5,kTimeStepping:1")
# pArr+=("Lpi:25,rho:0.75,kTimeStepping:1")


#---- run 2
# #SBATCH -a 0-0
# #SBATCH -o out_koop_rho25_AIfluid_rerun2_%a
# #SBATCH -e out_koop_rho25_AIfluid_rerun2_%a
# pArr=("Lpi:25,rho:1,seed:99,prefix:run2_")



#--- run 3
# #SBATCH -a 0-1
# #SBATCH -o out_koop_rho25_AIfluid_noskip_%a
# #SBATCH -e out_koop_rho25_AIfluid_noskip_%a
# pArr=("Lpi:25,rho:1,skipC:0")
# pArr+=("Lpi:25,rho:1,skipC:0,kTimeStepping:1")




#--- run 4
#SBATCH -a 0-7
#SBATCH -o out_koop_rho25_AIfluid_noskip_remainedcases_%a
#SBATCH -e out_koop_rho25_AIfluid_noskip_remainedcases_%a

pArr=("Lpi:25,rho:0,skipC:0")
pArr+=("Lpi:25,rho:0.25,skipC:0")
pArr+=("Lpi:25,rho:0.5,skipC:0")
pArr+=("Lpi:25,rho:0.75,skipC:0")
pArr+=("Lpi:25,rho:0,skipC:0,kTimeStepping:1")
pArr+=("Lpi:25,rho:0.25,skipC:0,kTimeStepping:1")
pArr+=("Lpi:25,rho:0.5,skipC:0,kTimeStepping:1")
pArr+=("Lpi:25,rho:0.75,skipC:0,kTimeStepping:1")





#---------------------------------------------

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 tensorboard/2.15.1-gfbf-2023a  h5py/3.9.0-foss-2023a matplotlib/3.7.2-gfbf-2023a Tkinter/3.11.3-GCCcore-12.3.0 

#ipython -c "%run train1D_MKS_fourier_Koop.ipynb"

#
#torchrun --standalone --nproc_per_node=4 train1d_Koop.py 1 
#

python train1d_Koop.py  ${pArr[$SLURM_ARRAY_TASK_ID]} 



#python train1d_Koop.py ${models[$SLURM_ARRAY_TASK_ID]}  ${LpiLpi[$SLURM_ARRAY_TASK_ID]} ${rhorho[$SLURM_ARRAY_TASK_ID]} ${kTStep[$SLURM_ARRAY_TASK_ID]} ${FTAdva[$SLURM_ARRAY_TASK_ID]} ${Linear[$SLURM_ARRAY_TASK_ID]} ${basist[$SLURM_ARRAY_TASK_ID]}

