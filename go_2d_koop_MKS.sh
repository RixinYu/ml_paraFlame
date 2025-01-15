#!/bin/env bash
#SBATCH -A NAISS2024-22-378 -p alvis

###-----   #SBATCH -t 110:00:00   # For Conv
#SBATCH -t 94:00:00     # For Fourier

#SBATCH --gpus-per-node=A100:1

###----   #SBATCH -a 0-5 (round 1)
###----   #SBATCH -a 0-5 (round 2)
###----   #SBATCH -a 0-1  # (round 3)
###----   #SBATCH -a 0-0  # (round 4)
###----   #SBATCH -a 0-0  # (round 5) 
###----   #SBATCH -a 0-2  # (round 6)  # The screen output from round 6 have mistakenly flushed those from previous round 5
#SBATCH -a 0-1  # (round 7)

#SBATCH -o out_2d_koop_MKS_round7_%a
#SBATCH -e out_2d_koop_MKS_round7_%a



module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 tensorboard/2.15.1-gfbf-2023a  h5py/3.9.0-foss-2023a matplotlib/3.7.2-gfbf-2023a Tkinter/3.11.3-GCCcore-12.3.0 

#ipython -c "%run train1D_MKS_fourier_Koop.ipynb"
#
#torchrun --standalone --nproc_per_node=4 train1d_Koop.py 1 
#

#--- round 1
# pArr=("Lpi:15,rho:0")
# pArr+=("Lpi:15,rho:0,kTimeStepping:1")
# pArr+=("Lpi:15,rho:1")
# pArr+=("Lpi:15,rho:1,kTimeStepping:1")
# pArr+=("Lpi:25,rho:0")
# pArr+=("Lpi:25,rho:0,kTimeStepping:1")

#---- round 2
# pArr=("Lpi:25,rho:1")
# pArr+=("Lpi:25,rho:1,kTimeStepping:1")
# pArr+=("model:kConv,Lpi:25,rho:1")
# pArr+=("model:kConv,Lpi:25,rho:1,kTimeStepping:1")
# pArr+=("model:kConv,Lpi:25,rho:0")
# pArr+=("model:kConv,Lpi:25,rho:0,kTimeStepping:1")

#---- round 3
#pArr=("Lpi:25,rho:1,prefix:rerun_,seed:10")
#pArr+=("Lpi:25,rho:0,kTimeStepping:1,seed:10")  # rerun the case with a new random seed (previous nonconverged run being interupted )

#--- round 4
#pArr=("Lpi:25,rho:1,kTimeStepping:1,prefix:rerun_,seed:10")

#--- round 5
#pArr=("Lpi:25,rho:1,prefix:rerun2_,seed:15,batchsize:40")


#--- round 6
# pArr=("Lpi:25,rho:0,kTimeStepping:1,prefix:rerun_,seed:6,gradientclip:20")
# pArr+=("Lpi:25,rho:1,kTimeStepping:1,prefix:rerun2_,seed:6,batchsize:35")
# pArr+=("Lpi:25,rho:1,prefix:rerun3_,seed:6,batchsize:40")

#--- round 7
pArr=("Lpi:25,rho:1,prefix:rerun4_,seed:7,batchsize:40")
pArr+=("Lpi:25,rho:1,kTimeStepping:1,prefix:rerun3_,seed:7,batchsize:35")

python train2d_koop_MKS.py  ${pArr[$SLURM_ARRAY_TASK_ID]} 



#python train1d_Koop.py ${models[$SLURM_ARRAY_TASK_ID]}  ${LpiLpi[$SLURM_ARRAY_TASK_ID]} ${rhorho[$SLURM_ARRAY_TASK_ID]} ${kTStep[$SLURM_ARRAY_TASK_ID]} ${FTAdva[$SLURM_ARRAY_TASK_ID]} ${Linear[$SLURM_ARRAY_TASK_ID]} ${basist[$SLURM_ARRAY_TASK_ID]}

