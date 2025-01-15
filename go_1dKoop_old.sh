#!/bin/env bash
#SBATCH -A NAISS2024-22-378 -p alvis
#SBATCH -t 8:00:00
#SBATCH --gpus-per-node=A40:1


####SBATCH -a 0-23
####SBATCH -a 0-3
#SBATCH -a 0-1
#SBATCH -o out_koop_batchrun3_%a
#SBATCH -e out_koop_batchrun3_%a

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 tensorboard/2.15.1-gfbf-2023a  h5py/3.9.0-foss-2023a matplotlib/3.7.2-gfbf-2023a Tkinter/3.11.3-GCCcore-12.3.0 

#ipython -c "%run train1D_MKS_fourier_Koop.ipynb"

#
#torchrun --standalone --nproc_per_node=4 train1d_Koop.py 1 
#


#
#               model_name, Lpi:int, rho:int, kTimeStepping:int ,   FourierTimeDIM:int, linearKoopmanAdv:int
#


#models=(kFNO kFNO kFNO kFNO  kFNO kFNO kFNO kFNO    kConv kConv kConv kConv   kConv kConv kConv kConv   kFNO kFNO kFNO kFNO    kFNO kFNO kFNO kFNO)
#LpiLpi=(  40  40   40  40     10   10   10   10      40     40    40    40     10    10     10    10     40   40   40   40      10   10   10  10)
#rhorho=(   0   0    1   1      0   0     1    1       0      0     1     1      0     0      1     1      0    1    0   1        0    1    0   1)
#kTStep=(  20   1   20   1     20   1    20    1      20      1    20     1     20     1     20     1     20   20   20  20       20   20   20  20)
#FTAdva=(   0   0    0   0      0   0     0    0       0      0     0     0      0     0      0     0      1    1    0   0        1    1    0   0)
#Linear=(   0   0    0   0      0   0     0    0       0      0     0     0      0     0      0     0      0    0    1   1        0    0    1   1) 


#models=(kFNO kFNO kFNO kFNO) 
#LpiLpi=( 40 40 40 40) 
#rhorho=(  1  1 1  1 )
#kTStep=( 20  1 20 1 )
#FTAdva=(  0  0 0  0 )
#Linear=(  0  0 0  0 )
#basist=(  o  o supr2 supr2)

models=(kFNO kFNO)
LpiLpi=( 40 40 )
rhorho=(  1  1 )
kTStep=( 20  1 )
FTAdva=(  0  0 )
Linear=(  0  0 )
basist=(  o  o )


python train1d_Koop_old.py ${models[$SLURM_ARRAY_TASK_ID]}  ${LpiLpi[$SLURM_ARRAY_TASK_ID]} ${rhorho[$SLURM_ARRAY_TASK_ID]} ${kTStep[$SLURM_ARRAY_TASK_ID]} ${FTAdva[$SLURM_ARRAY_TASK_ID]} ${Linear[$SLURM_ARRAY_TASK_ID]} ${basist[$SLURM_ARRAY_TASK_ID]}

