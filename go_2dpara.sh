#!/bin/env bash
#SBATCH -A NAISS2023-22-350 -p alvis
#SBATCH -t 2-00:00:00
#SBATCH --gpus-per-node=A40:4
#SBATCH -e out_pFNO2d_dct_ED6
#SBATCH -o out_pFNO2d_dct_ED6

module load PyTorch/1.13.1-foss-2022a-CUDA-11.7.0  TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0  SciPy-bundle/2022.05-foss-2022a  h5py/3.7.0-foss-2022a  matplotlib/3.5.2-foss-2022a Tkinter/3.10.4-GCCcore-11.3.0  IPython/8.5.0-GCCcore-11.3.0



#
torchrun --standalone --nproc_per_node=4 train2d_para.py 1
#

# python train2d_para.py 0

