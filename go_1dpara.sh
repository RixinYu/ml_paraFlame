#!/bin/env bash
#SBATCH -A NAISS2023-22-350 -p alvis
#SBATCH -t 1-24:00:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -e out_pConv1d
#SBATCH -o out_pConv1d

module load PyTorch/1.13.1-foss-2022a-CUDA-11.7.0  TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0  SciPy-bundle/2022.05-foss-2022a  h5py/3.7.0-foss-2022a  matplotlib/3.5.2-foss-2022a Tkinter/3.10.4-GCCcore-11.3.0  IPython/8.5.0-GCCcore-11.3.0


#ipython -c "%run train1D_MS_fourier_Para.ipynb"

#
#torchrun --standalone --nproc_per_node=4 train1d_para.py 1 
#

python train1d_para.py 0

