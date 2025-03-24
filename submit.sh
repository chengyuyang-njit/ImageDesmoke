#!/bin/bash -l
#SBATCH --job-name=unet_with_ssim_mse
#SBATCH --output=%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=cliu # Replace PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --time=23:59:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=4000M
conda activate py12
python train.py -c "/mmfs1/project/cliu/cy322/projects/ImageDesmoke/config_wulver_epoch_200_ssim_mse.json"