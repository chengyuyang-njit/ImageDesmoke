#!/bin/bash -l
#SBATCH --job-name=ablation-no-perc
#SBATCH --output=/mmfs1/project/cliu/cy322/projects/ImageDesmoke/sbatch_reports/outputs/%x.%j.out # %x.%j expands to slurm JobName.JobID
#SBATCH --error=/mmfs1/project/cliu/cy322/projects/ImageDesmoke/sbatch_reports/errors/%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=cliu # Replace PI_ucid which the NJIT UCID of PI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=3-00:00:00  # D-HH:MM:SS
#SBATCH --mem-per-cpu=4000M

conda activate py12
python train.py -c "/mmfs1/project/cliu/cy322/projects/ImageDesmoke/configs/ablation-no-perc.json"