#!/bin/bash
#SBATCH --job-name=SinGAN_train
#SBATCH --output=train_out.txt
#SBATCH -c 4
#SBATCH --time=10:00:00
#SBATCH -p gpu
#SBATCH --gres gpu:1
python3 main_train.py --input_name SMB-1-1.png 


