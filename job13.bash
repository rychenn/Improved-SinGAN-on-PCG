#!/bin/bash
#SBATCH --job-name=SinGAN_train13
#SBATCH --output=train_out13.txt
#SBATCH -c 4
#SBATCH --time=10:00:00
#SBATCH -p owners
#SBATCH --gres gpu:1
python3 main_train.py --input_name SMB-1-3.png 


