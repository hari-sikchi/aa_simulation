#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
SBATCH --mem=16G
#SBATCH --exclude=compute-0-[9,11,13]
SBATCH -o /home/hsikchi/out.txt
SBATCH -e /home/hsikchi/err.txt
set -x
set -u
set -e
module load singularity
module load cuda-80
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1  python train/train_straight_planner.py --algo trpo