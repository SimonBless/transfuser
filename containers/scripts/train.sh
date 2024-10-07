#!/bin/bash

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-gpu=8

pwd; hostname; date

echo "SLURM allocated GPUs: $SLURM_JOB_GPUS"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

#nvidia-smi
apptainer exec --nv --mount type=bind,src=mount_training,dst=/workspace/mount datagen_from_docker_conda_pytrees_cuda.sif python /workspace/mount/team_code_transfuser/train.py --batch_size 96 --logdir /workspace/mount/training_log/training_number_14_seed1_39_Highway_Val --root_dir /workspace/mount/training_dataset --parallel_training 0 --epochs 20 --load_file /workspace/mount/model_ckpt/transfuser/model_seed1_39.pth --lr 1e-4 --setting validation --val_every 1 --schedule_reduce_epoch_01 12 --schedule_reduce_epoch_02 16


