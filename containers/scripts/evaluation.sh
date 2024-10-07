#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8


echo "SLURM allocated GPUs: $SLURM_JOB_GPUS"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

export CUDA_VISIBLE_DEVICES=7

pwd; hostname; date

SINGULARITYENV_SDL_VIDEODRIVER=offscreen apptainer instance start --nv carla_as_instance.sif carla_instance
nvidia-smi
sleep 60s

apptainer exec --nv --mount type=bind,src=mount,dst=/workspace/mount datagen_from_docker_conda_pytrees_cuda.sif /workspace/mount/local_evaluation.sh

apptainer instance stop carla_instance
