#!/bin/bash

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-gpu=8

pwd; hostname; date
SINGULARITYENV_SDL_VIDEODRIVER=offscreen apptainer instance start --nv carla_as_instance.sif carla_instance
sleep 60s

apptainer exec --nv --mount type=bind,src=mount,dst=/workspace/mount datagen_from_docker_conda_pytrees_cuda.sif /workspace/mount/datagen.sh

apptainer instance stop carla_instance
