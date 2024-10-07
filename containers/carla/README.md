# Carla Docker Container Setup

This guide explains how to use the provided Docker file to create a Docker container and run the CARLA simulator within the container.

## Prerequisites

Before you begin, ensure the following requirements are met:

- Docker is installed on your machine.
- NVIDIA Docker runtime is installed (for GPU support).
- CARLA simulator Docker image (`carlasim/carla:0.9.10.1`) is available or built.

## Running CARLA in a Docker Container

To run the CARLA simulator inside a Docker container, follow these steps:

### 1. Build the Docker Image (if necessary)

If you haven't already built the Docker image, navigate to the folder containing the Dockerfile and build it with the following command:

```bash
sudo docker build -t carla:latest .
```

### 2. Run the carla simulation

To run CARLA with display support (using OpenGL), execute the following command:
```bash
sudo docker run -e DISPLAY=$DISPLAY --net=host --gpus all --runtime=nvidia --rm carla:latest /bin/bash CarlaUE4.sh -opengl --world-port=2000
```
To run it without an additional display:
sudo docker run -e DISPLAY=$DISPLAY --net=host --gpus all --runtime=nvidia --rm carla:latest /bin/bash DISPLAY= ./CarlaUE4.sh -opengl --world-port=2000