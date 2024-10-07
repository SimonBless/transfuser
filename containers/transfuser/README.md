# Transfuser Agent Docker Container

This README provides instructions on how to run the `transfuser-agent` Docker container, which is used to run an autonomous driving agent in CARLA. Additionally, it includes the resources and references used in creating the Docker image.

## Running the Transfuser-Agent Container

The `transfuser-agent` container runs a pre-trained agent capable of autonomous driving within CARLA. Follow the steps below to run the container.

### 1. Build the Docker Image

If you haven't built the Docker image yet, navigate to the directory containing the Dockerfile and run:

```bash
sudo docker build -t transfuser-agent:latest .
```


## Sources
1. https://fabiorosado.dev/blog/install-conda-in-docker
2. https://kevalnagda.github.io/conda-docker-tutorial
3. https://github.com/autonomousvision/transfuser


