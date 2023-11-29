#!/bin/bash

IMAGE_NAME_MP=docker_model

# Remove the existing Docker image with the same name if it exists
docker rmi $IMAGE_NAME_MP 2>/dev/null

# Build the Docker image using the Dockerfile in the current directory
docker build -t $IMAGE_NAME_MP -f Dockerfile .