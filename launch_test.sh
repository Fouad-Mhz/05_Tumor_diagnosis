#!/bin/bash

# Define an environment variable for the data path with a default value
DATA_PATH="C:\Users\Mhz\Desktop\Projet_Vif"
IMAGE_NAME_MP=docker_model

echo "DATA_PATH: $DATA_PATH"
echo "IMAGE_NAME_MP: $IMAGE_NAME_MP"

# Prompt the user to enter patient numbers
echo "Please enter patient IDs separated by commas (1 to 569):"
read -r patient_ids_csv

# Pass the patient_numbers to the Docker container
docker run -v "$DATA_PATH":/home/jovyan/my_model "$IMAGE_NAME_MP" python /home/jovyan/my_model/diagnostic.py $patient_ids_csv