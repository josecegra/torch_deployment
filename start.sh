#!/bin/bash
echo "pull docker image for profile picture classification..."
docker pull axiom1123/torch_deployment
echo "run container and forward to port 3030..."
docker run --detach -ti -p 0.0.0.0:3030:3030 axiom1123/torch_deployment:latest python torch_deployment/src/flask_api.py --model_path torch_deployment/model/model_checkpoint.pth --class_index_path torch_deployment/model/class_index_dict.json --endpoint_name classifier --host 0.0.0.0 --port 3030
