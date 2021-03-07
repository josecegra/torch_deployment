# Pytorch deployment with Flask

Deplying an image classifier with pytorch and Flask

`pip install -r requirements.txt`

Set up API

`python src/flask_api.py --model_path model/model_checkpoint.pth --class_index_path model/class_index_dict.json --endpoint_name classifier --host localhost --port 5000`

Make predictions

`python src/predict.py --img_dir test_images --XAI False --endpoint_name classifier --host localhost --port 5000`

The model can be trained using the repo: https://github.com/europeana/rd-img-classification-pilot

