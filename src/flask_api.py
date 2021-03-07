import os
from PIL import Image
import argparse
import json

from flask import Flask, jsonify, request

import sys
sys.path.append('../')

from torch_utils import *

def predict_images(model,filename_list,img_byte_list,saving_dir,XAI_bool):

    pred_dict = {'class_id':{},'class_name':{},'confidence':{},'XAI_path':{}}

    if XAI_bool == 'false':
        predictions_list, conf_list, XAI_path_list = predict_batches(model,img_byte_list,filename_list)
    else:
        predictions_list, conf_list, XAI_path_list = predict_XAI(model,img_byte_list,filename_list,saving_dir)

    for idx,fname,XAI_path,conf in zip(predictions_list,filename_list,XAI_path_list,conf_list):
        pred_dict['class_id'].update({fname:idx})
        pred_dict['class_name'].update({fname:class_index_dict[idx]})
        pred_dict['XAI_path'].update({fname:XAI_path})
        pred_dict['confidence'].update({fname:conf})

    return pred_dict


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--class_index_path', required=True)
    parser.add_argument('--endpoint_name', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--saving_dir', required=False)
    args = parser.parse_args()

    if args.saving_dir:
        saving_dir = args.saving_dir
    else:
        fpath = os.path.dirname(os.path.abspath(__file__))
        saving_dir = os.path.join(os.path.split(fpath)[0],'XAI')

    model, class_index_dict = load_model(args.model_path,args.class_index_path)

    app = Flask(__name__)
    @app.route(f'/{args.endpoint_name}', methods=['POST','GET'])
    def predict():
        if request.method == 'POST':
            XAI_bool = request.headers['XAI']
            filename_list = [request.files[k].filename for k in request.files.keys()]
            img_byte_list = [request.files[k].read() for k in request.files.keys()]

            pred_dict = predict_images(model,filename_list,img_byte_list,saving_dir,XAI_bool)
            return jsonify(pred_dict)
        if request.method == 'GET':
            return jsonify({'class_index_dict': class_index_dict})

    app.run(host=args.host, port=int(args.port))


