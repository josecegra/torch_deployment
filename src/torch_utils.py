import io
import os

import cv2
import torch
import numpy as np
from PIL import Image
import json

import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

def read_image_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert('RGB') 

class DataWrapper(Dataset):
  def __init__(self, img_byte_list,img_transform = None):
    self.img_byte_list = img_byte_list
    self.img_transform = img_transform

  def __len__(self):
    return len(self.img_byte_list)

  def __getitem__(self, idx):
    image_bytes = self.img_byte_list[idx]
    image = read_image_bytes(image_bytes)

    if self.img_transform:
      image = self.img_transform(image)
    return image


class ResNet(nn.Module):
    def __init__(self,size, output_size):
        super(ResNet, self).__init__()

        if size not in [18,34,50,101,152]:
            raise Exception('Wrong size for resnet')
        if size == 18:
            self.net = torchvision.models.resnet18(pretrained=True)
        elif size == 34:
            self.net = torchvision.models.resnet34(pretrained=True)
        elif size == 50:
            self.net = torchvision.models.resnet50(pretrained=True)
        elif size == 101:
            self.net = torchvision.models.resnet101(pretrained=True)
        elif size == 152:
            self.net = torchvision.models.resnet152(pretrained=True)

        #initialize the fully connected layer
        self.net.fc = nn.Linear(self.net.fc.in_features, output_size)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.net(x)
        out = self.sm(out)
        return out


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # this normalization is required https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def transform_image(image_bytes):
    base_transform = get_transform()
    image = read_image_bytes(image_bytes)   
    image = base_transform(image)
    return image.unsqueeze(0)

def load_model(model_path,class_index_path):

    with open(class_index_path,'r') as f:
        class_index_dict = json.load(f)
        class_index_dict = {int(k):v for k,v in class_index_dict.items()}

    model_name = 'resnet34'
    output_size = len(class_index_dict)
    if model_name.startswith("resnet"):
        size = int(model_name.replace("resnet",""))
        model = ResNet(size,output_size)
    model = nn.DataParallel(model)
    #for cpu only
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    return model, class_index_dict

def predict_batches(model,img_byte_list,filename_list):
    XAI_path_list = []
    predictions_list = []
    conf_list = []

    base_transform = get_transform()
    datawrapper = DataWrapper(img_byte_list,base_transform)
    dataloader = DataLoader(datawrapper,batch_size=4)
    predictions_list = []
    for img in dataloader:
        outputs = model(img)
        probabilities, predicted = torch.max(outputs.data, 1)
        predictions_list += list(predicted.cpu())
        conf_list += list(probabilities.cpu())

    predictions_list = [pred.item() for pred in predictions_list]
    conf_list = [conf.item() for conf in conf_list]
    XAI_path_list = [None for conf in conf_list]

    return predictions_list, conf_list, XAI_path_list

def predict_XAI(model,img_byte_list,filename_list,saving_dir):
    XAI_path_list = []
    predictions_list = []
    conf_list = []

    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)

    heatmap_layer = model.module.net.layer4[2].conv2

    for image_bytes,fname in zip(img_byte_list,filename_list):
        
        image = read_image_bytes(image_bytes)
        image_interpretable, label, conf = grad_cam(model, image, heatmap_layer, get_transform())

        XAI_img_path = os.path.join(saving_dir,fname)
        XAI_img = Image.fromarray(image_interpretable).convert('RGB')
        XAI_img.save(XAI_img_path)

        XAI_path_list.append(XAI_img_path)
        predictions_list.append(label.cpu().item())
        conf_list.append(conf.item())

    return predictions_list, conf_list, XAI_path_list



class InfoHolder():

    def __init__(self, heatmap_layer):
        self.gradient = None
        self.activation = None
        self.heatmap_layer = heatmap_layer

    def get_gradient(self, grad):
        self.gradient = grad

    def hook(self, model, input, output):
        output.register_hook(self.get_gradient)
        self.activation = output.detach()

def generate_heatmap(weighted_activation):
    raw_heatmap = torch.mean(weighted_activation, 0)
    heatmap = np.maximum(raw_heatmap.detach().cpu(), 0)
    heatmap /= torch.max(heatmap) + 1e-10
    return heatmap.numpy()

def superimpose(input_img, heatmap):
    img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.6 + img * 0.4)
    pil_img = cv2.cvtColor(superimposed_img,cv2.COLOR_BGR2RGB)
    return pil_img

def to_RGB(tensor):
    tensor = (tensor - tensor.min())
    tensor = tensor/(tensor.max() + 1e-10)
    image_binary = np.transpose(tensor.numpy(), (1, 2, 0))
    image = np.uint8(255 * image_binary)
    return image

def grad_cam(model, image,  heatmap_layer, transform, truelabel=None):
    
    input_tensor = transform(image)
    info = InfoHolder(heatmap_layer)
    heatmap_layer.register_forward_hook(info.hook)
    output = model(input_tensor.unsqueeze(0))[0]
    truelabel = truelabel if truelabel else torch.argmax(output)
    output[truelabel].backward()
    weights = torch.mean(info.gradient, [0, 2, 3])
    activation = info.activation.squeeze(0)
    weighted_activation = torch.zeros(activation.shape)
    for idx, (weight, activation) in enumerate(zip(weights, activation)):
        weighted_activation[idx] = weight * activation

    heatmap = generate_heatmap(weighted_activation)
    return superimpose(np.asarray(image),heatmap),truelabel,output[truelabel]