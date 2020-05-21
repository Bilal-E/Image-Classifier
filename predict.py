# CREATED BY: Bilal E.
# DATE: 21-May-2020
# 
#
# PURPOSE:
#   reads in an image and a checkpoint to predict image class and prints the probability with class name  
#
# EXPECTED USER INPUT:
#   python predict.py --arch <vgg or densenet (since there is only one checkpoint file that this application generates 
#   based on the training model; choose the same model as of training to avoid mismatch)> 
#   --img_path <complete path of image to be predicted> --lr <learning rate used when training(float)> 
#   --hidden_units <hidden units used when training(int)> --top_k <required no. of top K classes(int)>
#   --print_k <1 prints a list of top_k; 0 prints only max probability> --json_file <complete path to category names file>
#  
# CHECKPOINT FILE:
#   'model_checkpoint.pth', created by train.py 
#   <when tested in workspace 'model_checkpoint.pth' contained trained vgg16 model on flower dataset 'train', 
#    with a learning rate of 0.005, and hidden units 1024>
#
# EXAMPLE CALL TO PREDICT IMAGE USING ABOVE MENTIONED CHECKPOINT FILE: (default image path is set to one of 'rose' class image 
#   '/home/workspace/ImageClassifier/flowers/test/74/image_01191.jpg')
#   python predict.py --arch vgg --lr 0.005 --hidden_units 1024 --top_k 3 --print_k 1
# 


import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os
from functions_save_load_args import rebuild_load_checkpoint, input_args
from function_predict import predict
import json
from data_loaders import data_load
from PIL import Image


# Initialize argument parser
in_args = input_args()


# Load checkpoint
load_checkpoint = True

if load_checkpoint:
    model_loaded, optimizer, start_epoch =  rebuild_load_checkpoint(in_args.arch, '/home/workspace/ImageClassifier/model_checkpoint.pth')

    
    # device state (default is set to cuda)     
    device = torch.device("cuda" if in_args.gpu == "cuda" else "cpu")
    model_loaded = model_loaded.to(device)

    for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


    model_loaded.eval()


    print("Device : ", device)

    
# Predict top K classes
probs, classes = predict(in_args.img_path, model_loaded, in_args.top_k) 

with open(in_args.json_file, 'r') as f:
    cat_to_name = json.load(f)
   
    
if in_args.print_k == 0:    
   
    img_prob = max(float(i) for i in probs)   
    img_title = [cat_to_name[name] for name in classes][0]

    print('\nCategory Name: {} \nProbability = {:.3f}'.format(img_title, img_prob))

elif in_args.print_k == 1:
    
    img_prob = [float(j) for j in probs] 
    img_title = [cat_to_name[name] for name in classes]
    
    print('\nCategory Name: {} \nProbability = {}'.format(img_title, img_prob))











