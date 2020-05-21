# CREATED BY: Bilal E.
# DATE: 21-May-2020
# 
#
# PURPOSE:
#   train a dataset on a new network (models to be used: vgg or densenet); 
#   output Training loss, Validation loss, Validation accuracy, and Best value accuracy
#   and saves checkpoint to file 'model_checkpoint.pth' 
#
#
# EXPECTED USER INPUT:
#   python train.py --train_dir <path to desired directory for training> --arch <model(vgg or densenet)> 
#   --lr <learning rate for training(float)> --hidden_units <hidden units for training(int)> 
#   --epochs <epochs for training(int)> --gpu <train on gpu>
#
#
# EXAMPLE CALL:
#   python train.py --train_dir '/home/workspace/ImageClassifier/flowers/train' --arch vgg --lr 0.005 --hidden_units 1024 --epochs 30 --gpu cuda
# 


import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os
from workspace_utils import active_session
from functions_save_load_args import checkpoint_saver, input_args
from training import training
from data_loaders import data_load
from collections import OrderedDict
import time

in_args = input_args()


device = torch.device("cuda" if in_args.gpu == "cuda" else "cpu")
print(device)


# ARGS for training() given by user:
#   architecture(in_args.arch), n_epoch(in_args.epochs), hidden_units(in_args.hidden_units), learning_rate(in_args.lr)

with active_session(): #only use for long running tasks
    
    training(in_args.arch, in_args.epochs, in_args.hidden_units, in_args.lr)
    
    