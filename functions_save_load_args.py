# CREATED BY: Bilal E.
#
# PURPOSE: 
#   functions for argument parsing, saving checkpoint, and loading chcekpoint  


import torch
from torchvision import models
from torch import nn, optim
import argparse
import os
from collections import OrderedDict


def input_args():
    
    parser = argparse.ArgumentParser(description = '***  ***')
    
    
    parser.add_argument('--train_dir', type=str, default = '/home/workspace/ImageClassifier/flowers/train', help='path to train directory')

    parser.add_argument('--gpu', type=str, default='cuda', help='cuda or cpu')
    
    parser.add_argument('--arch', type=str, default='densenet', help='choose model: densenet or vgg')
   
    parser.add_argument('--lr', type=float, default=0.001, help='value for learning rate')
    
    parser.add_argument('--epochs', type=int, default=24, help='number of epochs')

    parser.add_argument('--hidden_units', type=int, default=620, help='number of hidden units ')
    
    parser.add_argument('--img_path', type=str, default='/home/workspace/ImageClassifier/flowers/test/74/image_01191.jpg', help='path to image')
    
    parser.add_argument('--json_file', type=str, default='/home/workspace/ImageClassifier/cat_to_name.json', help='path to JSON file to display category names')
   
    parser.add_argument('--top_k', type=int, default=1, help='specify top K classes')    
    
    parser.add_argument('--print_k', type=int, default=0, help='for print list of top_k, type 1/for print max prob only, type 0 (default)') 
    
    
    return parser.parse_args() 



def checkpoint_saver(checkpoint_dict):
   
    torch.save(checkpoint_dict, 'model_checkpoint.pth') 

    
    
def rebuild_load_checkpoint(check_model, check_filepath):
    
    in_args = input_args()

    n_input = None    
    n_output = 102

    
    if check_model == 'densenet':
        densenet161 = models.densenet161(pretrained=True)
    
        models_select = {'densenet': densenet161}

        model = models_select[check_model]

        for param in model.parameters():
            param.requires_grad = False

        
        n_input = 2208

        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(n_input, in_args.hidden_units)),
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(0.5)),
                                                ('fc2', nn.Linear(in_args.hidden_units, n_output)),
                                                ('output', nn.LogSoftmax(dim=1))]))                                              

        model.classifier = classifier

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr=in_args.lr, momentum=0.9)

    elif check_model == 'vgg':
        vgg16 = models.vgg16(pretrained=True)
    
        models_select = {'vgg': vgg16}

        model = models_select[check_model]

        for param in model.parameters():
            param.requires_grad = False

        n_input = 4096

        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(n_input, in_args.hidden_units)),
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(0.5)),
                                                ('fc2', nn.Linear(in_args.hidden_units, n_output)),
                                                ('output', nn.LogSoftmax(dim=1))]))                                              

        model.classifier[6] = classifier    

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier[6].parameters(), lr=in_args.lr, momentum=0.9)
        
        
    start_epoch = 0
    if os.path.isfile(check_filepath):

        print("\n=> loading checkpoint.. '{}'".format(check_filepath))
    
        checkpoint = torch.load(check_filepath)
 
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        model.load_state_dict(checkpoint['state_dict'])
        
        start_epoch = checkpoint['epochs'] + 1
        
        model.class_to_idx = checkpoint['class_to_idx'] 
   
    
    print('\nCHECKPOINT MODEL: ', in_args.arch)
    print('\nOPTIMIZER STATE: ', optimizer)
    print('\nEPOCHS TRAINED: ', start_epoch)
    
    return model, optimizer, start_epoch

