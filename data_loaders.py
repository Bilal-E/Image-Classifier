# CREATED BY: Bilal E.
#
# PURPOSE: 
#   function for creating dataloaders from datasets with transforms


import torch
from torchvision import datasets, transforms
from functions_save_load_args import input_args

in_args = input_args()


def data_load():
    
    
    #DIRECTORIES
    data_dir = '/home/workspace/ImageClassifier/flowers'
    train_dir = in_args.train_dir
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #TRANSFORMS
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(35),
                                           transforms.ColorJitter(brightness=1, contrast=0.8, saturation=0.6),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(226),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])


    test_transforms = transforms.Compose([transforms.Resize(226),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])

    #DATASETS
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms) 

    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

    #DATALOADERS
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 60, shuffle = True)

    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 45)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 45) 
    
    
    return trainloader, validloader, testloader, train_dataset