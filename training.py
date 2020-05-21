# CREATED BY: Bilal E.
#
# PURPOSE: 
#   function for training a dataset on a given network (vgg or densenet)
#
# ARGS:
#   achitecture: model name given by user 
#   n_epoch: training epochs given by user
#   hidden_units: no. of hidden units given by user
#   learning_rate: value of learning rate given by user
#

import torch
from torch import nn, optim
from torchvision import datasets, models
from data_loaders import data_load
from functions_save_load_args import checkpoint_saver, input_args
from collections import OrderedDict
import time

in_args = input_args()

def training(architecture, n_epoch, hidden_units, learning_rate):


    t = time.time()

    n_input = None    
    n_output = 102

    
    if architecture == 'densenet':
        
        densenet161 = models.densenet161(pretrained=True)
    
        models_select = {'densenet': densenet161}

        model = models_select[architecture]

        for param in model.parameters():
            param.requires_grad = False

        
        n_input = 2208

        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(n_input, hidden_units)),
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(0.5)),
                                                ('fc2', nn.Linear(hidden_units, n_output)),
                                                ('output', nn.LogSoftmax(dim=1))]))                                              

        model.classifier = classifier

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.9)

    elif architecture == 'vgg':

        
        vgg16 = models.vgg16(pretrained=True)
    
        models_select = {'vgg': vgg16}

        model = models_select[architecture]

        for param in model.parameters():
            param.requires_grad = False

        
        n_input = 4096

        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(n_input, hidden_units)),
                                                ('relu', nn.ReLU()),
                                                ('dropout', nn.Dropout(0.5)),
                                                ('fc2', nn.Linear(hidden_units, n_output)),
                                                ('output', nn.LogSoftmax(dim=1))]))                                              

        model.classifier[6] = classifier    

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier[6].parameters(), lr=learning_rate, momentum=0.9)

        
    epochs = n_epoch
    steps = 0
    running_loss = 0
    accuracy_per_epoch = []
    best_val_accuracy = None
    train_len = 80
    
    trainloader, validloader, testloader, train_dataset = data_load()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    t = time.time()
    
    for epoch in range(epochs):

        
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model.forward(inputs)
            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if steps % train_len == 0:

                test_loss = 0
                accuracy = 0

                model.eval()

                with torch.no_grad():

                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logits = model.forward(inputs)
                        batch_loss = criterion(logits, labels)

                        test_loss += batch_loss.item()

                        ps = torch.exp(logits)
                        top_p, top_class = ps.topk(1, dim=1)

                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
               
    
                running_loss = 0
                model.train()


        print(f'\nEpoch: {epoch+1}/{epochs}.. '
              f'\nTraining Loss = {running_loss/train_len:.3f}.. '
              f'\nValidation Loss = {test_loss/len(validloader):.3f}.. '
              f'\nAccuracy = {accuracy/len(validloader)*100:.1f}..')
        
        accuracy_per_epoch.append(accuracy/len(validloader)*100)
        
    time_elapsed = time.time() - t
    
    print('\n')
    print('\nTraining completed in {:.0f} min {:.0f} sec'.format(time_elapsed // 60, time_elapsed % 60))
    
    best_val_accuracy = max(float(val) for val in accuracy_per_epoch)    
    print("\nBest val accuracy = {:.1f}".format(best_val_accuracy))
            
    print('\n\n** Saving checkpoint ...')
            
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {'arch': architecture,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'epochs': epoch,
                  'learning_rate': learning_rate,
                  'class_to_idx': model.class_to_idx} 

    checkpoint_saver(checkpoint)
    
 
    
    
