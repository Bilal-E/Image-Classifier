# CREATED BY: Bilal E.
#
# PURPOSE: 
#   function for predicting image class and probability
#
# ARGS: 
#   image_path: path to image
#   model: model of the checkpoint loaded 
#   top_k: no. of top K classes


from PIL import Image
import argparse
import torch
from image_processing import process_image
from functions_save_load_args import input_args


def predict(image_path, model, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    in_args = input_args()
    
    device = torch.device("cuda" if in_args.gpu == "cuda" else "cpu")

    
    img = process_image(image_path)
    
    # Need to convert the image ndarray(process_image output) 
    # to input Tensor for the model to be used  
    img_input_model = torch.from_numpy(img).float().unsqueeze_(0)  
    
    #Tensor.to(device)
    out = img_input_model.to(device)
    
    output = model.forward(out)
    
    ps = torch.exp(output)
    
    # Top 5 probabilities with respective indices 
    top_5_ps, top_5_idx = ps.topk(top_k, dim=1) 
    top_5_ps = top_5_ps.tolist()[0]
    top_5_idx = top_5_idx.tolist()[0]
    
    class_idx = []
    top_5_labels = []

    # Get the class indices
    for idx in range(len(model.class_to_idx.items())):
        class_idx.append(list(model.class_to_idx.items())[idx][0])
    # Get class keys respective to class indices
    for label in range(top_k):
        top_5_labels.append(class_idx[top_5_idx[label]])   
    

    return top_5_ps, top_5_labels





