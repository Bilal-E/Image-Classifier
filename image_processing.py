# CREATED BY: Bilal E.
#
# PURPOSE: 
#   function for processing input image to be passed into model for prediction


import numpy as np
from PIL import Image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
  
    img = Image.open(image)
    img = img.resize((256, 256))
                                                #process to center crop 224x224 
    left = int(img.size[0]/2 - 224/2)           # 256/2 - 224/2 --> 16
    upper = int(img.size[1]/2 - 224/2)          # 256/2 - 224/2 --> 16
    right = left + 224                          # 16 + 224 --> 240
    lower = upper + 224                         # 16 + 224 --> 240

    img = img.crop((left, upper, right, lower)) # .crop(16,16,240,240) --> (224,224)
    img_ndarray = np.array(img) / 255 
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    img_ndarray = (img_ndarray - means) / stds
    
    img_ndarray = img_ndarray.transpose(2,0,1) #color channel: first, while retaining the order of the other two dimensions
    
    return img_ndarray 