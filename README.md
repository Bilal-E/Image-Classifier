# Image Classifier Command Line Application 

<br/>
<br/>

## TRAINING (train.py)

<br/>

### PURPOSE:
####   train a dataset on a new network (models to be used: vgg or densenet); 
####   output Training loss, Validation loss, Validation accuracy, and Best value accuracy
####   saves checkpoint to file 'model_checkpoint.pth' 

<br/>

### EXPECTED USER INPUT:
####   python train.py --train_dir 'path to desired directory for training' --arch 'model(vgg or densenet)' 
####   --lr 'learning rate for training(float)' --hidden_units 'hidden units for training(int)' 
####   --epochs 'epochs for training(int)' --gpu 'train on gpu'

<br/>

### EXAMPLE CALL:
```python train.py --train_dir '/dir' --arch vgg --lr 0.005 --hidden_units 1024 --epochs 30 --gpu cuda``` 

<br/>
<br/>

## PREDICTION (predict.py)

<br/>

### PURPOSE:
####   Reads in an image and a checkpoint to predict image class and prints the probability with class name  

<br/>

### EXPECTED USER INPUT:
####  python predict.py --arch 'vgg or densenet (since there is only one checkpoint file that this application generates 
####  based on the training model; choose the same model as of training to avoid mismatch)' 
####  --img_path 'complete path of image to be predicted' --lr 'learning rate used when training(float)' 
####  --hidden_units 'hidden units used when training(int)' --top_k 'required no. of top K classes(int)'
####  --print_k '1 prints a list of top_k; 0 prints only max probability' --json_file 'complete path to category names file'

<br/>

### CHECKPOINT FILE:
####   'model_checkpoint.pth', created by train.py 

<br/>

### EXAMPLE CALL   
```python predict.py --arch vgg --img_path /predict_img.jpg --lr 0.005 --hidden_units 1024 --top_k 3 --print_k 1```

<br/>

### Tested on 102 flower categories
### Model probability accuracy > 0.8
