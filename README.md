# Description
This is the official implementation of paper Landmark Detection with Learnable Connectivity Graph Convolutional Network

# Dependencies
Install these libraries with anaconda and pip
1. pytorch 1.8.0 cuda10.2
2. OpenCV 4.5.1
3. scikit-learn
4. easydict
5. tqdm
6. python 3.8
7. yacs
8. wandb

Training data is saved using [wandb](https://wandb.ai). Follow the instruction to create an account and login.
In case you don't want to log your training data to wandb, enable offline training option by using these command
in terminal
```bash
wandb offline
```
# Pretrained model and datasets
As 300W dataset is the combination of multiple datasets, we provide a single download link 
for convenience.

Download [WFLW dataset from here](https://wywu.github.io/projects/LAB/WFLW.html)

Download the pretrained weights and 300W dataset from [here](https://drive.google.com/drive/folders/178j9f_OA3TwUEDxX0k_z2Ri2DiVA2-Ij?usp=sharing)




# WFLW dataset
## Training
```bash
python scripts/train_wfw -i [image folder] --annotation [traning annotation file] --test_images [image folder] --test_annotation [test annotation file] --augmentation
```

## Evaluating
```bash
python scripts/evaluate_wflw.py -i [image folder] --annotation [test annotation file] --weights [pretrained weights]
```

## Single image prediction
```bash
python scripts/visualize_prediction.py -i temp/test/20.png --edge
```
Use "--edge" to visualize connections between landmarks

# 300W dataset
## Training
```bash
python python scripts/train_300w.py --annotation [dataset folder]
```

## Evaluating
```bash
python scripts/evaluate_300w.py -i [image folder] --annotation [test annotation file] --weights [pretrained weights]
```


# Acknowledgement
This repository reuse code from:
* HRNet Facial Landmark Detection: https://github.com/HRNet/HRNet-Facial-Landmark-Detection
* Objects as Points: https://github.com/xingyizhou/CenterNet

# LICENSE
Will be released under MIT License