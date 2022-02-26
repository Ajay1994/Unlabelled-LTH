# pytorch imports
import argparse
import os
import random
import shutil
import time
import sys
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataloader import ChestXrayDataSet
from torchvision import datasets, models, transforms

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, compute_AUCs
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
use_cuda = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))

from utils.misc import get_conv_zero_kernel
import argparse

transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASS_NAMES = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']
num_classes = len(CLASS_NAMES)
    

print("==> creating model ")
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
model = torch.nn.DataParallel(model)
model.cuda()

checkpoint = torch.load("results/0/model_best.pth.tar")
print(checkpoint.keys())
model.load_state_dict(checkpoint['state_dict'])


model.eval()
df = pd.read_csv("unlabelled_data.csv")
fopen = open("u_labels.csv", "w")
fopen.write("Image Index," + (",").join(CLASS_NAMES) + "\n")
for i in tqdm(range(0, len(df))):
    image = df.iloc[i, 0]
    image = Image.open(image)
    image = image.convert('RGB')
    image = transform_test(image)
    image = image.unsqueeze(0)
    output = model(image).detach().cpu().numpy()[0]
    output = [str(x) for x in output]
    fopen.write(df.iloc[i, 0]+ "," + (",").join(output) + "\n")
    fopen.flush()
    