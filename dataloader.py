import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import pandas as pd
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
random.seed(0)

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        self.df = pd.read_csv("nih_labels.csv")
        self.df = self.df[self.df['fold'] == mode]
        self.df = self.df.set_index("Image Index")
        self.PRED_LABEL = [
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
        
        self.transform = transform
        self.path_to_images = data_dir

    def __getitem__(self, idx):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')
        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                label[i] = self.df[self.PRED_LABEL[i].strip()
                                   ].iloc[idx].astype('int')
                
        if self.transform is not None:
            image = self.transform(image)
        
        return self.df.index[idx], image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.df)
    
    
class u_ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        self.df = pd.read_csv("u_labels.csv")
        self.df = self.df.set_index("Image Index")
        self.PRED_LABEL = [
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
        
        self.transform = transform
        self.path_to_images = data_dir

    def __getitem__(self, idx):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image = Image.open(os.path.join(self.df.index[idx]))
        image = image.convert('RGB')
        label = np.zeros(len(self.PRED_LABEL), dtype=float)
        
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('float') > 0):
                label[i] = self.df[self.PRED_LABEL[i].strip()
                                   ].iloc[idx].astype('float')
                
        if self.transform is not None:
            image = self.transform(image)
        
        return self.df.index[idx], image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.df)