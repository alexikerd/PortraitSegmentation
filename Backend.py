import os
from os import path

from PIL import Image

import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader






class ImageDataset(Dataset):

    def __init__(self,target,test=False,transform=None):
        self.root = path.abspath(path.curdir)
        self.transform = transform
        self.traintest = 'test' if test else 'train'
        self.data = os.listdir(self.root + f'/data/{self.traintest}/')
        self.target = target
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,index):

        if torch.is_tensor(index):
            index = index.tolist()

        item = Image.open(self.root + f'/data/{self.traintest}/' + self.data[index])
        target = Image.open(self.root + f'/{self.target}/{self.traintest}/' + self.data[index].replace('data',self.target))
        
        if self.transform:
            item = self.transform(item)
            # target = self.transform(target)
            target = transforms.ToTensor()(target)*255

        
        return item, target




