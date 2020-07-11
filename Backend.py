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






class T_ImageDataset(Dataset):

    def __init__(self,test=False,transform=None):
        self.root = path.abspath(path.curdir)
        self.transform = transform
        self.traintest = 'test' if test else 'train'
        self.data = os.listdir(self.root + f'/data/{self.traintest}/')
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self,index):

        if torch.is_tensor(index):
            index = index.tolist()

        item = Image.open(self.root + f'/data/{self.traintest}/' + self.data[index])
        target = Image.open(self.root + f'/trimap/{self.traintest}/' + self.data[index].replace("data","trimap"))
        
        if self.transform:
            item = self.transform(item)
            target = transforms.ToTensor()(target)*255

        
        return item, target






class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat((x1,x2),1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256,256)
        self.up1 = Up(512, 128, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        


    def forward(self, x):
        
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        
        if logits.shape[1]>1:
            logits = F.softmax(logits,dim=1)
            
        else:
            logits = torch.sigmoid(logits)

        return logits


class CompositionalLoss(nn.Module):
    
    
    def __init__(self) -> None:
        super(CompositionalLoss, self).__init__()
        self.epsilon: float = 1e-6
        self.dims = (1,2,3) 
        
    def forward(self,output,target):
        
       
        loss = torch.pow(torch.pow(output - target,2) + self.epsilon,0.5)
        
        return torch.mean(loss)


class GluNet(nn.Module):
    def __init__(self, bilinear=True):
        super(GluNet, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256,64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64,3)
        
        self.inc1 = DoubleConv(6, 64)
        self.down1_1 = Down(64, 128)
        self.down2_1 = Down(128, 256)
        self.down3_1 = Down(256,512)
        self.down4_1 = Down(512,512)
        self.up1_1 = Up(1024, 256, bilinear)
        self.up2_1 = Up(512, 128, bilinear)
        self.up3_1 = Up(256,64, bilinear)
        self.up4_1 = Up(128, 64, bilinear)
        self.outc1 = OutConv(64,1)
        


    def forward(self, image):
        
        
        
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x4,x5)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        
        trimap = F.softmax(self.outc(x),dim=1)
        x = torch.cat((image,trimap),1)


        x1 = self.inc1(x)
        x2 = self.down1_1(x1)
        x3 = self.down2_1(x2)
        x4 = self.down3_1(x3)
        x5 = self.down4_1(x4)
        x = self.up1_1(x4,x5)
        x = self.up2_1(x,x3)
        x = self.up3_1(x,x2)
        x = self.up4_1(x,x1)

        matte = torch.sigmoid(self.outc1(x))

        return trimap, matte




class Cropper():
    
    def __init__(self):
        
        self.transform = transforms.Compose([transforms.ColorJitter(brightness=0.75),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.root = path.abspath(path.curdir)        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.imgsize = (600,800)
        
        self.TNet = UNet(3,3)
        self.TNet.load_state_dict(torch.load(self.root + '/Models/Versions/COMP/TNET COMP v4'))
#         self.TNet.to(self.device)
        
        self.MNet = UNet(6,1)
        self.MNet.load_state_dict(torch.load(self.root + '/Models/Versions/COMP/MNET COMP v4'))
#         self.MNet.to(self.device)
        
        
    def __call__(self,image):
        
        self.TNet.to(self.device)
        
        image = image.resize(self.imgsize)
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            trimap = self.TNet(image)
            
        self.TNet.cpu()
        self.MNet.to(self.device)
            
        composite = torch.cat((image,trimap),1)
        
        with torch.no_grad():
            matte = self.MNet(composite).cpu().squeeze().numpy()
        
        self.MNet.cpu()
        
        trimap = trimap.squeeze().cpu().numpy()
        
        fusion = trimap[0] + trimap[1]*matte
        
        return matte, fusion