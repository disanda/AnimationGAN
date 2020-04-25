import torch
import torch.nn as nn
import loss_norm_gp

#---------------------------------第1版------------------------------
#kernel_size是4，stride是1-2-2-2-2, padding是0-1-1-1-1
class Generator_v1(nn.Module):
    def __init__(self,x_dim,c_dim=0):
        super().__init__()
        self.block1= nn.Sequential(
                nn.ConvTranspose2d(x_dim+c_dim,512,kernel_size=4,stride=1,padding=0),
                nn.BatchNorm2d(512),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.block2= nn.Sequential(
                nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(256),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.block3= nn.Sequential(
                nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(128),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.block4= nn.Sequential(
                nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(64),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.convT=nn.ConvTranspose2d(64,  1,  kernel_size=4, stride=2, padding=1)
        self.tanh=nn.Tanh()

    def forward(self, z, c=False):
        # z: (N, z_dim), c: (N, c_dim) or bool
        if type(c) == type(False):
           y=z
        else:
           y = torch.cat([z, c], 1)
        y = self.block1(y.view(y.size(0), y.size(1), 1, 1)) #1*1-->4*4,out_dim=512
        y = self.block2(y) # 4*4-->8*8
        y = self.block3(y) # 8*8-->16*16
        y = self.block4(y) # 16*16-->32*32
        y = self.tanh(self.convT(y))# 32*32-->64*64
        return y

class Discriminator_v1(nn.Module):
    def __init__(self,x_dim,c_dim=0):
        super().__init__()
        self.conv1=nn.Conv2d(x_dim + c_dim, 64,kernel_size=4, stride=2, padding=1)#out_dim:64
        self.lrelu=nn.LeakyReLU(0.2)
        self.block1=nn.Sequential(
                nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            )
        self.block2=nn.Sequential(
                nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)
            )
        self.block3=nn.Sequential(
                nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2)
            )
        self.conv2=nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)#out_dim:1
    def forward(self, x, c=False):
        # x: (N, x_dim, 32, 32), c: (N, c_dim) or bool
        if type(c)==type(False):
           y=x
           y = self.lrelu(self.conv1(x))#out_dim:64
        else:
           c = c.view(c.size(0), c.size(1), 1, 1) * torch.ones([c.size(0), c.size(1), x.size(2), x.size(3)], dtype=c.dtype, device=c.device)
           y = self.lrelu(self.conv1(torch.cat([x, c], 1)))#out_dim:64
        y = self.block1(y)#out_dim:128
        y = self.block2(y)#out_dim:256
        y = self.block3(y)#out_dim:512
        y = self.conv2(y)#out_dim:1
        return y

#---------------------------------MSG------------------------------

from torch.nn.functional import interpolate
class Generator_v2(nn.Module):
    def __init__(self, x_dim, c_dim=0):#x_dim=512
        super().__init__()
        dim = x_dim+c_dim
        self.block1=nn.Sequential(
                nn.ConvTranspose2d(x_dim+c_dim,512,4),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512,256,3,padding=1),
                nn.LeakyReLU(0.2)
            )#1*1->4*4
        self.block2=nn.Sequential(
                nn.Conv2d(256,256,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256,128,3,padding=1),
                nn.LeakyReLU(0.2)
            )#4*4->8*8
        self.block3=nn.Sequential(
                nn.Conv2d(128,128,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128,64,3,padding=1),
                nn.LeakyReLU(0.2)
            )
        self.block4=nn.Sequential(
                nn.Conv2d(64,64,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64,32,3,padding=1),
                nn.LeakyReLU(0.2)
            )
        self.block5=nn.Sequential(
                nn.Conv2d(32,32,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32,3,3,padding=1),
                nn.LeakyReLU(0.2)
            )
    def forward(self, z, c=False):
        # z: (N, z_dim), c: (N, c_dim)
        if type(c) ==type(False) :
            y=z
        else:
            y = torch.cat([z, c], 1)
        y = self.block1(y.view(y.size(0), y.size(1), 1, 1))
        y = interpolate(y, scale_factor=2)#4->8
        y = self.block2(y)
        y = interpolate(y, scale_factor=2)#8->16
        y = self.block3(y)
        y = interpolate(y, scale_factor=2)#16->32
        y = self.block4(y)
        y = interpolate(y, scale_factor=2)#32->64
        y = self.block5(y)
        return y

class Discriminator_v2(nn.Module):
    def __init__(self, x_dim, c_dim=0):#x_dim=512
        super().__init__()
        self.block1=nn.Sequential(
                nn.Conv2d(x_dim+c_dim,32,4),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32,32,3,padding=1),
                nn.LeakyReLU(0.2)
            )#1*1->4*4
        self.block2=nn.Sequential(
                nn.Conv2d(64,64,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64,64,3,padding=1),
                nn.LeakyReLU(0.2)
            )#4*4->8*8
        self.block3=nn.Sequential(
                nn.Conv2d(128,128,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128,128,3,padding=1),
                nn.LeakyReLU(0.2)
            )
        self.block4=nn.Sequential(
                nn.Conv2d(256,256,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256,256,3,padding=1),
                nn.LeakyReLU(0.2)
            )
        self.block5=nn.Sequential(
                nn.Conv2d(512,512,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512,512,3,padding=1),
                nn.LeakyReLU(0.2)
            )
        self.downSampler = nn.AvgPool2d(2)
        self.fc = nn.Conv2d(512, 1, 1)
    def forward(self, z, c=False):
        # z: (N, z_dim), c: (N, c_dim)
        if type(c) == type(False) :
            y=z
        else:
            y = torch.cat([z, c], 1)
        y = self.block1(y.view(y.size(0), y.size(1), 1, 1))
        y = self.downSampler(y)#64->32
        y = self.block2(y)
        y = self.downSampler(y)#32->16
        y = self.block3(y)
        y = self.downSampler(y)#16->8
        y = self.block4(y)
        y = self.downSampler(y)#8->4
        y = self.block5(y)
        y = self.fc(y)#4->1
        #y = y.view(-1)
        return y

#-----------------infoGAN--------------------多一个网络Q输出C即可
class Generator_v3(nn.Module):
    def __init__(self,x_dim,c_dim=0):
        super().__init__()
        self.block1= nn.Sequential(
                nn.ConvTranspose2d(x_dim+c_dim,512,kernel_size=4,stride=1,padding=0),
                nn.BatchNorm2d(512),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.block2= nn.Sequential(
                nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(256),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.block3= nn.Sequential(
                nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(128),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.block4= nn.Sequential(
                nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(64),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.convT=nn.ConvTranspose2d(64,  1,  kernel_size=4, stride=2, padding=1)
        self.tanh=nn.Tanh()

    def forward(self, z, c=False):
        # z: (N, z_dim), c: (N, c_dim) or bool
        if type(c) == type(False):
           y=z
        else:
           y = torch.cat([z, c], 1)
        y = self.block1(y.view(y.size(0), y.size(1), 1, 1)) #1*1-->4*4,out_dim=512
        y = self.block2(y) # 4*4-->8*8
        y = self.block3(y) # 8*8-->16*16
        y = self.block4(y) # 16*16-->32*32
        y = self.tanh(self.convT(y))# 32*32-->64*64
        return y

class Discriminator_v3(nn.Module):
    def __init__(self,x_dim,c_dim=0):
        super().__init__()
        self.conv1=nn.Conv2d(x_dim + c_dim, 64,kernel_size=4, stride=2, padding=1)#out_dim:64
        self.lrelu=nn.LeakyReLU(0.2)
        self.block1=nn.Sequential(
                nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            )
        self.block2=nn.Sequential(
                nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)
            )
        self.block3=nn.Sequential(
                nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2)
            )
        self.conv2=nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)#out_dim:1
    def forward(self, x, c=False):
        # x: (N, x_dim, 32, 32), c: (N, c_dim) or bool
        if type(c)==type(False):
           y=x
           y = self.lrelu(self.conv1(x))#out_dim:64
        else:
           c = c.view(c.size(0), c.size(1), 1, 1) * torch.ones([c.size(0), c.size(1), x.size(2), x.size(3)], dtype=c.dtype, device=c.device)
           y = self.lrelu(self.conv1(torch.cat([x, c], 1)))#out_dim:64
        y = self.block1(y)#out_dim:128
        y = self.block2(y)#out_dim:256
        y = self.block3(y)#out_dim:512
        y = self.conv2(y)#out_dim:1
        return y

class SelfSupervisedNet(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.conv1=nn.Conv2d(x_dim + c_dim, 64,kernel_size=4, stride=2, padding=1)#out_dim:64
        self.lrelu=nn.LeakyReLU(0.2)
        self.block1=nn.Sequential(
                nn.Conv2d(512,256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            )
        self.block2=nn.Sequential(
                nn.Conv2d(256,dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)
            )
    def forward(self,x):
        y = self.block1(x)
        y = self.block2(y)
        return y #放回各路c




# 打印网络参数
# a = Generator_v1(100)
# print('parameters:', sum(param.numel() for param in a.parameters()))




