import torch
import torch.nn as nn
import loss_norm_gp

#---------------------------------第1版------------------------------
class Generator_v1(nn.Module):
    def __init__(self,in_dim,c_dim=0):
        super().__init__()
        self.block1= nn.Sequential(
                nn.ConvTranspose2d(in_dim+c_dim,512,kernel_size=4,stride=1,padding=0),
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
        self.convT=nn.ConvTranspose2d(64,  3,  kernel_size=4, stride=2, padding=1)
        self.tanh=nn.Tanh()

    def forward(self, z, c=0):
        # z: (N, z_dim), c: (N, c_dim)
        if c == 0:
           y=z
        else:
           y = self.torch.cat([z, c], 1)
        y = self.block1(y.view(y.size(0), x.size(1), 1, 1)) #1*1-->4*4,out_dim=512
        y = self.block2(y) # 4*4-->8*8
        y = self.block3(y) # 8*8-->16*16
        y = self.block4(y) # 16*16-->32*32
        y = self.tanh(self.convT(y))
        return y

class Discriminator_v1(nn.Module):
    def __init__(self,x_dim,c_dim=0):
        super().__init__()
        self.conv1=nn.Conv2d(x_dim + c_dim, 64,kernel_size=4, stride=2, padding=1)#out_dim:64
        self.lrelu=nn.LeakyReLU(0.2)
        self.block1=nn.Sequential(
                nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2)
            )
        self.block2=nn.Sequential(
                nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2)
            )
        self.block3=nn.Sequential(
                nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2)
            )
        self.conv2=nn.Conv2d(dim*8, 1, kernel_size=4, stride=1, padding=0)#out_dim:1
    def forward(self, x, c=0):
        # x: (N, x_dim, 32, 32), c: (N, c_dim)
        if c==0:
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


#---------------------------------第2版------------------------------

from torch.nn.functional import interpolate
class Generator_v2(nn.Module):
    def __init__(self, z_dim, c_dim):
        super().__init__()
        self.block=nn.Sequential(
                nn.Conv2d(in_dim,out_dim,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_dim,out_dim,3,padding=1),
                nn.LeakyReLU(0.2)
            )
        self.convT1 = nn.ConvTranspose2d(z_dim+c_dim,z_dim+c_dim,4)
        self.lrelu = nn.LeakyReLU(0.2) 

        layers = []
        layers.append(ConvTranspose2d(z_dim+c_dim,z_dim+c_dim,4)) #padding =0
        layers.append(nn.LeakyReLU(0.2))
        layers.append(Conv2d(z_dim+c_dim,z_dim+c_dim,(3, 3), padding=1))
        layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
    def forward(self, z, c):
        # z: (N, z_dim), c: (N, c_dim)
        x = torch.cat([z, c], 1)
        x = self.net(x.view(x.size(0), x.size(1), 1, 1))
        return x





