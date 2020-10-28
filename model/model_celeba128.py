# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import loss_norm_gp
import torch.nn.utils.spectral_norm as spectral_norm

#128分辨率输出

#-----------------MWM-GAN-v1--------------------多一个网络Q输出C即可
class generator_mwm(nn.Module):
    def __init__(self, z_dim=200, output_channel=1, input_size=128, len_discrete_code=28, len_continuous_code=28):
        super().__init__()
        self.z_dim = z_dim
        self.output_dim = output_channel
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + self.len_discrete_code + self.len_continuous_code, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256*4*4),#[1024,64*64=4096]
            nn.BatchNorm1d(256*4*4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 2048, 4, 2, 1,bias=False), 
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1,bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512,  256,  4, 2, 1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 4, 2, 1,bias=False),
            nn.Tanh(),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input, dist_code, cont_code):
        x = torch.cat([input, dist_code, cont_code], 1)
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)#[-1,128,8,8]
        x = self.deconv(x)
        return x

class discriminator_mwm(nn.Module):
    # 输入是图片，输出是按照参数分为 [-1, output_dim] , [-1, len_continuous_code] , [-1 , len_continuous_code]
    def __init__(self, input_channel=1, output_dim=1, input_size=64, len_discrete_code=10, len_continuous_code=2):
        super().__init__()
        self.input_dim = input_channel
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        self.conv = nn.Sequential(
            spectral_nrom(nn.Conv2d(self.input_dim, 256, 4, 2, 1, bias = False)),#input_size/2
            nn.LeakyReLU(0.2),
            spectral_nrom(nn.Conv2d(256, 512, 4, 2, 1, bias = False)),
            nn.LeakyReLU(0.2),
            spectral_nrom(nn.Conv2d(512, 1024, 4, 2, 1, bias = False)),
            nn.LeakyReLU(0.2),
            spectral_nrom(nn.Conv2d(1024, 2048, 4, 2, 1, bias = False)),
            nn.LeakyReLU(0.2),             
            nn.Conv2d(2048, 256, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256*4*4, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 256*4*4) 
        x = self.fc(x)
        a = torch.sigmoid(x[:, self.output_dim])
        b = x[:, self.output_dim:self.output_dim + self.len_discrete_code]
        c = x[:, self.output_dim + self.len_discrete_code:]
        return a, b, c


