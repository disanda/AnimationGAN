# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import loss_norm_gp
#-----------------MWM-GAN-v1--------------------多一个网络Q输出C即可
class generator_mwm(nn.Module):
    def __init__(self, z_dim=100, output_channel=1, input_size=64, len_discrete_code=10, len_continuous_code=2):
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
            nn.Linear(1024, 128 * (self.input_size // 8) * (self.input_size // 8)),#[1024,128*8*8]-input_size=32
            nn.BatchNorm1d(128 * (self.input_size // 8) * (self.input_size // 8)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input, dist_code, cont_code):
        x = torch.cat([input, dist_code, cont_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 8), (self.input_size // 8))#[-1,128,8,8]
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
            nn.Conv2d(self.input_dim, 32, 4, 2, 1),#input_size/2
            #nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),#input_size/4
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),#input_size/8
            #nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 8) * (self.input_size // 8), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            # nn.Sigmoid(),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 8) * (self.input_size // 8))
        x = self.fc(x)
        a = torch.sigmoid(x[:, self.output_dim])
        b = x[:, self.output_dim:self.output_dim + self.len_discrete_code]
        c = x[:, self.output_dim + self.len_discrete_code:]
        return a, b, c


