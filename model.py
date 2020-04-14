import torch
import torch.nn as nn
import loss_norm_gp

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super().__init__()
        def dconv_bn_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1, output_padding=0):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        layers = []
        layers.append(dconv_bn_relu(z_dim + c_dim, 512, kernel_size=4, stride=1, padding=0)) # 1*1-->4*4,out_dim=512
        layers.append(dconv_bn_relu(512, 256, kernel_size=4, stride=2, padding=1)) # 4*4-->8*8
        layers.append(dconv_bn_relu(256, 128, kernel_size=4, stride=2, padding=1)) # 8*8-->16*16
        layers.append(dconv_bn_relu(128, 64,  kernel_size=4, stride=2, padding=1)) # -->32*32
        layers.append(nn.ConvTranspose2d(64,  3,  kernel_size=4, stride=2, padding=1)) #-->64*64
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, z, c):
        # z: (N, z_dim), c: (N, c_dim)
        x = torch.cat([z, c], 1)
        x = self.net(x.view(x.size(0), x.size(1), 1, 1))
        return x

class Discriminator_v1(nn.Module):
    def __init__(self, x_dim, c_dim, dim=64):
        super(DiscriminatorCGAN, self).__init__()
        norm_fn = _get_norm_fn_2d(norm)
        weight_norm_fn = _get_weight_norm_fn(weight_norm)
        def conv_norm_lrelu(in_dim, out_dim, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                nn.BatchNorm2d(out_dim)
                nn.LeakyReLU(0.2)
            )
        layers = []
        #downsamplings
        layers.append(nn.Conv2d(x_dim + c_dim, dim,kernel_size=4, stride=2, padding=1))#out_dim:64
        layers.append(nn.LeakyReLU(0.2)) 
        #logits
        layers.append(conv_norm_lrelu(dim,dim*2,kernel_size=4, stride=2, padding=1))#out_dim:128
        layers.append(conv_norm_lrelu(dim*2,dim*4,kernel_size=4, stride=2, padding=1))#out_dim:256
        layers.append(conv_norm_lrelu(dim*4,dim*8,kernel_size=4, stride=2, padding=1))#out_dim:512
        layers.append(nn.Conv2d(dim*8, 1, kernel_size=4, stride=1, padding=0))#out_dim:1

    def forward(self, x, c):
        # x: (N, x_dim, 32, 32), c: (N, c_dim)
        c = c.view(c.size(0), c.size(1), 1, 1) * torch.ones([c.size(0), c.size(1), x.size(2), x.size(3)], dtype=c.dtype, device=c.device)
        logit = self.ls(torch.cat([x, c], 1))
        return logit




