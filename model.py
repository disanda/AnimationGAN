import torch
import torch.nn as nn
import loss_norm_gp

#---------------------------------第1版------------------------------
#kernel_size是4，stride是1-2-2-2-2, padding是0-1-1-1-1
class Generator_v1(nn.Module):
    def __init__(self,x_dim,dim=0):
        super().__init__()
        self.block1= nn.Sequential(
                nn.ConvTranspose2d(x_dim+c_dim,512,kernel_size=4,stride=1),
                nn.BatchNorm2d(512),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
                #nn.LeakyReLU()
            )
        self.block2= nn.Sequential(
                nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(256),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
                #nn.LeakyReLU()
            )
        self.block3= nn.Sequential(
                nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(128),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
                #nn.LeakyReLU()
            )
        self.block4= nn.Sequential(
                nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(64),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
                #nn.LeakyReLU()
            )
        self.convT=nn.ConvTranspose2d(64,  1,  kernel_size=4, stride=2, padding=1)
        self.tanh=nn.Tanh()
        #self.LRelu=nn.LeakyReLU()

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
    def __init__(self,x_dim,dim=0):
        super().__init__()
        self.conv1=nn.Conv2d(x_dim + c_dim, 64,kernel_size=4, stride=2, padding=1)#64->32
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
           y = self.lrelu(self.conv1(torch.cat([x, c], 1)))
        y = self.block1(y)#32->32
        y = self.block2(y)#32->32
        y2 = self.block3(y)#32->32
        y1 = self.conv2(y2)#32->29 :[-1,c,29,29]
        return y1,y2


#---------------------------------第1版_改------------------------------
#kernel_size是4，stride是1-2-2-2-2, padding是0-1-1-1-1
class Generator_v1_1(nn.Module):
    def __init__(self,x_dim,dim=0):
        super().__init__()
        self.block1= nn.Sequential(
                nn.ConvTranspose2d(x_dim+dim,512,kernel_size=4,stride=1),
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

class Discriminator_v1_1(nn.Module):
    def __init__(self,x_dim,dim=0):
        super().__init__()
        self.conv1=nn.Conv2d(x_dim + dim, 64,kernel_size=4, stride=2, padding=1)#out_dim:64
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
        y = self.block1(y)#32->32
        y = self.block2(y)#32->32
        y2 = self.block3(y)#32->32
        y1 = self.conv2(y2)#32->29 :[-1,c,29,29]
        return y1,y2

class Mow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(512, 256, 4, 2, 1, bias=False),#in:[-1,512,32,32]->[-1,256,16,16]
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, 4, 2, 1, bias=False),#in:[-1,256,16,16]->[-1,128,8,8]
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 64, 4, 2, 1, bias=False),#in:[-1,128,8,8]->[-1,64,4,4]
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 32, 4, 1, 0, bias=False),#in:[-1,64,4,4]->[-1,32,1,1]
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.block5 = nn.Conv2d(32,4,1)#out:[-1,4,1,1]
    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.block5(y)
        y = y.squeeze()#[-1,4]
        return y







#---------------------------------第2版------------------------------
class Generator_v2(nn.Module):
    def __init__(self,x_dim,c_dim=0):
        super().__init__()
        self.block1= nn.Sequential(
                nn.ConvTranspose2d(x_dim+c_dim,1024,kernel_size=2),#这里卷积核是2，保证可以1*1->2*2
                nn.BatchNorm2d(1024),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.block2= nn.Sequential(
                nn.ConvTranspose2d(1024,512,kernel_size=3),#这里卷积核是2，保证可以2*2-->4*4
                nn.BatchNorm2d(512),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.block3= nn.Sequential(
                nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),#4*4->8*8
                nn.BatchNorm2d(256),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.block4= nn.Sequential(
                nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
                nn.BatchNorm2d(128),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.block5= nn.Sequential(
                nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(64),#'batch_norm', 'instance_norm','spectral_norm', 'weight_norm'
                nn.ReLU()
            )
        self.convT=nn.ConvTranspose2d(64,  1,  kernel_size=3, stride=2, padding=1)
        self.tanh=nn.Tanh()

    def forward(self, z, c=False):
        # z: (N, z_dim), c: (N, c_dim) or bool
        if type(c) == type(False):
           y=z
        else:
           y = torch.cat([z, c], 1)
        y = y.view(y.size(0), y.size(1), 1, 1)#[-1,dim]->[-1,dim,1,1]
        y = self.block1(y) #1*1-->4*4,out_dim=512
        y = self.block2(y) # 4*4-->8*8
        y = self.block3(y) # 8*8-->16*16
        y = self.block4(y) # 16*16-->32*32
        y = self.block5(y)
        y = self.tanh(self.convT(y))# 32*32-->64*64
        return y

class Discriminator_v2(nn.Module):
    def __init__(self,x_dim,c_dim=0):
        super().__init__()
        self.conv1=nn.Conv2d(x_dim + c_dim, 64,kernel_size=3, stride=2, padding=1)#out_dim:64
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
        self.conv2=nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0)#out_dim:1
    def forward(self, x, c=False):
        # x: (N, x_dim, 32, 32), c: (N, c_dim) or bool
        if type(c)==type(False):
           y=x
        else:
           c = c.view(c.size(0), c.size(1), 1, 1) * torch.ones([c.size(0), c.size(1), x.size(2), x.size(3)], dtype=c.dtype, device=c.device)
           y = torch.cat([x, c], 1)
        y = self.lrelu(self.conv1(y))#out_dim:64
        y = self.block1(y)#out_dim:128
        y = self.block2(y)#out_dim:256
        y = self.block3(y)#out_dim:512
        y = self.conv2(y)#out_dim:1
        return y



#---------------------------------MSG------------------------------

from torch.nn.functional import interpolate
class Generator_msg(nn.Module):
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
                nn.Conv2d(32,1,3,padding=1),#如果是RGB就是3，灰度图就是1
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

class Discriminator_msg(nn.Module):
    def __init__(self, x_dim, c_dim=0):#x_dim=512
        super().__init__()
        self.block1=nn.Sequential(
                nn.Conv2d(x_dim+c_dim,32,4),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32,64,3,padding=1),
                nn.LeakyReLU(0.2)
            )#1*1->4*4
        self.block2=nn.Sequential(
                nn.Conv2d(64,64,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64,128,3,padding=1),
                nn.LeakyReLU(0.2)
            )#4*4->8*8
        self.block3=nn.Sequential(
                nn.Conv2d(128,128,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128,256,3,padding=1),
                nn.LeakyReLU(0.2)
            )
        self.block4=nn.Sequential(
                nn.Conv2d(256,256,3,padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256,512,3,padding=1),
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
            c = c.view(c.size(0), c.size(1), 1, 1) * torch.ones([c.size(0), c.size(1), z.size(2), z.size(3)], dtype=c.dtype, device=c.device)
            y = torch.cat([z, c], 1)
        y = self.block1(y)
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
class generator_info(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, z_dim=100, output_channel=1, input_size=32, len_discrete_code=10, len_continuous_code=2):
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
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),#[1024,128*8*8]-input_size=32
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input, cont_code, dist_code):
        x = torch.cat([input, cont_code, dist_code], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))#[-1,128,8,8]
        x = self.deconv(x)
        return x

class discriminator_info(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    # 输入是图片，输出是按照参数分为 [-1, output_dim] , [-1, len_continuous_code] , [-1 , len_continuous_code]
    def __init__(self, input_channel=1, output_dim=1, input_size=32, len_discrete_code=10, len_continuous_code=2):
        super().__init__()
        self.input_dim = input_channel
        self.output_dim = output_dim
        self.input_size = input_size
        self.len_discrete_code = len_discrete_code  # categorical distribution (i.e. label)
        self.len_continuous_code = len_continuous_code  # gaussian distribution (e.g. rotation, thickness)
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),#input_size/2
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),#input_size/4
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim + self.len_continuous_code + self.len_discrete_code),
            # nn.Sigmoid(),
        )
        loss_norm_gp.initialize_weights(self)
    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)
        a = torch.sigmoid(x[:, self.output_dim])
        b = x[:, self.output_dim:self.output_dim + self.len_continuous_code]
        c = x[:, self.output_dim + self.len_continuous_code:]
        return a, b, c





# 打印网络参数
# a = Generator_v1(100)
# print('parameters:', sum(param.numel() for param in a.parameters()))




