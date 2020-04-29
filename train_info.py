import torch.optim as optim
import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
import itertools
import model
import argparse
from PIL import Image
import time

#-----------------------prepare of args-------------------
parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='experiment_name', default='InfoGAN_MNIST_v1_Supv')
args = parser.parse_args()


experiment_name = args.experiment_name
gpu_mode = True
SUPERVISED = True
batch_size = 64
z_dim_num = 100
c_d_num = 10
c_c_num = 2
input_dim = 112 # z =100 ,c_d =10 c_c = 2
input_size = 32
sample_num =100
epoch = 60


if not os.path.exists('./info_output/'):
    os.mkdir('./info_output/')

if not os.path.exists('./info_output/%s/'% experiment_name):
    os.mkdir('./info_output/%s/'% experiment_name)

save_dir = './info_output/%s/sample_training/' % experiment_name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#--------------------------data-----------------------
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(input_size, input_size), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),#Img2Tensor
     torchvision.transforms.Normalize(mean=[0.5], std=[0.5])# 取值范围(0,1)->(-1,1)
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)), #单通道改三通道
     #torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
     ]
)
train_loader = torch.utils.data.DataLoader(
    #dataset=torchvision.datasets.FashionMNIST('./data/', train=True, download=True, transform=transform),
    #dataset=torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform),
    dataset=torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=gpu_mode,
    drop_last=True
)

# fixed noise & condition
sample_z = torch.zeros((sample_num, z_dim_num))
for i in range(c_d_num):
	sample_z[i * c_d_num] = torch.rand(1, z_dim_num)#10
	for j in range(1, c_d_num):
		sample_z[i * c_d_num + j] = sample_z[i * c_d_num]#每10个的noize都相同
		temp = torch.zeros((c_d_num, 1))
for i in range(c_d_num):
	temp[i, 0] = i #每一个标签
	temp_d = torch.zeros((sample_num, 1))
for i in range(c_d_num):
		temp_d[i * c_d_num: (i + 1) * c_d_num] = temp #10个人的标签轮一遍
sample_d = torch.zeros((sample_num, c_d_num)).scatter_(1, temp_d.type(torch.LongTensor), 1)
sample_c = torch.zeros((sample_num, c_c_num))
# manipulating two continuous code
sample_z2 = torch.rand((1, z_dim_num)).expand(sample_num, z_dim_num) #[100,62],但是每个样本的noize相同
sample_d2 = torch.zeros(sample_num, c_d_num)#[100,10]
sample_d2[:, 0] = 1
temp_c = torch.linspace(-1, 1, 10)#10个-1->1的随机数
sample_c2 = torch.zeros((sample_num, 2))#[100,2]
for i in range(c_d_num):
	for j in range(c_d_num):
		sample_c2[i*c_d_num+j, 0] = temp_c[i]
		sample_c2[i*c_d_num+j, 1] = temp_c[j]
if gpu_mode == True:
	sample_z, sample_d, sample_c, sample_z2, sample_d2, sample_c2 = \
	sample_z.cuda(), sample_d.cuda(), sample_c.cuda(), \
	sample_z2.cuda(), sample_d2.cuda(), sample_c2.cuda()

#------------------------model setting-----------------

G = model.generator_info(z_dim=z_dim_num, output_dim=1, input_size=input_size, len_discrete_code=c_d_num, len_continuous_code=c_c_num)
D = model.discriminator_info(input_dim=1, output_dim=1, input_size=input_size, len_discrete_code=c_d_num, len_continuous_code=c_c_num)
G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
info_optimizer = optim.Adam(itertools.chain(G.parameters(), D.parameters()), lr=0.0002, betas=(0.5, 0.9))#G,D都更新
d_real_flag, d_fake_flag = torch.ones(batch_size, 1), torch.zeros(batch_size, 1)
if gpu_mode == True:
    d_real_flag, d_fake_flag = d_real_flag.cuda(), d_fake_flag.cuda()

if gpu_mode:
	G.cuda()
	D.cuda()
	BCE_loss = nn.BCELoss().cuda()
	CE_loss = nn.CrossEntropyLoss().cuda()
	MSE_loss = nn.MSELoss().cuda()
else:
	BCE_loss = nn.BCELoss()
	CE_loss = nn.CrossEntropyLoss()
	MSE_loss = nn.MSELoss()

train_hist = {}
train_hist['D_loss'] = []
train_hist['G_loss'] = []
train_hist['info_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []


#------------------train----------------------
D.train()
print('training start!!')
start_time = time.time()
for i in range(epoch):
	G.train()
	epoch_start_time = time.time()
	for j, (y, c_d) in enumerate(train_loader):
		z = torch.rand((batch_size, z_dim_num))
		if SUPERVISED == True:
			c_d = torch.zeros((batch_size, c_d_num)).scatter_(1, c_d.type(torch.LongTensor).unsqueeze(1), 1)
		else:
			c_d = torch.from_numpy(np.random.multinomial(1, c_d_num * [float(1.0 / c_d_num)],size=[batch_size])).type(torch.FloatTensor)#投骰子函数,随机化y_disc_
		c_c = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, 2))).type(torch.FloatTensor)
		if gpu_mode:
			y, z, c_d, c_c = y.cuda(), z.cuda(), c_d.cuda(), c_c.cuda()
# update D network
		D_optimizer.zero_grad()
		D_real, _, _ = D(y)
		D_real_loss = BCE_loss(D_real, d_real_flag)
		y_f = G(z, c_c, c_d)
		D_fake, _, _ = D(y_f)
		D_fake_loss = BCE_loss(D_fake, d_fake_flag)
		D_loss = D_real_loss + D_fake_loss
		train_hist['D_loss'].append(D_loss.item())
		D_loss.backward(retain_graph=True)
		D_optimizer.step()
# update G network
		G_optimizer.zero_grad()
		y_f = G(z, c_c, c_d)
		D_fake, D_cont, D_disc = D(y_f)
		G_loss = BCE_loss(D_fake, d_real_flag)
		train_hist['G_loss'].append(G_loss.item())
		G_loss.backward(retain_graph=True)
		G_optimizer.step()
# information loss
		disc_loss = CE_loss(D_disc, torch.max(c_d, 1)[1])
		cont_loss = MSE_loss(D_cont, c_c)
		info_loss = disc_loss + cont_loss
		train_hist['info_loss'].append(info_loss.item())
		info_loss.backward()
		info_optimizer.step()
		if ((j + 1) % 100) == 0:
			print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, info_loss: %.8f" %((i + 1), (j + 1), train_loader.dataset.__len__() // batch_size, D_loss.item(), G_loss.item(), info_loss.item()))
		train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
# save2img
	with torch.no_grad():
		G.eval()
		image_frame_dim = int(np.floor(np.sqrt(sample_num)))
		samples = G(sample_z, sample_c, sample_d)
		samples = (samples + 1) / 2
		torchvision.utils.save_image(samples, save_dir+'/%d_Epoch—c_d.png' % epoch, nrow=10)
		samples = G(sample_z2, sample_c2, sample_d2)
		samples = (samples + 1) / 2
		torchvision.utils.save_image(samples, save_dir + '/%d_Epoch-c_c.png' % epoch, nrow=10)

# others
ckpt_dir = './info_output/%s/checkpoints' % experiment_name
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
torch.save({'epoch': ep + 1,'G': G.state_dict()},'%s/Epoch_(%d).ckpt' % (ckpt_dir, ep + 1))



        # self.train_hist['total_time'].append(time.time() - start_time)
        # print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),self.i, self.train_hist['total_time'][0]))
        # print("Training finish!... save training results")
        # self.save()
        # self.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)