import torch.optim as optim
import torchvision
import torch
import torch.nn as nn
import os
import numpy as np
import itertools
import model_v3 as model
import argparse
from PIL import Image
import time
import utils
import tqdm
import random
import loss_norm_gp
import functools
#-----------------------prepare of args-------------------
parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='experiment_name', default='3dmmnist_wmw+_cd20_cc20_lamb20_fc3_v2')
args = parser.parse_args()



gpu_mode = True
#SUPERVISED = True
SUPERVISED = False
batch_size = 64
z_dim_num = 32
c_d_num = 20
c_c_num = 20
#input_dim: z =100 ,c_d =10 c_c = 2
input_size = 64
img_channel = 1
sample_num =400
epoch = 150
gp_mode = 'epoch150'
experiment_name = args.experiment_name+'_'+gp_mode

if not os.path.exists('./info_output/'):
    os.mkdir('./info_output/')

save_root='./info_output/%s/'%experiment_name
if not os.path.exists('./info_output/%s/'% experiment_name):
    os.mkdir('./info_output/%s/'% experiment_name)


save_dir = './info_output/%s/sample_training/' % experiment_name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

ckpt_dir = './info_output/%s/checkpoints' % experiment_name
if not os.path.exists(ckpt_dir):
	os.mkdir(ckpt_dir)



#--------------------------data-----------------------
# transform = torchvision.transforms.Compose(
#     [torchvision.transforms.Resize(size=(input_size, input_size), interpolation=Image.BICUBIC),
#      torchvision.transforms.ToTensor(),#Img2Tensor
#      torchvision.transforms.Normalize(mean=[0.5], std=[0.5])# 取值范围(0,1)->(-1,1)
#      #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)), #单通道改三通道
#      #torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

#      ]
# )
# train_loader = torch.utils.data.DataLoader(
#     #dataset=torchvision.datasets.FashionMNIST('./data/', train=True, download=True, transform=transform),
#     #dataset=torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform),
#     dataset=torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transform),
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=gpu_mode,
#     drop_last=True
# )

#celeba
# transform = torchvision.transforms.Compose([
#         torchvision.transforms.CenterCrop(160),
#         torchvision.transforms.Resize(64),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# data_dir = '/_yucheng/dataSet/celeba/'  # this path depends on your computer
# train_loader =  utils.load_celebA(data_dir, transform, batch_size, shuffle=True)

#face_3d
# transform = torchvision.transforms.Compose([
#         #torchvision.transforms.CenterCrop(160),
#         torchvision.transforms.Resize((64,64)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# path = '/_yucheng/dataSet/face3d//face3d'
# face3d_dataset = torchvision.datasets.ImageFolder(path, transform=transform)
# train_loader = torch.utils.data.DataLoader(face3d_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

#-------------moving-mnist--------------
train_set = utils.MovingMNIST(train=True,transform=torchvision.transforms.Normalize(mean=[127.5], std=[127.5]))#[0,255]->[-1,1]
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True,
                 drop_last=True
                 )


#nemo
# transform = torchvision.transforms.Compose([
#         #torchvision.transforms.CenterCrop(160),
#         torchvision.transforms.Resize((64,64)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# path = '/_yucheng/dataSet/nemo/nemo'
# face3d_dataset = torchvision.datasets.ImageFolder(path, transform=transform)
# train_loader = torch.utils.data.DataLoader(face3d_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

#------------ moco_actions----moco_shapes --------------
# transform = torchvision.transforms.Compose([
#         #torchvision.transforms.CenterCrop(160),
#         torchvision.transforms.Resize((64,64)),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# path = '/_yucheng/dataSet/moco/moco_actions/'
# #path = '/_yucheng/dataSet/moco/moco_shapes/'
# face3d_dataset = torchvision.datasets.ImageFolder(path, transform=transform)
# train_loader = torch.utils.data.DataLoader(face3d_dataset, batch_size=batch_size, shuffle=True,drop_last=True)



# 固定noise和cc，每c_d个变一次c_d
sample_z = torch.zeros((sample_num, z_dim_num))
temp = torch.zeros((c_d_num, 1))
for i in range(sample_num//c_d_num):
	sample_z[i * c_d_num] = torch.rand(1, z_dim_num)#为连续c_d个的样本赋值。
	for j in range(c_d_num):
		sample_z[i * c_d_num + j] = sample_z[i * c_d_num]#相同c_d的noize都相同

for i in range(c_d_num):
	temp[i, 0] = i #每一个标签
temp_d = torch.zeros((sample_num, 1))
for i in range(sample_num//c_d_num):
	temp_d[i * c_d_num: (i + 1) * c_d_num] = temp[i%c_d_num] #每c_d个的d一样
sample_d = torch.zeros((sample_num, c_d_num)).scatter_(1, temp_d.type(torch.LongTensor), 1)
sample_c = torch.zeros((sample_num, c_c_num))

# 观察单一变量，固定其他变量
sample_z2 = torch.rand((1, z_dim_num)).expand(sample_num, z_dim_num) #每个样本的noize相同
sample_d2 = torch.zeros(sample_num, c_d_num)#[200,c_d]

temp_c = torch.linspace(-1, 1, c_d_num)		#c_d_num个范围在-1->1的等差数列
sample_c2 = torch.zeros((sample_num, c_c_num))#[200,c_c]

for i in range(sample_num//c_d_num):		#每c_d个noise,c_d相同,c_c不同
	#d_label = random.randint(0,c_d_num-1)
	d_label = i%c_d_num
	sample_d2[i*c_d_num:(i+1)*c_d_num, d_label] = 1
	sample_c2[i*c_d_num:(i+1)*c_d_num,i%c_c_num] = temp_c

#gpu
if gpu_mode == True:
	sample_z, sample_d, sample_c, sample_z2, sample_d2, sample_c2 = \
	sample_z.cuda(), sample_d.cuda(), sample_c.cuda(), \
	sample_z2.cuda(), sample_d2.cuda(), sample_c2.cuda()

#------------------------model setting-----------------

G = model.generator_mwm(z_dim=z_dim_num, output_channel=img_channel, input_size=input_size, len_discrete_code=c_d_num, len_continuous_code=c_c_num)  
D = model.discriminator_mwm(input_channel=img_channel, output_dim=1, input_size=input_size, len_discrete_code=c_d_num, len_continuous_code=c_c_num)
G_optimizer = optim.Adam(G.parameters(),  betas=(0.5, 0.99),amsgrad=True)
D_optimizer = optim.Adam(D.parameters(), lr=0.0002,betas=(0.5, 0.99),amsgrad=True)
info_optimizer = optim.Adam(itertools.chain(G.parameters(), D.parameters()),lr=0.0001,betas=(0.6, 0.95),amsgrad=True)#G,D都更新
d_real_flag, d_fake_flag = torch.ones(batch_size), torch.zeros(batch_size)

with open(save_root+'setting.txt', 'w') as f:
	print('----',file=f)
	print(G,file=f)
	print('----',file=f)
	print(D,file=f)
	print('----',file=f)
	print('----',file=f)
	print(G_optimizer,file=f)
	print('----',file=f)
	print(D_optimizer,file=f)
	print('----',file=f)
	print(info_optimizer,file=f)
	print('----',file=f)


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

d_loss_fn, g_loss_fn = loss_norm_gp.get_losses_fn('wgan')


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
for i in tqdm.trange(epoch):
	G.train()
	epoch_start_time = time.time()
	j=0
	for y, c_d_true in tqdm.tqdm(train_loader):
		j = j + 1
		z = torch.rand((batch_size, z_dim_num))
		if SUPERVISED == True:
			c_d = torch.zeros((batch_size, c_d_num)).scatter_(1, c_d_true.type(torch.LongTensor).unsqueeze(1), 1)
		else:
			c_d = torch.from_numpy(np.random.multinomial(1, c_d_num * [float(1.0 / c_d_num)],size=[batch_size])).type(torch.FloatTensor)#投骰子函数,随机化y_disc_
		c_c = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, c_c_num))).type(torch.FloatTensor)
		if gpu_mode:
			y, z, c_d, c_c = y.cuda(), z.cuda(), c_d.cuda(), c_c.cuda()
# update D network
		D_optimizer.zero_grad()
		y_f = G(z, c_c, c_d)
		D_real, _, _ = D(y)
		D_fake, _, _ = D(y_f)
		D_real_loss = BCE_loss(D_real, d_real_flag)#1
		D_fake_loss = BCE_loss(D_fake, d_fake_flag)#0
		#D_real_loss, D_fake_loss = d_loss_fn(D_real, D_fake)
		#gp = loss_norm_gp.gradient_penalty(D, y, y_f, mode=gp_mode)
		#gp = loss_norm_gp.gradient_penalty(functools.partial(D), y, y_f, gp_mode='0-gp', sample_mode='line')
		gp=0
		D_loss = D_real_loss + D_fake_loss + gp
		train_hist['D_loss'].append(D_loss.item())
		D_loss.backward(retain_graph=True)
		D_optimizer.step()
# update G network
		G_optimizer.zero_grad()
		y_f = G(z, c_c, c_d)
		D_fake,D_disc,D_cont = D(y_f)
		G_loss = BCE_loss(D_fake, d_real_flag)
		#G_loss = g_loss_fn(D_fake)
		train_hist['G_loss'].append(G_loss.item())
		G_loss.backward(retain_graph=True)
		G_optimizer.step()
# information loss
		D_optimizer.zero_grad() #这两个网络不清零，梯度就会乱掉,训练失败
		G_optimizer.zero_grad()
		y_info = G(z, c_c, c_d)
		_,D_disc_info,D_cont_info = D(y_info)
		disc_loss = CE_loss(D_disc_info, torch.max(c_d, 1)[1])#第二个是将Label由one-hot转化为10进制数组
		# print('--------------')
		# print(D_cont.shape)
		# print(c_c.shape)
		# print('--------------')
		#gradient_penalty = loss_norm_gp.gradient_penalty(functools.partial(D),D_cont_info,c_c,gp_mode='lp', sample_mode='dragan',y=y_info)
		cont_loss = (D_cont_info - c_c)**2
		info_loss = disc_loss + cont_loss*20
		info_loss = info_loss.mean()
		train_hist['info_loss'].append(info_loss.item())
		info_loss.backward()
		info_optimizer.step()
		train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
		if ((j + 1) % 100) == 0:
			with open(save_root+'setting.txt', 'a') as f:
				print('----',file=f)
				print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f, info_loss: %.8f" %((i + 1), (j + 1), train_loader.dataset.__len__() // batch_size, D_loss.item(), G_loss.item(), info_loss.item()),file=f)
				print('----',file=f)
				#print("gp: %.8f" %(gradient_penalty.mean()),file=f)
# save2img
	with torch.no_grad():
		G.eval()
		image_frame_dim = int(np.floor(np.sqrt(sample_num)))
		samples = G(sample_z, sample_c, sample_d)
		samples = (samples + 1) / 2
		torchvision.utils.save_image(samples, save_dir+'/%d_Epoch—c_d.png' % i, nrow=20)
		samples = G(sample_z2, sample_c2, sample_d2)
		samples = (samples + 1) / 2
		torchvision.utils.save_image(samples, save_dir + '/%d_Epoch-c_c.png' % i, nrow=20)
		torch.save({'epoch': epoch + 1,'G': G.state_dict()},'%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1))#save model
		# with open(save_root+'setting.txt', 'a') as f:
		# 		print('----',file=f)
		# 		print(train_hist,file=f)
		# 		print('----',file=f)
		# print('-------------')
		# print(D_real.shape)
		# print(d_real_flag.shape)
		# print('--------------')

        # self.train_hist['total_time'].append(time.time() - start_time)
        # print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),self.i, self.train_hist['total_time'][0]))
        # print("Training finish!... save training results")
        # self.save()
        # self.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)