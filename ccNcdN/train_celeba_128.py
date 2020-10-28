import argparse
import json
import model
import loss_norm_gp
import numpy as np
import tensorboardX
import torch
import torchvision
import PIL.Image as Image
import os
import tqdm

# ===                                    param                                   =

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='experiment_name', default='Celeba128_dim256_cd28_cc28')
parser.add_argument('--info', dest='self_animation', default=True)
args = parser.parse_args()

z_dim = 128
epoch = 60
batch_size = 64
d_learning_rate = 0.0002
g_learning_rate = 0.001
n_d = 1
d_dim = 28 # 离散10 ，连续2 
c_dim = 28
experiment_name = args.experiment_name
gp_mode = 'none'#'dragan', 'wgan-gp'
gp_coef = 1.0
#info = False
info = args.self_animation

# save settings
if not os.path.exists('./output/%s' % experiment_name):
    os.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))#字典元素间用'，'分割，key和value间用':'分割

# others
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# ===                                  setting                                  =

# data
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(64, 64), interpolation=Image.BICUBIC),
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
    pin_memory=use_gpu,
    drop_last=True
)

# model
D = model.Discriminator_v1_1(x_dim=1, dim=dim).to(device)
G = model.Generator_v1_1(x_dim=z_dim, dim=dim).to(device)

#save model in txt
with open('./output/%s/setting.txt' % experiment_name, 'a') as f:
    print('----',file=f)
    print(G,file=f)
    print('----',file=f)
    print(D,file=f)



# gan loss function
d_loss_fn, g_loss_fn = loss_norm_gp.get_losses_fn('gan') #'gan', 'lsgan', 'wgan', 'hinge_v1', 'hinge_v2'


# optimizer
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))
info_optimizer = optim.Adam(itertools.chain(G.parameters(), D.parameters()),lr=0.0001,betas=(0.6, 0.95),amsgrad=True)#G,D都更新

# =                                    train                                   =

# load checkpoint
ckpt_dir = './output/%s/checkpoints' % experiment_name
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

# 加载预训练模型
# try:
#     ckpt = torch.load(ckpt_dir)
#     start_ep = ckpt['epoch']
#     D.load_state_dict(ckpt['D'])
#     G.load_state_dict(ckpt['G'])
#     d_optimizer.load_state_dict(ckpt['d_optimizer'])
#     g_optimizer.load_state_dict(ckpt['g_optimizer'])
# except:
#     print(' [*] No checkpoint!')
#     start_ep = 0
print('start training:')
start_ep = 0

# writer
writer = tensorboardX.SummaryWriter('./output/%s/summaries' % experiment_name)

# sample2img
save_dir = './output/%s/sample_training' % experiment_name
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

#保存真实样本,groundTrue
import time
now = time.asctime(time.localtime(time.time()))
if not os.path.exists('./output/%s/sample_training/TrueImg'%(experiment_name)):
    os.mkdir('./output/%s/sample_training/TrueImg'%(experiment_name))
torchvision.utils.save_image(list(train_loader)[0][0],'./output/%s/sample_training/TrueImg/%s.jpg'%(experiment_name,now), nrow=8)
#list(train_loader)[i][0].shape=[batch_size,3,64,64],是一组batch图片.. list(train_loader)[i][1].shape=[64],是图片的标签

# Sample
z_sample = torch.randn(100, z_dim).to(device) #z_sample:[100,100],100个样本
#c_sample = torch.tensor(np.concatenate([np.eye(d_dim)] * 10), dtype=z_sample.dtype).to(device)#c_sample:[100,10]，1-9连续标签
#c_sample = False
c_sample = torch.zeros([100,10]).to(device)
c_sample[:,5]=1 #全是6

# Training 
for ep in tqdm.trange(epoch):
    if start_ep != 0:
        ep = start_ep
    i = 0
    for x, c_dense in tqdm.tqdm(train_loader):
        step = ep * len(train_loader) + i + 1
        i+=1
        D.train()
        G.train()

# train D
        x = x.to(device)
        z = torch.randn(batch_size, z_dim).to(device)#[-1,10]
        c = torch.tensor(np.eye(d_dim)[c_dense.cpu().numpy()], dtype=z.dtype).to(device)#该操作类似one-hot c_dense是一个长度为batch_size=64的标签列表,维度为[-1,10]
        #c = False
        if info ==  True:
            mc = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, c_dim))).type(torch.FloatTensor)#[-1,2]
            mc = mc.to(device)
            c = torch.cat([c,mc],1)
        x_f = G(z, c).detach()
        x_gan_logit,_ = D(x, c)
        x_f_gan_logit,_ = D(x_f, c)

        d_x_gan_loss, d_x_f_gan_loss = d_loss_fn(x_gan_logit, x_f_gan_logit)
        gp = loss_norm_gp.gradient_penalty(D, x, x_f, mode=gp_mode)
        d_loss = d_x_gan_loss + d_x_f_gan_loss + gp * gp_coef

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        writer.add_scalar('D/d_gan_loss', (d_x_gan_loss + d_x_f_gan_loss).data.cpu().numpy(), global_step=step)
        writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)

# train G
        if step % n_d == 0:
            z = torch.randn(batch_size, z_dim).to(device)

            x_f = G(z, c)
            x_f_gan_logit,m_c = D(x_f, c)

            g_gan_loss = g_loss_fn(x_f_gan_logit)
            g_loss = g_gan_loss

            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            writer.add_scalar('G/g_gan_loss', g_gan_loss.data.cpu().numpy(), global_step=step)



# sample
        if step % 100 == 0:
            G.eval()#让BN和dropout失效
            z_sample = torch.randn(100, z_dim).to(device)#每次生成不固定noise
            if type(c_sample) == type(False):
                x_f_sample = (G(z=z_sample) + 1) / 2.0
                #print(x_f_sample.shape)
            elif info == True:
                temp_c = torch.linspace(-1, 1, 10)
                c1 = torch.zeros([100,1])
                c2 = torch.zeros([100,1])
                for i2 in range(100):
                    c1[i2]=temp_c[i2%10]
                    #c2[i2]=temp_c[i2%10]
                c_con = torch.cat([c1,c2],-1)#[-1,2]
                c_con = c_con.to(device)
                c_all = torch.cat([c_sample,c_con],-1)#[-1,10+2]
                c_all = c_all.to(device)
                # print('------------------')
                # print(z_sample.shape)#[-1,100]
                # print(c_sample.shape)#[-1,10]
                # print(c_con.shape)#[-1,10]
                # print(c_all.shape)#[-1,10]
                # print('------------------')
                x_f_sample = (G(z=z_sample, c=c_all) + 1) / 2.0
            else:
                x_f_sample = (G(z=z_sample, c=c_sample) + 1) / 2.0
            torchvision.utils.save_image(x_f_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, ep, i + 1, len(train_loader)), nrow=10)

    torch.save({'epoch': ep + 1,
                              'D': D.state_dict(),
                              'G': G.state_dict(),
                              'D_optimizer': d_optimizer.state_dict(),
                              'G_optimizer': g_optimizer.state_dict()},
                             '%s/Epoch_(%d).ckpt' % (ckpt_dir, ep + 1)
                )


