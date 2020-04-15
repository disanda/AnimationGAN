import argparse
import json

import model
import loss_norm_gp

import numpy as np
import PIL.Image as Image
import tensorboardX
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as tforms
import torchlib
import os

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', dest='experiment_name', default='CGAN_default')
args = parser.parse_args()

z_dim = 100
epoch = 50
batch_size = 64
d_learning_rate = 0.0002
g_learning_rate = 0.001
n_d = 1
c_dim = 10
experiment_name = args.experiment_name
norm = 'none' #'batch_norm', 'instance_norm'

# save settings
os.mkdir('./output/%s' % experiment_name)
with open('./output/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# others
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# ==============================================================================
# =                                   setting                                  =
# ==============================================================================

# data
transform = tforms.Compose(
    [tforms.Scale(size=(64, 64), interpolation=Image.BICUBIC),
     tforms.ToTensor(),
     tforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     tforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)
train_loader = torch.utils.data.DataLoader(
    dataset=dsets.FashionMNIST('./data/FashionMNIST', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=use_gpu,
    drop_last=True
)

# model
D = model.Discriminator_v1(x_dim=3, c_dim=c_dim, norm=norm, weight_norm=weight_norm).to(device)
G = model.Generator(z_dim=z_dim, c_dim=c_dim).to(device)

# gan loss function
d_loss_fn, g_loss_fn = loss_norm_gp.get_losses_fn(loss_mode)

# optimizer
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))

# =                                    train                                   =

# load checkpoint
ckpt_dir = './output/%s/checkpoints' % experiment_name
os.mkdir(ckpt_dir)
try:
    ckpt = torchlib.load_checkpoint(ckpt_dir)
    start_ep = ckpt['epoch']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    d_optimizer.load_state_dict(ckpt['d_optimizer'])
    g_optimizer.load_state_dict(ckpt['g_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_ep = 0

# writer
writer = tensorboardX.SummaryWriter('./output/%s/summaries' % experiment_name)

# run
z_sample = torch.randn(c_dim * 10, z_dim).to(device)
c_sample = torch.tensor(np.concatenate([np.eye(c_dim)] * 10), dtype=z_sample.dtype).to(device)
for ep in range(start_ep, epoch):
    for i, (x, c_dense) in enumerate(train_loader):
        step = ep * len(train_loader) + i + 1
        D.train()
        G.train()

        # train D
        x = x.to(device)
        z = torch.randn(batch_size, z_dim).to(device)
        c = torch.tensor(np.eye(c_dim)[c_dense.cpu().numpy()], dtype=z.dtype).to(device)

        x_f = G(z, c).detach()
        x_gan_logit = D(x, c)
        x_f_gan_logit = D(x_f, c)

        d_x_gan_loss, d_x_f_gan_loss = d_loss_fn(x_gan_logit, x_f_gan_logit)
        gp = model.gradient_penalty(D, x, x_f, mode=gp_mode)
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
            x_f_gan_logit = D(x_f, c)

            g_gan_loss = g_loss_fn(x_f_gan_logit)
            g_loss = g_gan_loss

            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            writer.add_scalar('G/g_gan_loss', g_gan_loss.data.cpu().numpy(), global_step=step)

        # display
        if step % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (ep, i + 1, len(train_loader)))

        # sample
        if step % 100 == 0:
            G.eval()
            x_f_sample = (G(z_sample, c_sample) + 1) / 2.0

            save_dir = './output/%s/sample_training' % experiment_name
            os.mkdir(save_dir)
            torchvision.utils.save_image(x_f_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, ep, i + 1, len(train_loader)), nrow=10)

    torchlib.save_checkpoint({'epoch': ep + 1,
                              'D': D.state_dict(),
                              'G': G.state_dict(),
                              'd_optimizer': d_optimizer.state_dict(),
                              'g_optimizer': g_optimizer.state_dict()},
                             '%s/Epoch_(%d).ckpt' % (ckpt_dir, ep + 1),
                             max_keep=2)
