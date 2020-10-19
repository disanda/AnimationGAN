#https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
import torch
from torch import nn
from torch import autograd
from torch import optim


def calc_gradient_penalty(netD, real_data, fake_data, batch_s=1):
    # print "real_data: ", real_data.size(), fake_data.size()
    use_cuda = True
    alpha = torch.rand(batch_s, 1)
    alpha = alpha.expand(batch_s, real_data.nelement()/batch_s).contiguous().view(batch_s, 3, 32, 32)
    alpha = alpha.cuda() if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

import torch
import numpy as np
import model_v2 as model
import gp
batch_size = 64
c_c_num = 20
c_d_num = 20
img_channel = 1
input_size =32
c_c = torch.from_numpy(np.random.uniform(-1, 1, size=(batch_size, c_c_num))).type(torch.FloatTensor)
D = model.discriminator_mwm(input_channel=img_channel, output_dim=1, input_size=input_size, len_discrete_code=c_d_num, len_continuous_code=c_c_num)
gradient_penalty = gp.calc_gradient_penalty(D,D_cont_info,c_c,batch_s=batch_size)