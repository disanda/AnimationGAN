#https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
import torch
from torch import nn
from torch import autograd
from torch import optim


def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    use_cuda = True
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, 32, 32)
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