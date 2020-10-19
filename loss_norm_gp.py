import torch
import torch.nn as nn
import torchlib
from torch.autograd import grad
import math

#                                 loss function                               

def get_losses_fn(mode):
    if mode == 'gan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = torch.nn.functional.binary_cross_entropy_with_logits(r_logit, torch.ones_like(r_logit))
            f_loss = torch.nn.functional.binary_cross_entropy_with_logits(f_logit, torch.zeros_like(f_logit))
            return r_loss, f_loss
        def g_loss_fn(f_logit):
            f_loss = torch.nn.functional.binary_cross_entropy_with_logits(f_logit, torch.ones_like(f_logit))
            return f_loss

    elif mode == 'lsgan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = torch.nn.functional.mse_loss(r_logit, torch.ones_like(r_logit))
            f_loss = torch.nn.functional.mse_loss(f_logit, torch.zeros_like(f_logit))
            return r_loss, f_loss
        def g_loss_fn(f_logit):
            f_loss = torch.nn.functional.mse_loss(f_logit, torch.ones_like(f_logit))
            return f_loss

    elif mode == 'wgan':
        def d_loss_fn(r_logit, f_logit):
            r_loss = -r_logit.mean()
            f_loss = f_logit.mean()
            return r_loss, f_loss
        def g_loss_fn(f_logit):
            f_loss = -f_logit.mean()
            return f_loss

    elif mode == 'hinge_v1':
        def d_loss_fn(r_logit, f_logit):
            r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
            f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
            return r_loss, f_loss
        def g_loss_fn(f_logit):
            f_loss = torch.max(1 - f_logit, torch.zeros_like(f_logit)).mean()
            return f_loss

    elif mode == 'hinge_v2':
        def d_loss_fn(r_logit, f_logit):
            r_loss = torch.max(1 - r_logit, torch.zeros_like(r_logit)).mean()
            f_loss = torch.max(1 + f_logit, torch.zeros_like(f_logit)).mean()
            return r_loss, f_loss
        def g_loss_fn(f_logit):
            f_loss = - f_logit.mean()
            return f_loss
    else:
        raise NotImplementedError
    return d_loss_fn, g_loss_fn

def m_loss(a,b1,b2):
    x1 = torch.exp(-b2)
    x2 = a - b1
    x3 = x2*x2*x1*(-0.5)
    loss = (b2+math.log(2*math.pi))/2-x3
    return torch.mean(loss)



#                                    gradient_penalty                                   
# =sample method=
def _sample_line(real, fake):
    shape = [real.size(0)] + [1] * (real.dim() - 1)#real.size(0),real第一维的个数/[1]*3 = [1,1,1] / real.dim()：real坐标数 如[1,1,1,1] dim为4
    alpha = torch.rand(shape, device=real.device)
    sample = real + alpha * (fake - real)
    return sample


def _sample_DRAGAN(real, fake):  # fake is useless
    beta = torch.rand_like(real)#维度和real一样的随机矩阵,值为(0,1)
    fake = real + 0.5 * real.std() * beta #fake只和real有关
    sample = _sample_line(real, fake)
    return sample


# =gradient penalty method=
def _norm(x):
    norm = x.view(x.size(0), -1).norm(p=2, dim=1)
    return norm


def _one_mean_gp(grad):
    norm = _norm(grad)
    gp = ((norm - 1)**2).mean()
    return gp


def _zero_mean_gp(grad):
    norm = _norm(grad)
    gp = (norm**2).mean()
    return gp


def _lipschitz_penalty(grad):
    norm = _norm(grad)
    gp = (torch.max(torch.zeros_like(norm), norm - 1)**2).mean()
    return gp


def gradient_penalty(f, real, fake, sample_mode, gp_mode, y):
    sample_fns = {
        'line': _sample_line,
        'real': lambda real, fake: real,
        'fake': lambda real, fake: fake,
        'dragan': _sample_DRAGAN,
    } #sample_mode

    gp_fns = {
        '1-gp': _one_mean_gp,
        '0-gp': _zero_mean_gp,
        'lp': _lipschitz_penalty,
    } #gp_mode
    if gp_mode == 'none':
        gp = torch.tensor(0, dtype=real.dtype, device=real.device)
    else:
        x = sample_fns[sample_mode](real, fake).detach() 
        x.requires_grad = True
        _,_,pred = f(y)
        grad = torch.autograd.grad(pred, x, grad_outputs=torch.ones_like(pred), create_graph=True, allow_unused=True)
        print(grad)
        gp = gp_fns[gp_mode](grad)
    return gp

#                                     utils                                   

def _get_norm_fn_2d(norm):  # 2d
    if norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return nn.InstanceNorm2d
    elif norm == 'none':
        return torchlib.NoOp
    else:
        raise NotImplementedError


def _get_weight_norm_fn(weight_norm):
    if weight_norm == 'spectral_norm':
        return torch.nn.utils.spectral_norm
    elif weight_norm == 'weight_norm':
        return torch.nn.utils.weight_norm
    elif weight_norm == 'none':
        return torchlib.identity
    else:
        return NotImplementedError

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()



#-----------------------network-----------------------
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


