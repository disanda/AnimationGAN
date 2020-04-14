import torch
import torch.nn as nn
import torchlib

from torch.autograd import grad


# ==============================================================================
# =                                loss function                               =
# ==============================================================================

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


# ==============================================================================
# =                                   others                                   =
# ==============================================================================

def gradient_penalty(f, real, fake, mode):
    device = real.device

    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = torch.rand(a.size()).to(device)
                b = a + 0.5 * a.std() * beta
            shape = [a.size(0)] + [1] * (a.dim() - 1)
            alpha = torch.rand(shape).to(device)
            inter = a + alpha * (b - a)
            return inter

        x = torch.tensor(_interpolate(real, fake), requires_grad=True)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        g = grad(pred, x, grad_outputs=torch.ones(pred.size()).to(device), create_graph=True)[0].view(x.size(0), -1)
        gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()

        return gp

    if mode == 'wgan-gp':
        gp = _gradient_penalty(f, real, fake)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'none':
        gp = torch.tensor(0.0).to(device)
    else:
        raise NotImplementedError

    return gp


# ==============================================================================
# =                                    utils                                   =
# ==============================================================================

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