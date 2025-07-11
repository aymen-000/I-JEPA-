import torch 
import torch.nn as nn 
import numpy as np 
import math 

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Applies DropPath (Stochastic Depth) regularization to the input tensor.

    During training, this function randomly drops entire residual paths 
    (i.e., sets the output of certain layers or blocks to zero) with probability `drop_prob`. 
    The remaining paths are scaled by `1 / (1 - drop_prob)` to preserve the expected output.

    Args:
        x (Tensor): Input tensor of shape (B, ...), where B is the batch size.
        drop_prob (float, optional): Probability of dropping a path. Defaults to 0.0.
        training (bool, optional): If True, apply DropPath; otherwise, return input as-is. Defaults to False.

    Returns:
        Tensor: Output tensor after applying DropPath.
    """
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor

    return output


def stem_conv(channels , strides  , bias) : 
    stem = []
    for i in range(len(channels) -2) : 
        stem = [nn.Conv2d(channels[i] , channels[i+1] , kernel_size=3 , stride=strides[i] , padding=1 ,bias=bias)]
        
        if not bias : 
            stem += [nn.BatchNorm2d(channels[i+1])] 
            
        stem += [nn.ReLU(inplace=True)] 
        
    return stem 




def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):

        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)


        tensor.uniform_(2 * l - 1, 2 * u - 1)


        tensor.erfinv_()


        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x