import torch 
import torch.nn as nn 
import numpy as np 
import math 
import logging
import sys 
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger() 


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



def sample_block_size(gen, scale, aspect_ratio_scale, height, width):
    """
    Samples a random block size (height, width) for masking based on scale and aspect ratio.

    Args:
        gen (torch.Generator): PyTorch random generator for reproducibility.
        scale (tuple): (min_scale, max_scale), fraction of total patches to be masked.
        aspect_ratio_scale (tuple): (min_ar, max_ar), allowed aspect ratio range.
        height (int): Total number of patch rows.
        width (int): Total number of patch columns.

    Returns:
        tuple: (block_height, block_width) - dimensions of the sampled block.
    """
    min_ar, max_ar = aspect_ratio_scale
    rand = torch.rand(1, generator=gen).item()

    total_patches = height * width
    min_s, max_s = scale
    mask_scale = min_s + rand * (max_s - min_s)

    max_keep = int(total_patches * mask_scale)

    aspect_ratio = min_ar + rand * (max_ar - min_ar)

    h = int(round(math.sqrt(max_keep * aspect_ratio)))
    w = int(round(math.sqrt(max_keep / aspect_ratio)))

    # Ensure the block fits in the grid
    while h >= height or w >= width:
        if h >= height:
            h -= 1
        if w >= width:
            w -= 1

    return (h, w)



def sample_block_mask(b_size, min_keep, height, width, acceptable_regions=None):
    """
    Generate a block mask of shape (height, width) with a rectangular region set to 1,
    such that the number of 1s exceeds min_keep. Optionally applies constraints.

    Args:
        b_size (tuple): Block size (h, w).
        min_keep (int): Minimum number of ones to keep the mask.
        height (int): Height of the full mask.
        width (int): Width of the full mask.
        accept_regions (optional): Constraint regions (used inside constrain_mask).
    
    Returns:
        mask (Tensor): Flattened tensor of indices where the block is set.
        mask_comp (Tensor): Complementary mask (1 where mask is 0).
    """
    def constrain_mask(mask,  tries=0):
        """ Helper to restrict given mask to a set of acceptable regions """
        N = max(int(len(acceptable_regions)-tries), 0)
        for k in range(N):
            mask *= acceptable_regions[k] 
    h, w = b_size
    tries = 0
    timeout = og_timeout = 20
    valid_mask = False

    while not valid_mask:
        top = torch.randint(0, height - h, (1,)).item()
        left = torch.randint(0, width - w, (1,)).item()

        mask = torch.zeros((height, width), dtype=torch.int32)
        mask[top:top + h, left:left + w] = 1

        if acceptable_regions is not None:
            constrain_mask(mask,  tries)  

        nonzero_mask = torch.nonzero(mask.flatten(), as_tuple=False)

        valid_mask = len(nonzero_mask) > min_keep

        if not valid_mask:
            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = og_timeout
                print(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')

    mask = nonzero_mask.squeeze()

    mask_comp = torch.ones((height, width), dtype=torch.int32)
    mask_comp[top:top + h, left:left + w] = 0

    return mask, mask_comp
    
    
    
    
def set_seed(seed) : 
    torch.manual_seed(seed)
    np.random.seed(seed)


def config_device() : 
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)





