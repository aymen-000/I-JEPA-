import torch 
import torch.nn as nn 
from src.models.utils import stem_conv



class PatchEmbed(nn.Module) : 
    """Imge to patch embedding"""
    
    def __init__(self , img_size=224 , patch_size=16 , in_chans=3 , embed_dim=768):  
        super().__init__() 
        self.img_size = img_size 
        self.num_patches = (img_size // patch_size)**2
        self.patch_size = patch_size 
        
        
        self.proj = nn.Conv2d(in_chans , embed_dim , kernel_size=patch_size , stride=patch_size)
        
        
    def forward(self, x) : 
        batch , chans , height , weidth = x.shape 
        x = self.proj(x).flatten(2).transpose(1,2) # (B , C, H , W) => (B , embed_dims , H , W ) => (B , embed_dims , H*W) => (B  , H*W , embed_dims) 
        
        return x
    
    
class ConvEmbed(nn.Module) : 
    """
     3 by 3 convolution 
    """
    
    def __init__(self, channs , strides , img_size=224 , in_chans=3 , batch_norm =True):
        super().__init__()
        bias = not batch_norm
        channels = [in_chans] + channels
        stem = stem_conv(channels , strides , bias)
        
        stem += [nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=strides[-1])] 
        
        self.stem = nn.Sequential(*stem)
        
        

        
    def forward(self, x):
        p = self.stem(x)
        return p.flatten(2).transpose(1, 2)
    
    
    
