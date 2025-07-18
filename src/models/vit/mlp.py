import torch.nn as nn 
import numpy as np 

from src.help.utils import drop_path , stem_conv


class MLP(nn.Module) : 
    def __init__(self, in_features , hidden_features=None , out_features=None , act_layer=nn.GELU ,drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features , hidden_features) 
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features , out_features)
        
        self.drop = nn.Dropout(drop)
        
    def forward(self ,x) : 
        x = self.fc1(x) 
        x = self.act(x) 
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x 
    
    
    
class Attention(nn.Module) : 
    def __init__(self , dim , num_heads=8 , qkv_bias=False , qk_scale=None , attn_drop=0. , proj_drop=0. ) :
        super().__init__()
        self.num_heads = num_heads 
        heads_dim = dim // num_heads 
        self.scale = qk_scale or heads_dim ** -0.5
        
        
        self.qkv = nn.Linear(dim , dim*3 , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim , dim)
        
        self.proj_drop = nn.Dropout(proj_drop)
        
        
    def forward(self , x) : 
        B , N , C = x.shape 
        qkv = self.qkv(x).reshape(B , N , 3 , self.num_heads , C//self.num_heads).permute(2 , 0 , 3, 1 , 4) 
        q , k , v = qkv[0] , qkv[1] , qkv[2]
        
        attn = (q @ k.transpose(-2 , -1)) * self.scale
        
        attn = attn.softmax(dim=-1)
        
        attn = self.attn_drop(attn)  
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn
    
    
class DropPath(nn.Module) : 
    """
        Drop path per sample  
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath , self).__init__()
        self.drop_prob = drop_prob 
        
        
    def forward(self , x) : 
        return drop_path(x , self.drop_prob , self.training)
    
    
class Block(nn.Module):
    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        

        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, ret_attn=False):
        # Attention + DropPath + Residual
        y, attn = self.attn(self.norm1(x))
        
        if ret_attn:
            return attn

        x = x + self.drop_path(y)
        
        # MLP + DropPath + Residual
        y = self.mlp(self.norm2(x))
        x = x + self.drop_path(y)

        return x
    
    
    

    
    
    

    
    
    
