import math
import numpy as np 
from functools import partial

import torch 
import torch.nn as nn 
from src.models.vit.pos_encode import sinus_pos_embedding
from src.models.vit.mlp import Block
from src.help.utils import *
from src.models.vit.image_embedding import PatchEmbed
class VisionTransformerPredictor(nn.Module) : 
    """VIT""" 
    
    def __init__(self, 
                 num_patchs , embed_dim=768 , pred_embed_dim = 384 , depth=6 , num_heads=12 , mlp_ratio=4.0 , qkv_bias=True , qk_scale=None , drop_rate=0.0 , drop_path_rate=0.0 , norm_layer=nn.LayerNorm , init_std=0.02 ,attn_drop_rate=0.0,  **kwargs):
        super().__init__(**kwargs)
        self.predictor_embed = nn.Linear(embed_dim , pred_embed_dim , bias=True) 
        self.mask_token = nn.Parameter(torch.zeros(1,1,pred_embed_dim))
        
        
        dpr = [x.item() for x in torch.linspace(0 , drop_path_rate , depth)]
        
        self.pred_pos_embed = nn.Parameter(torch.zeros(1 , num_patchs , pred_embed_dim) , requires_grad=False) 
        
        pred_pos_embed = sinus_pos_embedding(self.pred_pos_embed.shape[-1] , int(num_patchs**.5) )
        
        self.pred_pos_embed.data.copy_(torch.from_numpy(pred_pos_embed).float().squeeze(0))
        
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=pred_embed_dim , num_heads=num_heads , mlp_ratio=mlp_ratio , 
                    qk_scale=qk_scale , 
                    qkv_bias=qkv_bias , 
                    drop=drop_rate , 
                    attn_drop=attn_drop_rate , 
                    drop_path=dpr[i] ,
                    norm_layer=norm_layer 
                ) for i in range(depth)
            ]
        )
        
        self.predictor_norm = norm_layer(pred_embed_dim)
        
        self.predictor_porj = nn.Linear(pred_embed_dim , embed_dim , bias=True)
        
        self.init_std = init_std 
        trunc_normal_(self.mask_token , std=self.init_std)
        
        self.apply(self._init_weights) 
        self.fix_init_weight() 
        
        
    def fix_init_weight(self) : 
        def rescale(param , layer_id) : 
            param.div_(math.sqrt(2.0*layer_id)) 
            
        for layer_id , layer in enumerate(self.predictor_blocks) : 
            rescale(layer.attn.proj.weight.data , layer_id+1) 
            rescale(layer.mlp.fc2.weight.data , layer_id+1) 
            
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self , x , masks_x , masks) : 
        assert masks is not None 
        assert masks_x is not None 
        
        if not isinstance(masks_x , list) : 
            masks_x = [masks_x]
            
        if not isinstance(masks , list): 
            masks = [masks]
            
        B = len(x) // len(masks_x) # correct batch size for multiple mask 
        
        x = self.predictor_embed(x) 
        x_pos_embed = self.pred_pos_embed.repeat(B , 1,1) 
        x += apply_masks(x_pos_embed , masks_x)
        
        
        _ , N_ctxt , D = x.shape 
        
        pos_embed = self.pred_pos_embed.repeat(B , 1 , 1)
        pos_embed = apply_masks(pos_embed , masks) # not visible 
        
        pos_embed = repeat_interleave_batch(pos_embed , B , repeat=len(masks_x)) 
        
        pred_tokens = self.mask_token.repeat(pos_embed.size(0) , pos_embed.size(1) , 1) 
        
        pred_tokens += pos_embed 
        
        
        x = x.repeat(len(masks) , 1 , 1) 
        x = torch.cat([x , pred_tokens] , dim=1) 
        
        for blk in self.predictor_blocks : 
            x = blk(x) 
        
        
        x = self.predictor_norm(x) 
        
        x = x[: , N_ctxt:] 
        x = self.predictor_porj(x) 
        return x 
        
        
        
        
class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = sinus_pos_embedding(self.pos_embed.shape[-1],
                                            int(self.patch_embed.num_patches**.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

     
        x = self.patch_embed(x)
        B, N, D = x.shape

    
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed


        if masks is not None:
            x = apply_masks(x, masks)

   
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model




VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
}
        
    
         
        