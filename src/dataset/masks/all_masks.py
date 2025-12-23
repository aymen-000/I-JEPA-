import math 
import torch 
from multiprocessing import Value
from src.help.utils import * 

class MutiBlockMaskCollector(object): 
    
    def __init__(self, 
                 input_size=(64, 64), 
                 patch_size=8, 
                 enc_mask_scale=(0.2, 0.8), 
                 pred_mask_scale=(0.2, 0.8), 
                 aspect_ratio=(0.3, 3.0),  
                 nenc=1, 
                 npred=2, 
                 min_keep=4, 
                 allow_overlap=False): 
        
        super(MutiBlockMaskCollector, self).__init__() 
        if not isinstance(input_size, tuple): 
            input_size = (input_size,) * 2 
            
        self.enc_mask_scale = enc_mask_scale 
        
        # Convert pixel dimensions to patch grid dimensions
        self.height = input_size[0] // patch_size  # 64 // 8 = 8 patches
        self.width = input_size[1] // patch_size   # 64 // 8 = 8 patches
        
        self.patch_size = patch_size 
        self.pred_mask_scale = pred_mask_scale 
        self.aspect_ratio = aspect_ratio 
        self.min_keep = min_keep 
        self.allow_overlap = allow_overlap 
        self._itr_counter = Value('i', -1)
        self.nenc = nenc 
        self.npred = npred
        
        # Debug print
        print(f"MaskCollator initialized:")
        print(f"  Input size (pixels): {input_size}")
        print(f"  Patch size: {patch_size}")
        print(f"  Grid size (patches): {self.height} x {self.width}")
        print(f"  Total patches: {self.height * self.width}")
        
    def step(self): 
        i = self._itr_counter
        with i.get_lock(): 
            i.value += 1 
            v = i.value 
        return v 
    
    def __call__(self, batch):
        B = len(batch) 
        
        collated_batch = torch.utils.data.default_collate(batch)
        
        seed = self.step()
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        # Sample block sizes for predictor and encoder
        p_size = sample_block_size(
            gen=g, 
            scale=self.pred_mask_scale, 
            aspect_ratio_scale=self.aspect_ratio,  # FIXED: Use aspect_ratio
            height=self.height, 
            width=self.width 
        )
        e_size = sample_block_size(
            gen=g, 
            scale=self.enc_mask_scale,  # FIXED: Was using pred_mask_scale for both!
            aspect_ratio_scale=self.aspect_ratio,  # FIXED: Use aspect_ratio
            height=self.height, 
            width=self.width 
        )
        
        collated_masks_pred, collated_masks_enc = [], []  
        
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        
        for _ in range(B):
            masks_p, masks_c = [], [] 
            
            for _ in range(self.npred): 
                mask, mask_c = sample_block_mask(
                    p_size, 
                    min_keep=self.min_keep, 
                    width=self.width, 
                    height=self.height
                ) 
                
                masks_p.append(mask) 
                masks_c.append(mask_c)
                
                min_keep_pred = min(min_keep_pred, len(mask))
                
            collated_masks_pred.append(masks_p)
            
            # Acceptable regions for encoder (avoid overlap with predictor)
            accep_regions = masks_c
            
            try: 
                if self.allow_overlap: 
                    accep_regions = None 
            except Exception as e: 
                print(f"Encountered exception in mask-generator: {e}")
                
            masks_e = [] 
            for _ in range(self.nenc): 
                mask, _ = sample_block_mask(
                    e_size,
                    min_keep=self.min_keep, 
                    width=self.width, 
                    height=self.height,  
                    acceptable_regions=accep_regions
                )
                
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask)) 
                
            collated_masks_enc.append(masks_e)
            
        # Trim masks to minimum keep length
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
            
            
class RandomMaskCollector(object): 
    def __init__(self, 
                 ratio=(0.4, 0.6), 
                 input_size=(64, 64),  
                 patch_size=8):
        
        super(RandomMaskCollector, self).__init__()
        if not isinstance(input_size, tuple): 
            input_size = (input_size,) * 2 
            
        self.patch_size = patch_size 
        
        # CRITICAL FIX: Convert to patch grid dimensions
        self.height = input_size[0] // patch_size  # Number of patches in height
        self.width = input_size[1] // patch_size   # Number of patches in width
        
        self.ratio = ratio 
        self._itr_counter = Value('i', -1) 
        
        # Debug print
        print(f"RandomMaskCollator initialized:")
        print(f"  Input size (pixels): {input_size}")
        print(f"  Patch size: {patch_size}")
        print(f"  Grid size (patches): {self.height} x {self.width}")
        print(f"  Total patches: {self.height * self.width}")
        
    def step(self): 
        i = self._itr_counter
        with i.get_lock(): 
            i.value += 1 
            v = i.value 
        return v 
    
    def __call__(self, batch): 
        B = len(batch) 
        
        collated_batch = torch.utils.data.default_collate(batch) 
        
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed) 
        
        ratio = self.ratio 
        ratio = ratio[0] + torch.rand(1, generator=g).item() * (ratio[1] - ratio[0])
        
        num_patches = self.height * self.width 
        num_keep = int(num_patches * (1. - ratio)) 
        
        collated_masks_pred, collated_masks_enc = [], [] 
        for _ in range(B): 
            m = torch.randperm(num_patches, generator=g)  # FIXED: Use generator
            
            collated_masks_enc.append([m[:num_keep]])
            collated_masks_pred.append([m[num_keep:]])
            
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        
        return collated_batch, collated_masks_enc, collated_masks_pred