import os
import torch
import torch.nn as nn
from models.augNaCLIP.clip import load


class TeacherCLIP(nn.Module):
    def __init__(self, pretrained, gaussian_std, arch, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        
        clip_model, _ = load(
            name=pretrained,
            device=None,
            jit=False
        )
        
        clip_vit = clip_model.visual
        
        # change to float32
        clip_vit = clip_vit.to(dtype=torch.float32)
        # Set initial parameters for ViT in CLIP
        attn_strategy='naclip'
        gaussian_std=gaussian_std
        # clip_vit.set_params("reduced", attn_strategy, gaussian_std)
        clip_vit.set_params(arch, attn_strategy, gaussian_std)
        
        self.vit = clip_vit
    
    def forward(self, args_dict, shifted_images, shifted_idxs):
        augmented_img_feats = self.vit.forward(
            args_dict=args_dict,
            shifted_images=shifted_images,
            shifted_idxs=shifted_idxs
        )
        
        return augmented_img_feats