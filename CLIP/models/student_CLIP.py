import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.CLIP.clip import load
from models.CLIP.model import LayerNorm, QuickGELU
from collections import OrderedDict


class ModifiedLastLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    # def attention(self, x: torch.Tensor):
    #     self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    #     return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        Forward function for computing the value features for dense prediction (i.e., features for every image patch).
        """
        # Get the weights and biases for the value projection, multihead attention uses 3 * embed_dim for the input projection
        v_in_proj_weight = self.attn.in_proj_weight[-self.attn.embed_dim:]
        v_in_proj_bias = self.attn.in_proj_bias[-self.attn.embed_dim:]

        v_in = F.linear(self.ln_1(x), v_in_proj_weight, v_in_proj_bias)
        v_out = F.linear(v_in, self.attn.out_proj.weight, self.attn.out_proj.bias)

        # Using the value features works the best. Adding this to 'x' or feeding 'v' to the LayerNorm then MLP degrades the performance
        return v_out


class StudentCLIP(nn.Module):
    def __init__(self, pretrained, num_registers=16, patch_size=16, input_resolution=(448, 448)):
        super().__init__()
        self.resolution = input_resolution
        self.patch_size = patch_size
        
        clip_model, _ = load(
            name=pretrained,
            device='cpu',
            jit=False
        )
        
        clip_vit = clip_model.visual
        
        # change to float32
        clip_vit = clip_vit.to(dtype=torch.float32)
        
        # Update position embedding
        if self.resolution != (224, 224):
            old_patch_pos_embed = torch.clone(clip_vit.positional_embedding.detach())
            # print(clip_vit.positional_embedding.dtype)
            # print(clip_vit.conv1.weight.dtype)
            clip_vit.positional_embedding.data = self.interpolate_pos_encoding(
                old_patch_pos_embed,
                w=self.resolution[0],
                h=self.resolution[1]
            )
            
        # Modify the last layer, using the value output
        # Load the final residual block state
        last_resblock = clip_vit.transformer.resblocks[-1]
        last_resblock_state_dict = last_resblock.state_dict()

        d_model = clip_vit.transformer.width
        n_head = last_resblock.attn.num_heads
        
        last_layer = ModifiedLastLayer(
            d_model=d_model,
            n_head=n_head,
            attn_mask=None
        )
        
        last_layer.load_state_dict(last_resblock_state_dict, strict=True)
        clip_vit.transformer.resblocks[-1] = last_layer
        
        # Create content embeddings for the new register tokens
        embed_dim = clip_vit.class_embedding.shape[0]
        self.register_tokens = nn.Parameter(
            torch.randn(num_registers, embed_dim) / torch.sqrt(torch.tensor(embed_dim)),
        )
        
        self.vit = clip_vit
    
    def forward(self, x):
        img_w, img_h = x.shape[-2], x.shape[-1]
        x = self.vit.conv1(x)  # --> [B, C, H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # --> [B, N_patches, C]

        batch_size = x.shape[0]
        
        cls_token = self.vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        
        x = torch.cat([cls_token, x], dim=1)
        
        if (img_w, img_h) != self.resolution:
            pos_embed = self.interpolate_pos_encoding(
                    pos_embed=self.vit.positional_embedding,
                    w=img_w,
                    h=img_h
                ).to(x.device, x.dtype)
        else:
            pos_embed = self.vit.positional_embedding
        x = x + pos_embed

        register_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(x.dtype)  # --> [B, num_register_tokens, C]
        
        x = torch.cat([x, register_tokens], dim=1)  # --> [B, 1 + N_patches + num_register_tokens, C]

        x = self.vit.ln_pre(x)
        x = x.permute(1, 0, 2)  # [B, N, C] -> [N, B, C]
        x = self.vit.transformer(x)
        x = x.permute(1, 0, 2)  # [N, B, C] -> [B, N, C]
        x = self.vit.ln_post(x)

        if self.vit.proj is not None:
            x = x @ self.vit.proj

        return x
    
    def interpolate_pos_encoding(self, pos_embed, w, h):
        if len(pos_embed.shape) == 3:
            pos_embed = pos_embed.squeeze(0)
        
        npatch = self.resolution[0] // self.patch_size
        N = pos_embed.shape[0] - 1
        if npatch == N and w == h:
            return pos_embed
        class_pos_embed = pos_embed[[0]]
        patch_pos_embed = pos_embed[1:]
        dim = pos_embed.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), mode='bicubic',
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)




