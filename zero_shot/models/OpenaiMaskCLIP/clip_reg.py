import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from collections import OrderedDict
import torch
from clip import load
from model import *
# from models.OpenaiMaskCLIP.clip import load
# from models.OpenaiMaskCLIP.model import * 
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union



class ModifiedResBlock(torch.nn.Module):
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

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # Return the first element (attention output)
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

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


class ModifiedImageEncoder(nn.Module):
    def __init__(self, original_image_encoder, num_register_tokens=2, patch_size=16, input_resolutions=(448,448), upsampled_pos_grad=False):
        super().__init__()
        self.requires_grad = upsampled_pos_grad
        
        self.original_image_encoder = original_image_encoder
        
        # Load the final residual block state
        last_resblock = self.original_image_encoder.transformer.resblocks[-1]
        last_resblock_state_dict = last_resblock.state_dict()

        d_model = self.original_image_encoder.transformer.width
        n_head = last_resblock.attn.num_heads

        # Create a modified final residual block
        new_resblock = ModifiedResBlock(
            d_model=d_model,
            n_head=n_head,
            attn_mask=None
        )
        new_resblock.load_state_dict(last_resblock_state_dict, strict=True)
        self.original_image_encoder.transformer.resblocks[-1] = new_resblock
        
        # Create content embeddings for the new register tokens
        embed_dim = self.original_image_encoder.class_embedding.shape[0]
        self.register_tokens = nn.Parameter(
            torch.randn(num_register_tokens, embed_dim) / torch.sqrt(torch.tensor(embed_dim)),
            requires_grad=self.requires_grad
        )
        
        self.input_resolutions = input_resolutions
        self.patch_size = patch_size
        if self.input_resolutions != (224, 224):
            self.original_image_encoder.positional_embedding = self.upsampling_pos_embedding(image_shape=self.input_resolutions)

    def forward(self, x):
        img_h, img_w = x.shape[-2], x.shape[-1]
        x = self.original_image_encoder.conv1(x)  # --> [B, C, H', W']
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # --> [B, N_patches, C]

        batch_size = x.shape[0]

        # cls_token = (
        #     self.original_image_encoder.class_embedding
        #     .to(x.device, x.dtype)
        #     .unsqueeze(0)
        #     .expand(batch_size, -1, -1)   # --> [B, 1, C]
        # )
        cls_token = self.original_image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        
        x = torch.cat([cls_token, x], dim=1)
        
        if (img_h, img_w) != self.input_resolutions:
            pos_embed = self.upsampling_pos_embedding(image_shape=(img_h, img_w)).to(x.device, x.dtype)
        # pos_embed = self.original_image_encoder.positional_embedding.to(x.device, x.dtype)
        else:
            pos_embed = self.original_image_encoder.positional_embedding
        x = x + pos_embed

        register_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(x.dtype)  # --> [B, num_register_tokens, C]
        
        x = torch.cat([x, register_tokens], dim=1)  # --> [B, 1 + N_patches + num_register_tokens, C]

        # pos_embed = self.original_image_encoder.positional_embedding.to(x.device, x.dtype)
        # seq_len_pos_embed = pos_embed.shape[0]
        # x[:, :seq_len_pos_embed, :] += pos_embed
        # x[:, :seq_len_pos_embed, :] = x[:, :seq_len_pos_embed, :] + pos_embed


        x = self.original_image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # [B, N, C] -> [N, B, C]
        x = self.original_image_encoder.transformer(x)
        x = x.permute(1, 0, 2)  # [N, B, C] -> [B, N, C]
        x = self.original_image_encoder.ln_post(x)

        if self.original_image_encoder.proj is not None:
            x = x @ self.original_image_encoder.proj

        return x
    
    def upsampling_pos_embedding(self, image_shape):

        # ------------------------------------------------------------
        # 2.2 Upscale the positional embeddings from 224×224 -> 448x448
        #     (224/16=14 patches → 448/16=28 patches)
        # ------------------------------------------------------------
        # shape: (1 + 14×14, 768)
        # if image_shape == (224, 224):
        #     return self.original_image_encoder.positional_embedding
        
        # The first token is [CLS], next tokens are patch embeddings
        old_patch_pos_embed = self.original_image_encoder.positional_embedding.clone()
        patch_pos_embed = old_patch_pos_embed[1:]  # shape: (14×14, 768)
        # print(patch_pos_embed.shape)
        num_patches, hidden_size = patch_pos_embed.shape
        patches = int(math.sqrt(num_patches))
        assert patches ** 2 == num_patches
        # Reshape to (1, 14, 14, 768) => permute => interpolate => (1, 28, 28, 768)
        patch_pos_embed_2d = patch_pos_embed.reshape(-1, patches, patches, hidden_size).permute(0, 3, 1, 2)
        # print(f"patch_pos_embed_2d: {patch_pos_embed_2d.shape}")        
        
        # row_resolution, col_resolution = input_resolution
        patch_pos_embed_v2 = F.interpolate(
            patch_pos_embed_2d,
            size=(image_shape[0] // self.patch_size, image_shape[1] // self.patch_size),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        
        # Flatten back to (28×28, 768)
        flat = patch_pos_embed_v2.permute(0, 2, 3, 1).view(1, -1, hidden_size)
        # print(f"flat: {flat.shape}")
        # Recombine with the [CLS] token
        new_patch_pos_embed = torch.cat([old_patch_pos_embed[:1], flat[0]], dim=0)
        
        return nn.Parameter(new_patch_pos_embed.clone(), requires_grad=self.requires_grad)
    
    def forward_intermediates(
        self,
        x: torch.Tensor,
        n: Union[int, List[int], Tuple[int]] = 1,
        return_prefix_tokens: bool = False,
        norm: bool = True,
        output_fmt: str = "NLC",
        intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass with intermediate layer outputs.
        
        Args:
            x: Input tensor
            n: Layer indices to return (last n if int, specific if list)
            return_prefix_tokens: Whether to return prefix tokens (cls + register)
            norm: Apply layer norm to intermediates
            output_fmt: Output format ('NLC' or 'NCHW')
            intermediates_only: Only return intermediates, not final output
        """
        # Initial processing (same as forward)
        img_h, img_w = x.shape[-2], x.shape[-1]
        x = self.original_image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        
        batch_size = x.shape[0]
        
        # cls_token = (
        #     self.original_image_encoder.class_embedding
        #     .to(x.device, x.dtype)
        #     .unsqueeze(0)
        #     .expand(batch_size, -1, -1)
        # )
        cls_token = self.original_image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        
        x = torch.cat([cls_token, x], dim=1)
        if (img_h, img_w) != self.input_resolutions:
            pos_embed = self.upsampling_pos_embedding(image_shape=(img_h, img_w)).to(x.device, x.dtype)
        # pos_embed = self.original_image_encoder.positional_embedding.to(x.device, x.dtype)
        else:
            pos_embed = self.original_image_encoder.positional_embedding
        x = x + pos_embed

        register_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(x.dtype)
        x = torch.cat([x, register_tokens], dim=1)
        
        x = self.original_image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # [B, N, C] -> [N, B, C]
        
        # Prepare for intermediate storage
        intermediates = []
        transformer = self.original_image_encoder.transformer
        
        # Process through transformer layers while storing intermediates
        for i, layer in enumerate(transformer.resblocks):
            x = layer(x)
            
            # Store intermediate if needed
            if isinstance(n, (list, tuple)) and i in n or \
            isinstance(n, int) and i >= (len(transformer.resblocks) - n):
                intermediate = x.permute(1, 0, 2)  # [N, B, C] -> [B, N, C]
                
                if norm:
                    intermediate = self.original_image_encoder.ln_post(intermediate)
                else:
                    intermediate = x.clone()
                
                intermediates.append(intermediate)
        
        x = x.permute(1, 0, 2)  # [N, B, C] -> [B, N, C]
        
        # Final processing
        if not intermediates_only:
            x = self.original_image_encoder.ln_post(x)
            if self.original_image_encoder.proj is not None:
                x = x @ self.original_image_encoder.proj
            intermediates[-1] = x
        
        # Process intermediates
        if intermediates:
            if isinstance(n, int) and len(intermediates) > n:
                intermediates = intermediates[-n:]
        
        # Reshape if needed
        if output_fmt == "NCHW" and intermediates:
            # h = w = int(math.sqrt(intermediates[0].shape[1] - 1 - self.register_tokens.shape[0]))
            h = img_h // self.patch_size
            w = img_w // self.patch_size
            for i, feat in enumerate(intermediates):
                # Split cls, patches, and register tokens
                cls_token = feat[:, :1]
                patches = feat[:, 1:1+h*w]
                registers = feat[:, 1+h*w:]
                
                # Reshape patches
                patches = patches.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
                
                if return_prefix_tokens:
                    intermediates[i] = (cls_token, registers, patches)
                else:
                    intermediates[i] = patches
        
        # if intermediates_only:
        #     return intermediates
        # return (x, intermediates) if intermediates else x
        return intermediates
    

def student_model_initialization(pretrained, num_register_tokens, patch_size, input_resolution, pos_grad):
    # Load the CLIP model
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = load("ViT-B/16", device=device)
    model, _ = load(pretrained, device=None, jit=False)
    # Freeze all parameters except registers and positional embeddings
    # for param in model.parameters():
    #     param.requires_grad = False

    # Replace the image encoder with our modified version
    regViT = ModifiedImageEncoder(
        model.visual, 
        num_register_tokens=num_register_tokens,
        patch_size = patch_size,
        input_resolutions=input_resolution,
        upsampled_pos_grad=pos_grad
    )    

    # model = model.to(torch.float32)
    regViT = regViT.to(torch.float32)
    
    for param in regViT.parameters():
        param.requires_grad = False
        
    # activate register tokens
    regViT.register_tokens.requires_grad = True   
    
    # activate the positional embedding
    regViT.original_image_encoder.positional_embedding.requires_grad = True
       
    # activate embedding layers        
    regViT.original_image_encoder.conv1.weight.requires_grad = True
    bias_exist = False
    if regViT.original_image_encoder.conv1.bias is not None:
        regViT.original_image_encoder.conv1.bias.requires_grad = True
        bias_exist = True
        
    # activate the second last transformer block
    for param in regViT.original_image_encoder.transformer.resblocks[-2].parameters():
        param.requires_grad = True
    
    # activate the last transformer block
    for param in regViT.original_image_encoder.transformer.resblocks[-1].parameters():
        param.requires_grad = True
        
    # print(f"We are fine-tuning Position Embedding, {num_register_tokens} Registers, Conv Layer and Transformer Resblocks[-1],[-2]")

    return regViT, bias_exist




def load_model(checkpoint_path, num_register_tokens, input_resolution, lr=1e-4, weight_decay=1e-5, device='cpu'):
    model, optimizer = student_model_initialization(
        num_register_tokens=num_register_tokens, 
        input_resolution=input_resolution, 
        lr=lr, 
        weight_decay=weight_decay, 
        device=device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model, optimizer
