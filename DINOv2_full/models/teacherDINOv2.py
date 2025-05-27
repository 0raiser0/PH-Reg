import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math

    

"--------------------"
class AugDINOv2Base(nn.Module):
    def __init__(self, pretrained_path='facebook/dinov2-base', attn_implementation='eager', weight_frozen=True):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path = pretrained_path,
                                               attn_implementation=attn_implementation)
        
        # Freeze all weights
        if weight_frozen:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False
                    
        self.config = self.model.config
        self.embed = self.model.embeddings
        self.encoder = self.model.encoder
        self.layernorm = self.model.layernorm # The last layernorm after 12-layer transforms
        
        # Fine-tuning positional embeddings
        # self.model.embeddings.position_embeddings.requires_grad = True
        
        self.num_layers = self.config.num_hidden_layers
        
    def forward_images(self, images, output_attentions, output_hidden_states):
        h, w = images.shape[-2], images.shape[-1]
        n_patches = (h // self.config.patch_size, w // self.config.patch_size)
        x = self.embed(images) # DINO v2 includes 'interpolate_pos_encoding'
            
        hidden_states = (x,) if output_hidden_states else None
        attentions = () if output_attentions else None
        
        """
        See:https://github.com/huggingface/transformers/blob/49b5ab6a27511de5168c72e83318164f1b4adc43/src/transformers/models/dinov2/modeling_dinov2.py#L263C9-L264C1
        Dinov2SelfAttention always returns a Tuple.
        
        See: https://github.com/huggingface/transformers/blob/49b5ab6a27511de5168c72e83318164f1b4adc43/src/transformers/models/dinov2/modeling_dinov2.py#L427C9-L427C16
        If output_attentions == True: -> Dinov2Layer returns a Tuple including 'layeroutput' and 'attention weights'
        else: -> It returns a Tuple including 'layeroutput'
        """
        for layer in self.encoder.layer: # [:-1]
            x = layer(x, output_attentions=output_attentions)
            
            if output_hidden_states:
                hidden_states = hidden_states + (x[0],)
            
            if output_attentions:
                attentions = attentions + (x[1],)
            
            x = x[0]
            
        # update last_hidden_state
        norm_output = self.layernorm(x)
        # Use norm_output as the last element in 'hidden_states
        
        output = {}
        output['last_hidden_state'] = norm_output # norm_output
        output['hidden_states'] = hidden_states
        output['attentions'] = attentions
        
        return output
    
    def forward(self, args_dict, shifted_images, shifted_idxs, output_attentions, output_hidden_states):
        num_patches = args_dict['resolution'] // args_dict['patch_size']
        #  merge batch and counts of images -> [batch_size*counts, 3, resolution, resolution]
        shifted_images = shifted_images.reshape(-1, 3, args_dict['resolution'], args_dict['resolution'])
        shifted_idxs = shifted_idxs.reshape(-1, 2, num_patches, num_patches)
        # print(shifted_idxs[10]) # shifted_idxs[10] should be equal to shifted_idxs[0]
        teacher_img_output = self.forward_images(
            images=shifted_images,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
       
        teacher_img_feats = teacher_img_output['last_hidden_state']
        teacher_img_patches = teacher_img_feats[:, 1:, :]
        out_hidden_size = teacher_img_patches.shape[-1]
        # B_counts, _, out_hidden_size = teacher_img_patches.shape
        teacher_img_patches = teacher_img_patches.reshape(-1, num_patches, num_patches, out_hidden_size)
        
        # [batch_size*counts, num_patches, num_patches, out_hidden_size]
        final_maps, countribution_maps = self.recover_images(
            shifted_idxs=shifted_idxs,
            teacher_img_patches=teacher_img_patches
        )
        # [batch_size, counts, num_patches, num_patches, hidden_size]
        "Final maps need to be divided by contribution maps."
        final_maps = final_maps.reshape(-1, args_dict['counts'], num_patches, num_patches, out_hidden_size)
        countribution_maps = countribution_maps.reshape(-1, args_dict['counts'], num_patches, num_patches)
        final_maps = final_maps.sum(dim=1) / countribution_maps.sum(dim=1).unsqueeze(-1)
        # [batch_size, num_patches, num_patches, hidden_size]
        out_img_feats = final_maps.permute(0, 3, 1, 2).contiguous()  # (B, 512, num_patches, num_patches)
        out_img_feats = out_img_feats.flatten(2).transpose(1, 2)
        
        teacher_img_output['last_hidden_state'] = out_img_feats
        return teacher_img_output
    
    def recover_images(self, shifted_idxs, teacher_img_patches):
        B, num_patches, _, out_hidden_size = teacher_img_patches.shape
        final_maps_zero = torch.zeros_like(
            input=teacher_img_patches,
            dtype=teacher_img_patches.dtype,
            device=teacher_img_patches.device
        )
        
        contribution_map_zero = torch.zeros(
            size=(B, num_patches, num_patches),
            dtype=teacher_img_patches.dtype,
            device=teacher_img_patches.device
        )
        # Get valid shifting postion, valid_mask: (B, H, W)
        valid_mask = (shifted_idxs >= 0).all(dim=1)

        # Get indexes of all valid shifting postions
        batch_idx, valid_h, valid_w = valid_mask.nonzero(as_tuple=True)
        # size: batch_size * num_patches * num_patches

        # According to valid postion indexes，we can get shifting positions
        # shifted_idxs: (B, 2, H, W)，x_coords: shifted_idxs[b, 0, h, w]
        # y_coords: shifted_idxs[b, 1, h, w]
        x_coords = shifted_idxs[batch_idx, 0, valid_h, valid_w]
        y_coords = shifted_idxs[batch_idx, 1, valid_h, valid_w]

        # Extract valid patches from patch_feats_2d (B, H, W, hidden_size)
        patch_feats_valid = teacher_img_patches[batch_idx, valid_h, valid_w] 
        # shape: (B, h, w, hidden_size)
        
        # Accumalate to corresponding positions in final_maps (B, H, W, hidden_size)
        # x_coords, y_coords correspond to the second, third dimension in final_maps, respectively
        # We select all values along the hidden_size(fourth) dimension. So we not show it in the code.
        # final_maps.index_put_((batch_idx, x_coords, y_coords), patch_feats_valid, accumulate=True)
        final_maps_zero[batch_idx, x_coords, y_coords] = patch_feats_valid
        # [batch_size, counts, num_patches, num_patches, hidden_size]
        contribution_map_zero[batch_idx, x_coords, y_coords] = 1
        
        return final_maps_zero, contribution_map_zero