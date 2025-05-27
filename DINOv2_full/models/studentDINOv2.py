import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math


"""
Use Registers & DINOv2 Base - Patch14 as a Student Model
2-Registers & DINOv2 Base, modifying the last layer where we only use value as dense feature output
When distilling, we only update registers and position embedding.
"""
 

    
class RegDINOv2Base(nn.Module):
    def __init__(self, pretrained_path='facebook/dinov2-base', attn_implementation='eager', num_registers=16, weight_frozen=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path = pretrained_path,
                                               attn_implementation=attn_implementation)
        # Freeze all weights
        if weight_frozen:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.requires_grad = False

        # Fine-tuning positional embeddings
        # self.model.embeddings.position_embeddings.requires_grad = True

        self.embed = self.model.embeddings
        self.encoder = self.model.encoder
        # The last layernorm after 12-layer transforms
        self.layernorm = self.model.layernorm 

        hidden_size = self.model.config.hidden_size
        # Fine-tuning registers
        "scale register values with (float(hidden_size) ** 0.5)"
        self.registers = nn.Parameter(torch.randn(1, num_registers, hidden_size) / (float(hidden_size) ** 0.5))

    def forward(self, images, output_attentions, output_hidden_states):
        x = self.embed(images) # DINO v2 includes 'interpolate_pos_encoding'
        x = torch.cat([self.registers.expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        # I do not add pos_embedding to registers
        
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
        output['last_hidden_state'] = norm_output
        output['hidden_states'] = hidden_states
        output['attentions'] = attentions
        
        return output