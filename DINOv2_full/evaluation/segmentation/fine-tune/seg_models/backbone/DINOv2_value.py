import sys
sys.path.append('/data/chenyinjie/CYJcode/traindistill/DINOv2_full')

from mmengine.model import BaseModule, ModuleList
from mmseg.registry import MODELS

import torch
# from transformers import AutoModel
from models.studentDINOv2 import RegDINOv2Base

@MODELS.register_module()
class DistileedDINOv2(BaseModule):
    def __init__(self, pretrained, distilled_weight, patch_size, freeze_weights, out_indices, get_intermediates):
        super().__init__()
        self.out_indices = out_indices
        self.get_intermediates = get_intermediates
        self.patch_size = patch_size
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'else')
        self.model = RegDINOv2Base(
            pretrained_path=pretrained,
            attn_implementation='eager',
            num_registers=16,
            weight_frozen=True
        )
        self.model = self.model.to(torch.float32).to(device)
        
        pretrained_dict = torch.load(distilled_weight)
        self.model.load_state_dict(pretrained_dict)
        
        model_dict = self.model.state_dict()
        unmatched_keys = []
        for k in pretrained_dict:
            if k not in model_dict:
                unmatched_keys.append((k, "Key not in model"))
            elif model_dict[k].shape != pretrained_dict[k].shape:
                unmatched_keys.append((k, f"Shape mismatch: model={model_dict[k].shape}, pretrained={pretrained_dict[k].shape}"))
        print(unmatched_keys)
        
        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.eval()
        
    def forward(self, img):
        with torch.no_grad():
            if self.get_intermediates:
                # feature_maps = self.forward_intermediates(
                #     x=img,
                #     n=self.out_indices,
                #     return_prefix_tokens=False,
                #     norm=True,
                #     output_fmt="NCHW",
                #     intermediates_only=True
                # )
                
                # if not isinstance(feature_maps, tuple):
                #     feature_maps = tuple(feature_maps)
                raise ValueError("the method 'forward_intermediates' is not defined.")
            else:
                feature_maps = ()
                b, _, h, w = img.shape
                hidden_states = self.model(img, output_attentions=False, output_hidden_states=False)
                last_states = hidden_states['last_hidden_state']
                patches = last_states[:, 1+16:, :]
                patches = patches.reshape(b, h//self.patch_size, w//self.patch_size, -1)
                patches = patches.permute(0, 3, 1, 2)
                feature_maps = feature_maps + (patches,)
            
        return feature_maps
    
    def forward_intermediates(self, x, n, return_prefix_tokens, norm, output_fmt, intermediates_only):
        batch_size, _, img_h, img_w = x.shape
        x = self.model.embed(x)
        
        intermediates = []
        # Process through transformer layers while storing intermediates
        for i, layer in enumerate(self.model.encoder.layer[:-1]):
            x = layer(x, output_attentions=False)
            x = x[0] # self.model.encoder.layer returns a tuple
            
            # Store intermediate if needed
            if isinstance(n, (list, tuple)) and i in n or \
            isinstance(n, int) and i >= (len(self.model.encoder.layer) - n):                
                if norm:
                    intermediate = self.model.layernorm(x)
                else:
                    intermediate = x
                
                intermediates.append(intermediate)
                
        value_output = self.model.last_layer(x)
        if isinstance(n, (list, tuple)) and 11 in n:
            if norm:
                intermediate = self.model.layernorm(value_output)
            else:
                intermediate = value_output
                
            intermediates.append(intermediate)
                
        
        # Final processing
        if not intermediates_only:
            x = self.model.layernorm(value_output)
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
                patches = feat[:, 1:]
                
                # Reshape patches
                patches = patches.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
                # convert to 'batch_size, channels, height, width
                if return_prefix_tokens:
                    intermediates[i] = (cls_token, patches)
                else:
                    intermediates[i] = patches
        
        # if intermediates_only:
        #     return intermediates
        # return (x, intermediates) if intermediates else x
        return intermediates