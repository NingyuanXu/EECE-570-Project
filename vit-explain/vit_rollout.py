from numpy.lib.nanfunctions import nanmean
import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2
from sklearn.decomposition import PCA

def rollout(attentions, discard_ratio, head_fusion, mode):
    # temporal (196, 12, 30, 30)
    # spatial (30, 12, 197, 197)
    with torch.no_grad():
      if mode == "spatial_only":
        result = torch.eye(attentions[0].size(-1))
        for attention in attentions:
          if head_fusion == "mean":
              attention_heads_fused = attention.mean(axis=1)
          elif head_fusion == "max":
              attention_heads_fused = attention.max(axis=1)[0]
          elif head_fusion == "min":
              attention_heads_fused = attention.min(axis=1)[0]
          else:
              raise "Attention head fusion type Not supported"
          # Drop the lowest attentions, but don't drop the class token
          flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
          _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
          indices = indices[indices != 0]
          flat[0, indices] = 0
          attention_heads_fused = flat.view(attention_heads_fused.shape)   
          I = torch.eye(attention_heads_fused.size(-1))
          a = (attention_heads_fused + 1.0*I)/2
          a = a / a.sum(dim=-1).unsqueeze(-1).expand_as(a)
          result = torch.matmul(a, result)
          # print(result.shape) # (30, 197, 197)
        # Look at the total attention between the class token and the image patches
        mask = result[:, 0, 1 :]
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(mask.size(0), width, width).numpy()
        mask = mask / np.max(mask)
        # print(mask.shape) # (30, 14, 14)
        return mask  
          
      # Spatial Temporal
      else:
        temp_result = torch.eye(attentions[0].size(-1))
        spat_result = torch.eye(attentions[1].size(-1))
        for idx, attention in enumerate(attentions):
          if head_fusion == "mean":
              attention_heads_fused = attention.mean(axis=1)
          elif head_fusion == "max":
              attention_heads_fused = attention.max(axis=1)[0]
          elif head_fusion == "min":
              attention_heads_fused = attention.min(axis=1)[0]
          else:
              raise "Attention head fusion type Not supported"
          # Drop the lowest attentions, but don't drop the class token
          flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
          _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
          indices = indices[indices != 0]
          flat[0, indices] = 0
          attention_heads_fused = flat.view(attention_heads_fused.shape)   
          I = torch.eye(attention_heads_fused.size(-1))
          a = (attention_heads_fused + 1.0*I)/2
          a = a / a.sum(dim=-1).unsqueeze(-1).expand_as(a)
          if idx % 2 == 0:
            temp_result = torch.matmul(a, temp_result) 
          else:
            spat_result = torch.matmul(a, spat_result)   
        #print(temp_result.shape) # (196, 30, 30)
        #print(spat_result.shape) # (30, 197, 197)  
        spat_mask = spat_result[:, 0, 1 :]
        width = int(spat_mask.size(-1)**0.5)
        spat_mask = spat_mask.reshape(spat_mask.size(0), width, width).numpy()
        spat_mask = spat_mask / np.max(spat_mask)
        temp_mask = temp_result.view(196, 900)
        pca = PCA(n_components=30)
        temp_mask = pca.fit_transform(temp_mask)
        temp_mask = torch.Tensor(temp_mask).view(196, 30)
        temp_mask = temp_mask.view(14, 14, 30).permute(2, 0, 1).numpy()
        mask = spat_mask + temp_mask
        print(mask.shape) # (30, 14, 14)
        return mask  

class VITAttentionRollout:
    def __init__(self, model, head_fusion="mean", mode="spatial_temporal", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.mode = mode
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
          if mode == "spatial_only":
            if 'attn.attn_drop' in name and 'temporal_attn' not in name:
              module.register_forward_hook(self.get_attention)
          else:
            if 'attn.attn_drop' in name:
              module.register_forward_hook(self.get_attention)
        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
        
        return rollout(self.attentions, self.discard_ratio, self.head_fusion, self.mode)