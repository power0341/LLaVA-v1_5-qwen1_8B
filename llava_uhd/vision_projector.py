import torch
import torch.nn as nn
import re
from resampler import Resampler
import math

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'resample':
        target_sequence_length = getattr(config, 'resampler_seq_len', 64)
        grid_size = int(math.sqrt(target_sequence_length))
        resampler = Resampler(
            grid_size=grid_size,
            embed_dim = getattr(config, 'resampler_embed_dim', 2048),  # 保持与视觉模型输出的 embed_dim 一致
            num_heads = getattr(config, 'resampler_num_heads', 8),  # 保持与视觉模型输出的 num_heads 一致
            kv_dim=getattr(config, 'resampler_kv_dim', 1024),  # 保持与视觉模型输出的 kv_dim 一致
        )
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')
    
    return resampler

    
