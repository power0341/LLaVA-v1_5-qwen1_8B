from dataclasses import dataclass, field
from typing import Optional
import torch
from .depth_anything_encoder import DepthAnythingVisionTower
from .clip_encoder import CLIPVisionTower
import torch.nn.functional as F
import math
from einops import rearrange

@dataclass
class VisionConfig:
    image_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")

def parse_subargs(args):
    clip_args = VisionConfig()
    clip_args.image_tower = args.dual_tower_clip_image_tower
    clip_args.mm_vision_select_layer = args.dual_tower_clip_mm_vision_select_layer
    clip_args.mm_vision_select_feature = args.dual_tower_clip_mm_vision_select_feature
    da_args = VisionConfig()
    da_args.image_tower = args.dual_tower_depth_anything_image_tower
    da_args.mm_vision_select_layer = args.dual_tower_depth_anything_mm_vision_select_layer
    da_args.mm_vision_select_feature = args.dual_tower_depth_anything_mm_vision_select_feature
    return clip_args, da_args


class DualTowerEncoder(torch.nn.Module):
    def __init__(self, image_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = image_tower
        self.towers = torch.nn.ModuleList()
        clip_args, da_args = parse_subargs(args)

        self.towers.append(CLIPVisionTower(clip_args.image_tower, clip_args, True))        
        self.towers.append(DepthAnythingVisionTower(da_args.image_tower, da_args, True))
        
        if not delay_load:
            self.load_model()
        
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        for tower in self.towers:
            tower.load_model(device_map=device_map)
        self.image_processor = self.towers[0].image_processor
        self.is_loaded = True

    def convert_image_tensors(self, image):
        image_mean0 = self.towers[0].image_processor.image_mean
        image_std0 = self.towers[0].image_processor.image_std
        image_mean1 = self.towers[1].image_processor.image_mean
        image_std1 = self.towers[1].image_processor.image_std
        image *= torch.tensor(image_std0).to(image.device)[None, :, None, None]
        image += torch.tensor(image_mean0).to(image.device)[None, :, None, None]
        image = F.interpolate(image, size=(self.towers[1].image_processor.size['height'], self.towers[1].image_processor.size['width']), mode="bicubic", align_corners=False)
        image -= torch.tensor(image_mean1).to(image.device)[None, :, None, None]
        image /= torch.tensor(image_std1).to(image.device)[None, :, None, None]
        return image

    def predict_align_and_stack(self, image):
        visual_features0 = self.towers[0](image)
        image = self.convert_image_tensors(image)
        visual_features1 = self.towers[1](image)
        tower0_dim = self.towers[0].num_patches_per_side
        tower1_dim = self.towers[1].num_patches_per_side
        visual_features1 = rearrange(visual_features1, "b (h w) c -> b c h w", h=tower1_dim, w=tower1_dim)
        visual_features1 = F.interpolate(visual_features1, (tower0_dim, tower0_dim), mode="bilinear", align_corners=False)
        visual_features1 = rearrange(visual_features1, "b c h w -> b (h w) c", h=tower0_dim, w=tower0_dim)
        return torch.concat([visual_features0, visual_features1], dim=-1)

    @torch.no_grad()
    def forward(self, images):

        if type(images) is list:
            image_features = []
            for image in images:
                image_features.append(self.predict_align_and_stack(image))
        else:
            image_features = self.predict_align_and_stack(images)    
        
        return image_features

    @property
    def dtype(self):
        return self.towers[0].vision_tower.dtype

    @property
    def device(self):
        return self.towers[0].vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.towers[0].vision_tower.config
        else:
            return self.towers[0].cfg_only

    @property
    def hidden_size(self):
        return self.towers[0].hidden_size + self.towers[1].hidden_size
    
    @property
    def num_patches_per_side(self):
        return self.towers[0].config.image_size // self.towers[0].config.patch_size

    @property
    def num_patches(self):
        return (self.towers[0].config.image_size // self.towers[0].config.patch_size) ** 2
