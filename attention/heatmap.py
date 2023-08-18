from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, Tuple, Set

import numpy as np
import PIL.Image
from PIL import Image
import torch
import torch.nn.functional as F

from .utils import compute_token_merge_indices, auto_autocast

__all__ = ['GlobalHeatMap', 'RawHeatMapCollection', 'WordHeatMap']


def compute_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape[0] != b.shape[0]:
        a = F.interpolate(a.unsqueeze(0).unsqueeze(0).float(), size=b.shape, mode='bicubic').squeeze()
        a[a < 1] = 0
        a[a >= 1] = 1

    intersection = (a * b).sum()
    union = a.sum() + b.sum() - intersection

    return (intersection / (union + 1e-8)).item()


def compute_ioa(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape[0] != b.shape[0]:
        a = F.interpolate(a.unsqueeze(0).unsqueeze(0).float(), size=b.shape, mode='bicubic').squeeze()
        a[a < 1] = 0
        a[a >= 1] = 1

    intersection = (a * b).sum()
    area = a.sum()

    return (intersection / (area + 1e-8)).item()

def convert_rgba_to_rgb(rgba_image):
    # Check if the image is RGBA
    if rgba_image.mode != 'RGBA':
        raise ValueError("Input image must be in RGBA mode")

    # Create a new image with RGB mode and gray background
    rgb_image = Image.new("RGB", rgba_image.size, (255, 255, 255))
    
    # Composite the RGBA image onto the RGB image
    rgb_image.paste(rgba_image, mask=rgba_image.split()[3])
    
    return rgb_image

def get_heatmap(im, heat_map, word=None, crop=None, color_normalize=True, ax=None):

    with auto_autocast(dtype=torch.float32):
        im = np.array(im)

        if crop is not None:
            heat_map = heat_map.squeeze()[crop:-crop, crop:-crop]
            im = im[crop:-crop, crop:-crop]

        im = torch.from_numpy(im).float() / 255
        im_heatmap = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)

        # Convert the torch tensor back to a numpy array and then to a Pillow Image
        im_with_heatmap_np = (im_heatmap.cpu().numpy() * 255).astype(np.uint8)
        heatmap_image = Image.fromarray(im_with_heatmap_np)
        
        return convert_rgba_to_rgb(heatmap_image)

class WordHeatMap:
    def __init__(self, heatmap: torch.Tensor, word: str = None, word_idx: int = None):
        self.word = word
        self.word_idx = word_idx
        self.heatmap = heatmap

    @property
    def value(self):
        return self.heatmap

    def get_heatmap_image(self, width, height, **expand_kwargs):
        black_image = Image.new('RGB', (width, height), color='black')
        return get_heatmap(
            black_image,
            self.expand_as(black_image, **expand_kwargs)
        )
        
    def expand_as(self, image, absolute=False, threshold=None, plot=False, **plot_kwargs):
        # type: (PIL.Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor
        im = self.heatmap.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(image.size[0], image.size[1]), mode='bicubic')

        if not absolute:
            im = (im - im.min()) / (im.max() - im.min() + 1e-8)

        if threshold:
            im = (im > threshold).float()

        im = im.cpu().detach().squeeze()

        if plot:
            self.plot_overlay(image, **plot_kwargs)

        return im

    def compute_ioa(self, other: 'WordHeatMap'):
        return compute_ioa(self.heatmap, other.heatmap)


class GlobalHeatMap:
    def __init__(self, tokenizer: Any, prompt: str, heat_maps: torch.Tensor):
        self.tokenizer = tokenizer
        self.heat_maps = heat_maps
        self.prompt = prompt
        self.compute_word_heat_map = lru_cache(maxsize=50)(self.compute_word_heat_map)

    def compute_word_heat_map(self, word: str, word_idx: int = None, offset_idx: int = 0) -> WordHeatMap:
        merge_idxs, word_idx = compute_token_merge_indices(self.tokenizer, self.prompt, word, word_idx, offset_idx)
        return WordHeatMap(self.heat_maps[merge_idxs].mean(0), word, word_idx)


RawHeatMapKey = Tuple[int, int, int]  # factor, layer, head

class RawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)

    def update(self, factor: int, layer_idx: int, head_idx: int, heatmap: torch.Tensor):
        with auto_autocast(dtype=torch.float32):
            key = (factor, layer_idx, head_idx)
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
