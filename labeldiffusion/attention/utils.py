#   Original Paper Title: "What the DAAM: Interpreting Stable Diffusion Using Cross Attention"
#   Original Paper URL:  https://arxiv.org/abs/2210.04885
#   Original Implementation URL: https://github.com/castorini/daam
#   Original License: MIT License - Copyright (c) 2022 Castorini
# --------------------------------------------------------------------------------------------
#   LabelDiffusion - Automatic Labeling of Stable Diffusion Pipelines
#   Copyright (C) 2023  Michael Shenoda
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import random
from typing import TypeVar
import numpy as np
import torch

__all__ = ['create_generator', 'compute_token_merge_indices', 'auto_device', 'auto_autocast']

T = TypeVar('T')

def auto_device(obj: T = torch.device('cpu')) -> T:
    if isinstance(obj, torch.device):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        return obj.to('cuda')

    return obj

def auto_autocast(*args, **kwargs):
    if not torch.cuda.is_available():
        kwargs['enabled'] = False

    return torch.cuda.amp.autocast(*args, **kwargs)

def create_generator(seed: int, device:str=None, auto_select_device:bool=False) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    gen = None
    
    if auto_select_device:
        gen = torch.Generator(device=auto_device())
    elif device != None:
        gen = torch.Generator(device=device)
    else:
        gen = torch.Generator()
    
    gen.manual_seed(seed)

    return gen

def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0):
    merge_idxs = []
    tokens = tokenizer.tokenize(prompt.lower())
    if word_idx is None:
        word = word.lower()
        search_tokens = tokenizer.tokenize(word)
        start_indices = [x + offset_idx for x in range(len(tokens)) if tokens[x:x + len(search_tokens)] == search_tokens]
        for indice in start_indices:
            merge_idxs += [i + indice for i in range(0, len(search_tokens))]
        if not merge_idxs:
            raise ValueError(f'Search word {word} not found in prompt!')
    else:
        merge_idxs.append(word_idx)

    return [x + 1 for x in merge_idxs], word_idx  # Offset by 1.

