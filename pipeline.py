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

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from ultralytics import YOLO
from . import trace
from . import label_attention_map, label_image

__all__ = ["LabelDiffusion", "LabelDiffusionImg2Img", "LabelDiffusionInpaint"]

class LabelDiffusion():
    def __init__(self, pipe:StableDiffusionPipeline, yolo:YOLO=None, class_map:dict=None):
        self.txt2img = pipe
        self.txt2img.enable_xformers_memory_efficient_attention()
        self.yolo = yolo
        self.class_map = class_map
        
    def __call__(self, object_word, prompt, negative_prompt, num_inference_steps, generator, width, height, unsupervised=False, **kwargs):
        if unsupervised:
            return self.process_unsupervised(object_word, prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs)
        else:
            return self.process(object_word, prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs)
    
    def process(self, object_word, prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs):
        self.txt2img.to(generator.device)
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            with trace(self.txt2img) as tc:
                out = self.txt2img(prompt=prompt, 
                                negative_prompt=negative_prompt, 
                                num_inference_steps=num_inference_steps, 
                                generator=generator, 
                                width=width, height=height, 
                                **kwargs)

                width, height = out.images[0].size

                heat_map = tc.compute_global_heat_map()

                attention_heat_map = heat_map.compute_word_heat_map(object_word).get_heatmap_image(width, height)

                labels, binary_mask = label_image(out.images[0], self.yolo, self.class_map)

                return out, labels, binary_mask, attention_heat_map
            
    def process_unsupervised(self, object_word, prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs):
        self.txt2img.to(generator.device)
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            with trace(self.txt2img) as tc:
                out = self.txt2img(prompt=prompt, 
                                negative_prompt=negative_prompt, 
                                num_inference_steps=num_inference_steps, 
                                generator=generator, 
                                width=width, height=height, 
                                **kwargs)

                width, height = out.images[0].size

                heat_map = tc.compute_global_heat_map()

                attention_heat_map = heat_map.compute_word_heat_map(object_word).get_heatmap_image(width, height)

                labels, semantic_mask = label_attention_map(attention_heat_map, 70)

                return out, labels, semantic_mask, attention_heat_map

class LabelDiffusionImg2Img():
    def __init__(self, pipe:StableDiffusionImg2ImgPipeline, yolo:YOLO=None, class_map:dict=None):
        self.pipe = pipe
        self.pipe.enable_xformers_memory_efficient_attention()
        self.yolo = yolo
        self.class_map = class_map
        
    def __call__(self, image, object_word, prompt, negative_prompt, num_inference_steps, generator, width, height, unsupervised=False, **kwargs):
        if unsupervised:
            return self.process_unsupervised(image, object_word, prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs)
        else:
            return self.process(image, object_word, prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs)
    
    def process(self, image, object_word, prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs):
        self.pipe.to(generator.device)
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            with trace(self.pipe) as tc:
                out = self.pipe(prompt=prompt, 
                                   image=image,
                                negative_prompt=negative_prompt, 
                                num_inference_steps=num_inference_steps, 
                                generator=generator,
                                **kwargs)

                width, height = out.images[0].size

                heat_map = tc.compute_global_heat_map()

                attention_heat_map = heat_map.compute_word_heat_map(object_word).get_heatmap_image(width, height)

                labels, binary_mask = label_image(out.images[0], self.yolo, self.class_map)

                return out, labels, binary_mask, attention_heat_map
            
    def process_unsupervised(self, image, object_word, prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs):
        self.pipe.to(generator.device)
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            with trace(self.pipe) as tc:
                out = self.pipe(prompt=prompt, 
                                   image=image,
                                negative_prompt=negative_prompt, 
                                num_inference_steps=num_inference_steps, 
                                generator=generator,
                                **kwargs)

                width, height = out.images[0].size

                heat_map = tc.compute_global_heat_map()

                attention_heat_map = heat_map.compute_word_heat_map(object_word).get_heatmap_image(width, height)

                labels, semantic_mask = label_attention_map(attention_heat_map, 70)

                return out, labels, semantic_mask, attention_heat_map  
            
class LabelDiffusionInpaint():
    def __init__(self, pipe:StableDiffusionInpaintPipeline, yolo:YOLO=None, class_map:dict=None):
        self.pipe = pipe
        self.pipe.enable_xformers_memory_efficient_attention()
        self.yolo = yolo
        self.class_map = class_map
        
    def __call__(self, image, image_mask, inpaint_prompt, negative_prompt, num_inference_steps, generator, width, height, unsupervised=False, **kwargs):
        if unsupervised:
            return self.process_unsupervised(image, image_mask, inpaint_prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs)
        else:
            return self.process(image, image_mask, inpaint_prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs)
                        
    def process(self, image, image_mask, inpaint_prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs):
        self.pipe.to(generator.device)
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            with trace(self.pipe) as tc:
                out = self.pipe(prompt=inpaint_prompt, 
                                    image=image, 
                                    mask_image=image_mask,
                                    negative_prompt=negative_prompt, 
                                    num_inference_steps=num_inference_steps, 
                                    generator=generator, 
                                    width=width, height=height)
                width, height = out.images[0].size

                heat_map = tc.compute_global_heat_map()

                attention_heat_map = heat_map.compute_word_heat_map(inpaint_prompt).get_heatmap_image(width, height)
                labels = []
                semantic_mask = None
                if isinstance(self.yolo, YOLO):
                    labels, semantic_mask = label_image(out.images[0], self.yolo, self.class_map)

                return out, labels, semantic_mask, attention_heat_map
            
    def process_unsupervised(self, image, image_mask, inpaint_prompt, negative_prompt, num_inference_steps, generator, width, height, **kwargs):
        self.pipe.to(generator.device)
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            with trace(self.pipe) as tc:
                out = self.pipe(image=image, 
                                mask_image=image_mask,
                                prompt=inpaint_prompt, 
                                negative_prompt=negative_prompt, 
                                num_inference_steps=num_inference_steps, 
                                generator=generator, 
                                width=width, height=height, 
                                **kwargs)

                width, height = out.images[0].size

                heat_map = tc.compute_global_heat_map()

                attention_heat_map = heat_map.compute_word_heat_map(inpaint_prompt).get_heatmap_image(width, height)

                labels, semantic_mask = label_attention_map(attention_heat_map, 70)

                return out, labels, semantic_mask, attention_heat_map