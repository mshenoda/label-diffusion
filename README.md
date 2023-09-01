# LabelDiffusion
### [**DiffuGen**](https://github.com/mshenoda/diffugen)'s core module for Labeling Stable Diffusion Pipeline


Combines the capabilities of diffusion models with two distinct labeling techniques: unsupervised and supervised.
Comes with its dedicated integral labeling pipeline for each diffusion task: Text-to-Image, Image-to-Image, and Inpainting

### Unsupervised method 
Uses cross attention heatmap to provide sematnic segmentation then finds contours to localize polygons and bounding boxes

### Supervised method
Utilizes existing model architectures to provide instance segemenations and bounding boxes;
Currently uses [YOLOv8-Seg](https://github.com/ultralytics/ultralytics)


## Installation
### PyTorch Dependency
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 
```
pip3 install labeldiffusion
```
