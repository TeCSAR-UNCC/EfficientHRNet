# EfficientHRNet

EfficientHRNet is a family of scalable and efficient networks created by unifiying EfficientNet and HigherHRNet for Multi-person human pose estimation. 

Our code is based on the 

1) Official implementation of HigherHRNet - https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation

2) PyTorch implementation of EfficientNet - https://github.com/narumiruna/efficientnet-pytorch


We provide a formulation for jointly scaling our backbone EfficientNet below the baseline B0 and the rest of EfficientHRNet with it. Ultimately, we are able to create a family
of highly accurate and efficient 2D human pose estimators that is flexible enough to provide lightweight solution for a variety of application and device requirements.

## Environment Setup ## 

* Pytorch >= 1.1.0

* Dependencies as listed in requirements.txt

* COCO Keypoints Dataset along with COCOAPI as given in https://cocodataset.org/#download and https://github.com/cocodataset/cocoapi

The code is developed using python 3.6. NVIDIA GPUs are needed. The code is developed and tested using NVIDIA V100 and Titan V GPU cards. Other platforms or GPU cards are not fully tested.


## Configurations for EfficientHRNet models ## 

Config files are found at experiments/coco/higher_hrnet/ 

Varying the following parameters provide different EfficientHRNet models ranging from H0 to H-4 :

  * scale_factor
  * input_size
  * width_mult
  * depth_mult

More details on scaling can be found in our paper: https://arxiv.org/pdf/2007.08090.pdf


## Training and Testing ## 

Distributed training is supported.

Single-scale and multi-scale testing is supported.

Training and validation scripts can be found at tools/


## TODO: ##

Make pretrained models public. 
