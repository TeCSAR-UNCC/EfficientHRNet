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


## EfficientHRNet Training and Evaluation ## 

Distributed training is supported. Config settings can be customized based on user requirements. Training and validation scripts can be found at tools/

### Training on COCO (Nvidia GPUs) ###

(Single GPU training) CUDA_VISIBLE_DEVICES=0 python3 tools/dist_train.py --cfg experiments/coco/higher-hrnet/config.yaml 

(Distributed training) CUDA_VISIBLE_DEVICES=0,1 python3 tools/dist_train.py --cfg experiments/coco/higher-hrnet/config.yaml --dist-url tcp://127.0.0.1:12345

### Testing on COCO (Nvidia GPUs) ###

Single-scale and multi-scale testing is supported.

(Single scale testing) python3 tools/valid.py --cfg experiments/coco/higher-hrnet/config.yaml TEST.MODEL_FILE /path/to/model.pth 

(Multi-scale testing) python3 tools/valid.py --cfg experiments/coco/higher-hrnet/config.yaml TEST.MODEL_FILE /path/to/model.pth TEST.SCALE_FACTOR [0.5,1.0,1.5]

## TODO: ##

Make pretrained models public. 
