# EfficientHRNet

EfficientHRNet is a family of scalable and efficient networks created by unifiying EfficientNet and HigherHRNet for Multi-person human pose estimation. A preprint of our paper can be found [here.](https://arxiv.org/abs/2007.08090)

Our code is based on the 

1) Official implementation of [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)

2) PyTorch implementation of [EfficientNet](https://github.com/narumiruna/efficientnet-pytorch)


We provide a formulation for jointly scaling our backbone EfficientNet below the baseline B0 and the rest of EfficientHRNet with it. Ultimately, we are able to create a family
of highly accurate and efficient 2D human pose estimators that is flexible enough to provide lightweight solution for a variety of application and device requirements.


## Environment Setup ## 

* Pytorch >= 1.1.0

* Dependencies as listed in requirements.txt

* COCO Keypoints Dataset along with COCOAPI as given in https://cocodataset.org/#download and https://github.com/cocodataset/cocoapi

The code was developed using python 3.6 and NVIDIA GPUs, both of which are required.


## Configurations for EfficientHRNet models ## 

Config files are found at experiments/coco/higher_hrnet/ 

Varying the following parameters provide different EfficientHRNet models ranging from H0 to H-4 :

  * scale_factor
  * input_size
  * width_mult
  * depth_mult

More details on scaling can be found in our paper: https://arxiv.org/pdf/2007.08090.pdf

Examples can be seen in the Example Configs folder.


## EfficientHRNet Training and Evaluation ## 

Distributed training is supported. Config settings can be customized based on user requirements. Training and validation scripts can be found at tools/


### Training on COCO (Nvidia GPUs) ###

<b>Single GPU training example:</b>
```
CUDA_VISIBLE_DEVICES=0 python3 tools/dist_train.py --cfg experiments/coco/higher-hrnet/config.yaml 
```
<b>Distributed training example:</b>
```
CUDA_VISIBLE_DEVICES=0,1 python3 tools/dist_train.py --cfg experiments/coco/higher-hrnet/config.yaml --dist-url tcp://127.0.0.1:12345
```

### Testing on COCO (Nvidia GPUs) ###

Both single-scale and multi-scale testing are supported.

<b>Single scale testing:</b>
```
python3 tools/valid.py --cfg experiments/coco/higher-hrnet/config.yaml TEST.MODEL_FILE /path/to/model.pth 
```
<b>Multi-scale testing:</b>
```
python3 tools/valid.py --cfg experiments/coco/higher-hrnet/config.yaml TEST.MODEL_FILE /path/to/model.pth TEST.SCALE_FACTOR [0.5,1.0,1.5]
```

## Pretrained Models ##

COCO17 pretrained models for EfficientHRNet H0 to H-4 can be download [here.](https://drive.google.com/drive/folders/1FcJ1bawqWb1yAkcqb2sJfMsePMwupsWJ?usp=sharing)

|     Method     | Input Size | Parameters | FLOPs |  AP  | AP<sub>multi-scale</sub> |
|:--------------:|:----------:|:----------:|:-----:|:----:|:----:|
| H<sub>0</sub>  |    512     |   23.3M    | 25.6B | 64.0 | 67.1 |
| H<sub>-1</sub> |    480     |   16M      | 14.2B | 59.1 | 62.3 |
| H<sub>-2</sub> |    448     |   10.3M    | 7.7B  | 52.8 | 55.0 |
| H<sub>-3</sub> |    416     |   6.9M     | 4.2B  | 44.5 | 45.5 |
| H<sub>-4</sub> |    384     |   3.7M     | 2.1B  | 35.5 | 39.7 |


Compact EfficientNet ImageNet trained weights can be downloaded [here.](https://drive.google.com/drive/folders/1AZMYacfDcZv4QePcYONtg2in7oVmmSwV?usp=sharing)

|                |            |        |ImageNet    |       | Cifar-100  |       |
|:--------------:|:----------:|:------:|:----------:|:-----:|:----------:|:-----:|   
|     Method     | Input Size |  FLOPs | Parameters | Top-1 | Parameters | Top-1 |
| B0  |    512     |  0.4B  |    5.3M    |  75   |    4.1M    |  81.9 |
| B<sub>-1</sub> |    480     |  0.3B  |    4.5M    |  73.8 |    3.5M    |  81.4 |
| B<sub>-2</sub> |    448     |  0.2B  |    3.4M    |  71.3 |    2.5M    |  79.8 |
| B<sub>-3</sub> |    416     |  0.1B  |    2.8M    |  68.5 |    1.9M    |  78.2 |
| B<sub>-4</sub> |    384     |  0.05B |    1.3M    |  65.6 |    1.3M    |  74.3 |


## Citation ##

If you would like use EfficientHRNet in your work, please use the following citation.

```
@misc{neff2020efficienthrnet,
      title={EfficientHRNet: Efficient Scaling for Lightweight High-Resolution Multi-Person Pose Estimation}, 
      author={Christopher Neff and Aneri Sheth and Steven Furgurson and Hamed Tabkhi},
      year={2020},
      eprint={2007.08090},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

We also recommend citing EfficientNet and HigherHRNet, which inspired this work.
