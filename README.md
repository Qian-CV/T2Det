# T<sup>2</sup>Det: Twin-tower detector with geometric invariance for oriented object detection

![](docs/Figure/Fig1.png)

## Introduction

This repository is the official implementation of "T<sup>2</sup>Det: Twin-tower detector with geometric invariance for oriented object detection" at [Please stay tuned!]

The master branch is built on MMRotate which works with **PyTorch 1.8+**.

T<sup>2</sup>Det's train/test configure files are placed under configs/exp_configs/t2det/

How to utilize the dynamic perception of T<sup>2</sup>Det can be referenced to [here](docs/en/tutorials/dynamic_perception.md).

## Deep Learning Experiments

### Source of Pre-trained models

* CSPNeXt-m: pre-trained checkpoint supported by Openmmlab([link](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth)).
* ResNet: pre-trained ResNet50 supported by Pytorch.

### Results and models


#### 1. VEDAI

|                    Model                     |  mAP  | Angle | lr schd | Batch Size |                           Configs                            |                           Download                           |
| :------------------------------------------: | :---: | :---: | :-----: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [RTMDet-M](https://arxiv.org/abs/2212.07784) | 83.32 | le90  |   6x    |     4      |                                                              | [model](https://drive.google.com/file/d/1WCXqfqfS9sslkJ2OOk8A1RmxOteLShaV/view?usp=sharing) |
|               T<sup>2</sup>Det               | 85.15 | le90  |   6x    |     4      | [t2det-vedai](./configs/exp_configs/t2det/VEDAI/t2det_rtmdet_m-6x-vedai.py) | [model](https://drive.google.com/file/d/1SSPv49ms1Vs9tGHj48TZ3zdjqaVAWASo/view?usp=sharing) \|[log](./tools/work_dirs/PG-DRFNet/VEDAI_log.log) |

#### 2. HRSC2016

|  Model  |  mAP  | Angle | lr schd | Batch Size |                           Configs                            |                           Download                           |
| :-----: | :---: | :---: | :-----: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| T<sup>2</sup>Det | 90.66 | le90  |   6x    |     8      | [t2det-hrsc2016](./configs/exp_configs/t2det/hrsc/t2det_rtmdet_m-6x-hrsc.py) | [model](https://drive.google.com/file/d/1RT7sitAzAcmMcXiLQ4nYavxJBmvXoiBz/view?usp=sharing) \| [log](./tools/work_dirs/PG-DRFNet/DOTA_log.log) |


For example, when dataset is VEDAI and method is T<sup>2</sup>Det, you can train by running the following

```bash
python tools/train.py \
  --config configs/PG-DRFNet/pg_drfnet-6x-dota2.py \
  --work-dir work_dirs/PG-DRFNet \
  --load_from path/to/pre-trained/model \
```

and if you want test the VEDAI results, you can run  as follows

```bash
python tools/test.py \
  --config configs/PG-DRFNet/pg_drfnet-6x-dota2.py \
  --checkpoint path/to/gvt/model \
  --cfg-options test_dataloader.dataset.ann_file=''  test_dataloader.dataset.data_prefix.img_path=test/images/ test_evaluator.format_only=True test_evaluator.merge_patches=True test_evaluator.outfile_prefix='path/to/save_dir'
```

### Hyperparameters Configuration

Detailed hyperparameters config can be found in configs/base/ and configs/PG-DRFNet/

## Installation

[MMRotate](https://github.com/open-mmlab/mmrotate/tree/1.x) depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install -U openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/Qian-CV/PG-DRFNet.git
cd PG-DRFNet
pip install -v -e .
```

## Get Started

Please see [here](docs/en/get_started.md) for the basic usage of MMRotate.
We also provide some tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [dynamic perception](docs/en/tutorials/dynamic_perception.md)
- [useful tools](docs/en/tutorials/useful_tools.md)

## Acknowledgments

The code is developed based on the following repositories. We appreciate their nice implementations.

|  Method  |                Repository                 |
| :------: | :---------------------------------------: |
|  RTMDet  | https://github.com/open-mmlab/mmdetection |
| RTMDet-R |  https://github.com/open-mmlab/mmrotate   |
|  ECANet  |    https://github.com/BangguWu/ECANet     |
|  QFocal  |     https://github.com/implus/GFocal      |

## Cite this repository

If you use this software in your work, please cite it using the following metadata. Liuqian Wang, Jing Zhang, et. al. (2024). T<sup>2</sup>Det by BJUT-AI&VBD [Computer software]. https://github.com/Qian-CV/T2Det.git
