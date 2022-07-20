# Semantic-guided Multi-mask Image Harmonization

## Introduction

This is the official code of the paper: Semantic-guided Multi-mask Image Harmonization

## Quick Start

### Data Preparation

In this paper, we constructs two benchmarks HScene and HLIP, we also conduct expriments on [iHarmony4](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4).

The datasets of HScene and HLIP can be download from [Google Drive](https://drive.google.com/drive/folders/1Kd32kfQFtHhnXWvqzwvX9Q33H_HD-R_N?usp=sharing). The composite dataset in /datasets/HScene(HLIP)/test/composite is the test dataset we generated.

Download the datasets and put them under the SgMMH folder.

### Train and Test

Our code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR), thanks for its excellent projects.

We provide a training and a test examples: train.sh, test.sh

One quick training command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/dist_train.sh 4 \
options/train/OMGAN/train_OM_Mask_HScene.yml
```

One quick testing command:

```bash
CUDA_VISIBLE_DEVICES=0  ./scripts/dist_test.sh 1 \
options/test/OMGAN/test_OM_HScene.yml
```

One quick evaluation command:

```bash
python basicsr/metrics/calculate_lpips.py --path results/test_OM_HScene/visualization/HScene
```

### Pretrained Model

Sg-MMH: [Google Drive](https://drive.google.com/drive/folders/1Gg1U9IFikOlXCId-IUlyQEr7I7LBKN2S?usp=sharing)

Download the model and change the path in each yml files. 

We also revise [HarmonyTransformer](https://github.com/zhenglab/HarmonyTransformer) with our operator masks, and provide the pretrained model in [Google Drive](https://drive.google.com/drive/folders/10IQg3RWCBg0tR6b5booiz8I8wlU8FtLP?usp=sharing).

### Acknowledgement

For the whole code framework and some of the data modules and model functions used in the source code, we need to acknowledge the repo of [BasicSR](https://github.com/XPixelGroup/BasicSR), [DoveNet](https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4/tree/master/DoveNet), [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix),[HarmonyTransformer](https://github.com/zhenglab/HarmonyTransformer).

