# <span id="title"> Location-aware Graph Convolutional Networks for Video Question Answering </span>

This repo holds the codes for the L-GCN framework presented on AAAI 2020

**Location-aware Graph Convolutional Networks for Video Question Answering**
Deng Huang, Peihao Chen, Runhao Zeng, Qing Du, Mingkui Tan, Chuang Gan, *AAAI 2020*, New York.

[[Paper](https://arxiv.org/abs/2008.09105)]

![overview.png](https://i.loli.net/2020/08/21/KRk98ciC6eY2hGd.png)

# Contents

---
- [Usage Guide](#usage-guide)
    - [Code Preparation](#code)
    - [Module Preparation](#module)
    - [Data Preparation](#data)
    - [Training](#training)
- [Other Info](#other-info)
    - [Citation](#citation)
    - [Contact](#contact)
---



#  Usage Guide

## <span id="code"> Code Preparation <font size=3>[[back to top](#title)]</font> </span>

Clone this repo with git

```bash
git clone https://github.com/SunDoge/L-GCN.git
cd L-GCN
```

## <span id="module"> Module Preparation <font size=3>[[back to top](#title)]</font> </span>

This repo is based on Pytorch>=1.2

Other modules can be installed by running

```bash
pip install -r requirements.txt
python -m spacy download en
```

## <span id="data"> Data Preparation <font size=3>[[back to top](#title)]</font> </span>

### Data Processing

#### Save frames
Extract frames by following the instructions in [tgif-qa].

```bash
./save-frames.sh data/tgif/{gifs,frames}
```

Some GIF cannot be read by ffmpeg, you can use imagemagick to save the frames.

```bash
convert img.gif img/%d.jpg
```

#### Split frames

Since there are too many frames to process, we split them into N parts.

```bash
python -m scripts.split_n_parts
```

#### Get bboxes

Extract bboxes using [Mask R-CNN]. Check the script for more args.

```bash
python -m scripts.extract_bboxes_with_maskrcnn \
-f data/tgif/frames \
-o data/tgif/bboxes \
-c /path/to/e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml
```

#### Merge bboxes

```bash
python -m scripts.merge_bboxes \
--bboxes data/tgif/bboxes
-o data/tgif/bbox_features
```

#### Extract bbox features

```bash
python -m scripts.extract_resnet152_features_with_bboxes \
-i data/tgif/bboxes \
-o data/tgif/resnet152_layer4_features
```

#### Extract pool5 features

```bash
python -m scripts.extract_resnet152_features \
-i data/tgif/frames
```


## <span id="training"> Training <font size=3>[[back to top](#title)]</font> </span>

Use the following command to train L-GCN

```bash
python train.py -c config/resnet152-bbox/$TASK_CONFIG -e $PATH_TO_SAVE_RESULT
```

- `$TASK_CONFIG` denotes the config of task, there are four choice: `action.conf`, `transition.conf`, `frameqa.conf`, `count.conf`

- `$PATH_TO_SAVE_RESULT` denotes the path to save the result



# Other Info

## <span id="citation"> Citation <font size=3>[[back to top](#title)]</font> </span>

Please cite the following paper if you feel L-GCN useful to your research

```
@inproceedings{L-GCN2020AAAI,
  author    = {Deng Huang and
               Peihao Chen and
               Runhao Zeng and
               Qing Du and
               Mingkui Tan and
               Chuang Gan},
  title     = {Location-aware Graph Convolutional Networks for Video Question Answering},
  booktitle = {AAAI},
  year      = {2020},
}
```

## <span id="contact"> Contact <font size=3>[[back to top](#title)]</font> </span>

For any question, please file an issue or contact

```
im.huangdeng@gmail.com
phchencs@gmail.com
```

[Mask R-CNN]: (https://github.com/facebookresearch/maskrcnn-benchmark)
[tgif-qa]: (https://github.com/YunseokJANG/tgif-qa)
