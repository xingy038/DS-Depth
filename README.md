# DS-Depth: Dynamic and Static Depth Estimation via a Fusion Cost Volume

[![arXiv](https://img.shields.io/badge/arXiv-2308.07225-b31b1b.svg)](https://arxiv.org/abs/2308.07225)
[![IEEE](https://img.shields.io/badge/DOI-10.1109/TCSVT.2023.3305776-blue.svg)](https://doi.org/10.1109/TCSVT.2023.3305776)

> **DS-Depth: Dynamic and Static Depth Estimation via a Fusion Cost Volume**<br>
> [Paper](https://arxiv.org/abs/2308.07225)<br>
> Xingyu Miao, Yang Bai, Haoran Duan, Yawen Huang, Fan Wan, Xinxing Xu,
Yang Long, Yefeng Zheng<br>
> Accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)



## Setup

To get started, please create the conda environment by running

```bash
cd DSdepth
conda env create -f environment.yaml
conda activate dsdepth
```

## Train

To train a KITTI model, run:

```bash
python -m dsdepth.train \
    --data_path <your_KITTI_path> \
    --log_dir <your_save_path> \
    --model_name <your_model_name>
```


For instructions on downloading the KITTI dataset, see [Monodepth2](https://github.com/nianticlabs/monodepth2)

To train a CityScapes model, run:

```bash
python -m dsdepth.train \
    --data_path <your_preprocessed_cityscapes_path> \
    --log_dir <your_save_path> \
    --model_name <your_model_name> \
    --dataset cityscapes_preprocessed \
    --split cityscapes_preprocessed \
    --freeze_teacher_epoch 5 \
    --height 192 --width 512
```

This assumes you have already preprocessed the CityScapes dataset.
If you have not yet processed the CityScapes data set, please refer to [ManyDepth](https://github.com/nianticlabs/manydepth) for processing.


## Evaluation

### KITTI dataset

First you have run `export_gt_depth.py` to extract ground truth files.

To evaluate a model on KITTI, run:

```bash
python -m dsdepth.evaluate_depth \
    --data_path <your_KITTI_path> \
    --load_weights_folder <your_model_path>
    --eval_mono
    --eval_split eigen
```

### Cityscapes dataset

The ground truth depth files [Here](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip).

To evaluate a model on Cityscapes, run:

```bash
python -m dsdepth.evaluate_depth \
    --data_path <your_cityscapes_path> \
    --load_weights_folder <your_model_path>
    --eval_mono \
    --eval_split cityscapes
```
And to evaluate a model on Cityscapes (Dynamic region only), run:

```bash
python -m dsdepth.evaluate_depth_dynamic \
    --data_path <your_cityscapes_path> \
    --load_weights_folder <your_model_path>
    --eval_mono \
    --eval_split cityscapes
```

Please make sure you switch the dynamic region dataloader. And the dynamic object masks for Cityscapes dataset can download from [Here](https://github.com/AutoAILab/DynamicDepth).

##  Pretrained weights

You can download weights for some pretrained models here:

* [KITTI (AbsRel 0.095)](https://drive.google.com/file/d/1nK_YX-ZMWQF5GPDW0i-J0tIsh3mUQQqy/view?usp=drive_link)
* [CityScapes (AbsRel 0.100)](https://drive.google.com/file/d/1T8a5SyYZAd6CHnegPcLbqC7AlF69SuWZ/view?usp=drive_link)

If you have any concern with this paper or implementation, welcome to open an issue or email me at
[xingyu.miao@durham.ac.uk](xingyu.miao@durham.ac.uk).

## Citation

If you find this code useful for your research, please consider citing the following paper:

```latex

```

## Acknowledgments
Our training code is build upon [ManyDepth](https://github.com/nianticlabs/manydepth).



