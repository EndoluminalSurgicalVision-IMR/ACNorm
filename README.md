# ACNorm
## The code of our paper:
Chuyan Zhang, Yuncheng Yang, Xin You, Hao Zheng, Yun Gu, "AC-Norm: Effective Tuning for Medical Image Analysis via Affine Collaborative Normalization"

## Introduction
We propose the AC-Norm as a new normalization technique tailored for effective fine-tuning. Besides, a metric called AC-Corr is introduced for fast transferability estimation among a zoo of pretrained models.

## How to perform fine-tuning?
In your own model, replace the original BatchNorm module with AC-Norm module in affine_co_norm.py.

## How to perform transferability estimation?
After a fast one-epoch fine-tuning, run get_ac_corr.py to get the estimated transferability value. Then, you can rank the 
transferability values of diverse pretrained models (self-supervised learned via different pretext tasks or fully supervised learned on different datasets) and pick the best one for continual fine-tuning.
