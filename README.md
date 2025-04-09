# Rethinking Semi-Supervised Imbalanced Node Classification from Bias-Variance Decomposition (NeurIPS 2023).

This repository is the official implementation of "Rethinking Semi-Supervised Imbalanced Node Classification from Bias-Variance Decomposition" (NeurIPS 2023). 

[[Project Page](https://divinyan.com/ReVar/) [Arxiv](https://arxiv.org/abs/2310.18765) [OpenReview](https://openreview.net/forum?id=0gvtoxhvMY&noteId=bBcG4XGOE8) [Slides](https://nips.cc/media/neurips-2023/Slides/73050.pdf) [Poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202023/73050.png?t=1702115401.9719656)]

Authors: Liang Yan, Gengchen Wei, Chen Yang, Shengzhong Zhang, Zengfeng Huang.

## Introduction

![variance_imbalance](figures/variance_imbalance.png)
![revar](figures/revar.png)
This paper introduces a new approach to address the issue of class imbalance in graph neural networks (GNNs) for learning on graph-structured data. Our approach integrates imbalanced node classification and Bias-Variance Decomposition, establishing a theoretical framework that closely relates data imbalance to model variance. We also leverage graph augmentation technique to estimate the variance, and design a regularization term to alleviate the impact of imbalance. This work provides a novel theoretical perspective for addressing the problem of imbalanced node classification in GNNs.

## Environment
```bash
conda create -n "revar" python=3.8.13
source activate revar
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install torch_geometric
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/pyg_lib-0.3.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.16%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1%2Bpt112cu113-cp38-cp38-linux_x86_64.whl
```

## The Implementation of Baselines and the Configuration of Hyperparameters
- For the implementation and hyperparameters setting of **Re-Weight, PC Softmax, BalancedSoftmax, TAM**, please refer to [TAM](https://github.com/Jaeyun-Song/TAM).
- For the implementation and hyperparameters setting of **GraphSmote**, please refer to [GraphSmote](https://github.com/TianxiangZhao/GraphSmote).
- For the implementation and hyperparameters setting of **Renode**, please refer to [Renode](https://github.com/victorchen96/ReNode).
- For the implementation and hyperparameters setting of **GraphENS**, please refer to [GraphENS](https://github.com/JoonHyung-Park/GraphENS).

We strictly adhere to the hyperparameter settings as specified in these papers. For detailed information, please refer to the respective publications.

## The hyperparameter settings for each experiment:
###  CiteSeer_semi_GAT_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset CiteSeer --de_1 0.45 --de_2 0.4 --decay 0.01 --df_1 0.5 --df_2 0.4 --dim 128 --epochs 2000 --imb_ratio 10.0 --lam 0.25 --lam2 3 --layers 2 --lr 0.001 --n_head 8 --net GAT --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.21 --thres 0.7
```

### CiteSeer_semi_GCN_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset CiteSeer --de_1 0.65 --de_2 0.35 --decay 0.01 --df_1 0.4 --df_2 0.2 --dim 512 --epochs 2000 --imb_ratio 10.0 --lam 0.25 --lam2 2.85 --layers 4 --lr 0.005 --n_head 8 --net GCN --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.13 --thres 0.6
```

### CiteSeer_semi_SAGE_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset CiteSeer --de_1 0.65 --de_2 0.15 --decay 0.01 --df_1 0.7 --df_2 0.15 --dim 256 --epochs 2000 --imb_ratio 10.0 --lam 0.25 --lam2 1.25 --layers 2 --lr 0.0005 --n_head 8 --net SAGE --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.08 --thres 0.6
```

### Computers_semi_GAT_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset Computers-semi --de_1 0.45 --de_2 0.35 --decay 0.01 --df_1 0.7 --df_2 0.15 --dim 512 --epochs 2000 --imb_ratio 10.0 --lam 0.35 --lam2 1.5 --layers 3 --lr 0.0005 --n_head 8 --net GAT --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.21 --thres 0.99
```

### Computers_semi_GCN_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset Computers-semi --de_1 0.7 --de_2 0.2 --decay 0.01 --df_1 0.4 --df_2 0.1 --dim 256 --epochs 2000 --imb_ratio 10.0 --lam 0.35 --lam2 3 --layers 3 --lr 0.0001 --n_head 8 --net GCN --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.26 --thres 0.66
```

### Computers_semi_SAGE_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset Computers-semi --de_1 0.4 --de_2 0.1 --decay 0.01 --df_1 0.6 --df_2 0.15 --dim 64 --epochs 2000 --imb_ratio 10.0 --lam 0.5 --lam2 3 --layers 4 --lr 0.0005 --n_head 8 --net SAGE --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.08 --thres 0.66
```

### Cora_semi_GAT_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset Cora --de_1 0.55 --de_2 0.1 --decay 0.01 --df_1 0.6 --df_2 0.3 --dim 512 --epochs 2000 --imb_ratio 10.0 --lam 3 --lam2 0.35 --layers 3 --lr 0.01 --n_head 8 --net GAT --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.08 --thres 0.99
```

### Cora_semi_GCN_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset Cora --de_1 0.45 --de_2 0.3 --decay 0.01 --df_1 0.65 --df_2 0.45 --dim 128 --epochs 2000 --imb_ratio 10.0 --lam 0.5 --lam2 0.35 --layers 4 --lr 0.01 --n_head 8 --net GCN --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.16 --thres 0.9
```

### Cora_semi_SAGE_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset Cora --de_1 0.6 --de_2 0.1 --decay 0.01 --df_1 0.7 --df_2 0.4 --dim 512 --epochs 2000 --imb_ratio 10.0 --lam 0.25 --lam2 0.5 --layers 3 --lr 0.0005 --n_head 8 --net SAGE --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.05 --thres 0.8
```

### PubMed_semi_GAT_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset PubMed --de_1 0.65 --de_2 0.4 --decay 0.01 --df_1 0.4 --df_2 0.45 --dim 512 --epochs 2000 --imb_ratio 10.0 --lam 2.15 --lam2 1.5 --layers 3 --lr 0.1 --n_head 8 --net GAT --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.23 --thres 0.9
```

### PubMed_semi_GCN_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset PubMed --de_1 0.65 --de_2 0.15 --decay 0.01 --df_1 0.4 --df_2 0.1 --dim 512 --epochs 2000 --imb_ratio 10.0 --lam 3 --lam2 3 --layers 2 --lr 0.1 --n_head 8 --net GCN --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.13 --thres 0.93
```

### PubMed_semi_SAGE_10
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset PubMed --de_1 0.5 --de_2 0.15 --decay 0.01 --df_1 0.4 --df_2 0.45 --dim 512 --epochs 2000 --imb_ratio 10.0 --lam 2.65 --lam2 3 --layers 2 --lr 0.1 --n_head 8 --net SAGE --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.16 --thres 0.96
```






### Computers_random_GAT
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset Computers-random --de_1 0.5 --de_2 0.45 --decay 0.01 --df_1 0.45 --df_2 0.1 --dim 128 --epochs 2000 --imb_ratio 1.0 --lam 0.35 --lam2 1.25 --layers 4 --lr 0.001 --n_head 8 --net GAT --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.23 --thres 0.6
```
### Computers_random_GCN
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset Computers-random --de_1 0.65 --de_2 0.15 --decay 0.01 --df_1 0.7 --df_2 0.1 --dim 512 --epochs 2000 --imb_ratio 1.0 --lam 3 --lam2 2.85 --layers 2 --lr 0.0005 --n_head 8 --net GCN --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.23 --thres 0.83
```

### Computers_random_SAGE
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset Computers-random --de_1 0.4 --de_2 0.2 --decay 0.01 --df_1 0.4 --df_2 0.15 --dim 128 --epochs 2000 --imb_ratio 1.0 --lam 1 --lam2 1 --layers 4 --lr 0.0005 --n_head 8 --net SAGE --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.23 --thres 0.99
```

### CS_random_GAT
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset CS-random --de_1 0.55 --de_2 0.2 --decay 0.01 --df_1 0.7 --df_2 0.4 --dim 512 --epochs 2000 --imb_ratio 1.0 --lam 2 --lam2 0.5 --layers 2 --lr 0.0001 --n_head 8 --net GAT --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.05 --thres 0.63
```

### CS_random_GCN
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset CS-random --de_1 0.45 --de_2 0.3 --decay 0.01 --df_1 0.7 --df_2 0.2 --dim 512 --epochs 2000 --imb_ratio 1.0 --lam 0.85 --lam2 0.5 --layers 2 --lr 0.001 --n_head 8 --net GCN --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.16 --thres 0.7
```

### CS_random_SAGE
```bash
python main.py --balancedmask False --chebgcn_para 2 --classcenter True --datadir /tmp/data --dataset CS-random --de_1 0.7 --de_2 0.4 --decay 0.01 --df_1 0.45 --df_2 0.2 --dim 512 --epochs 2000 --imb_ratio 1.0 --lam 1.5 --lam2 0.5 --layers 4 --lr 0.001 --n_head 8 --net SAGE --patience 200 --project rvgnn --repetitions 5 --supervised True --tau 0.13 --thres 0.6
```

## Configuration
All the algorithms and models are implemented in Python and PyTorch Geometric. Experiments are
conducted on a server with an NVIDIA 3090 GPU (24 GB memory) and an Intel(R) Xeon(R) Silver
4210R CPU @ 2.40GHz.

## Citation
```
@inproceedings{
yan2023rethinking,
title={Rethinking Semi-Supervised Imbalanced Node Classification from Bias-Variance Decomposition},
author={Divin Yan and Gengchen Wei and Chen Yang and Shengzhong Zhang and Zengfeng Huang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=0gvtoxhvMY}
}
```

## Acknowledgement
This work is supported by National Natural Science Foundation of China No.U2241212, No.62276066. We extend our gratitude to Jaeyun-Song for their meticulous organization of the baselines implementation within the [TAM framework](https://github.com/Jaeyun-Song/TAM).
