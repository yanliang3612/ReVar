# Rethinking Semi-Supervised Imbalanced Node Classification from Bias-Variance Decomposition (NeurIPS 2023).

## Introduction

Official Pytorch implementation of NeurIPS 2023 paper "[Rethinking Semi-Supervised Imbalanced Node Classification from Bias-Variance Decomposition](https://arxiv.org/abs/2310.18765)"

This paper introduces a new approach to address the issue of class imbalance in graph neural networks (GNNs) for learning on graph-structured data. Our approach integrates imbalanced node classification and Bias-Variance Decomposition, establishing a theoretical framework that closely relates data imbalance to model variance. We also leverage graph augmentation technique to estimate the variance, and design a regularization term to alleviate the impact of imbalance. This work provides a novel theoretical perspective for addressing the problem of imbalanced node classification in GNNs.

![variance_imbalance](figures/variance_imbalance.png)
