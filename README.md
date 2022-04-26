# Superglue-Jittor

Code based on [SuperGluePretrainedNetwork](https://github.com/Skydes/SuperGluePretrainedNetwork)

## Introduction

This is a jittor implementation of paper: [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763)

we improve the superglue model with followings:

图神经网络模块：借鉴自Transformer的简单变形应用，由特征点位置（position）和描述子（descriptor）共同组成输入嵌入模块（Input Embedding），并在累叠的Self-Attention层（同一图像的内部注意力计算）中额外扩展了Cross-Attention层（不同图像的跨图注意力计算）。在默认全连接网络图注意学习中，该模块的复杂度为O(N2)，在运行时间和显存空间占用上都比通常的卷积上都高出一个数量级。根据其计算量决定性的资源消耗性质，我们对矩阵运算进行分块优化，在不影响并行数的前提下极大降低显存占用。

最优传输匹配模块：原版采用的是Sinkhorn来迭代求解近似的匹配矩阵。我们根据实际实验效果将其替换成更加简化的dual-Softmax算子来优化迭代计算。


To show the matching result, we integrate our model into a Hieratical-Localization pipeline.



## Dependencies

python3

jittor

## Contents

1. **test_gpu/superglue.py**: jittor implementation of superglue model.

2. **test_gpu/test_gpu.py**: evaluate the speed of our implementation.

   for more information of evaluating time perfoemance, see test_gpu/README.md

3. pipeline_inloc_aslfeat.py: a pipeline of Hieratical-Localization with Inloc dataset.

## Checkpoints

you can get checkpoints for aslfeat model at https://research.altizure.com/data/aslfeat_models/aslfeat.tar and put it in /third_party/aslfeat/pretrained/
you can get corresponding superglue model at https://cloud.tsinghua.edu.cn/f/5e2701c1416640af8a02/?dl=1 and put it in /pack

## Dataset
you can get test data at https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/InLoc/ or http://www.ok.sc.e.titech.ac.jp/INLOC/. Image-only is enough for this project.
