# Superglue-Jittor

Code based on [SuperGluePretrainedNetwork](https://github.com/Skydes/SuperGluePretrainedNetwork)

## Introduction

This is a jittor implementation of paper: [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763)

The superglue-jittor network is integrated into **[Hierarchical-Localization](https://github.com/cvg/Hierarchical-Localization)** 



## Dependencies

python3

jittor

## Contents

1. **test_gpu/superglue.py**: jittor implementation of superglue model.

2. **test_gpu/test_gpu.py**: evaluate the speed of our implementation.

   for more information of evaluating time perfoemance, see test_gpu/README.md

3. pipeline_inloc_aslfeat.py: a pipeline of Hieratical-Localization with Inloc dataset.