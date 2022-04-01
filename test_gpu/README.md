# superglue jittor 版本使用方法

superglue 性能优化，相比 PyTorch 性能提升20%～30%， 显存节省2-3倍

```
# 默认配置, fp32, no tensorcore
# 3090 上 627 ms， 显存 2817 mb
python3 ./test_gpu.py

# fp32, 开启 tensorcore
# 3090 上 493 ms， 显存 2817 mb
use_tensorcore=1 python3 ./test_gpu.py

# fp32, 开启 tensorcore, conv 显存优化
# 3090 上 523 ms， 显存 2223 mb
conv_opt=1 use_tensorcore=1 python3 ./test_gpu.py

# fp16, 开启 tensorcore
# 3090 上 254 ms， 显存 2329 mb
use_fp16=1 python3 ./test_gpu.py

# fp16, 开启 tensorcore， conv 显存优化
# 3090 上 258 ms， 显存 1350 mb
conv_opt=1 use_fp16=1 python3 ./test_gpu.py


# fp16, 开启 tensorcore， conv 显存优化
# 切块大小1节省显存
# 3090 上 310 ms， 显存 970 mb
split_size=1 conv_opt=1 use_fp16=1 python3 ./test_gpu.py


```