# IMP_GCN

This is our implementation for the paper:

Fan Liu, Zhiyong Cheng*, Lei Zhu, Zan Gao and Liqiang Nie*. [Interest-aware Message-Passing GCN for Recommendation]


## Environment Settings
- Tensorflow-gpu version:  1.3.0

## Example to run the codes.

# gowalla
Run IMP_GCN.py
```
python IMP_GCN.py --dataset gowalla  --regs [1e-4] --embed_size 64 --layer_size [64,64,64,64,64,64] --lr 0.001 --batch_size 2048 --epoch 2000 --groups 3 --Ks [20,10] --gpu_id 0
```
