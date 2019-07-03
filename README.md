# Idea

Replay based learning for CNNs
# CIFAR10 PyTorch experiments
Code based on https://github.com/kuangliu/pytorch-cifar

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Accuracy
| Model             | Acc.        | Time        |
| ----------------- | ----------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |ToDo
| [VGG16 + replay](https://arxiv.org/abs/1409.1556)              | ToDo      |ToDo

## Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,150)`
- `0.01` for epoch `[150,250)`
- `0.001` for epoch `[250,350)`

Resume the training with `python main.py --resume --lr=0.01`
