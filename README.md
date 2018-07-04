Implementation of a number of MobileNet variants, based on https://github.com/marvis/pytorch-mobilenet.
Imagenet data is processed [as described here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)

Options Implemented:
- `Residual Connection`: Residual connection around each depthwise separable convolution. 
- `Squeeze-and-Excitation Channel Attention`: Based on the [Squeeze-and-Excitation paper](https://arxiv.org/abs/1709.01507), I add the squeeze and excite block after every depthwise separable convolution. 
- `Group Convolutions`: For the 3x3 convolutions in the depthwise seperable convolutions, I allow them to have group size of greater than 1. 

Learning rate schedule: I use Nesterov and cosine learning rate starting at LR = 0.05 and train it for 90 epochs.


| Architecture  | Explanation   | Accuracy  |
|:------------- |:-------------| :-----:|
| mobilenet | Baseline architecture | 71.84|
| mobilenetg4 | Using group size of 4 in the 3x3s of dep-sep conv| 73.184 |
| mobilenetr  | Using residual connections | 71.94|
| mobilenetra  | Using residual connections and squeeze-and-excite blocks | 73.48|
| mobilenetrag4  | Using residual connections, squeeze-excite blocks and group size of 4 in 3x3s | 74.13|

`Command to train`: python main.py -a <ARCH> -b 256 --cosineLR --lr 0.05 --nesterov /imagenet/
where <ARCH> is one of the options from the first column in the above table.


