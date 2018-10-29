# ResidualAttentionNetwork-pytorch
A pytorch code about Residual Attention Network.  

This code is based on two  projects from 

https://github.com/liudaizong/Residual-Attention-Network 
and 
https://github.com/fwang91/residual-attention-network/blob/master/imagenet_model/Attention-92-deploy.prototxt

The first project is the pytorch code, but i think some network detail is not good. So I modify it according to 
the architechure of the Attention-92-deploy.prototxt.

And I also add the ResidualAttentionModel_92 for training imagenet,
ResidualAttentionModel_448input for larger image input,
and ResidualAttentionModel_92_32input_update for training cifar10.



# paper referenced
Residual Attention Network for Image Classification (CVPR-2017 Spotlight)
By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang


# how to train?
first, download the data from http://www.cs.toronto.edu/~kriz/cifar.html
make sure the varible 
# 
is_train = True
#
CUDA_VISIBLE_DEVICES=0 python train.py

CUDA_VISIBLE_DEVICES=0 python train_mixup.py(with mixup) 

you can train on ResidualAttentionModel_56 or ResidualAttentionModel_448input, only should modify the code in train.py
from  "from model.residual_attention_network import ResidualAttentionModel_92 as ResidualAttentionModel" to
"from model.residual_attention_network import ResidualAttentionModel_56 as ResidualAttentionModel"

# how to test?
make sure the varible 
#
is_train = False
#
CUDA_VISIBLE_DEVICES=0 python train.py

CUDA_VISIBLE_DEVICES=0 python train_mixup.py(with mixup) 

# result
1. cifar-10: Acc-95.4(Top-1 err 4.6) with ResidualAttentionModel_92_32input_update(higher than paper top-1 err 4.99)

2. cifar-10: Acc-96.65(Top-1 err 3.35) with ResidualAttentionModel_92_32input_update(with mixup).

3. cifar-10: Acc-96.84(Top-1 err 3.16) with ResidualAttentionModel_92_32input_update(with mixup, with simpler attention module).

Thanks to @PistonY, who give me the advice of mixup.
More details for mixup you can reference the project https://github.com/facebookresearch/mixup-cifar10

the paper only give the archietcture details of attention_92 for imagenet with 224 input but not for cifar10. So I build the net following my understanding. I have not struggled for optimizing the code, so maybe you can do better based my code.

# model file： 
model_92_sgd.pkl is the trained model file, accuracy of 0.954
