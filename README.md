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
and ResidualAttentionModel_92_32input for training cifar10.



# paper referenced
Residual Attention Network for Image Classification (CVPR-2017 Spotlight)
By Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Chen Li, Honggang Zhang, Xiaogang Wang, Xiaoou Tang


# how to train?
first, download the data from http://www.cs.toronto.edu/~kriz/cifar.html
make sure the varible 
is_train = True
then python train.py

you can train on ResidualAttentionModel_56 or ResidualAttentionModel_448input, only should modify the code in train.py
from  "from model.residual_attention_network import ResidualAttentionModel_92 as ResidualAttentionModel" to
"from model.residual_attention_network import ResidualAttentionModel_56 as ResidualAttentionModel"

# how to test?
make sure the varible 
is_train = False
then python train.py

# result
I have test on ResidualAttentionModel_92 on cifar10 test set, the result is as following:
Accuracy of the model on the test images: 86 %
Accuracy of plane : 88 %
Accuracy of   car : 93 %
Accuracy of  bird : 79 %
Accuracy of   cat : 74 %
Accuracy of  deer : 85 %
Accuracy of   dog : 79 %
Accuracy of  frog : 89 %
Accuracy of horse : 90 %
Accuracy of  ship : 92 %
Accuracy of truck : 91 %
