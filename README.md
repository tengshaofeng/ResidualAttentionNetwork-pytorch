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
# 
is_train = True
#
then python train.py

you can train on ResidualAttentionModel_56 or ResidualAttentionModel_448input, only should modify the code in train.py
from  "from model.residual_attention_network import ResidualAttentionModel_92 as ResidualAttentionModel" to
"from model.residual_attention_network import ResidualAttentionModel_56 as ResidualAttentionModel"

# how to test?
make sure the varible 
#
is_train = False
#
then python train.py

# result
I have test on ResidualAttentionModel_92_32input on cifar10 test set, the result is as following:
# 
Accuracy of the model on the test images: 92.31 %
#
Accuracy of plane : 94 %

Accuracy of   car : 96 %

Accuracy of  bird : 88 %

Accuracy of   cat : 85 %

Accuracy of  deer : 93 %

Accuracy of   dog : 86 %

Accuracy of  frog : 94 %

Accuracy of horse : 94 %

Accuracy of  ship : 94 %

Accuracy of truck : 94 %

#
the paper only give the archietcture details of attention_92 for imagenet with 224 input but not for cifar10. So I build the net following my understanding. I have not struggled for optimizing the code, so maybe you can do better based my code.
For example, you can add subtract the mean for preprocessing, you can avgpooling when feature map is 2,2 not 4,4 in my code, so many tricks.
