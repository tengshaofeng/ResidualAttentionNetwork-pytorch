from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
from torchvision.utils import save_image
import os
import cv2
from model.basic_layers import ResidualBlock
from model.attention_module import AttentionModule_pre as AttentionModule

# Image Preprocessing 
transform = transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor()])

test_dataset = datasets.CIFAR10(root='./data/',
                              train=False, 
                              transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=20, 
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class ResidualAttentionModel(nn.Module):
    def __init__(self):
        super(ResidualAttentionModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionModule(256, 256, (56,56), (28,28), (14,14))
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionModule(512, 512, (28,28), (14,14), (7,7))
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionModule(1024, 1024, (14,14), (7,7), (4,4))
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Linear(2048,10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mpool1(out)
        # print(out.data)
        out = self.residual_block1(out)
        out1 = self.attention_module1(out)
        out = self.residual_block2(out1)
        out2 = self.attention_module2(out)
        out = self.residual_block3(out2)
        # print(out.data)
        out3 = self.attention_module3(out)
        out = self.residual_block4(out3)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out, x, out1, out2, out3

model = ResidualAttentionModel().cuda()
model.load_state_dict(torch.load('model.pkl'))
print(model)

# Test
correct = 0
total = 0
#
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
num = 0

for images, labels in test_loader:
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())
    outputs, image, feature1, feature2, feature3 = model(images)
    #save image
    print('process batch %d' % num)
    image = image.view(image.size(0), 3, 224, 224)
    feature1 = feature1[:,0:3,:,:]#.view(feature1.size(0), 3, 28, 28)
    feature2 = feature2[:,0:3,:,:]#.view(feature2.size(0), 3, 28, 28)
    feature3 = feature3[:,0:3,:,:]#.view(feature3.size(0), 3, 28, 28)
    if not os.path.exists('./output'):
      os.mkdir('./output')
    save_image(denorm(image.data), './output/image-%d.png' %(num+1))
    save_image(denorm(feature1.data), './output/feature1-%d.png' %(num+1))
    save_image(denorm(feature2.data), './output/feature2-%d.png' %(num+1))
    save_image(denorm(feature3.data), './output/feature3-%d.png' %(num+1))
    num += 1
    #predict
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()
    #
    c = (predicted == labels.data).squeeze()
    for i in range(4):
        label = labels.data[i]
        class_correct[label] += c[i]
        class_total[label] += 1

print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

