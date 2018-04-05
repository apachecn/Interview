#!/usr/bin/python
# coding: utf-8

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

from models import AlexNet

"""
参考链接：https://github.com/pytorch/examples/blob/27e2a46c1d1505324032b1d94fc6ce24d5b67e97/imagenet/main.py
"""
# 1. 加载数据


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        '''
        目标：获取所有图片地址，并根据训练、验证、测试划分数据
        '''
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg 
        if self.test:
            imgs = sorted(
                imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # 划分训练、验证集，验证:训练 = 3:7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:

            # 数据转换操作，测试验证和训练的数据转换有所区别

            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # 测试集和验证集
            if self.test or not train:
                self.transforms = T.Compose(
                    [T.Resize(224), T.CenterCrop(224), T.ToTensor(), normalize])
            # 训练集
            else:
                self.transforms = T.Compose([
                    T.Resize(256), T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(), T.ToTensor(), normalize
                ])

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        对于测试集，没有label，返回图片id，如1000.jpg返回1000
        '''
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)


traindir = '/opt/data/kaggle/playground/dogs-vs-cats/sample_train'
train_dataset = DogCat(traindir, train=True)
loader_train = data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=1)

testdir = '/opt/data/kaggle/playground/dogs-vs-cats/sample_test'
test_dataset = DogCat(testdir, train=True)
loader_test = data.DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=1)


# 2. 创建 CNN 模型
cnn = AlexNet()
print(cnn)
# 3. 设置优化器和损失函数
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.005, betas=(0.9, 0.99))  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# 4. 训练模型
EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
# training and testing
for epoch in range(EPOCH):
        num = 0
        # gives batch data, normalize x when iterate train_loader
        for step, (x, y) in enumerate(loader_train):
            b_x = Variable(x)  # batch x
            b_y = Variable(y)  # batch y

            output = cnn(b_x)  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            # print('-'*30, step)
            if step % 20 == 0:
                num += 1
                for _, (x_t, y_test) in enumerate(loader_test):
                    x_test = Variable(x_t)  # batch x
                    test_output = cnn(x_test)
                    pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    accuracy = sum(pred_y == y_test) / float(y_test.size(0))
                    print('Epoch: ', epoch, '| Num: ',  num, '| Step: ',  step, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)
