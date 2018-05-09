#!/usr/bin/python3
# coding: utf-8
'''
Created on 2017-12-18
Update  on 2018-03-27
Author: 片刻
Github: https://github.com/apachecn/kaggle
Result: 
    BATCH_SIZE = 10 and EPOCH = 10; [10,  4000] loss: 0.069
    BATCH_SIZE = 10 and EPOCH = 15; [10,  4000] loss: 0.069
'''
# import csv
import pandas as pd

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class CustomedDataSet(Dataset):
    def __init__(self, train=True):
        self.train = train
        if self.train:
            trainX = pd.read_csv(
                '/opt/data/kaggle/getting-started/digit-recognizer/input/train.csv'
                # names=["ImageId", "Label"]
            )
            trainY = trainX.label.as_matrix().tolist()
            trainX = trainX.drop(
                'label', axis=1).as_matrix().reshape(trainX.shape[0], 1, 28, 28)
            self.datalist = trainX
            self.labellist = trainY
        else:
            testX = pd.read_csv(
                '/opt/data/kaggle/getting-started/digit-recognizer/input/test.csv'
            )
            self.testID = testX.index
            testX = testX.as_matrix().reshape(testX.shape[0], 1, 28, 28)
            self.datalist = testX

    def __getitem__(self, index):
        if self.train:
            return torch.Tensor(
                self.datalist[index].astype(float)), self.labellist[index]
        else:
            return torch.Tensor(self.datalist[index].astype(float))

    def __len__(self):
        return self.datalist.shape[0]


train_data = CustomedDataSet()
test_data = CustomedDataSet(train=False)

BATCH_SIZE = 150
train_loader = DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(
                kernel_size=2
            ),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7,
                             10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(
            x.size(0),
            -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization


cnn = CNN()
# print(cnn)  # net architecture

LR = 0.001  # learning rate
optimizer = torch.optim.Adam(
    cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
print(u'开始训练')
EPOCH = 5  # train the training data n times, to save time, we just train 1 epoch
for epoch in range(EPOCH):
    running_loss = 0.0

    for step, (x, y) in enumerate(
            train_loader
    ):  # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x)  # batch x
        b_y = Variable(y)  # batch y

        output = cnn(b_x)[0]  # 输入训练数据
        loss = loss_func(output, b_y)  # 计算误差
        optimizer.zero_grad()  # 清空上一次梯度
        loss.backward()  # 误差反向传递
        optimizer.step()  # 优化器参数更新

        # 每1000批数据打印一次平均loss值
        running_loss += loss.data[
            0]  # loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
        if step % 500 == 499:  # 每2000批打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / 500))
            running_loss = 0.0
print('Finished Training')

# correct = 0
# total = 0
# for img, label in test_loader:
#     img = Variable(img, volatile=True)
#     label = Variable(label, volatile=True)

#     outputs = cnn(img)
#     _, predicted = torch.max(outputs[0], 1)
#     # print('1-', type(label), '-------', label)
#     # print('2-', type(predicted), '-------', predicted)
#     total += label.size(0)
#     num_correct = (predicted == label).sum()
#     correct += num_correct.data[0]

# print('Accuracy of the network on the %d test images: %.3f %%' % (total, 100 * correct / total))

# I just can't throw all of test data into the network,since it was so huge that my GPU memory cann't afford it
ans = torch.LongTensor()  # build a tensor to concatenate answers
for img in test_loader:
    img = Variable(img)
    outputs = cnn(img)
    _, predicted = torch.max(outputs[0], 1)
    # print('type(predicted) = ', type(predicted), predicted)
    ans = torch.cat([ans, predicted.data], 0)

testLabel = ans.numpy()  # only tensor on cpu can transform to the numpy array

# # 结果输出保存
# def saveResult(result, csvName):
#     with open(csvName, 'w') as myFile:
#         myWriter = csv.writer(myFile)
#         myWriter.writerow(["ImageId", "Label"])
#         index = 0
#         for r in result:
#             index += 1
#             myWriter.writerow([index, int(r)])

#     print('Saved successfully...')  # 保存预测结果

# saveResult(testLabel,
#            '/opt/data/kaggle/getting-started/digit-recognizer/output/Result_pytorch_CNN.csv')

# 提交结果
submission_df = pd.DataFrame(
    data={'ImageId': test_data.testID+1,
          'Label': testLabel})
# print(submission_df.head(10))
submission_df.to_csv(
    '/opt/data/kaggle/getting-started/digit-recognizer/output/Result_pytorch_CNN.csv',
    columns=["ImageId", "Label"],
    index=False)
