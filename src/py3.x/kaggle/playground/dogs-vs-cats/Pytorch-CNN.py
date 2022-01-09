# -*- coding: utf-8 -*-

'''
 PyTorch 版本的 CNN 实现 Dogs vs Cats
'''

# 引入相应的库函数
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

# 我们暂时使用 AlexNet 模型做第一次测试
from models import AlexNet

'''
参考链接：
https://github.com/pytorch/examples/blob/27e2a46c1d1505324032b1d94fc6ce24d5b67e97/imagenet/main.py
'''

'''
这里我们使用了 torch.utils.data 中的一些函数，比如 Dataset
class torch.utils.data.Dataset
表示 Dataset 的抽象类
所有其他的数据集都应继承该类。所有子类都应该重写 __len__ ，提供数据集大小的方法，和 __getitem__ ，支持从 0 到 len(self) 整数索引的方法
'''
# --------------------------- 1.加载数据 start ----------------------------------------------------------------
class GetData(data.Dataset):
    def __init__(self, root, transforms=None, train=True, test=Flase):
        '''
        Desc:
            获取全部的数据，并根据我们的要求，将数据划分为 训练、验证、测试数据集
        Args:
            self --- none
            root --- 数据存在路径
            transforms --- 对数据的转化，这里默认是 None
            train ---- 标注是否是训练集
            test  ---- 标注是否是测试集
        Returns:
            None
        '''
        # 设置 测试集数据
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1：即测试数据集， D:/dataset/dogs-vs-cats/test1
        # train: 即训练数据集，D:/dataset/dogs-vs-cats/train
        if self.test:
            # 提取 测试数据集的序号，
            # 如 x = 'd:/path/123.jpg'，
            # x.split('.') 得到 ['d:/path/123', 'jpg'] 
            # x.split('.')[-2] 得到 d:/path/123
            # x.split('.')[-2].split('/') 得到 ['d:', 'path', '123']
            # x.split('.')[-2].split('/')[-1] 得到 123
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            # 如果不是测试集的话，就是训练集，我们只切分一下，仍然得到序号，123
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        
        # 获取图片的数量
        imgs_num = len(imgs)

        # 划分训练、验证集，验证集:训练集 = 3:7
        # 判断是否为测试集
        if self.test:
            # 如果是 测试集，那么 就直接赋值
            self.imgs = imgs
        # 判断是否为 训练集
        elif train:
            # 如果是训练集，那么就把数据集的开始位置的数据 到 70% 部分的数据作为训练集
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            # 这种情况就是划分验证集啦,从 70% 部分的数据 到达数据的末尾，全部作为验证集
            self.imgs = imgs[int(0.7 * imgs_num):]

        # 数据的转换操作，测试集，验证集，和训练集的数据转换有所区别
        if transforms is None:
            # 如果转换操作没有设置，那我们设置一个转换 
            '''
            几个常见的 transforms 用的转换：
            1、数据归一化 --- Normalize(mean, std) 是通过下面公式实现数据归一化 channel = (channel-mean)/std
            2、class torchvision.transforms.Resize(size, interpolation=2) 将输入的 PIL 图像调整为给定的大小
            3、class torchvision.transforms.CenterCrop(size) 在中心裁剪给定的 PIL 图像，参数 size 是期望的输出大小
            4、ToTensor() 是将 PIL.Image(RGB) 或者 numpy.ndarray(H X W X C) 从 0 到 255 的值映射到 0~1 的范围内，并转化为 Tensor 形式
            5、transforms.Compose() 这个是将多个 transforms 组合起来使用
            '''
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # 测试集 和 验证集 的转换
            # 判断如果是测试集或者不是训练集（也就是说是验证集），就应用我们下边的转换
            if self.test or not train:
                self.trainsforms = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), normalize])
            else:
                # 如果是测试集的话，使用另外的转换
                self.transforms = T.Compose([T.Resize(256), T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), normalize])

    def __len__(self):
        '''
        Desc:
            继承 Dataset 基类，重写 __len__ 方法，提供数据集的大小
        Args:
            self --- 无
        Return:
            数据集的长度
        '''
        return len(self.imgs)

    def __getitem__(self, index):
        '''
        Desc:
            继承 Dataset 基类，重写 __getitem__ 方法，支持整数索引，范围从 0 到 len(self) 
            返回一张图片的数据
            对于测试集，没有label，返回图片 id，如 985.jpg 返回 985
            对于训练集，是具有 label ，返回图片 id ，以及相对应的 label，如 dog.211.jpg 返回 id 为 211，label 为 dog
        Args:
            self --- none
            index --- 索引
        Return:
            返回一张图片的数据
            对于测试集，没有label，返回图片 id，如 985.jpg 返回 985
            对于训练集，是具有 label ，返回图片 id ，以及相对应的 label，如 dog.211.jpg 返回 id 为 211，label 为 dog
        '''
        img_path = self.imgs[index]
        # 判断，如果是测试集的数据的话，那就返回对应的序号，比如 d:path/123.jpg 返回 123
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            # 如果不是测试集的数据，那么会有相应的类别（label），也就是对应的dog 和 cat，dog 为 1，cat 为0
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        # 这里使用 Pillow 模块，使用 Image 打开一个图片
        data = Image.open(img_path)
        # 使用我们定义的 transforms ，将图片转换，详情参考：https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-pil-image
        # 默认的 transforms 设置的是 none
        data = self.transforms(data)
        # 将转换完成的 data 以及对应的 label（如果有的话），返回
        return data,label
    
# 训练数据集的路径
train_path = 'D:/dataset/dogs-vs-cats/train'
# 从训练数据集的存储路径中提取训练数据集
train_dataset = GetData(train_path, train=True)
# 将训练数据转换成 mini-batch 形式
load_train = data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=1)

'''
utils.data.DataLoader() 解析
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
数据加载器。组合数据集和采样器，并在数据集上提供单个或多个进程迭代器。
参数：
dataset(Dataset) --- 从这之中加载数据的数据集。
batch_size (int, 可选) --- 每个 batch 加载多少个样本（默认值为：1）
shuffle (bool, 可选) --- 设置为 True 时，会在每个 epoch 时期重新组合数据（默认值：False）
sampler (Sampler, 可选) --- 定义从数据集中抽取样本的策略。如果指定，那么 shuffle 必须是 False 。
batch_sampler (Sampler, 可选) --- 类似采样器，但一次返回一批量的索引（index）。与 batch_size, shuffle, sampler 和 drop_last 相互排斥。
num_workers (int, 可选) --- 设置有多少个子进程用于数据加载。0 表示数据将在主进程中加载。（默认：0）
collate_fn (callable, 可选) --- 合并样本列表以形成 mini-batch 
pin_memory (bool, 可选) ---  如果为 True，数据加载器会在 tensors 返回之前将 tensors 复制到 CUDA 固定内存中。
drop_last (bool, 可选) --- 如果 dataset size （数据集大小）不能被 batch size （批量大小）整除，则设置为 True 以删除最后一个 incomplete batch（未完成批次）。
                          如果设置为 False 和 dataset size（数据集大小）不能被 batch size（批量大小）整除，则最后一批将会更小。（默认：False）
timeout (numeric, 可选) --- 如果是正值，则为从 worker 收集 batch 的超时值。应该始终是非负的。（默认：0）
worker_init_fn (callable, 可选) --- 如果不是 None，那么将在每个工人子进程上使用 worker id（在 [0，num_workers - 1] 中的 int）作为输入，在 seeding 和加载数据之前调用这个子进程。（默认：无）
'''

# 测试数据的获取
# 首先设置测试数据的路径
test_path = 'D:/dataset/dogs-vs-cats/test1'
# 从测试数据集的存储路径中提取测试数据集
test_path = GetData(test_path, test=True)
# 将测试数据转换成 mini-batch 形式
loader_test = data.DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=1)

# --------------------------- 1.加载数据 end ----------------------------------------------------------------

# --------------------------- 2.构建 CNN 模型 start ----------------------------------------------------------------
# 调用我们现成的 AlexNet() 模型
cnn = AlexNet()
# 将模型打印出来观察一下
print(cnn)

# --------------------------- 2.构建 CNN 模型 end ------------------------------------------------------------------

# --------------------------- 3.设置相应的优化器和损失函数 start ------------------------------------------------------------------

'''
1、torch.optim 是一个实现各种优化算法的软件包。
比如我们这里使用的就是 Adam() 这个方法
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) 这个类就实现了 adam 算法。
params(iterable) --- 可迭代的参数来优化或取消定义参数组
lr(float, 可选) --- 学习率（默认值 1e-3）
beta(Tuple[float, float], 可选) --- 用于计算梯度及其平方的运行平均值的系数（默认值：（0.9，0.999））
eps (float, 可选) ---- 添加到分母以提高数值稳定性（默认值：1e-8）
weight_decay (float, 可选) --- 权重衰减（L2 惩罚）（默认值：0）
amsgrad (boolean, 可选) ---- 是否使用该算法的AMSGrad变体来自论文关于 Adam 和 Beyond 的融合  


2、还有这里我们使用的损失函数 
class torch.nn.CrossEntropyLoss(weight=None, size_average=True, ignore_index=-100, reduce=True)
交叉熵损失函数
具体的请看：http://pytorch.apachecn.org/cn/docs/0.3.0/nn.html   
'''
# 3. 设置优化器和损失函数
# 这里我们使用 Adam 优化器，使用的损失函数是 交叉熵损失
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.005, betas=(0.9, 0.99))  # 优化所有的 cnn 参数
loss_func = nn.CrossEntropyLoss()  # 目标 label 不是 one-hotted 类型的

# --------------------------- 3.设置相应的优化器和损失函数 end ------------------------------------------------------------------

# --------------------------- 4.训练 CNN 模型 start ------------------------------------------------------------------

# 4. 训练模型
# 设置训练模型的次数，这里我们设置的是 10 次，也就是用我们的训练数据集对我们的模型训练 10 次，为了节省时间，我们可以只训练 1 次
EPOCH = 10
# 训练和测试
for epoch in range(EPOCH):
        num = 0
        # 给出 batch 数据，在迭代 train_loader 的时候对 x 进行 normalize
        for step, (x, y) in enumerate(loader_train):
            b_x = Variable(x)  # batch x
            b_y = Variable(y)  # batch y

            output = cnn(b_x)  # cnn 的输出
            loss = loss_func(output, b_y)  # 交叉熵损失
            optimizer.zero_grad()  # 在这一步的训练步骤上，进行梯度清零
            loss.backward()  # 反向传播，并进行计算梯度
            optimizer.step()  # 应用梯度

            # 可以打印一下
            # print('-'*30, step)
            if step % 20 == 0:
                num += 1
                for _, (x_t, y_test) in enumerate(loader_test):
                    x_test = Variable(x_t)  # batch x
                    test_output = cnn(x_test)
                    pred_y = torch.max(test_output, 1)[1].data.squeeze()
                    accuracy = sum(pred_y == y_test) / float(y_test.size(0))
                    print('Epoch: ', epoch, '| Num: ',  num, '| Step: ',  step, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.4f' % accuracy)

# --------------------------- 4. 训练 CNN 模型 end ------------------------------------------------------------------