

# 1.导入需要的库
## 1.1.导入一些必要的库，如pandas、numpy、matplotlib、sklearn
## 1.2.导入keras（tensorflow backend），用来搭建神经网络


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
#将那些用matplotlib绘制的图显示在页面里而不是弹出一个窗口
%matplotlib inline   

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # 转换成 one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
```

    Using TensorFlow backend.


# 2.数据准备工作
## 2.1.导入数据


```python
# Load the data
train = pd.read_csv(r'''/home/cd/kaggle-master/datasets/getting-started/digit-recognizer/input/train.csv''')
test = pd.read_csv(r'''/home/cd/kaggle-master/datasets/getting-started/digit-recognizer/input/test.csv''')

X_train = train.values[:,1:]
Y_train = train.values[:,0]
test=test.values
```

## 2.2.标准化


```python
# Normalization
X_train = X_train / 255.0
test = test / 255.0
```

## 2.3.将数组维度变成（28，28，1）
### 之前用pandas导入数据的时候会将数据变成一维数组


```python
X_train = X_train.reshape(-1,28,28,1)
test = test.reshape(-1,28,28,1)
```

## 2.4.将标签编码为one-hot编码
### 如：2 -> [0,0,1,0,0,0,0,0,0,0]            
###        7 -> [0,0,0,0,0,0,0,1,0,0]



```python
Y_train = to_categorical(Y_train, num_classes = 10)
```

## 2.5.将训练集随机划分成训练集和验证集
### 设置随机数种子


```python
random_seed = 2
```


```python
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
```

# 3.CNN
## 3.1.定义cnn模型
### cnn结构为[[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out


```python
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

```

## 3.2.设置优化器
### 优化算法选用自适应学习率算法
[RMSprop](http://blog.csdn.net/bvl10101111/article/details/72616378)
### 选择默认参数


```python
# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
```

## 3.3.编译模型
### 优化算法选择RMSprop
### 损失函数选择categorical_crossentropy，亦称作多类的对数损失
### 性能评估方法选择准确率


```python
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
```


```python
epochs = 30 
batch_size = 86
```

### 设置学习率退火器
### *当评价指标不在提升时，减少学习率
### *监测量是val_acc，当3个epoch过去而模型性能不提升时，学习率减少的动作会触发
### *factor：每次减少学习率的因子，学习率将以lr = lr×factor的形式被减少
### *min_lr：学习率的下限

### *回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。


```python
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
```

## 3.4.设置图片生成器
### 用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。


```python
datagen = ImageDataGenerator(
        featurewise_center=False,  # 使输入数据集去中心化（均值为0）, 按feature执行
        samplewise_center=False,  # 使输入数据的每个样本均值为0
        featurewise_std_normalization=False,  # 将输入除以数据集的标准差以完成标准化, 按feature执行
        samplewise_std_normalization=False,  # 将输入的每个样本除以其自身的标准差
        zca_whitening=False,  # 对输入数据施加ZCA白化
        rotation_range=10,  # 数据增强时图片随机转动的角度
        zoom_range = 0.1, # 随机缩放的幅度
        width_shift_range=0.1,  # 图片宽度的某个比例，数据增强时图片水平偏移的幅度
        height_shift_range=0.1,  # 图片高度的某个比例，数据增强时图片竖直偏移的幅度
        horizontal_flip=False,  # 进行随机水平翻转
        vertical_flip=False)  # 进行随机竖直翻转

```

### 计算依赖于数据的变换所需要的统计信息(均值方差等


```python
datagen.fit(X_train)
```

## 3.5.训练模型
### fit_generator：利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率
### datagen.flow（）：生成器函数，接收numpy数组和标签为参数,生成经过数据增强或标准化后的batch数据,并在一个无限循环中不断的返回batch数据
### callbacks=[learning_rate_reduction]：回调函数，这个list中的回调函数将会在训练过程中的适当时机被调用



```python
import datetime
starttime = datetime.datetime.now()

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

endtime = datetime.datetime.now()

print ((endtime - starttime).seconds)


```

    Epoch 1/30
     - 10s - loss: 0.0352 - acc: 0.9900 - val_loss: 0.0174 - val_acc: 0.9943
    Epoch 2/30
     - 10s - loss: 0.0322 - acc: 0.9908 - val_loss: 0.0176 - val_acc: 0.9952
    Epoch 3/30
     - 10s - loss: 0.0353 - acc: 0.9895 - val_loss: 0.0177 - val_acc: 0.9950
    Epoch 4/30
     - 9s - loss: 0.0320 - acc: 0.9908 - val_loss: 0.0172 - val_acc: 0.9950
    Epoch 5/30
     - 9s - loss: 0.0350 - acc: 0.9903 - val_loss: 0.0173 - val_acc: 0.9952
    Epoch 6/30
    
    Epoch 00006: reducing learning rate to 1e-05.
     - 10s - loss: 0.0346 - acc: 0.9897 - val_loss: 0.0170 - val_acc: 0.9948
    Epoch 7/30
     - 10s - loss: 0.0350 - acc: 0.9897 - val_loss: 0.0171 - val_acc: 0.9948
    Epoch 8/30
     - 10s - loss: 0.0338 - acc: 0.9907 - val_loss: 0.0170 - val_acc: 0.9950
    Epoch 9/30
     - 10s - loss: 0.0355 - acc: 0.9896 - val_loss: 0.0172 - val_acc: 0.9948
    Epoch 10/30
     - 9s - loss: 0.0335 - acc: 0.9907 - val_loss: 0.0174 - val_acc: 0.9948
    Epoch 11/30
     - 9s - loss: 0.0318 - acc: 0.9903 - val_loss: 0.0171 - val_acc: 0.9948
    Epoch 12/30
     - 10s - loss: 0.0348 - acc: 0.9902 - val_loss: 0.0173 - val_acc: 0.9948
    Epoch 13/30
     - 10s - loss: 0.0337 - acc: 0.9902 - val_loss: 0.0170 - val_acc: 0.9950
    Epoch 14/30
     - 10s - loss: 0.0344 - acc: 0.9902 - val_loss: 0.0172 - val_acc: 0.9948
    Epoch 15/30
     - 9s - loss: 0.0339 - acc: 0.9900 - val_loss: 0.0171 - val_acc: 0.9950
    Epoch 16/30
     - 10s - loss: 0.0338 - acc: 0.9904 - val_loss: 0.0168 - val_acc: 0.9948
    Epoch 17/30
     - 10s - loss: 0.0342 - acc: 0.9902 - val_loss: 0.0166 - val_acc: 0.9950
    Epoch 18/30
     - 10s - loss: 0.0358 - acc: 0.9903 - val_loss: 0.0169 - val_acc: 0.9950
    Epoch 19/30
     - 9s - loss: 0.0339 - acc: 0.9903 - val_loss: 0.0166 - val_acc: 0.9950
    Epoch 20/30
     - 10s - loss: 0.0356 - acc: 0.9903 - val_loss: 0.0166 - val_acc: 0.9950
    Epoch 21/30
     - 10s - loss: 0.0350 - acc: 0.9900 - val_loss: 0.0165 - val_acc: 0.9952
    Epoch 22/30
     - 10s - loss: 0.0350 - acc: 0.9899 - val_loss: 0.0169 - val_acc: 0.9950
    Epoch 23/30
     - 10s - loss: 0.0353 - acc: 0.9898 - val_loss: 0.0171 - val_acc: 0.9948
    Epoch 24/30
     - 9s - loss: 0.0325 - acc: 0.9904 - val_loss: 0.0167 - val_acc: 0.9948
    Epoch 25/30
     - 10s - loss: 0.0359 - acc: 0.9892 - val_loss: 0.0168 - val_acc: 0.9948
    Epoch 26/30
     - 10s - loss: 0.0349 - acc: 0.9901 - val_loss: 0.0163 - val_acc: 0.9948
    Epoch 27/30
     - 10s - loss: 0.0328 - acc: 0.9908 - val_loss: 0.0166 - val_acc: 0.9948
    Epoch 28/30
     - 10s - loss: 0.0331 - acc: 0.9910 - val_loss: 0.0166 - val_acc: 0.9952
    Epoch 29/30
     - 10s - loss: 0.0343 - acc: 0.9903 - val_loss: 0.0173 - val_acc: 0.9943
    Epoch 30/30
     - 9s - loss: 0.0346 - acc: 0.9902 - val_loss: 0.0166 - val_acc: 0.9948
    287


## 可以看到准确率大概在0.995左右，训练时间为287s (GPU加速后)

### 参考
[代码](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)


[keras文档](https://keras-cn.readthedocs.io/en/latest/)


### 预测和提交结果
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("datasets/getting-started/digit-recognizer/ouput/Result_keras_CNN.csv",index=False)