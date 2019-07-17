# XDStudent

> Author: https://www.kaggle.com/xiuchengwang

> From: https://www.kaggle.com/xiuchengwang/xdstudent

> License: [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

> Score: 0.9999

In [1]:

```py
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

```

```
['test', 'sample_submission.csv', 'train.csv', 'train']

```

In [2]:

```py
from keras.layers import *
from keras.models import Model, Sequential, load_model
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K

```

```
Using TensorFlow backend.

```

In [3]:

```py
train = pd.read_csv('../input/train.csv')
train.head()

```

Out[3]:

|  | id | has_cactus |
| --- | --- | --- |
| 0 | 0004be2cfeaba1c0361d39e2b000257b.jpg | 1 |
| --- | --- | --- |
| 1 | 000c8a36845c0208e833c79c1bffedd1.jpg | 1 |
| --- | --- | --- |
| 2 | 000d1e9a533f62e55c289303b072733d.jpg | 1 |
| --- | --- | --- |
| 3 | 0011485b40695e9138e92d0b3fb55128.jpg | 1 |
| --- | --- | --- |
| 4 | 0014d7a11e90b62848904c1418fc8cf2.jpg | 1 |
| --- | --- | --- |

In [4]:

```py
train['has_cactus'].value_counts()

```

Out[4]:

```
1    13136
0     4364
Name: has_cactus, dtype: int64
```

In [5]:

```py
import matplotlib.pyplot as plt
import tqdm

img = plt.imread('../input/train/train/'+ train['id'][0])
img.shape

```

Out[5]:

```
(32, 32, 3)
```

In [6]:

```py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

x_train, x_test, y_train, y_test = train_test_split(train['id'], train['has_cactus'], test_size = 0.1, random_state = 32)

```

In [7]:

```py
x_train_arr = []
for images in tqdm.tqdm(x_train):
    img = plt.imread('../input/train/train/' + images)
    x_train_arr.append(img)

x_train_arr = np.array(x_train_arr)
print(x_train_arr.shape)

```

```
100%|██████████| 15750/15750 [00:18<00:00, 861.12it/s]

```

```
(15750, 32, 32, 3)

```

In [8]:

```py
x_test_arr = []
for images in tqdm.tqdm(x_test):
    img = plt.imread('../input/train/train/' + images)
    x_test_arr.append(img)

x_test_arr = np.array(x_test_arr)
print(x_test_arr.shape)

```

```
100%|██████████| 1750/1750 [00:02<00:00, 759.34it/s]
```

```
(1750, 32, 32, 3)

```

In [9]:

```py
x_train_arr = x_train_arr.astype('float32')
x_test_arr = x_test_arr.astype('float32')
x_train_arr = x_train_arr/255
x_test_arr = x_test_arr/255

```

In [10]:

```py
from keras.applications.densenet import DenseNet201
from keras.layers import *

inputs = Input((32, 32, 3))
base_model = DenseNet201(include_top=False, input_shape=(32, 32, 3))#, weights=None
x = base_model(inputs)
out1 = GlobalMaxPooling2D()(x)
out2 = GlobalAveragePooling2D()(x)
out3 = Flatten()(x)
out = Concatenate(axis=-1)([out1, out2, out3])
out = Dropout(0.5)(out)
out = Dense(256, name="3_")(out)
out = BatchNormalization()(out)
out = Activation("relu")(out)
out = Dense(1, activation="sigmoid", name="3_2")(out)
model = Model(inputs, out)
model.summary()

```

```
Downloading data from https://github.com/keras-team/keras-applications/releases/download/densenet/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5
74842112/74836368 [==============================] - 2s 0us/step
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 3)    0                                            
__________________________________________________________________________________________________
densenet201 (Model)             (None, 1, 1, 1920)   18321984    input_1[0][0]                    
__________________________________________________________________________________________________
global_max_pooling2d_1 (GlobalM (None, 1920)         0           densenet201[1][0]                
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 1920)         0           densenet201[1][0]                
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 1920)         0           densenet201[1][0]                
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 5760)         0           global_max_pooling2d_1[0][0]     
                                                                 global_average_pooling2d_1[0][0] 
                                                                 flatten_1[0][0]                  
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 5760)         0           concatenate_1[0][0]              
__________________________________________________________________________________________________
3_ (Dense)                      (None, 256)          1474816     dropout_1[0][0]                  
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 256)          1024        3_[0][0]                         
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 256)          0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
3_2 (Dense)                     (None, 1)            257         activation_1[0][0]               
==================================================================================================
Total params: 19,798,081
Trainable params: 19,568,513
Non-trainable params: 229,568
__________________________________________________________________________________________________

```

In [11]:

```py
base_model.Trainable=True

set_trainable=False
for layer in base_model.layers:
    layer.trainable = True

```

In [12]:

```py
model.compile('rmsprop', loss = "binary_crossentropy", metrics=["accuracy"])

```

In [13]:

```py
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

batch_size = 128
epochs = 36

filepath="weights_resnet.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
learning_rate_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=4, verbose=1, mode='max', min_delta=0.0, cooldown=0, min_lr=0)
early_stop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=25, verbose=1, mode='max', baseline=None, restore_best_weights=True)

train_datagen = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.1,
    vertical_flip=True,
    horizontal_flip=True)

train_datagen.fit(x_train_arr)
history = model.fit_generator(
    train_datagen.flow(x_train_arr, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(x_test_arr, y_test),
    callbacks=[learning_rate_reduce, checkpoint] 
)

```

```
Epoch 1/36
123/123 [==============================] - 75s 611ms/step - loss: 0.0981 - acc: 0.9632 - val_loss: 0.5698 - val_acc: 0.8554

Epoch 00001: val_acc improved from -inf to 0.85543, saving model to weights_resnet.hdf5
Epoch 2/36
123/123 [==============================] - 26s 211ms/step - loss: 0.0569 - acc: 0.9842 - val_loss: 2.3376 - val_acc: 0.7554

Epoch 00002: val_acc did not improve from 0.85543
Epoch 3/36
123/123 [==============================] - 25s 204ms/step - loss: 0.0379 - acc: 0.9883 - val_loss: 1.3842 - val_acc: 0.7783

Epoch 00003: val_acc did not improve from 0.85543
Epoch 4/36
123/123 [==============================] - 25s 203ms/step - loss: 0.0445 - acc: 0.9864 - val_loss: 0.1151 - val_acc: 0.9686

Epoch 00004: val_acc improved from 0.85543 to 0.96857, saving model to weights_resnet.hdf5
Epoch 5/36
123/123 [==============================] - 26s 215ms/step - loss: 0.0437 - acc: 0.9873 - val_loss: 0.2282 - val_acc: 0.9726

Epoch 00005: val_acc improved from 0.96857 to 0.97257, saving model to weights_resnet.hdf5
Epoch 6/36
123/123 [==============================] - 26s 208ms/step - loss: 0.0288 - acc: 0.9900 - val_loss: 0.3356 - val_acc: 0.9149

Epoch 00006: ReduceLROnPlateau reducing learning rate to 0.0006000000284984708.

Epoch 00006: val_acc did not improve from 0.97257
Epoch 7/36
123/123 [==============================] - 26s 209ms/step - loss: 0.0160 - acc: 0.9945 - val_loss: 0.0146 - val_acc: 0.9954

Epoch 00007: val_acc improved from 0.97257 to 0.99543, saving model to weights_resnet.hdf5
Epoch 8/36
123/123 [==============================] - 25s 207ms/step - loss: 0.0143 - acc: 0.9954 - val_loss: 0.0158 - val_acc: 0.9937

Epoch 00008: val_acc did not improve from 0.99543
Epoch 9/36
123/123 [==============================] - 25s 200ms/step - loss: 0.0254 - acc: 0.9949 - val_loss: 0.0097 - val_acc: 0.9977

Epoch 00009: val_acc improved from 0.99543 to 0.99771, saving model to weights_resnet.hdf5
Epoch 10/36
123/123 [==============================] - 25s 207ms/step - loss: 0.0120 - acc: 0.9959 - val_loss: 0.1715 - val_acc: 0.9514

Epoch 00010: ReduceLROnPlateau reducing learning rate to 0.0003600000170990825.

Epoch 00010: val_acc did not improve from 0.99771
Epoch 11/36
123/123 [==============================] - 26s 215ms/step - loss: 0.0082 - acc: 0.9975 - val_loss: 0.0119 - val_acc: 0.9966

Epoch 00011: val_acc did not improve from 0.99771
Epoch 12/36
123/123 [==============================] - 25s 206ms/step - loss: 0.0092 - acc: 0.9968 - val_loss: 0.4189 - val_acc: 0.9029

Epoch 00012: val_acc did not improve from 0.99771
Epoch 13/36
123/123 [==============================] - 25s 205ms/step - loss: 0.0072 - acc: 0.9978 - val_loss: 0.0366 - val_acc: 0.9874

Epoch 00013: val_acc did not improve from 0.99771
Epoch 14/36
123/123 [==============================] - 27s 219ms/step - loss: 0.0077 - acc: 0.9973 - val_loss: 0.0312 - val_acc: 0.9920

Epoch 00014: ReduceLROnPlateau reducing learning rate to 0.00021600000327453016.

Epoch 00014: val_acc did not improve from 0.99771
Epoch 15/36
123/123 [==============================] - 26s 212ms/step - loss: 0.0048 - acc: 0.9986 - val_loss: 0.0163 - val_acc: 0.9960

Epoch 00015: val_acc did not improve from 0.99771
Epoch 16/36
123/123 [==============================] - 25s 203ms/step - loss: 0.0047 - acc: 0.9986 - val_loss: 0.0555 - val_acc: 0.9806

Epoch 00016: val_acc did not improve from 0.99771
Epoch 17/36
123/123 [==============================] - 26s 211ms/step - loss: 0.0060 - acc: 0.9982 - val_loss: 0.0492 - val_acc: 0.9943

Epoch 00017: val_acc did not improve from 0.99771
Epoch 18/36
123/123 [==============================] - 24s 198ms/step - loss: 0.0045 - acc: 0.9986 - val_loss: 0.0067 - val_acc: 0.9977

Epoch 00018: ReduceLROnPlateau reducing learning rate to 0.00012960000021848827.

Epoch 00018: val_acc did not improve from 0.99771
Epoch 19/36
123/123 [==============================] - 25s 206ms/step - loss: 0.0060 - acc: 0.9973 - val_loss: 0.0070 - val_acc: 0.9971

Epoch 00019: val_acc did not improve from 0.99771
Epoch 20/36
123/123 [==============================] - 27s 217ms/step - loss: 0.0048 - acc: 0.9989 - val_loss: 0.0065 - val_acc: 0.9983

Epoch 00020: val_acc improved from 0.99771 to 0.99829, saving model to weights_resnet.hdf5
Epoch 21/36
123/123 [==============================] - 25s 203ms/step - loss: 0.0235 - acc: 0.9974 - val_loss: 0.0203 - val_acc: 0.9949

Epoch 00021: val_acc did not improve from 0.99829
Epoch 22/36
123/123 [==============================] - 24s 199ms/step - loss: 0.0040 - acc: 0.9990 - val_loss: 0.0149 - val_acc: 0.9966

Epoch 00022: ReduceLROnPlateau reducing learning rate to 7.775999838486313e-05.

Epoch 00022: val_acc did not improve from 0.99829
Epoch 23/36
123/123 [==============================] - 26s 211ms/step - loss: 0.0039 - acc: 0.9988 - val_loss: 0.0084 - val_acc: 0.9977

Epoch 00023: val_acc did not improve from 0.99829
Epoch 24/36
123/123 [==============================] - 25s 200ms/step - loss: 0.0382 - acc: 0.9951 - val_loss: 0.0088 - val_acc: 0.9983

Epoch 00024: val_acc did not improve from 0.99829
Epoch 25/36
123/123 [==============================] - 25s 201ms/step - loss: 0.0035 - acc: 0.9989 - val_loss: 0.0054 - val_acc: 0.9983

Epoch 00025: val_acc did not improve from 0.99829
Epoch 26/36
123/123 [==============================] - 26s 209ms/step - loss: 0.0246 - acc: 0.9970 - val_loss: 0.0059 - val_acc: 0.9977

Epoch 00026: ReduceLROnPlateau reducing learning rate to 4.6655999904032795e-05.

Epoch 00026: val_acc did not improve from 0.99829
Epoch 27/36
123/123 [==============================] - 26s 210ms/step - loss: 0.0032 - acc: 0.9989 - val_loss: 0.0100 - val_acc: 0.9983

Epoch 00027: val_acc did not improve from 0.99829
Epoch 28/36
123/123 [==============================] - 25s 205ms/step - loss: 0.0025 - acc: 0.9993 - val_loss: 0.0063 - val_acc: 0.9971

Epoch 00028: val_acc did not improve from 0.99829
Epoch 29/36
123/123 [==============================] - 26s 211ms/step - loss: 0.0222 - acc: 0.9981 - val_loss: 0.0077 - val_acc: 0.9977

Epoch 00029: val_acc did not improve from 0.99829
Epoch 30/36
123/123 [==============================] - 26s 212ms/step - loss: 0.0300 - acc: 0.9969 - val_loss: 0.0085 - val_acc: 0.9977

Epoch 00030: ReduceLROnPlateau reducing learning rate to 2.799360081553459e-05.

Epoch 00030: val_acc did not improve from 0.99829
Epoch 31/36
123/123 [==============================] - 25s 203ms/step - loss: 0.0233 - acc: 0.9966 - val_loss: 0.0155 - val_acc: 0.9960

Epoch 00031: val_acc did not improve from 0.99829
Epoch 32/36
123/123 [==============================] - 26s 212ms/step - loss: 0.0036 - acc: 0.9981 - val_loss: 0.0150 - val_acc: 0.9966

Epoch 00032: val_acc did not improve from 0.99829
Epoch 33/36
123/123 [==============================] - 26s 209ms/step - loss: 0.0239 - acc: 0.9969 - val_loss: 0.0077 - val_acc: 0.9983

Epoch 00033: val_acc did not improve from 0.99829
Epoch 34/36
123/123 [==============================] - 25s 201ms/step - loss: 0.0158 - acc: 0.9967 - val_loss: 0.0112 - val_acc: 0.9966

Epoch 00034: ReduceLROnPlateau reducing learning rate to 1.6796160707599483e-05.

Epoch 00034: val_acc did not improve from 0.99829
Epoch 35/36
123/123 [==============================] - 25s 201ms/step - loss: 9.8774e-04 - acc: 0.9997 - val_loss: 0.0110 - val_acc: 0.9977

Epoch 00035: val_acc did not improve from 0.99829
Epoch 36/36
123/123 [==============================] - 26s 211ms/step - loss: 0.0012 - acc: 0.9996 - val_loss: 0.0098 - val_acc: 0.9971

Epoch 00036: val_acc did not improve from 0.99829

```

In [14]:

```py
train_pred = model.predict(x_train_arr, verbose= 1)
valid_pred = model.predict(x_test_arr, verbose= 1)

train_acc = roc_auc_score(np.round(train_pred), y_train)
valid_acc = roc_auc_score(np.round(valid_pred), y_test)

```

```
15750/15750 [==============================] - 24s 2ms/step
1750/1750 [==============================] - 2s 1ms/step

```

In [15]:

```py
confusion_matrix(np.round(valid_pred), y_test)

```

Out[15]:

```
array([[ 442,    0],
       [   5, 1303]])
```

In [16]:

```py
sample = pd.read_csv('../input/sample_submission.csv')

```

In [17]:

```py
test = []
for images in tqdm.tqdm(sample['id']):
    img = plt.imread('../input/test/test/' + images)
    test.append(img)

test = np.array(test)

```

```
100%|██████████| 4000/4000 [00:04<00:00, 909.30it/s]

```

In [18]:

```py
test = test/255
test_pred = model.predict(test, verbose= 1)

```

```
4000/4000 [==============================] - 5s 1ms/step

```

In [19]:

```py
sample['has_cactus'] = test_pred
sample.head()

```

Out[19]:

|  | id | has_cactus |
| --- | --- | --- |
| 0 | 000940378805c44108d287872b2f04ce.jpg | 9.999971e-01 |
| --- | --- | --- |
| 1 | 0017242f54ececa4512b4d7937d1e21e.jpg | 9.999970e-01 |
| --- | --- | --- |
| 2 | 001ee6d8564003107853118ab87df407.jpg | 9.930815e-08 |
| --- | --- | --- |
| 3 | 002e175c3c1e060769475f52182583d0.jpg | 1.431321e-05 |
| --- | --- | --- |
| 4 | 0036e44a7e8f7218e9bc7bf8137e4943.jpg | 9.999958e-01 |
| --- | --- | --- |

In [20]:

```py
sample.to_csv('sub.csv', index= False)

```