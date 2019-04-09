# Simple CNN Model with Keras

> Author: https://www.kaggle.com/frules11

> From: https://www.kaggle.com/frules11/cnn-model-with-keras

> License: [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

> Score: 1.0000

In [1]:

```
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
['test', 'train', 'train.csv', 'sample_submission.csv']

```

In [2]:

```
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tqdm import tqdm
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

```

```
Using TensorFlow backend.

```

In [3]:

```
class DataLoader:
    def __init__(self, npy_file: str = "npy_data"):
        self.npy_file = npy_file
        self.csv_name = "../input/train.csv"
        self.df = self.read_csv()
        self.n_classes = 2

        os.makedirs(self.npy_file, exist_ok=True)

    def read_csv(self):
        df = pd.read_csv(self.csv_name)

        return df

    def read_data(self, load_from_npy: bool = True, size2resize: tuple = (75, 75), make_gray: bool = True,
                  save: bool = True, categorical: bool = False, n_classes: int = 2):

        x_data = []
        y_data = []

        if load_from_npy:
            try:
                x_data = np.load(fr"{self.npy_file}/x_data.npy")
                y_data = np.load(fr"{self.npy_file}/y_data.npy")
            except FileNotFoundError:
                load_from_npy = False
                print("NPY files not found!")
                pass

        if not load_from_npy:
            x_data = []
            y_data = []

            for dir_label in tqdm(self.df.values):
                img = cv2.imread(os.path.join("../input", "train/train", dir_label[0]))

                if make_gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img = cv2.resize(img, size2resize)

                x_data.append(img)
                y_data.append(int(dir_label[1]))

                del img

            x_data = np.array(x_data)
            y_data = np.array(y_data)

            if save:
                np.save(fr"{self.npy_file}/x_data.npy", x_data)
                np.save(fr"{self.npy_file}/y_data.npy", y_data)

        if categorical:
            y_data = tf.keras.utils.to_categorical(y_data, num_classes=n_classes)

        if not categorical:
            y_data = y_data.reshape(-1, 1)

        if load_from_npy and make_gray:
            try:
                x_data_2 = [cv2.cvtColor(n, cv2.COLOR_BGR2GRAY) for n in x_data]
                x_data = x_data_2
            except cv2.error:
                pass

        if make_gray:
            x_data = np.expand_dims(x_data, axis=-1)

        return x_data, y_data

    def read_test_data(self, load_from_npy: bool = True, size2resize: tuple = (75, 75), make_gray: bool = True,
                  save: bool = True, categorical: bool = False, n_classes: int = 2):

        test_df = pd.read_csv("../input/sample_submission.csv")

        x_data = []
        y_data = []

        if load_from_npy:
            try:
                x_data = np.load(fr"{self.npy_file}/x_data_test.npy")
                y_data = np.load(fr"{self.npy_file}/y_data_test.npy")
            except FileNotFoundError:
                load_from_npy = False
                print("NPY files not found!")
                pass

        if not load_from_npy:
            x_data = []
            y_data = []

            for dir_label in tqdm(test_df.values):
                img = cv2.imread(os.path.join("../input", "test/test", dir_label[0]))

                if make_gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                img = cv2.resize(img, size2resize)

                x_data.append(img)
                y_data.append(int(dir_label[1]))

                del img

            x_data = np.array(x_data)
            y_data = np.array(y_data)

            if save:
                np.save(fr"{self.npy_file}/x_data_test.npy", x_data)
                np.save(fr"{self.npy_file}/y_data_test.npy", y_data)

        if categorical:
            y_data = tf.keras.utils.to_categorical(y_data, num_classes=n_classes)

        if not categorical:
            y_data = y_data.reshape(-1, 1)

        if load_from_npy and make_gray:
            try:
                x_data_2 = [cv2.cvtColor(n, cv2.COLOR_BGR2GRAY) for n in x_data]
                x_data = x_data_2
            except cv2.error:
                pass

        if make_gray:
            x_data = np.expand_dims(x_data, axis=-1)

        return x_data, y_data

```

In [4]:

```
class TrainWithKeras:
    def __init__(self, x_data, y_data, lr: float = 0.001, epochs: int = 10, batch_size: int = 32,
                 loss: str = "categorical_crossentropy", model_path: str = "model.h5"):
        self.x_data = x_data
        self.y_data = y_data
        self.model_path = model_path

        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = Adam(lr=lr)
        self.loss = loss

    def make_model(self, summarize: bool = True):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), strides=1, activation="relu",
                         input_shape=(self.x_data.shape[1], self.x_data.shape[2], self.x_data.shape[3])))
        model.add(MaxPooling2D())
        model.add(Conv2D(128, (3, 3), strides=1, activation="relu"))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (3, 3), strides=1, activation="relu"))
        model.add(MaxPooling2D())
        model.add(Conv2D(512, (3, 3), strides=1, activation="relu"))
        model.add(Dropout(0.3))

        model.add(Conv2D(1024, (3, 3), strides=1, activation="relu"))

        model.add(Flatten())

        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation="softmax"))

        if summarize:
            model.summary()

        return model

    def compile(self, kmodel: Sequential):
        kmodel.compile(loss=self.loss, optimizer=self.optimizer, metrics=["acc"])

        return kmodel

    def train(self, kmodel: Sequential, save: bool = True):
        history = kmodel.fit(self.x_data, self.y_data, batch_size=self.batch_size, epochs=self.epochs,
                             validation_split=0.0)

        if save:
            kmodel.save(self.model_path)

        return history, kmodel

```

In [5]:

```
class MakeSubmission:
    def __init__(self, x_test: np.array, model_path: str, csv_path: str):
        self.x_test = x_test
        self.model_path = model_path
        self.csv_path = csv_path

        self.model = tf.keras.models.load_model(self.model_path)
        self.df = pd.read_csv(self.csv_path)

        preds = self.make_predictions()

        submission = pd.DataFrame({'id': self.df['id'], 'has_cactus': preds})
        submission.to_csv("sample_submission.csv", index=False)

    def make_predictions(self, make_it_ready: bool = True):
        preds = self.model.predict(self.x_test)

        if make_it_ready:
            preds = [np.argmax(n) for n in preds]

        return preds

```

In [6]:

```
os.makedirs("models", exist_ok=True)

dl = DataLoader()
X_data, Y_data = dl.read_data(True, (32, 32), False, True, True, 2)

```

```
  0%|          | 71/17500 [00:00<00:24, 703.67it/s]
```

```
NPY files not found!

```

```
100%|██████████| 17500/17500 [00:25<00:00, 692.41it/s]

```

In [7]:

```
trainer = TrainWithKeras(X_data, Y_data, model_path="models/model.h5", epochs=50, batch_size=1024, lr=0.0002)
model = trainer.make_model()
model = trainer.compile(model)

histroy = trainer.train(model)

```

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 30, 30, 64)        1792      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 15, 15, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 13, 13, 128)       73856     
_________________________________________________________________
dropout_1 (Dropout)          (None, 13, 13, 128)       0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 13, 13, 128)       512       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 11, 11, 256)       295168    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 256)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 3, 512)         1180160   
_________________________________________________________________
dropout_2 (Dropout)          (None, 3, 3, 512)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 1, 1024)        4719616   
_________________________________________________________________
flatten_1 (Flatten)          (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dropout_3 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 2050      
=================================================================
Total params: 7,322,754
Trainable params: 7,322,498
Non-trainable params: 256
_________________________________________________________________
Epoch 1/50
17500/17500 [==============================] - 5s 291us/step - loss: 0.4455 - acc: 0.7814
Epoch 2/50
17500/17500 [==============================] - 2s 88us/step - loss: 0.1582 - acc: 0.9434
Epoch 3/50
17500/17500 [==============================] - 2s 88us/step - loss: 0.0925 - acc: 0.9665
Epoch 4/50
17500/17500 [==============================] - 2s 88us/step - loss: 0.0686 - acc: 0.9747
Epoch 5/50
17500/17500 [==============================] - 2s 88us/step - loss: 0.0623 - acc: 0.9774
Epoch 6/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0475 - acc: 0.9833
Epoch 7/50
17500/17500 [==============================] - 2s 88us/step - loss: 0.0484 - acc: 0.9819
Epoch 8/50
17500/17500 [==============================] - 2s 88us/step - loss: 0.0393 - acc: 0.9862
Epoch 9/50
17500/17500 [==============================] - 2s 89us/step - loss: 0.0394 - acc: 0.9858
Epoch 10/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0325 - acc: 0.9884
Epoch 11/50
17500/17500 [==============================] - 2s 86us/step - loss: 0.0468 - acc: 0.9848
Epoch 12/50
17500/17500 [==============================] - 1s 85us/step - loss: 0.0314 - acc: 0.9895
Epoch 13/50
17500/17500 [==============================] - 1s 86us/step - loss: 0.0414 - acc: 0.9859
Epoch 14/50
17500/17500 [==============================] - 1s 85us/step - loss: 0.0295 - acc: 0.9893
Epoch 15/50
17500/17500 [==============================] - 1s 86us/step - loss: 0.0203 - acc: 0.9929
Epoch 16/50
17500/17500 [==============================] - 2s 86us/step - loss: 0.0251 - acc: 0.9920
Epoch 17/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0193 - acc: 0.9935
Epoch 18/50
17500/17500 [==============================] - 2s 86us/step - loss: 0.0247 - acc: 0.9914
Epoch 19/50
17500/17500 [==============================] - 2s 86us/step - loss: 0.0179 - acc: 0.9938
Epoch 20/50
17500/17500 [==============================] - 2s 86us/step - loss: 0.0149 - acc: 0.9946
Epoch 21/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0119 - acc: 0.9957
Epoch 22/50
17500/17500 [==============================] - 2s 86us/step - loss: 0.0110 - acc: 0.9963
Epoch 23/50
17500/17500 [==============================] - 1s 85us/step - loss: 0.0120 - acc: 0.9958
Epoch 24/50
17500/17500 [==============================] - 2s 86us/step - loss: 0.0228 - acc: 0.9921
Epoch 25/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0134 - acc: 0.9960
Epoch 26/50
17500/17500 [==============================] - 2s 88us/step - loss: 0.0104 - acc: 0.9963
Epoch 27/50
17500/17500 [==============================] - 2s 86us/step - loss: 0.0136 - acc: 0.9954
Epoch 28/50
17500/17500 [==============================] - 1s 85us/step - loss: 0.0086 - acc: 0.9970
Epoch 29/50
17500/17500 [==============================] - 2s 86us/step - loss: 0.0096 - acc: 0.9967
Epoch 30/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0080 - acc: 0.9971
Epoch 31/50
17500/17500 [==============================] - 1s 86us/step - loss: 0.0204 - acc: 0.9934
Epoch 32/50
17500/17500 [==============================] - 1s 86us/step - loss: 0.0141 - acc: 0.9948
Epoch 33/50
17500/17500 [==============================] - 2s 86us/step - loss: 0.0067 - acc: 0.9974
Epoch 34/50
17500/17500 [==============================] - 1s 85us/step - loss: 0.0070 - acc: 0.9975
Epoch 35/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0129 - acc: 0.9950
Epoch 36/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0072 - acc: 0.9976
Epoch 37/50
17500/17500 [==============================] - 2s 88us/step - loss: 0.0204 - acc: 0.9926
Epoch 38/50
17500/17500 [==============================] - 2s 88us/step - loss: 0.0097 - acc: 0.9964
Epoch 39/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0054 - acc: 0.9983
Epoch 40/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0055 - acc: 0.9979
Epoch 41/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0059 - acc: 0.9978
Epoch 42/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0095 - acc: 0.9965
Epoch 43/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0057 - acc: 0.9979
Epoch 44/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0058 - acc: 0.9983
Epoch 45/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0055 - acc: 0.9982
Epoch 46/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0036 - acc: 0.9986
Epoch 47/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0049 - acc: 0.9983
Epoch 48/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0055 - acc: 0.9981
Epoch 49/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0044 - acc: 0.9984
Epoch 50/50
17500/17500 [==============================] - 2s 87us/step - loss: 0.0031 - acc: 0.9989

```

In [8]:

```
X_data_test, Y_data_test = dl.read_test_data(True, (32, 32), False, True, False)
ms = MakeSubmission(X_data_test, "models/model.h5", "../input/sample_submission.csv")

```

```
  2%|▏         | 62/4000 [00:00<00:06, 614.87it/s]
```

```
NPY files not found!

```

```
100%|██████████| 4000/4000 [00:05<00:00, 669.71it/s]

```