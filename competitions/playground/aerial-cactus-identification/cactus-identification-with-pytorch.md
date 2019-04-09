# Cactus Identification with Pytorch

> Author: https://www.kaggle.com/nelsongriffiths

> From: https://www.kaggle.com/nelsongriffiths/cactus-identification-with-pytorch

> License: [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

In [1]:

```
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import cv2
import matplotlib.pyplot as plt

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

```

```
CUDA is available!  Training on GPU ...

```

# Data Prep

First, I am going to import our data sources and take a look at what we are working with. We have a csv file that contains our target variable and a folder with our cactus images.

In [2]:

```
df = pd.read_csv('../input/train.csv')
df.head()

```

Out[2]:

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

In [3]:

```
df['has_cactus'].value_counts(normalize=True)

```

Out[3]:

```
1    0.750629
0    0.249371
Name: has_cactus, dtype: float64
```

In [4]:

```
train_df, val_df = train_test_split(df, stratify = df.has_cactus, test_size=.2)

```

In [5]:

```
#Checking that validation set has same proportions as original training data
val_df['has_cactus'].value_counts(normalize=True)

```

Out[5]:

```
1    0.750571
0    0.249429
Name: has_cactus, dtype: float64
```

In [6]:

```
#Build a class for our data to put our images and target variables into our pytorch dataloader
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

class DataSet(torch.utils.data.Dataset):
    def __init__(self, labels, data_directory, transform=None):
        super().__init__()
        self.labels = labels.values
        self.data_dir = data_directory
        self.transform=transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        name, label = self.labels[index]
        img_path = os.path.join(self.data_dir, name)
        img = cv2.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

```

In [7]:

```
batch_size = 32

# Transform training data with random flips and normalize it to prepare it for dataloader
train_transforms = transforms.Compose([transforms.ToPILImage(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

val_transforms = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_data = DataSet(train_df,'../input/train/train', transform = train_transforms)
val_data = DataSet(val_df,'../input/train/train', transform = val_transforms)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
val_data_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)

```

In [8]:

```
#Checking what our cactus look like
fig,ax = plt.subplots(1,3,figsize=(15,5))

for i, idx in enumerate(train_df[train_df['has_cactus']==1]['id'][0:3]):
  path = os.path.join('../input/train/train',idx)
  ax[i].imshow(cv2.imread(path))

```

![](cactus-identification-with-pytorch_files/__results___8_0.png)In [9]:

```
#Building a CNN from scratch

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding = 1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2*16*16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(p = .25)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 2*16*16)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))

        return x

```

In [10]:

```
model = Net()
if train_on_gpu:
    model = model.cuda()

epochs = 30
learning_rate = .003

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

```

In [11]:

```
#Training and validation for model

best_loss = np.Inf
best_model = Net()
if train_on_gpu:
    best_model.cuda()

for epoch in range(1, epochs+1):
    train_loss = 0
    val_loss = 0

    model.train()
    for images, labels in train_data_loader:

        if train_on_gpu:
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        #print('Loss: {}'.format(loss.item()))

    model.eval()
    for images, labels in val_data_loader:

        if train_on_gpu:
            images = images.cuda()
            labels = labels.cuda()

        out = model(images)
        loss = criterion(out, labels)

        val_loss += loss.item()

    train_loss = train_loss/len(train_data_loader.dataset)
    val_loss = val_loss/len(val_data_loader.dataset)  
    print('Epoch: {}  \tTraining Loss: {:.6f}  \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

    #Saving the weights of the best model according to validation score
    if val_loss < best_loss:
        print('Improved Model Score - Updating Best Model Parameters...')
        best_model.load_state_dict(model.state_dict())

```

```
Epoch: 1 	Training Loss: 0.007002 	Validation Loss: 0.003353
Improved Model Score - Updating Best Model Parameters...
Epoch: 2 	Training Loss: 0.003053 	Validation Loss: 0.002155
Improved Model Score - Updating Best Model Parameters...
Epoch: 3 	Training Loss: 0.002027 	Validation Loss: 0.001604
Improved Model Score - Updating Best Model Parameters...
Epoch: 4 	Training Loss: 0.001602 	Validation Loss: 0.001545
Improved Model Score - Updating Best Model Parameters...
Epoch: 5 	Training Loss: 0.001286 	Validation Loss: 0.001005
Improved Model Score - Updating Best Model Parameters...
Epoch: 6 	Training Loss: 0.001038 	Validation Loss: 0.001039
Improved Model Score - Updating Best Model Parameters...
Epoch: 7 	Training Loss: 0.000975 	Validation Loss: 0.001050
Improved Model Score - Updating Best Model Parameters...
Epoch: 8 	Training Loss: 0.000826 	Validation Loss: 0.000844
Improved Model Score - Updating Best Model Parameters...
Epoch: 9 	Training Loss: 0.000768 	Validation Loss: 0.000574
Improved Model Score - Updating Best Model Parameters...
Epoch: 10 	Training Loss: 0.000666 	Validation Loss: 0.000445
Improved Model Score - Updating Best Model Parameters...
Epoch: 11 	Training Loss: 0.000604 	Validation Loss: 0.000394
Improved Model Score - Updating Best Model Parameters...
Epoch: 12 	Training Loss: 0.000607 	Validation Loss: 0.000716
Improved Model Score - Updating Best Model Parameters...
Epoch: 13 	Training Loss: 0.000459 	Validation Loss: 0.000416
Improved Model Score - Updating Best Model Parameters...
Epoch: 14 	Training Loss: 0.000544 	Validation Loss: 0.000305
Improved Model Score - Updating Best Model Parameters...
Epoch: 15 	Training Loss: 0.000390 	Validation Loss: 0.000316
Improved Model Score - Updating Best Model Parameters...
Epoch: 16 	Training Loss: 0.000345 	Validation Loss: 0.000183
Improved Model Score - Updating Best Model Parameters...
Epoch: 17 	Training Loss: 0.000329 	Validation Loss: 0.000158
Improved Model Score - Updating Best Model Parameters...
Epoch: 18 	Training Loss: 0.000285 	Validation Loss: 0.000158
Improved Model Score - Updating Best Model Parameters...
Epoch: 19 	Training Loss: 0.000244 	Validation Loss: 0.000087
Improved Model Score - Updating Best Model Parameters...
Epoch: 20 	Training Loss: 0.000260 	Validation Loss: 0.000235
Improved Model Score - Updating Best Model Parameters...
Epoch: 21 	Training Loss: 0.000208 	Validation Loss: 0.000061
Improved Model Score - Updating Best Model Parameters...
Epoch: 22 	Training Loss: 0.000214 	Validation Loss: 0.000726
Improved Model Score - Updating Best Model Parameters...
Epoch: 23 	Training Loss: 0.000208 	Validation Loss: 0.000097
Improved Model Score - Updating Best Model Parameters...
Epoch: 24 	Training Loss: 0.000172 	Validation Loss: 0.000169
Improved Model Score - Updating Best Model Parameters...
Epoch: 25 	Training Loss: 0.000192 	Validation Loss: 0.000245
Improved Model Score - Updating Best Model Parameters...
Epoch: 26 	Training Loss: 0.000222 	Validation Loss: 0.000091
Improved Model Score - Updating Best Model Parameters...
Epoch: 27 	Training Loss: 0.000112 	Validation Loss: 0.000061
Improved Model Score - Updating Best Model Parameters...
Epoch: 28 	Training Loss: 0.000119 	Validation Loss: 0.000155
Improved Model Score - Updating Best Model Parameters...
Epoch: 29 	Training Loss: 0.000139 	Validation Loss: 0.000022
Improved Model Score - Updating Best Model Parameters...
Epoch: 30 	Training Loss: 0.000132 	Validation Loss: 0.000035
Improved Model Score - Updating Best Model Parameters...

```

In [12]:

```
#Check model accuracy
best_model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_data_loader:
        if train_on_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = best_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))

```

```
Test Accuracy: 99.96428571428571 %

```