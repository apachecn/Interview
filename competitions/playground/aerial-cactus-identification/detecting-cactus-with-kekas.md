# Detecting cactus with kekas

> Author: https://www.kaggle.com/artgor

> From: https://www.kaggle.com/artgor/detecting-cactus-with-kekas

> License: [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## General information

![](detecting-cactus-with-kekas_files/cactus_0163.jpg)

Researchers in Mexico have created the VIGIA project, aiming to build a system for autonomous surveillance of protected areas. One of the first steps is being able to recognize the vegetation in the area. In this competition we are trying to identify whether there is a cactus in the image.

In this kernel I use kekas ([https://github.com/belskikh/kekas](https://github.com/belskikh/kekas)) as a wrapper for Pytorch.

Most of the code is taken from my other kernel: [https://www.kaggle.com/artgor/cancer-detection-with-kekas](https://www.kaggle.com/artgor/cancer-detection-with-kekas)

CodeIn [1]:

```
# libraries
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score
import cv2

```

Some of good libraries for DL aren't available in Docker with GPU by default, so it is necessary to install them. (don't forget to turn on internet connection in kernels).

In [2]:

```
!pip install albumentations > /dev/null 2>&1
!pip install pretrainedmodels > /dev/null 2>&1
!pip install kekas > /dev/null 2>&1
!pip install adabound > /dev/null 2>&1

```

CodeIn [3]:

```
# more imports
import albumentations
from albumentations import torch as AT
import pretrainedmodels
import adabound

from kekas import Keker, DataOwner, DataKek
from kekas.transformations import Transformer, to_torch, normalize
from kekas.metrics import accuracy
from kekas.modules import Flatten, AdaptiveConcatPool2d
from kekas.callbacks import Callback, Callbacks, DebuggerCallback
from kekas.utils import DotDict

```

```
/opt/conda/lib/python3.6/site-packages/kekas/keker.py:9: UserWarning: Error 'No module named 'apex''' during importing apex library. To use mixed precison you should install it from https://github.com/NVIDIA/apex
  warnings.warn(f"Error '{e}'' during importing apex library. To use mixed precison"

```

## Data overview

In [4]:

```
labels = pd.read_csv('../input/train.csv')
fig = plt.figure(figsize=(25, 8))
train_imgs = os.listdir("../input/train/train")
for idx, img in enumerate(np.random.choice(train_imgs, 20)):
    ax = fig.add_subplot(4, 20//4, idx+1, xticks=[], yticks=[])
    im = Image.open("../input/train/train/" + img)
    plt.imshow(im)
    lab = labels.loc[labels['id'] == img, 'has_cactus'].values[0]
    ax.set_title(f'Label: {lab}')

```

![](detecting-cactus-with-kekas_files/__results___6_0.png)

Images were resized, so I can see almost nothing in them...

Kekas accepts pandas DataFrame as an input and iterates over it to get image names and labels

In [5]:

```
test_img = os.listdir('../input/test/test')
test_df = pd.DataFrame(test_img, columns=['id'])
test_df['has_cactus'] = -1
test_df['data_type'] = 'test'

labels['has_cactus'] = labels['has_cactus'].astype(int)
labels['data_type'] = 'train'

labels.head()

```

Out[5]:

|  | id | has_cactus | data_type |
| --- | --- | --- | --- |
| 0 | 0004be2cfeaba1c0361d39e2b000257b.jpg | 1 | train |
| --- | --- | --- | --- |
| 1 | 000c8a36845c0208e833c79c1bffedd1.jpg | 1 | train |
| --- | --- | --- | --- |
| 2 | 000d1e9a533f62e55c289303b072733d.jpg | 1 | train |
| --- | --- | --- | --- |
| 3 | 0011485b40695e9138e92d0b3fb55128.jpg | 1 | train |
| --- | --- | --- | --- |
| 4 | 0014d7a11e90b62848904c1418fc8cf2.jpg | 1 | train |
| --- | --- | --- | --- |

In [6]:

```
labels.loc[labels['data_type'] == 'train', 'has_cactus'].value_counts()

```

Out[6]:

```
1    13136
0     4364
Name: has_cactus, dtype: int64
```

We have some disbalance in the data, but it isn't too big.

In [7]:

```
# splitting data into train and validation
train, valid = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)

```

### Reader function

At first it is necessary to create a reader function, which will open images. It accepts i and row as input (like from pandas iterrows). The function should return a dictionary with image and label. [:,:,::-1] - is a neat trick which converts BGR images to RGB, it works faster that converting to RGB by usual means.

In [8]:

```
def reader_fn(i, row):
    image = cv2.imread(f"../input/{row['data_type']}/{row['data_type']}/{row['id']}")[:,:,::-1] # BGR -> RGB
    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}

```

### Data transformation

Next step is defining data transformations and augmentations. This differs from standard PyTorch way. We define resizing, augmentations and normalizing separately, this allows to easily create separate transformers for train and valid/test data.

At first we define augmentations. We create a function with a list of augmentations (I prefer albumentation library: [https://github.com/albu/albumentations](https://github.com/albu/albumentations))

In [9]:

```
def augs(p=0.5):
    return albumentations.Compose([
        albumentations.HorizontalFlip(),
        albumentations.VerticalFlip(),
        albumentations.RandomBrightness(),
    ], p=p)

```

Now we create a transforming function. It heavily uses Transformer from kekas.

*   The first step is defining resizing. You can change arguments of function if you want images to have different height and width, otherwis you can leave it as it is.
*   Next step is defining augmentations. Here we provide the key of image which is defined in reader_fn;
*   The third step is defining final transformation to tensor and normalizing;
*   After this we can compose separate transformations for train and valid/test data;

In [10]:

```
def get_transforms(dataset_key, size, p):

    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))

    AUGS = Transformer(dataset_key, lambda x: augs()(image=x)["image"])

    NRM_TFMS = transforms.Compose([
        Transformer(dataset_key, to_torch()),
        Transformer(dataset_key, normalize())
    ])

    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])

    return train_tfms, val_tfms

```

In [11]:

```
train_tfms, val_tfms = get_transforms("image", 32, 0.5)

```

Now we can create a DataKek, which is similar to creating dataset in Pytorch. We define the data, reader function and transformation.Then we can define standard PyTorch DataLoader.

In [12]:

```
train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = DataLoader(train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True)
val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)

```

In [13]:

```
test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = DataLoader(test_dk, batch_size=batch_size, num_workers=workers, shuffle=False)

```

### Building a neural net

Here we define the architecture of the neural net.

*   Pre-trained backbone is taken from pretrainedmodels: [https://github.com/Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) Here I take densenet169
*   We also define changes to the architecture. For example, we take off the last layer and add a custom head with nn.Sequential. AdaptiveConcatPool2d is a layer in kekas, which concats AdaptiveMaxPooling and AdaptiveAveragePooling

In [14]:

```
class Net(nn.Module):
    def __init__(
            self,
            num_classes: int,
            p: float = 0.2,
            pooling_size: int = 2,
            last_conv_size: int = 1664,
            arch: str = "densenet169",
            pretrained: str = "imagenet") -> None:
        """A simple model to finetune.

 Args:
 num_classes: the number of target classes, the size of the last layer's output
 p: dropout probability
 pooling_size: the size of the result feature map after adaptive pooling layer
 last_conv_size: size of the flatten last backbone conv layer
 arch: the name of the architecture form pretrainedmodels
 pretrained: the mode for pretrained model from pretrainedmodels
 """
        super().__init__()
        net = pretrainedmodels.__dict__[arch](pretrained=pretrained)
        modules = list(net.children())[:-1]  # delete last layer
        # add custom head
        modules += [nn.Sequential(
            # AdaptiveConcatPool2d is a concat of AdaptiveMaxPooling and AdaptiveAveragePooling 
            # AdaptiveConcatPool2d(size=pooling_size),
            Flatten(),
            nn.BatchNorm1d(1664),
            nn.Dropout(p),
            nn.Linear(1664, num_classes)
        )]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return logits

```

The data for training needs to be transformed one more time - we define DataOwner, which contains all the data. For now let's define it for train and valid. Next we define model and loss. As I choose BCEWithLogitsLoss, we can set the number of classes for output to 1.

In [15]:

```
dataowner = DataOwner(train_dl, val_dl, None)
model = Net(num_classes=1)
criterion = nn.BCEWithLogitsLoss()

```

```
Downloading: "http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth" to /tmp/.torch/models/densenet169-f470b90a4.pth
57372314it [00:03, 14671003.61it/s]

```

And now we define what will the model do with the data. For example we could slice the output and take only a part of it. For now we will simply return the output of the model.

In [16]:

```
def step_fn(model: torch.nn.Module,
            batch: torch.Tensor) -> torch.Tensor:
    """Determine what your model will do with your data.

 Args:
 model: the pytorch module to pass input in
 batch: the batch of data from the DataLoader

 Returns:
 The models forward pass results
 """

    inp = batch["image"]
    return model(inp)

```

Defining custom metrics

In [17]:

```
def bce_accuracy(target: torch.Tensor,
                 preds: torch.Tensor,
                 thresh: bool = 0.5) -> float:
    target = target.cpu().detach().numpy()
    preds = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target, preds)

def roc_auc(target: torch.Tensor,
                 preds: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    return roc_auc_score(target, preds)

```

### Keker

Now we can define the Keker - the core Kekas class for training the model.

Here we define everything which is necessary for training:

*   the model which was defined earlier;
*   dataowner containing the data for training and validation;
*   criterion;
*   step function;
*   the key of labels, which was defined in the reader function;
*   the dictionary with metrics (there can be several of them);
*   The optimizer and its parameters;

In [18]:

```
keker = Keker(model=model,
              dataowner=dataowner,
              criterion=criterion,
              step_fn=step_fn,
              target_key="label",
              metrics={"acc": bce_accuracy, 'auc': roc_auc},
              opt=torch.optim.SGD,
              opt_params={"momentum": 0.99})

```

In [19]:

```
keker.unfreeze(model_attr="net")

layer_num = -1
keker.freeze_to(layer_num, model_attr="net")

```

In [20]:

```
keker.kek_one_cycle(max_lr=1e-2,                  # the maximum learning rate
                    cycle_len=5,                  # number of epochs, actually, but not exactly
                    momentum_range=(0.95, 0.85),  # range of momentum changes
                    div_factor=25,                # max_lr / min_lr
                    increase_fraction=0.3,        # the part of cycle when learning rate increases
                    logdir='train_logs')
keker.plot_kek('train_logs')

```

```
Epoch 1/5: 100% 218/218 [00:52<00:00,  4.94it/s, loss=0.0502, val_loss=0.0337, acc=0.9896, auc=0.9995]
Epoch 2/5: 100% 218/218 [00:42<00:00,  6.22it/s, loss=0.0131, val_loss=0.0152, acc=0.9955, auc=0.9999]
Epoch 3/5: 100% 218/218 [00:41<00:00,  6.37it/s, loss=0.0146, val_loss=0.0112, acc=0.9955, auc=1.0000]
Epoch 4/5: 100% 218/218 [00:43<00:00,  5.73it/s, loss=0.0097, val_loss=0.0109, acc=0.9955, auc=1.0000]
Epoch 5/5: 100% 218/218 [00:41<00:00,  6.24it/s, loss=0.0107, val_loss=0.0092, acc=0.9957, auc=1.0000]

```

<svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525" style="background: rgb(255, 255, 255) none repeat scroll 0% 0%;"><g class="cartesianlayer"><g class="subplot xy"><g class="xaxislayer-above"><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(80,0)">0</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="500" data-math="N" transform="translate(245.32,0)">500</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1000" data-math="N" transform="translate(410.65,0)">1000</text></g></g><g class="yaxislayer-above"><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(0,427.87)">0</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.1" data-math="N" transform="translate(0,383.54)">0.1</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.2" data-math="N" transform="translate(0,339.2)">0.2</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.3" data-math="N" transform="translate(0,294.86)">0.3</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.4" data-math="N" transform="translate(0,250.53)">0.4</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.5" data-math="N" transform="translate(0,206.19)">0.5</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.6" data-math="N" transform="translate(0,161.85)">0.6</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.7" data-math="N" transform="translate(0,117.51)">0.7</text></g></g></g></g></svg><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525"><g class="infolayer"><g class="legend" pointer-events="all" transform="translate(540.02, 100)"><g class="scrollbox" transform="translate(0, 0)" clip-path="url(#legend105b76)"><g class="groups"><g class="traces" style="opacity: 1;" transform="translate(0, 14.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="train/batch/loss" data-math="N">train/batch/loss</text></g><g class="traces" style="opacity: 1;" transform="translate(0, 33.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="val/batch/loss" data-math="N">val/batch/loss</text></g></g></g></g><g class="g-gtitle"><text class="gtitle" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(68, 68, 68); opacity: 1; font-weight: normal; white-space: pre;" x="346.7" y="50" text-anchor="middle" dy="0em" data-unformatted="batch/loss" data-math="N">batch/loss</text></g></g></svg>[<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 132 132" height="1em" width="1em"><title>plotly-logomark</title></svg>](https://plot.ly/)<svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525" style="background: rgb(255, 255, 255) none repeat scroll 0% 0%;"><g class="cartesianlayer"><g class="subplot xy"><g class="xaxislayer-above"><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(80,0)">0</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="500" data-math="N" transform="translate(246.42,0)">500</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1000" data-math="N" transform="translate(412.84,0)">1000</text></g></g><g class="yaxislayer-above"><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.6" data-math="N" transform="translate(0,382.21)">0.6</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.7" data-math="N" transform="translate(0,315.97)">0.7</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.8" data-math="N" transform="translate(0,249.73)">0.8</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.9" data-math="N" transform="translate(0,183.49)">0.9</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1" data-math="N" transform="translate(0,117.25)">1</text></g></g></g></g></svg><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525"><g class="infolayer"><g class="legend" pointer-events="all" transform="translate(543.0799999999999, 100)"><g class="scrollbox" transform="translate(0, 0)" clip-path="url(#legende39934)"><g class="groups"><g class="traces" style="opacity: 1;" transform="translate(0, 14.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="train/batch/acc" data-math="N">train/batch/acc</text></g><g class="traces" style="opacity: 1;" transform="translate(0, 33.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="val/batch/acc" data-math="N">val/batch/acc</text></g></g></g></g><g class="g-gtitle"><text class="gtitle" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(68, 68, 68); opacity: 1; font-weight: normal; white-space: pre;" x="346.7" y="50" text-anchor="middle" dy="0em" data-unformatted="batch/acc" data-math="N">batch/acc</text></g></g></svg>[<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 132 132" height="1em" width="1em"><title>plotly-logomark</title></svg>](https://plot.ly/)<svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525" style="background: rgb(255, 255, 255) none repeat scroll 0% 0%;"><g class="cartesianlayer"><g class="subplot xy"><g class="xaxislayer-above"><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(80,0)">0</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="500" data-math="N" transform="translate(246.06,0)">500</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1000" data-math="N" transform="translate(412.11,0)">1000</text></g></g><g class="yaxislayer-above"><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.5" data-math="N" transform="translate(0,433.09)">0.5</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.6" data-math="N" transform="translate(0,369.92)">0.6</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.7" data-math="N" transform="translate(0,306.75)">0.7</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.8" data-math="N" transform="translate(0,243.59)">0.8</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.9" data-math="N" transform="translate(0,180.42000000000002)">0.9</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1" data-math="N" transform="translate(0,117.25)">1</text></g></g></g></g></svg><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525"><g class="infolayer"><g class="legend" pointer-events="all" transform="translate(542.06, 100)"><g class="scrollbox" transform="translate(0, 0)" clip-path="url(#legend3f014a)"><g class="groups"><g class="traces" style="opacity: 1;" transform="translate(0, 14.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="train/batch/auc" data-math="N">train/batch/auc</text></g><g class="traces" style="opacity: 1;" transform="translate(0, 33.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="val/batch/auc" data-math="N">val/batch/auc</text></g></g></g></g><g class="g-gtitle"><text class="gtitle" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(68, 68, 68); opacity: 1; font-weight: normal; white-space: pre;" x="346.7" y="50" text-anchor="middle" dy="0em" data-unformatted="batch/auc" data-math="N">batch/auc</text></g></g></svg>[<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 132 132" height="1em" width="1em"><title>plotly-logomark</title></svg>](https://plot.ly/)<svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525" style="background: rgb(255, 255, 255) none repeat scroll 0% 0%;"><g class="cartesianlayer"><g class="subplot xy"><g class="xaxislayer-above"><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(80,0)">0</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="200" data-math="N" transform="translate(177.89,0)">200</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="400" data-math="N" transform="translate(275.78,0)">400</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="600" data-math="N" transform="translate(373.66,0)">600</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="800" data-math="N" transform="translate(471.55,0)">800</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1000" data-math="N" transform="translate(569.44,0)">1000</text></g></g><g class="yaxislayer-above"><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(0,440.69)">0</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.002" data-math="N" transform="translate(0,376)">0.002</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.004" data-math="N" transform="translate(0,311.31)">0.004</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.006" data-math="N" transform="translate(0,246.62)">0.006</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.008" data-math="N" transform="translate(0,181.94)">0.008</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.01" data-math="N" transform="translate(0,117.25)">0.01</text></g></g></g></g></svg><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525"><g class="infolayer"><g class="g-gtitle"><text class="gtitle" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(68, 68, 68); opacity: 1; font-weight: normal; white-space: pre;" x="346.7" y="50" text-anchor="middle" dy="0em" data-unformatted="batch/lr" data-math="N">batch/lr</text></g></g></svg>[<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 132 132" height="1em" width="1em"><title>plotly-logomark</title></svg>](https://plot.ly/)In [21]:

```
keker.kek_one_cycle(max_lr=1e-3,                  # the maximum learning rate
                    cycle_len=5,                  # number of epochs, actually, but not exactly
                    momentum_range=(0.95, 0.85),  # range of momentum changes
                    div_factor=25,                # max_lr / min_lr
                    increase_fraction=0.2,        # the part of cycle when learning rate increases
                    logdir='train_logs1')
keker.plot_kek('train_logs1')

```

```
Epoch 1/5: 100% 218/218 [00:41<00:00,  6.29it/s, loss=0.0149, val_loss=0.0089, acc=0.9963, auc=1.0000]
Epoch 2/5: 100% 218/218 [00:41<00:00,  6.34it/s, loss=0.0221, val_loss=0.0098, acc=0.9966, auc=1.0000]
Epoch 3/5: 100% 218/218 [00:42<00:00,  6.39it/s, loss=0.0122, val_loss=0.0094, acc=0.9960, auc=1.0000]
Epoch 4/5: 100% 218/218 [00:41<00:00,  5.93it/s, loss=0.0119, val_loss=0.0087, acc=0.9963, auc=1.0000]
Epoch 5/5: 100% 218/218 [00:41<00:00,  6.48it/s, loss=0.0106, val_loss=0.0086, acc=0.9974, auc=1.0000]

```

<svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525" style="background: rgb(255, 255, 255) none repeat scroll 0% 0%;"><g class="cartesianlayer"><g class="subplot xy"><g class="xaxislayer-above"><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(80,0)">0</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="500" data-math="N" transform="translate(245.32,0)">500</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1000" data-math="N" transform="translate(410.65,0)">1000</text></g></g><g class="yaxislayer-above"><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(0,428.18)">0</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.02" data-math="N" transform="translate(0,387.22)">0.02</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.04" data-math="N" transform="translate(0,346.27)">0.04</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.06" data-math="N" transform="translate(0,305.31)">0.06</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.08" data-math="N" transform="translate(0,264.35)">0.08</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.1" data-math="N" transform="translate(0,223.39)">0.1</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.12" data-math="N" transform="translate(0,182.44)">0.12</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.14" data-math="N" transform="translate(0,141.48)">0.14</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.16" data-math="N" transform="translate(0,100.52)">0.16</text></g></g></g></g></svg><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525"><g class="infolayer"><g class="legend" pointer-events="all" transform="translate(540.02, 100)"><g class="scrollbox" transform="translate(0, 0)" clip-path="url(#legend7718e1)"><g class="groups"><g class="traces" style="opacity: 1;" transform="translate(0, 14.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="train/batch/loss" data-math="N">train/batch/loss</text></g><g class="traces" style="opacity: 1;" transform="translate(0, 33.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="val/batch/loss" data-math="N">val/batch/loss</text></g></g></g></g><g class="g-gtitle"><text class="gtitle" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(68, 68, 68); opacity: 1; font-weight: normal; white-space: pre;" x="346.7" y="50" text-anchor="middle" dy="0em" data-unformatted="batch/loss" data-math="N">batch/loss</text></g></g></svg>[<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 132 132" height="1em" width="1em"><title>plotly-logomark</title></svg>](https://plot.ly/)<svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525" style="background: rgb(255, 255, 255) none repeat scroll 0% 0%;"><g class="cartesianlayer"><g class="subplot xy"><g class="xaxislayer-above"><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(80,0)">0</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="500" data-math="N" transform="translate(246.42,0)">500</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1000" data-math="N" transform="translate(412.84,0)">1000</text></g></g><g class="yaxislayer-above"><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.96" data-math="N" transform="translate(0,382.21)">0.96</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.97" data-math="N" transform="translate(0,315.97)">0.97</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.98" data-math="N" transform="translate(0,249.73)">0.98</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.99" data-math="N" transform="translate(0,183.49)">0.99</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1" data-math="N" transform="translate(0,117.25)">1</text></g></g></g></g></svg><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525"><g class="infolayer"><g class="legend" pointer-events="all" transform="translate(543.0799999999999, 100)"><g class="scrollbox" transform="translate(0, 0)" clip-path="url(#legend46d60f)"><g class="groups"><g class="traces" style="opacity: 1;" transform="translate(0, 14.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="train/batch/acc" data-math="N">train/batch/acc</text></g><g class="traces" style="opacity: 1;" transform="translate(0, 33.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="val/batch/acc" data-math="N">val/batch/acc</text></g></g></g></g><g class="g-gtitle"><text class="gtitle" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(68, 68, 68); opacity: 1; font-weight: normal; white-space: pre;" x="346.7" y="50" text-anchor="middle" dy="0em" data-unformatted="batch/acc" data-math="N">batch/acc</text></g></g></svg>[<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 132 132" height="1em" width="1em"><title>plotly-logomark</title></svg>](https://plot.ly/)<svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525" style="background: rgb(255, 255, 255) none repeat scroll 0% 0%;"><g class="cartesianlayer"><g class="subplot xy"><g class="xaxislayer-above"><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(80,0)">0</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="500" data-math="N" transform="translate(246.06,0)">500</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1000" data-math="N" transform="translate(412.11,0)">1000</text></g></g><g class="yaxislayer-above"><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.9975" data-math="N" transform="translate(0,427.36)">0.9975</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.998" data-math="N" transform="translate(0,365.34)">0.998</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.9985" data-math="N" transform="translate(0,303.32)">0.9985</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.999" data-math="N" transform="translate(0,241.29)">0.999</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.9995" data-math="N" transform="translate(0,179.26999999999998)">0.9995</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1" data-math="N" transform="translate(0,117.25)">1</text></g></g></g></g></svg><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525"><g class="infolayer"><g class="legend" pointer-events="all" transform="translate(542.06, 100)"><g class="scrollbox" transform="translate(0, 0)" clip-path="url(#legend8b7908)"><g class="groups"><g class="traces" style="opacity: 1;" transform="translate(0, 14.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="train/batch/auc" data-math="N">train/batch/auc</text></g><g class="traces" style="opacity: 1;" transform="translate(0, 33.5)"><text class="legendtext user-select-none" text-anchor="start" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" x="40" y="4.680000000000001" data-unformatted="val/batch/auc" data-math="N">val/batch/auc</text></g></g></g></g><g class="g-gtitle"><text class="gtitle" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(68, 68, 68); opacity: 1; font-weight: normal; white-space: pre;" x="346.7" y="50" text-anchor="middle" dy="0em" data-unformatted="batch/auc" data-math="N">batch/auc</text></g></g></svg>[<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 132 132" height="1em" width="1em"><title>plotly-logomark</title></svg>](https://plot.ly/)<svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525" style="background: rgb(255, 255, 255) none repeat scroll 0% 0%;"><g class="cartesianlayer"><g class="subplot xy"><g class="xaxislayer-above"><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(80,0)">0</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="200" data-math="N" transform="translate(177.89,0)">200</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="400" data-math="N" transform="translate(275.78,0)">400</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="600" data-math="N" transform="translate(373.66,0)">600</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="800" data-math="N" transform="translate(471.55,0)">800</text></g><g class="xtick"><text text-anchor="middle" x="0" y="458" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="1000" data-math="N" transform="translate(569.44,0)">1000</text></g></g><g class="yaxislayer-above"><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0" data-math="N" transform="translate(0,440.69)">0</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.0002" data-math="N" transform="translate(0,376)">0.0002</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.0004" data-math="N" transform="translate(0,311.31)">0.0004</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.0006" data-math="N" transform="translate(0,246.63)">0.0006</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.0008" data-math="N" transform="translate(0,181.94)">0.0008</text></g><g class="ytick"><text text-anchor="end" x="79" y="4.199999999999999" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 12px; fill: rgb(68, 68, 68); fill-opacity: 1; white-space: pre;" data-unformatted="0.001" data-math="N" transform="translate(0,117.25)">0.001</text></g></g></g></g></svg><svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xlink="http://www.w3.org/1999/xlink" width="693.4" height="525"><g class="infolayer"><g class="g-gtitle"><text class="gtitle" style="font-family: &quot;Open Sans&quot;, verdana, arial, sans-serif; font-size: 17px; fill: rgb(68, 68, 68); opacity: 1; font-weight: normal; white-space: pre;" x="346.7" y="50" text-anchor="middle" dy="0em" data-unformatted="batch/lr" data-math="N">batch/lr</text></g></g></svg>[<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 132 132" height="1em" width="1em"><title>plotly-logomark</title></svg>](https://plot.ly/)

### Predicting and TTA

Simply predicting on test data is okay, but it is better to use TTA - test time augmentation. Let's see how it can be done with Kekas.

*   define augmentations;
*   define augmentation function;
*   create objects with these augmentations;
*   put these objects into a single dictionary;

In [22]:

```
preds = keker.predict_loader(loader=test_dl)

```

```
Predict: 100% 63/63 [00:11<00:00,  6.02it/s]

```

In [23]:

```
# flip_ = albumentations.HorizontalFlip(always_apply=True)
# transpose_ = albumentations.Transpose(always_apply=True)

# def insert_aug(aug, dataset_key="image", size=224): 
#     PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))

#     AUGS = Transformer(dataset_key, lambda x: aug(image=x)["image"])

#     NRM_TFMS = transforms.Compose([
#         Transformer(dataset_key, to_torch()),
#         Transformer(dataset_key, normalize())
#     ])

#     tfm = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
#     return tfm

# flip = insert_aug(flip_)
# transpose = insert_aug(transpose_)

# tta_tfms = {"flip": flip, "transpose": transpose}

# # third, run TTA
# keker.TTA(loader=test_dl,                # loader to predict on 
#           tfms=tta_tfms,                # list or dict of always applying transforms
#           savedir="tta_preds1",  # savedir
#           prefix="preds")               # (optional) name prefix. default is 'preds'

```

In [24]:

```
# prediction = np.zeros((test_df.shape[0], 1))
# for i in os.listdir('tta_preds1'):
#     pr = np.load('tta_preds1/' + i)
#     prediction += pr
# prediction = prediction / len(os.listdir('tta_preds1'))

```

In [25]:

```
test_preds = pd.DataFrame({'imgs': test_df.id.values, 'preds': preds.reshape(-1,)})
test_preds.columns = ['id', 'has_cactus']
test_preds.to_csv('sub.csv', index=False)
test_preds.head()

```

Out[25]:

|  | id | has_cactus |
| --- | --- | --- |
| 0 | 79ac4cc3b082e0a1defe1be601806efd.jpg | 4.901506 |
| --- | --- | --- |
| 1 | e880364d6521c6f3a27748ec62b0e335.jpg | 9.305355 |
| --- | --- | --- |
| 2 | 74912492b6cdf28c4bfb9c8e1d35af3e.jpg | 8.171706 |
| --- | --- | --- |
| 3 | 078cfa961183b30693ea2f13f5ff6d17.jpg | 8.186723 |
| --- | --- | --- |
| 4 | 7fd729184ef182899ce3e7a174fb9bc0.jpg | 9.269026 |
| --- | --- | --- |