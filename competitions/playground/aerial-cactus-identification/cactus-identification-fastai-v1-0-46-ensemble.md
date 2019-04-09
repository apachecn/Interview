# Cactus Identification fastai v1.0.46 ensemble

> Author: https://www.kaggle.com/mnpinto

> From: https://www.kaggle.com/mnpinto/cactus-identification-fastai-v1-0-46-ensemble

> License: [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

> Score: 0.9996

# Cactus Identification fastai baseline

In [1]:

```
import fastai
from fastai.vision import *
from sklearn.model_selection import KFold

```

In [2]:

```
# Copy pretrained model weights to the default path
!mkdir '/tmp/.torch'
!mkdir '/tmp/.torch/models/'
#!cp '../input/resnet18/resnet18.pth' '/tmp/.torch/models/resnet18-5c106cde.pth'
#!cp '../input/densenet121/densenet121.pth' '/tmp/.torch/models/densenet121-a639ec97.pth'
!cp '../input/densenet201/densenet201.pth' '/tmp/.torch/models/densenet201-c1103571.pth'

```

In [3]:

```
fastai.__version__

```

Out[3]:

```
'1.0.46'
```

In [4]:

```
data_path = Path('../input/aerial-cactus-identification')
df = pd.read_csv(data_path/'train.csv')
df.head()

```

Out[4]:

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

In [5]:

```
sub_csv = pd.read_csv(data_path/'sample_submission.csv')
sub_csv.head()

```

Out[5]:

|  | id | has_cactus |
| --- | --- | --- |
| 0 | 000940378805c44108d287872b2f04ce.jpg | 0.5 |
| --- | --- | --- |
| 1 | 0017242f54ececa4512b4d7937d1e21e.jpg | 0.5 |
| --- | --- | --- |
| 2 | 001ee6d8564003107853118ab87df407.jpg | 0.5 |
| --- | --- | --- |
| 3 | 002e175c3c1e060769475f52182583d0.jpg | 0.5 |
| --- | --- | --- |
| 4 | 0036e44a7e8f7218e9bc7bf8137e4943.jpg | 0.5 |
| --- | --- | --- |

In [6]:

```
def create_databunch(valid_idx):
    test = ImageList.from_df(sub_csv, path=data_path/'test', folder='test')
    data = (ImageList.from_df(df, path=data_path/'train', folder='train')
            .split_by_idx(valid_idx)
            .label_from_df()
            .add_test(test)
            .transform(get_transforms(flip_vert=True, max_rotate=20.0), size=128)
            .databunch(path='.', bs=64)
            .normalize(imagenet_stats)
           )
    return data

```

**5 fold ensemble**

In [7]:

```
kf = KFold(n_splits=5, random_state=379)
epochs = 6
lr = 1e-2
preds = []
for train_idx, valid_idx in kf.split(df):
    data = create_databunch(valid_idx)
    learn = create_cnn(data, models.densenet201, metrics=[accuracy])
    learn.fit_one_cycle(epochs, slice(lr))
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr/400, lr/4))
    learn.fit_one_cycle(epochs, slice(lr/800, lr/8))
    preds.append(learn.get_preds(ds_type=DatasetType.Test))

```

Total time: 06:07

| epoch | train_loss | valid_loss | accuracy | time |
| --- | --- | --- | --- | --- |
| 1 | 0.056767 | 0.018588 | 0.994000 | 01:08 |
| 2 | 0.022490 | 0.010378 | 0.995429 | 01:00 |
| 3 | 0.025892 | 0.003561 | 0.998857 | 01:00 |
| 4 | 0.012079 | 0.038481 | 0.996571 | 00:59 |
| 5 | 0.006535 | 0.002324 | 0.999143 | 00:57 |
| 6 | 0.003622 | 0.002164 | 0.999143 | 01:01 |

 66.67% [4/6 04:47<02:23]

| epoch | train_loss | valid_loss | accuracy | time |
| --- | --- | --- | --- | --- |
| 1 | 0.007291 | 0.003572 | 0.998286 | 01:12 |
| 2 | 0.014519 | 0.005793 | 0.998286 | 01:11 |
| 3 | 0.008674 | 0.003707 | 0.998286 | 01:10 |
| 4 | 0.007575 | 0.001763 | 0.999143 | 01:13 |

 91.28% [199/218 00:58<00:05 0.0037]In [8]:

```
ens = torch.cat([preds[i][0][:,1].view(-1, 1) for i in range(5)], dim=1)
ens  = (ens.mean(1)>0.5).long(); ens[:10]

```

Out[8]:

```
tensor([1, 1, 0, 0, 1, 1, 1, 1, 1, 0])
```

In [9]:

```
sub_csv['has_cactus'] = ens

```

In [10]:

```
sub_csv.to_csv('submission.csv', index=False)

```