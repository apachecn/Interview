# Simple_FastAI_exercise

> Author: https://www.kaggle.com/kenseitrg

> From: https://www.kaggle.com/kenseitrg/simple-fastai-exercise

> License: [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

> Score: 1.0000

In [1]:

```py
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

```

In [2]:

```py
from pathlib import Path
from fastai import *
from fastai.vision import *
import torch

```

In [3]:

```py
data_folder = Path("../input")
#data_folder.ls()

```

In [4]:

```py
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")

```

In [5]:

```py
test_img = ImageList.from_df(test_df, path=data_folder/'test', folder='test')
trfm = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
train_img = (ImageList.from_df(train_df, path=data_folder/'train', folder='train')
        .split_by_rand_pct(0.01)
        .label_from_df()
        .add_test(test_img)
        .transform(trfm, size=128)
        .databunch(path='.', bs=64, device= torch.device('cuda:0'))
        .normalize(imagenet_stats)
       )

```

In [6]:

```py
#train_img.show_batch(rows=3, figsize=(7,6))

```

In [7]:

```py
learn = cnn_learner(train_img, models.densenet161, metrics=[error_rate, accuracy])

```

```
Downloading: "https://download.pytorch.org/models/densenet161-8d451a50.pth" to /tmp/.torch/models/densenet161-8d451a50.pth
115730790it [00:06, 17660476.43it/s]

```

In [8]:

```py
#learn.lr_find()
#learn.recorder.plot()

```

In [9]:

```py
lr = 3e-02
learn.fit_one_cycle(5, slice(lr))

```

Total time: 06:22

| epoch | train_loss | valid_loss | error_rate | accuracy | time |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.064186 | 0.090407 | 0.017143 | 0.982857 | 01:25 |
| 1 | 0.039902 | 0.001073 | 0.000000 | 1.000000 | 01:15 |
| 2 | 0.027812 | 0.003891 | 0.000000 | 1.000000 | 01:13 |
| 3 | 0.012305 | 0.000548 | 0.000000 | 1.000000 | 01:14 |
| 4 | 0.002986 | 0.001019 | 0.000000 | 1.000000 | 01:14 |

In [10]:

```py
#learn.unfreeze()
#learn.lr_find()
#learn.recorder.plot()

```

In [11]:

```py
#learn.fit_one_cycle(1, slice(1e-06))

```

In [12]:

```py
#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_top_losses(9, figsize=(7,6))

```

In [13]:

```py
preds,_ = learn.get_preds(ds_type=DatasetType.Test)

```

In [14]:

```py
test_df.has_cactus = preds.numpy()[:, 0]

```

In [15]:

```py
test_df.to_csv('submission.csv', index=False)

```