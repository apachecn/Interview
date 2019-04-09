# Fast fastai with condensenet

> Author: https://www.kaggle.com/interneuron

> From: https://www.kaggle.com/interneuron/fast-fastai-with-condensenet

> License: [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0)

> Score: 1.0000

forked from [https://www.kaggle.com/kenseitrg/simple-fastai-exercise](https://www.kaggle.com/kenseitrg/simple-fastai-exercise)

In [1]:

```
!pip install pytorchcv

```

```
Collecting pytorchcv
  Downloading https://files.pythonhosted.org/packages/20/8c/c9a820af0a5d56c4f5803a3138319ce76907c2b6db61fd9edd9dec483bb9/pytorchcv-0.0.42-py2.py3-none-any.whl (280kB)
    100% |████████████████████████████████| 286kB 10.6MB/s 
Requirement already satisfied: requests in /opt/conda/lib/python3.6/site-packages (from pytorchcv) (2.21.0)
Requirement already satisfied: numpy in /opt/conda/lib/python3.6/site-packages (from pytorchcv) (1.16.2)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests->pytorchcv) (2019.3.9)
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests->pytorchcv) (3.0.4)
Requirement already satisfied: urllib3<1.25,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests->pytorchcv) (1.22)
Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests->pytorchcv) (2.6)
Installing collected packages: pytorchcv
Successfully installed pytorchcv-0.0.42

```

In [2]:

```
!pip install fastai==1.0.47

```

```
Collecting fastai==1.0.47
  Downloading https://files.pythonhosted.org/packages/4b/92/134c4ce85851f6c9156e3363c7d396716a17dc9915b4921b490f96a5a4f2/fastai-1.0.47-py3-none-any.whl (205kB)
    100% |████████████████████████████████| 215kB 33.0MB/s 
Requirement already satisfied: torch>=1.0.0 in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (1.0.1.post2)
Requirement already satisfied: packaging in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (17.1)
Requirement already satisfied: dataclasses; python_version < "3.7" in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (0.6)
Requirement already satisfied: requests in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (2.21.0)
Requirement already satisfied: nvidia-ml-py3 in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (7.352.0)
Requirement already satisfied: scipy in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (1.1.0)
Requirement already satisfied: numpy>=1.15 in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (1.16.2)
Requirement already satisfied: pyyaml in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (3.12)
Requirement already satisfied: bottleneck in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (1.2.1)
Requirement already satisfied: fastprogress>=0.1.19 in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (0.1.20)
Requirement already satisfied: matplotlib in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (3.0.3)
Requirement already satisfied: pandas in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (0.23.4)
Requirement already satisfied: beautifulsoup4 in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (4.6.0)
Requirement already satisfied: Pillow in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (5.1.0)
Requirement already satisfied: typing in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (3.6.4)
Requirement already satisfied: spacy>=2.0.18 in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (2.0.18)
Requirement already satisfied: numexpr in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (2.6.5)
Requirement already satisfied: torchvision in /opt/conda/lib/python3.6/site-packages (from fastai==1.0.47) (0.2.2)
Requirement already satisfied: pyparsing>=2.0.2 in /opt/conda/lib/python3.6/site-packages (from packaging->fastai==1.0.47) (2.2.0)
Requirement already satisfied: six in /opt/conda/lib/python3.6/site-packages (from packaging->fastai==1.0.47) (1.12.0)
Requirement already satisfied: urllib3<1.25,>=1.21.1 in /opt/conda/lib/python3.6/site-packages (from requests->fastai==1.0.47) (1.22)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.6/site-packages (from requests->fastai==1.0.47) (2019.3.9)
Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/lib/python3.6/site-packages (from requests->fastai==1.0.47) (2.6)
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/lib/python3.6/site-packages (from requests->fastai==1.0.47) (3.0.4)
Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.6/site-packages (from matplotlib->fastai==1.0.47) (0.10.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.6/site-packages (from matplotlib->fastai==1.0.47) (1.0.1)
Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/lib/python3.6/site-packages (from matplotlib->fastai==1.0.47) (2.6.0)
Requirement already satisfied: pytz>=2011k in /opt/conda/lib/python3.6/site-packages (from pandas->fastai==1.0.47) (2018.4)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/conda/lib/python3.6/site-packages (from spacy>=2.0.18->fastai==1.0.47) (1.0.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/conda/lib/python3.6/site-packages (from spacy>=2.0.18->fastai==1.0.47) (2.0.2)
Requirement already satisfied: preshed<2.1.0,>=2.0.1 in /opt/conda/lib/python3.6/site-packages (from spacy>=2.0.18->fastai==1.0.47) (2.0.1)
Requirement already satisfied: thinc<6.13.0,>=6.12.1 in /opt/conda/lib/python3.6/site-packages (from spacy>=2.0.18->fastai==1.0.47) (6.12.1)
Requirement already satisfied: plac<1.0.0,>=0.9.6 in /opt/conda/lib/python3.6/site-packages (from spacy>=2.0.18->fastai==1.0.47) (0.9.6)
Requirement already satisfied: ujson>=1.35 in /opt/conda/lib/python3.6/site-packages (from spacy>=2.0.18->fastai==1.0.47) (1.35)
Requirement already satisfied: dill<0.3,>=0.2 in /opt/conda/lib/python3.6/site-packages (from spacy>=2.0.18->fastai==1.0.47) (0.2.9)
Requirement already satisfied: regex==2018.01.10 in /opt/conda/lib/python3.6/site-packages (from spacy>=2.0.18->fastai==1.0.47) (2018.1.10)
Requirement already satisfied: setuptools in /opt/conda/lib/python3.6/site-packages (from kiwisolver>=1.0.1->matplotlib->fastai==1.0.47) (39.1.0)
Requirement already satisfied: msgpack<0.6.0,>=0.5.6 in /opt/conda/lib/python3.6/site-packages (from thinc<6.13.0,>=6.12.1->spacy>=2.0.18->fastai==1.0.47) (0.5.6)
Requirement already satisfied: msgpack-numpy<0.4.4 in /opt/conda/lib/python3.6/site-packages (from thinc<6.13.0,>=6.12.1->spacy>=2.0.18->fastai==1.0.47) (0.4.3.2)
Requirement already satisfied: cytoolz<0.10,>=0.9.0 in /opt/conda/lib/python3.6/site-packages (from thinc<6.13.0,>=6.12.1->spacy>=2.0.18->fastai==1.0.47) (0.9.0.1)
Requirement already satisfied: wrapt<1.11.0,>=1.10.0 in /opt/conda/lib/python3.6/site-packages (from thinc<6.13.0,>=6.12.1->spacy>=2.0.18->fastai==1.0.47) (1.10.11)
Requirement already satisfied: tqdm<5.0.0,>=4.10.0 in /opt/conda/lib/python3.6/site-packages (from thinc<6.13.0,>=6.12.1->spacy>=2.0.18->fastai==1.0.47) (4.31.1)
Requirement already satisfied: toolz>=0.8.0 in /opt/conda/lib/python3.6/site-packages (from cytoolz<0.10,>=0.9.0->thinc<6.13.0,>=6.12.1->spacy>=2.0.18->fastai==1.0.47) (0.9.0)
Installing collected packages: fastai
  Found existing installation: fastai 1.0.50.post1
    Uninstalling fastai-1.0.50.post1:
      Successfully uninstalled fastai-1.0.50.post1
Successfully installed fastai-1.0.47

```

In [3]:

```
import time
start = time.time()

```

In [4]:

```
import numpy as np 
import pandas as pd 
from pytorchcv.model_provider import get_model as ptcv_get_model

```

In [5]:

```
from pathlib import Path
from fastai import *
from fastai.vision import *
import torch

```

In [6]:

```
data_folder = Path("../input")
#data_folder.ls()

```

In [7]:

```
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/sample_submission.csv")

```

In [8]:

```
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

In [9]:

```
def md(f=None):
    mdl = ptcv_get_model('condensenet74_c4_g4', pretrained=True)
    mdl.features.final_pool = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
    return mdl

```

In [10]:

```
learn = cnn_learner(train_img, md, metrics=[error_rate, accuracy])

```

```
Downloading /tmp/.torch/models/condensenet74_c4_g4-0828-5ba55049.pth.zip from https://github.com/osmr/imgclsmob/releases/download/v0.0.4/condensenet74_c4_g4-0828-5ba55049.pth.zip...

```

In [11]:

```
#learn.lr_find()
#learn.recorder.plot()

```

In [12]:

```
lr = 3.5e-02
learn.fit_one_cycle(5, slice(lr))

```

Total time: 04:54

| epoch | train_loss | valid_loss | error_rate | accuracy | time |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.083160 | 0.177766 | 0.040000 | 0.960000 | 01:06 |
| 1 | 0.054930 | 0.036326 | 0.011429 | 0.988571 | 00:58 |
| 2 | 0.040563 | 0.048152 | 0.011429 | 0.988571 | 00:58 |
| 3 | 0.009462 | 0.001094 | 0.000000 | 1.000000 | 00:56 |
| 4 | 0.004515 | 0.001212 | 0.000000 | 1.000000 | 00:54 |

In [13]:

```
preds,_ = learn.get_preds(ds_type=DatasetType.Test)

```

In [14]:

```
test_df.has_cactus = preds.numpy()[:, 0]

```

In [15]:

```
test_df.to_csv('submission.csv', index=False)

```

In [16]:

```
end = time.time() 
print(end - start)

```

```
313.3328809738159

```