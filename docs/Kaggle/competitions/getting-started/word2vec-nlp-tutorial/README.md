# 比赛分析

步骤:

```
一. 数据分析
1. 下载并加载数据
2. 总体预览:了解每列数据的含义,数据的格式等
3. 数据初步分析,使用统计学与绘图:初步了解数据之间的相关性,为构造特征工程以及模型建立做准备

二. 特征工程
1.根据业务,常识,以及第二步的数据分析构造特征工程.
2.将特征转换为模型可以辨别的类型(如处理缺失值,处理文本进行等)

三. 模型选择
1.根据目标函数确定学习类型,是无监督学习还是监督学习,是分类问题还是回归问题等.
2.比较各个模型的分数,然后取效果较好的模型作为基础模型.

四. 模型融合

五. 修改特征和模型参数
1.可以通过添加或者修改特征,提高模型的上限.
2.通过修改模型的参数,是模型逼近上限
```

* * * 

* 比赛地址: https://www.kaggle.com/c/word2vec-nlp-tutorial
* 参考地址: https://www.cnblogs.com/zhao441354231/p/6056914.html
* 参考地址: https://blog.csdn.net/lijingpengchina/article/details/52250765


# 加载包


```python
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
```

# 读取数据


```python
root_dir = "/opt/data/kaggle/getting-started/word2vec-nlp-tutorial"
# 载入数据集
train = pd.read_csv('%s/%s' % (root_dir, 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
test = pd.read_csv('%s/%s' % (root_dir, 'testData.tsv'), header=0, delimiter="\t", quoting=3)

print(train.shape)
print(train.columns.values)
print(train.head(3))
print(test.head(3))
```

    (25000, 3)
    ['id' 'sentiment' 'review']
             id  sentiment                                             review
    0  "5814_8"          1  "With all this stuff going down at the moment ...
    1  "2381_9"          1  "\"The Classic War of the Worlds\" by Timothy ...
    2  "7759_3"          0  "The film starts with a manager (Nicholas Bell...
               id                                             review
    0  "12311_10"  "Naturally in a film who's main themes are of ...
    1    "8348_2"  "This movie is a disaster within a disaster fi...
    2    "5828_4"  "All in all, this is a movie for kids. We saw ...



```python
# 去除评论中的HTML标签
print('\n处理前: \n', train['review'][0])

example1 = BeautifulSoup(train['review'][0], "html.parser")

import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub('[^a-zA-Z]',  # 搜寻的pattern
                      ' ',           # 用来替代的pattern(空格)
                      example1.get_text())  # 待搜索的text 

print(letters_only)
lower_case = letters_only.lower()  # Convert to lower case
words = lower_case.split()  # Split into word

print('\n处理后: \n', words)
```

    
    处理前: 
     "With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."
     With all this stuff going down at the moment with MJ i ve started listening to his music  watching the odd documentary here and there  watched The Wiz and watched Moonwalker again  Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent  Moonwalker is part biography  part feature film which i remember going to see at the cinema when it was originally released  Some of it has subtle messages about MJ s feeling towards the press and also the obvious message of drugs are bad m kay Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring  Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him The actual feature film bit when it finally starts is only on for    minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord  Why he wants MJ dead so bad is beyond me  Because MJ overheard his plans  Nah  Joe Pesci s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno  maybe he just hates MJ s music Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence  Also  the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene Bottom line  this movie is for people who like MJ on one level or another  which i think is most people   If not  then stay away  It does try and give off a wholesome message and ironically MJ s bestest buddy in this movie is a girl  Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty  Well  with all the attention i ve gave this subject    hmmm well i don t know because people can be different behind closed doors  i know this for a fact  He is either an extremely nice but stupid guy or one of the most sickest liars  I hope he is not the latter  
    
    处理后: 
     ['with', 'all', 'this', 'stuff', 'going', 'down', 'at', 'the', 'moment', 'with', 'mj', 'i', 've', 'started', 'listening', 'to', 'his', 'music', 'watching', 'the', 'odd', 'documentary', 'here', 'and', 'there', 'watched', 'the', 'wiz', 'and', 'watched', 'moonwalker', 'again', 'maybe', 'i', 'just', 'want', 'to', 'get', 'a', 'certain', 'insight', 'into', 'this', 'guy', 'who', 'i', 'thought', 'was', 'really', 'cool', 'in', 'the', 'eighties', 'just', 'to', 'maybe', 'make', 'up', 'my', 'mind', 'whether', 'he', 'is', 'guilty', 'or', 'innocent', 'moonwalker', 'is', 'part', 'biography', 'part', 'feature', 'film', 'which', 'i', 'remember', 'going', 'to', 'see', 'at', 'the', 'cinema', 'when', 'it', 'was', 'originally', 'released', 'some', 'of', 'it', 'has', 'subtle', 'messages', 'about', 'mj', 's', 'feeling', 'towards', 'the', 'press', 'and', 'also', 'the', 'obvious', 'message', 'of', 'drugs', 'are', 'bad', 'm', 'kay', 'visually', 'impressive', 'but', 'of', 'course', 'this', 'is', 'all', 'about', 'michael', 'jackson', 'so', 'unless', 'you', 'remotely', 'like', 'mj', 'in', 'anyway', 'then', 'you', 'are', 'going', 'to', 'hate', 'this', 'and', 'find', 'it', 'boring', 'some', 'may', 'call', 'mj', 'an', 'egotist', 'for', 'consenting', 'to', 'the', 'making', 'of', 'this', 'movie', 'but', 'mj', 'and', 'most', 'of', 'his', 'fans', 'would', 'say', 'that', 'he', 'made', 'it', 'for', 'the', 'fans', 'which', 'if', 'true', 'is', 'really', 'nice', 'of', 'him', 'the', 'actual', 'feature', 'film', 'bit', 'when', 'it', 'finally', 'starts', 'is', 'only', 'on', 'for', 'minutes', 'or', 'so', 'excluding', 'the', 'smooth', 'criminal', 'sequence', 'and', 'joe', 'pesci', 'is', 'convincing', 'as', 'a', 'psychopathic', 'all', 'powerful', 'drug', 'lord', 'why', 'he', 'wants', 'mj', 'dead', 'so', 'bad', 'is', 'beyond', 'me', 'because', 'mj', 'overheard', 'his', 'plans', 'nah', 'joe', 'pesci', 's', 'character', 'ranted', 'that', 'he', 'wanted', 'people', 'to', 'know', 'it', 'is', 'he', 'who', 'is', 'supplying', 'drugs', 'etc', 'so', 'i', 'dunno', 'maybe', 'he', 'just', 'hates', 'mj', 's', 'music', 'lots', 'of', 'cool', 'things', 'in', 'this', 'like', 'mj', 'turning', 'into', 'a', 'car', 'and', 'a', 'robot', 'and', 'the', 'whole', 'speed', 'demon', 'sequence', 'also', 'the', 'director', 'must', 'have', 'had', 'the', 'patience', 'of', 'a', 'saint', 'when', 'it', 'came', 'to', 'filming', 'the', 'kiddy', 'bad', 'sequence', 'as', 'usually', 'directors', 'hate', 'working', 'with', 'one', 'kid', 'let', 'alone', 'a', 'whole', 'bunch', 'of', 'them', 'performing', 'a', 'complex', 'dance', 'scene', 'bottom', 'line', 'this', 'movie', 'is', 'for', 'people', 'who', 'like', 'mj', 'on', 'one', 'level', 'or', 'another', 'which', 'i', 'think', 'is', 'most', 'people', 'if', 'not', 'then', 'stay', 'away', 'it', 'does', 'try', 'and', 'give', 'off', 'a', 'wholesome', 'message', 'and', 'ironically', 'mj', 's', 'bestest', 'buddy', 'in', 'this', 'movie', 'is', 'a', 'girl', 'michael', 'jackson', 'is', 'truly', 'one', 'of', 'the', 'most', 'talented', 'people', 'ever', 'to', 'grace', 'this', 'planet', 'but', 'is', 'he', 'guilty', 'well', 'with', 'all', 'the', 'attention', 'i', 've', 'gave', 'this', 'subject', 'hmmm', 'well', 'i', 'don', 't', 'know', 'because', 'people', 'can', 'be', 'different', 'behind', 'closed', 'doors', 'i', 'know', 'this', 'for', 'a', 'fact', 'he', 'is', 'either', 'an', 'extremely', 'nice', 'but', 'stupid', 'guy', 'or', 'one', 'of', 'the', 'most', 'sickest', 'liars', 'i', 'hope', 'he', 'is', 'not', 'the', 'latter']



```python
def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613
    '''
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    # 返回words
    return words


# 预处理数据
label = train['sentiment']
train_data = []
for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
test_data = []
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

# 预览数据
print(train_data[0], '\n')
print(test_data[0])
```

    with all this stuff going down at the moment with mj i ve started listening to his music watching the odd documentary here and there watched the wiz and watched moonwalker again maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent moonwalker is part biography part feature film which i remember going to see at the cinema when it was originally released some of it has subtle messages about mj s feeling towards the press and also the obvious message of drugs are bad m kay visually impressive but of course this is all about michael jackson so unless you remotely like mj in anyway then you are going to hate this and find it boring some may call mj an egotist for consenting to the making of this movie but mj and most of his fans would say that he made it for the fans which if true is really nice of him the actual feature film bit when it finally starts is only on for minutes or so excluding the smooth criminal sequence and joe pesci is convincing as a psychopathic all powerful drug lord why he wants mj dead so bad is beyond me because mj overheard his plans nah joe pesci s character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno maybe he just hates mj s music lots of cool things in this like mj turning into a car and a robot and the whole speed demon sequence also the director must have had the patience of a saint when it came to filming the kiddy bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene bottom line this movie is for people who like mj on one level or another which i think is most people if not then stay away it does try and give off a wholesome message and ironically mj s bestest buddy in this movie is a girl michael jackson is truly one of the most talented people ever to grace this planet but is he guilty well with all the attention i ve gave this subject hmmm well i don t know because people can be different behind closed doors i know this for a fact he is either an extremely nice but stupid guy or one of the most sickest liars i hope he is not the latter 
    
    naturally in a film who s main themes are of mortality nostalgia and loss of innocence it is perhaps not surprising that it is rated more highly by older viewers than younger ones however there is a craftsmanship and completeness to the film which anyone can enjoy the pace is steady and constant the characters full and engaging the relationships and interactions natural showing that you do not need floods of tears to show emotion screams to show fear shouting to show dispute or violence to show anger naturally joyce s short story lends the film a ready made structure as perfect as a polished diamond but the small changes huston makes such as the inclusion of the poem fit in neatly it is truly a masterpiece of tact subtlety and overwhelming beauty


## 特征处理

直接丢给计算机这些词文本，计算机是无法计算的，因此我们需要把文本转换为向量，有几种常见的文本向量处理方法，比如： 

1. 单词计数 
2. TF-IDF向量 
3. Word2vec向量 

我们先使用TF-IDF来试一下。

### 2.TF-IDF向量


```python
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
# 参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613

"""
min_df: 最小支持度为2（词汇出现的最小次数）
max_features: 默认为None，可设为int，对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集
strip_accents: 将使用ascii或unicode编码在预处理步骤去除raw document中的重音符号
analyzer: 设置返回类型
token_pattern: 表示token的正则表达式，需要设置analyzer == 'word'，默认的正则表达式选择2个及以上的字母或数字作为token，标点符号默认当作token分隔符，而不会被当作token
ngram_range: 词组切分的长度范围
use_idf: 启用逆文档频率重新加权
use_idf：默认为True，权值是tf*idf，如果设为False，将不使用idf，就是只使用tf，相当于CountVectorizer了。
smooth_idf: idf平滑参数，默认为True，idf=ln((文档总数+1)/(包含该词的文档数+1))+1，如果设为False，idf=ln(文档总数/包含该词的文档数)+1
sublinear_tf: 默认为False，如果设为True，则替换tf为1 + log(tf)
stop_words: 设置停用词，设为english将使用内置的英语停用词，设为一个list可自定义停用词，设为None不使用停用词，设为None且max_df∈[0.7, 1.0)将自动根据当前的语料库建立停用词表
"""
tfidf = TFIDF(min_df=2,
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}',
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # 去掉英文停用词

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all)
# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print('TF-IDF处理结束.')

print("train: \n", np.shape(train_x[0]))
print("test: \n", np.shape(test_x[0]))

```

    TF-IDF处理结束.
    train: 
     (1, 810866)
    test: 
     (1, 810866)


### 朴素贝叶斯训练


```python
# 朴素贝叶斯训练

from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB() # (alpha=1.0, class_prior=None, fit_prior=True)
# 为了在预测的时候使用
model_NB.fit(train_x, label)

from sklearn.model_selection import cross_val_score
import numpy as np

print("多项式贝叶斯分类器10折交叉验证得分:  \n", cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc'))
print("\n多项式贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')))

```

    多项式贝叶斯分类器10折交叉验证得分:  
     [0.95134592 0.94728448 0.951648   0.94707712 0.95122816 0.94939968
     0.95240704 0.95434432 0.94438528 0.94930816]
    
    多项式贝叶斯分类器10折交叉验证得分:  0.949842816



```python
test_predicted = np.array(model_NB.predict(test_x))
print('保存结果...')

submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(10))
submission_df.to_csv('/Users/jiangzl/Desktop/submission_br.csv',columns = ['id','sentiment'], index = False)

# nb_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
# nb_output['id'] = test['id']
# nb_output = nb_output[['id', 'sentiment']]
# nb_output.to_csv('nb_output.csv', index=False)
print('结束.')

'''
1.提交最终的结果到kaggle，AUC为：0.85728，排名300左右，50%的水平
2. ngram_range = 3, 三元文法，AUC为0.85924
'''
```

    保存结果...
               id  sentiment
    0  "12311_10"          1
    1    "8348_2"          0
    2    "5828_4"          1
    3    "7186_2"          1
    4   "12128_7"          1
    5    "2913_8"          1
    6    "4396_1"          0
    7     "395_2"          0
    8   "10616_1"          0
    9    "9074_9"          1
    结束.





    '\n1.提交最终的结果到kaggle，AUC为：0.85728，排名300左右，50%的水平\n2. ngram_range = 3, 三元文法，AUC为0.85924\n'



### 逻辑回归


```python
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV

# 设定grid search的参数
grid_values = {'C': [1, 15, 30, 50]}  
# grid_values = {'C': [30]}
# 设定打分为roc_auc
"""
penalty: l1 or l2, 用于指定惩罚中使用的标准。
"""
model_LR = GridSearchCV(LR(penalty='l2', dual=True, random_state=0), grid_values, scoring='roc_auc', cv=20)
model_LR.fit(train_x, label)
# 20折交叉验证
# GridSearchCV(cv=20, 
#         estimator=LR(C=1.0, 
#             class_weight=None, 
#             dual=True, 
#             fit_intercept=True, 
#             intercept_scaling=1, 
#             penalty='l2', 
#             random_state=0, 
#             tol=0.0001),
#         fit_params={}, 
#         iid=True,
#         n_jobs=1,
#         param_grid={'C': [30]}, 
#         pre_dispatch='2*n_jobs',
#         refit=True,
#         scoring='roc_auc', 
#         verbose=0)

# 输出结果
# print(model_LR.grid_scores_, '\n', model_LR.best_params_, model_LR.best_params_)
print(model_LR.cv_results_, '\n', model_LR.best_params_, model_LR.best_score_)
```

    {'mean_fit_time': array([0.77368994, 1.95680232, 2.88316183, 3.50976259]), 'std_fit_time': array([0.05099312, 0.19345662, 0.39457327, 0.50422455]), 'mean_score_time': array([0.00273149, 0.0025926 , 0.00262785, 0.00249476]), 'std_score_time': array([0.0001698 , 0.00014623, 0.00014215, 0.00024111]), 'param_C': masked_array(data=[1, 15, 30, 50],
                 mask=[False, False, False, False],
           fill_value='?',
                dtype=object), 'params': [{'C': 1}, {'C': 15}, {'C': 30}, {'C': 50}], 'split0_test_score': array([0.95273728, 0.95990784, 0.960192  , 0.9602816 ]), 'split1_test_score': array([0.96081408, 0.96953856, 0.96975104, 0.96994816]), 'split2_test_score': array([0.9583616 , 0.96794112, 0.96825856, 0.96836352]), 'split3_test_score': array([0.95249152, 0.96079104, 0.96123136, 0.96137984]), 'split4_test_score': array([0.96460288, 0.9721088 , 0.9724672 , 0.97263104]), 'split5_test_score': array([0.95881216, 0.96733184, 0.96779008, 0.96797184]), 'split6_test_score': array([0.95679232, 0.96563968, 0.96596736, 0.96606976]), 'split7_test_score': array([0.95171072, 0.96053248, 0.96105216, 0.96125952]), 'split8_test_score': array([0.95526656, 0.9604096 , 0.96051712, 0.96053248]), 'split9_test_score': array([0.94979328, 0.95777024, 0.95817472, 0.95834368]), 'split10_test_score': array([0.95965952, 0.9672192 , 0.9675264 , 0.96764672]), 'split11_test_score': array([0.95329024, 0.96009472, 0.96019712, 0.96021504]), 'split12_test_score': array([0.96268544, 0.97140224, 0.97184256, 0.97202944]), 'split13_test_score': array([0.9571968 , 0.96615936, 0.9666048 , 0.96676864]), 'split14_test_score': array([0.95916544, 0.96551936, 0.96583168, 0.96596992]), 'split15_test_score': array([0.96279296, 0.96956928, 0.96978176, 0.96979968]), 'split16_test_score': array([0.95332096, 0.96132352, 0.96161792, 0.96173568]), 'split17_test_score': array([0.94883328, 0.9570816 , 0.95749632, 0.95771136]), 'split18_test_score': array([0.9528448 , 0.96074496, 0.96114176, 0.9612672 ]), 'split19_test_score': array([0.96429824, 0.97186048, 0.972032  , 0.97212416]), 'mean_test_score': array([0.9567735 , 0.9646473 , 0.9649737 , 0.96510246]), 'std_test_score': array([0.0046911 , 0.00476416, 0.00475249, 0.00475557]), 'rank_test_score': array([4, 3, 2, 1], dtype=int32), 'split0_train_score': array([0.99254593, 1.        , 1.        , 1.        ]), 'split1_train_score': array([0.99230078, 1.        , 1.        , 1.        ]), 'split2_train_score': array([0.9923811, 1.       , 1.       , 1.       ]), 'split3_train_score': array([0.9924227, 1.       , 1.       , 1.       ]), 'split4_train_score': array([0.9923401, 1.       , 1.       , 1.       ]), 'split5_train_score': array([0.9924475, 1.       , 1.       , 1.       ]), 'split6_train_score': array([0.99238184, 1.        , 1.        , 1.        ]), 'split7_train_score': array([0.99249388, 1.        , 1.        , 1.        ]), 'split8_train_score': array([0.99257082, 1.        , 1.        , 1.        ]), 'split9_train_score': array([0.99253744, 1.        , 1.        , 1.        ]), 'split10_train_score': array([0.99235201, 1.        , 1.        , 1.        ]), 'split11_train_score': array([0.99243953, 1.        , 1.        , 1.        ]), 'split12_train_score': array([0.99236668, 1.        , 1.        , 1.        ]), 'split13_train_score': array([0.99248181, 1.        , 1.        , 1.        ]), 'split14_train_score': array([0.99254685, 1.        , 1.        , 1.        ]), 'split15_train_score': array([0.99240575, 1.        , 1.        , 1.        ]), 'split16_train_score': array([0.99240521, 1.        , 1.        , 1.        ]), 'split17_train_score': array([0.99248037, 1.        , 1.        , 1.        ]), 'split18_train_score': array([0.99243375, 1.        , 1.        , 1.        ]), 'split19_train_score': array([0.99242053, 1.        , 1.        , 1.        ]), 'mean_train_score': array([0.99243773, 1.        , 1.        , 1.        ]), 'std_train_score': array([7.34564551e-05, 0.00000000e+00, 2.48253415e-17, 2.48253415e-17])} 
     {'C': 50} 0.965102464



```python
model_LR = LR(penalty='l2', dual=True, random_state=0)
model_LR.fit(train_x, label)

test_predicted = np.array(model_LR.predict(test_x))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(10))
submission_df.to_csv('/Users/jiangzl/Desktop/submission_br.csv',columns = ['id','sentiment'], index = False)
print('结束.')

'''
1. 提交最终的结果到kaggle，AUC为：0.88956，排名260左右，比之前贝叶斯模型有所提高
2. 三元文法，AUC为0.89076
'''
```

    保存结果...
               id  sentiment
    0  "12311_10"          1
    1    "8348_2"          0
    2    "5828_4"          1
    3    "7186_2"          1
    4   "12128_7"          1
    5    "2913_8"          1
    6    "4396_1"          0
    7     "395_2"          0
    8   "10616_1"          0
    9    "9074_9"          1
    结束.





    '\n1. 提交最终的结果到kaggle，AUC为：0.88956，排名260左右，比之前贝叶斯模型有所提高\n2. 三元文法，AUC为0.89076\n'



## 3.Word2vec向量

神经网络语言模型L = SUM[log(p(w|contect(w))]，即在w的上下文下计算当前词w的概率，由公式可以看到，我们的核心是计算p(w|contect(w)， Word2vec给出了构造这个概率的一个方法。


```python
import gensim
import nltk
from nltk.corpus import stopwords

tokenizer = nltk.data.load('/opt/data/nlp/nltk_data/tokenizers/punkt/english.pickle')
    
def review_to_wordlist(review, remove_stopwords=False):
    # review = BeautifulSoup(review, "html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review)

    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # print(words)
    return(words)


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    '''
    1. 将评论文章，按照句子段落来切分(所以会比文章的数量多很多)
    2. 返回句子列表，每个句子由一堆词组成
    '''
    review = BeautifulSoup(review, "html.parser").get_text()
    # raw_sentences 句子段落集合
    raw_sentences = tokenizer.tokenize(review)
    # print(raw_sentences)
    
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            # 获取句子中的词列表
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


sentences = []
for i, review in enumerate(train["review"]):
    # print(i, review)
    sentences += review_to_sentences(review, tokenizer, True)

```


```python
print(np.shape(train["review"]))
print(np.shape(sentences))
```

    (25000,)
    (267192,)



```python
unlabeled_train = pd.read_csv("%s/%s" % (root_dir, "unlabeledTrainData.tsv"), header=0, delimiter="\t", quoting=3 )
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print('预处理 unlabeled_train data...')
```

    预处理 unlabeled_train data...



```python
print(np.shape(train_data))
print(np.shape(sentences))
```

    (25000,)
    (1035107,)


> 构建word2vec模型


```python
import time
from gensim.models import Word2Vec
# 模型参数
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

```


```python
%%time
# 训练模型
print("训练模型中...")
model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count=min_word_count, \
            window=context, sample=downsampling)
print("训练完成")
```

    训练模型中...
    训练完成
    CPU times: user 7min 12s, sys: 6.7 s, total: 7min 19s
    Wall time: 2min 29s



```python
print('保存模型...')
model.init_sims(replace=True)
model_name = "%s/%s" % (root_dir, "300features_40minwords_10context")
model.save(model_name)
print('保存结束')
```

    保存模型...
    保存结束


> 预处理


```python
model.wv.doesnt_match("man woman child kitchen".split())
```




    'kitchen'




```python
model.wv.doesnt_match("france england germany berlin".split())
```




    'berlin'




```python
model.wv.doesnt_match("paris berlin london austria".split())
```




    'paris'




```python
# help(model.wv.most_similar)
model.wv.most_similar("man", topn=5)
```




    [('woman', 0.5622519850730896),
     ('lady', 0.5539723634719849),
     ('lad', 0.5375600457191467),
     ('men', 0.4897556006908417),
     ('monk', 0.48445409536361694)]




```python
model.wv.most_similar("queen", topn=5)
```




    [('princess', 0.6005384922027588),
     ('bride', 0.5296590328216553),
     ('queens', 0.5233569145202637),
     ('eva', 0.5130444765090942),
     ('brunette', 0.505348265171051)]




```python
model.wv.most_similar("awful", topn=5)
```




    [('terrible', 0.7551479935646057),
     ('abysmal', 0.7124387621879578),
     ('horrible', 0.7055309414863586),
     ('atrocious', 0.6951155066490173),
     ('horrendous', 0.6731454730033875)]




```python
model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
```




    [('princess', 0.4474681615829468)]



> 使用Word2vec特征


```python
def makeFeatureVec(words, model, num_features):
    '''
    对段落中的所有词向量进行取平均操作
    '''
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.

    # Index2word包含了词表中的所有词，为了检索速度，保存到set中
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # 取平均
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    '''
    给定一个文本列表，每个文本由一个词列表组成，返回每个文本的词向量平均值
    '''
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        if counter % 5000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))

        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1

    return reviewFeatureVecs
```


```python
%time trainDataVecs = getAvgFeatureVecs(train_data, model, num_features)
print(np.shape(trainDataVecs))
```

    Review 0 of 25000


    /Users/jiangzl/.virtualenvs/python3.6/lib/python3.6/site-packages/ipykernel_launcher.py:13: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      del sys.path[0]


    Review 5000 of 25000
    Review 10000 of 25000
    Review 15000 of 25000
    Review 20000 of 25000
    CPU times: user 5min 27s, sys: 4.14 s, total: 5min 31s
    Wall time: 5min 58s
    (25000, 300)



```python
%time testDataVecs = getAvgFeatureVecs(test_data, model, num_features)
print(np.shape(testDataVecs))
```

    Review 0 of 25000


    /Users/jiangzl/.virtualenvs/python3.6/lib/python3.6/site-packages/ipykernel_launcher.py:13: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      del sys.path[0]


    Review 5000 of 25000
    Review 10000 of 25000
    Review 15000 of 25000
    Review 20000 of 25000
    CPU times: user 5min 10s, sys: 3.6 s, total: 5min 14s
    Wall time: 5min 30s
    (25000, 300)


### 高斯贝叶斯+Word2vec训练


```python
from sklearn.naive_bayes import GaussianNB as GNB

model_GNB = GNB()
model_GNB.fit(trainDataVecs, label)

from sklearn.cross_validation import cross_val_score
import numpy as np

print("高斯贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_GNB, trainDataVecs, label, cv=10, scoring='roc_auc')))

print('保存结果...')
result = model_GNB.predict( testDataVecs )
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': result})
print(submission_df.head(10))
submission_df.to_csv('/Users/jiangzl/Desktop/gnb_word2vec.csv',columns = ['id','sentiment'], index = False)
print('结束.')

"""
从验证结果来看，没有超过基于TF-IDF多项式贝叶斯模型
"""
```

    高斯贝叶斯分类器10折交叉验证得分:  0.6163932159999999
    保存结果...
               id  sentiment
    0  "12311_10"          0
    1    "8348_2"          0
    2    "5828_4"          1
    3    "7186_2"          0
    4   "12128_7"          0
    5    "2913_8"          0
    6    "4396_1"          0
    7     "395_2"          1
    8   "10616_1"          1
    9    "9074_9"          1
    结束.





    '\n从验证结果来看，没有超过基于TF-IDF多项式贝叶斯模型\n'



### 随机森林+Word2vec训练


```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier( n_estimators = 100, n_jobs=2)

print("Fitting a random forest to labeled training data...")
%time forest = forest.fit( trainDataVecs, label )
print("随机森林分类器10折交叉验证得分: ", np.mean(cross_val_score(forest, trainDataVecs, label, cv=10, scoring='roc_auc')))

# 测试集
result = forest.predict( testDataVecs )

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': result})
print(submission_df.head(10))
submission_df.to_csv('/Users/jiangzl/Desktop/rf_word2vec.csv',columns = ['id','sentiment'], index = False)
print('结束.')

"""
改用随机森林之后，效果有提升，但是依然没有超过基于TF-IDF多项式贝叶斯模型
"""
```

    Fitting a random forest to labeled training data...
    CPU times: user 43.8 s, sys: 347 ms, total: 44.2 s
    Wall time: 23.1 s
    随机森林分类器10折交叉验证得分:  0.6428176640000001
    保存结果...
               id  sentiment
    0  "12311_10"          1
    1    "8348_2"          1
    2    "5828_4"          0
    3    "7186_2"          1
    4   "12128_7"          0
    5    "2913_8"          0
    6    "4396_1"          0
    7     "395_2"          0
    8   "10616_1"          1
    9    "9074_9"          1
    结束.





    '\n改用随机森林之后，效果有提升，但是依然没有超过基于TF-IDF多项式贝叶斯模型\n'




```python
# 加载训练好的词向量

from gensim.models.word2vec import Word2Vec

model = Word2Vec.load_word2vec_format("vector.txt", binary=False)  # C text format
# model = Word2Vec.load_word2vec_format("vector.bin", binary=True)  # C
```


```python
# 加载 google 的词向量，查看单词之间关系

from gensim.models.word2vec import Word2Vec 
model = Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
```


```python
# 测试预测效果

print(model.most_similar(positive=["woman", "king"], negative=["man"], topn=5))
print(model.most_similar(positive=["biggest", "small"], negative=["big"], topn=5))
print(model.most_similar(positive=["ate", "speak"], negative=["eat"], topn=5))
```


```python
import numpy as np

with open("food_words.txt", "r") as infile:
    food_words = infile.readlines()
    
with open("sports_words.txt", "r") as infile:
    food_words = infile.readlines()
    
with open("weather_words.txt", "r") as infile:
    food_words = infile.readlines()
    
def getWordVecs(words):
    vec = []
    for word in words:
        word = word.replace("\n", "")
        try:
            vecs.append(model[word].reshape((1, 300)))
        except KeyError:
            continue
    
    # numpy提供了numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接
    """
    >>> a=np.array([1,2,3])
    >>> b=np.array([11,22,33])
    >>> c=np.array([44,55,66])
    >>> np.concatenate((a,b,c),axis=0)  # 默认情况下，axis=0可以不写
    array([ 1,  2,  3, 11, 22, 33, 44, 55, 66]) #对于一维数组拼接，axis的值不影响最后的结果
    """
    vecs = np.concatenate(vecs)
    return np.array(vecs, dtype="float")

food_vecs = getWordVecs(food_words)
sports_vecs = getWordVecs(sports_words)
weather_vecs = getWordVecs(weather_words)
```


```python
# 利用 TSNE 和 matplotlib 对分类结果进行可视化处理

from sklearn.manifold import TSEN
import matplotlib.pyplot as plt

ts = TSEN(2)
reduced_vecs = ts.fit_transform(np.concatenate((food_vecs, sports_vecs, weather_vecs)))

for i in range(len(reduced_vecs)):
    if i < len(food_vecs):
        color = "b"
    elif i >= len(food_vecs) and i <(len(food_vecs)+len(sports_vecs)):
        color = "r"
    else:
        color = "g"
    
    plt.plot(reduced_vecs[i, 0], reduced_vecs[i, 1], marker="0", color=color, marksize=8)
```


```python
# 首先，我们导入数据并构建 Word2Vec 模型：

from sklearn.cross_validation import train_ _test_ _split
from gensim.models.word2vec import Word2Vec

with open('twitter.data/pos_ tweets.txt', 'r') as infile:
    pos_tweets= infile.readlines()

with open(' twitter_ data/neg_ tweets.txt', 'r') as infile:
    neg_ _tweets = infile.readlines()

# use 1for positive sentiment,0 for negative
Y= np.concatenate((np.ones( len (pos_tweets )) ，np.zeros(len(neg_tweets))))

x_train,x_test,y_train,y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2)
# Do some very minor text preprocessing

def cleanText(corpus):
    corpus= [z.lower( ).replace(' \n' , '').split() for z in corpus]
    return corpus

x_ train= cleanText(x_ train)
x_ test= cleanText (x_ _test)

n _dim= 300
#Initialize model and build vocab
imdb_w2v= Word2Vec(size=n dim, min_count=10)
imdb_w2v.build_vocab(x_ _train)
#Train the model over train_ _reviews (this may take several minutes)
imdb_w2v.train( x_train)
```


```python
# 接下来，为了利用下面的函数获得推文中所有词向量的平均值，我们必须构建作为输入文本的词向量。

def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1，size))
    count= 0.

    for word in text :
        try:
            vec += imdb_w2v[word].reshape( (1，size) )
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec 1'= count
    return vec
```


```python
# 调整数据集的量纲是数据标准化处理的一部分，我们通常将数据集转化成服从均值为零的高斯分布，这说明数值大于均值表示乐观，反之则表示悲观。为了使模型更有效，许多机器学习模型需要预先处理数据集的量纲，特别是文本分类器这类具有许多变量的模型。

from sklearn.preprocessing import scale

train_vecs = np.concatenate([buildWordVector(z ，n_dim) for z in x_train])
train_vecs= scale(train_vecs)

# Train word2vec on test tweets
imdb_w2v.train(x_test)

```


```python
# 最后我们需要建立测试集向量并对其标准化处理：

#Build test tweet vectors then scale
test_vecs = np.concatenate( [buildWordVector( Z，n _dim) for z in x _test ])
test_vecs = scale(test_vecs)

```


```python
"""
接下来我们想要通过计算测试集的预测精度和 ROC 曲线来验证分类器的有效性。 ROC 曲线衡量当模型参数调整的时候，其真阳性率和假阳性率的变化情况。在我们的案例中，我们调整的是分类器模型截断阈值的概率。一般来说，ROC 曲线下的面积（AUC）越大，该模型的表现越好。你可以在这里找到更多关于 ROC 曲线的资料

（https://en.wikipedia.org/wiki/Receiver_operating_characteristic）

在这个案例中我们使用罗吉斯回归的随机梯度下降法作为分类器算法。
"""

#Use classification algorithm (i.e.Stochastic Logistic Regression) on training set, then assess model performance on test set

from sklearn.linear model import SGDClassifier
lr = SGDClassifier(loss='log' ，penalty='11' )
lr.fit(train_vecs, y_train)
print' Test Accuracy: %.2f' % r.score(test vecs, y_test )



```


```python
# 随后我们利用 matplotlib 和 metric 库来构建 ROC 曲线

#Crea t e ROC curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

pred_probas = lr.predict_proba(test_vecs)[:, 1]

fpr, tpr, _ = roc_curve(y_test, pred_probas )
roc_auc = auc(fpr, tpr)

plt.plot(fpr,tpr,label='area = %.2f' % roc_ auc)
plt.plot([0,1]，[0，1],'k--')
plt. xlim( [0. 0 ，1. 0 ])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.show()
```
