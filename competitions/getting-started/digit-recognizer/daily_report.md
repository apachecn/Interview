
# 2018-03-20

## @xxxx100

   拜读很多大佬的代码，学习怎样编写掉包的python代码，但是对sk-learn的文档还是不熟悉，下一步打算看看文档，继续看代码。
   
## @hduyyg

1.  已完成

    1.  knn+pca的调参测试： score=0.975

        ~~~python
        import functions
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        data, label, test_data = functions.read_data_from_csv()
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)
        def genearte_knn_model():
            weights = 'distance'
            for n_neighbors in range(3, 7):
                for metric in ['euclidean', 'manhattan']:
                    model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                weights=weights,
                                                metric=metric)
                    print('n_neighbors={}\n weights={}\n metric={} \n'.format(n_neighbors, weights, metric))
                    yield model
        def generate_pca_model():
            for n_components in range(30, 33):
                model = PCA(n_components=n_components)
                print('n_components={}\n'.format(n_components))
                yield model

        for knn_model in genearte_knn_model():
            for pca_model in generate_pca_model():
                pca_model.fit(x_train)
                new_x_train = pca_model.transform(x_train)
                new_x_test = pca_model.transform(x_test)
                
                knn_model.fit(new_x_train, y_train)
                score = knn_model.score(new_x_test, y_test)
                print('score={}\n'.format(score))
        ~~~

        结果统计：

        1.  knn的n_neighbors的有效区间在[3, 5]；
        2.  knn的weights选取distance效果较好
        3.  pca降维，以30位分界线，30以下（不包括30），分类效果比较差，30以上，比较好，区别不是特别大。

    2.  knn+lda：score=0.80-0.90

        ~~~ python
        import functions
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        data, label, test_data = functions.read_data_from_csv()
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)


        def genearte_knn_model():
            weights = 'distance'
            for n_neighbors in range(1, 7):
                for metric in ['euclidean', 'manhattan']:
                    model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                weights=weights,
                                                metric=metric)
                    print('n_neighbors={}\n weights={}\n metric={} \n'.format(n_neighbors, weights, metric))
                    yield model
        def generate_lda_model():
            for n_components in range(5, 10):
                model = LDA(n_components=n_components)
                print('n_components={}\n'.format(n_components))
                yield model

        for knn_model in genearte_knn_model():
            for lda_model in generate_lda_model():
                lda_model.fit(x_train, y_train)
                new_x_train = lda_model.transform(x_train)
                new_x_test = lda_model.transform(x_test)
                
                knn_model.fit(new_x_train, y_train)
                score = knn_model.score(new_x_test, y_test)
                print('score={}\n'.format(score))
        ~~~

        其效果基本等同于LDA分类的效果。

    3.  LLE，跑不出来

        ~~~ python
        import functions
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.manifold import LocallyLinearEmbedding as LLE

        data, label, test_data = functions.read_data_from_csv()
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42)

        def genearte_knn_model():
            weights = 'distance'
            for n_neighbors in range(1, 7):
                for metric in ['euclidean', 'manhattan']:
                    model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                                weights=weights,
                                                metric=metric)
                    print('n_neighbors={}\n weights={}\n metric={} \n'.format(n_neighbors, weights, metric))
                    yield model
        def generate_lle_model():
            n_neighbors = 5
            for n_components in range(2, 6):
                model = LLE(n_neighbors=n_neighbors, n_components=n_components)
                print('lle:\n n_neighbors={}\n n_components={}\n'.format(n_neighbors, n_components))
                yield model

        for knn_model in genearte_knn_model():
            for lle_model in generate_lle_model():
                lle_model.fit(x_train)
                new_x_train = lle_model.transform(x_train)
                new_x_test = lle_model.transform(x_test)
                
                knn_model.fit(new_x_train, y_train)
                score = knn_model.score(new_x_test, y_test)
                print('score={}\n'.format(score))
        ~~~

        我看到一篇<a href="http://www.cnblogs.com/pinard/p/6266408.html">博客</a>讲到，LLE算法学习的流形只能是不闭合的，那可能是我用错了，LLE根本不适合数字图片。

    4.  学习了以下LLE降维，大致知道是个什么东西了。

2.  下一步计划

    1.  按照README-V2的规划，完成基于朴素贝叶斯K近邻的快速图像分类算法。

3.  随笔

    1.  看数学原理看的头痛，暂时只要会用就好了。
    
# 2018-03-19

## @huangzijian

1.  已完成

	1. 阅读了[kaggle手写数字识别README](https://github.com/huangzijian888/kaggle/blob/dev/competitions/getting-started/digit-recognizer/README.md)

	2. 学习了[README语法](https://github.com/huangzijian888/README)
2.  下一步计划

	1. 发现一个博客[Gain](https://wmpscc.github.io/)很多干货 后期可以到这里来查阅
	
	2. 再多看一点kaggle比赛项目的开发流程 学习经验
	
3.  随笔

	1. 越学习越发现自己知识的贫瘠 这也许就是努力的理由吧

## @rujinshi

1.  已完成

    1.  [用scikit-learn学习主成分分析(PCA)](http://www.cnblogs.com/pinard/p/6243025.html)。自己在用的时候出现了一些问题，还没有彻底熟悉这个方法。熟悉后，补一下这边遇到的坑。

    2.  [Sklearn-train_test_split随机划分训练集和测试集](http://blog.csdn.net/zahuopuboss/article/details/54948181)，与之匹配的还有cross_val_score的使用。

    3.  做了优化。score:0.97185，Ranking: 1247/1920

2.  下一步计划

    1.  学习他人在Kernel上分享的思路。尤其对于数据预处理。我这边根本没有涉及对缺失值处理，查看分布这些操作。

3.  随笔
 
## @xxxx100

1.  已完成

    1.  在kaggle上看了几篇kernal ,在machine learning in action 中看到相关knn代码。了解优化操作，尽量跟大佬脚步。
2.  下一步计划

3.  随笔


# 2018-03-18

## @huangzijian

1.  已完成

    1.  群内大佬推荐了一个学习数学的网站 [数学乐](http://www.shuxuele.com/) 在上面看了微积分相关内容
	
    2.  看了一下组内小伙伴推荐的 [深度学习的网站](http://zh.gluon.ai/) 教程很详细 非常适合我
	
    3.  [k-近邻算法](http://blog.csdn.net/xuelabizp/article/details/50931493) 
	
    4.  [图像分类与KNN](http://blog.csdn.net/han_xiaoyang/article/details/49949535)

2.  下一步计划
	
    1.  过一遍 [数据科学入门篇3：数据处理利器Pandas使用手册](https://zhuanlan.zhihu.com/p/25184830)
	
    2.  详细阅读[kaggle数字识别V1](https://github.com/huangzijian888/kaggle/blob/dev/competitions/getting-started/digit-recognizer/README.md)流程和代码

3.  随笔

## @lianjizhe

1.已完成

	1.今天在小伙伴的帮助下对整个github流程有了一定的了解,知道该怎么操作了
	2.上网看了一些关于github的操作博客,会创建仓库,上传代码到自己的仓库了

2.下一步计划

	1.学习git的知识,熟悉命令操作
	2.上网看关于数字识别的相关博客,好好学习别人的思路
	
3.随笔

	1.很喜欢这个组织,一上来什么都不懂,小伙伴直接远程教我操作
	2.希望接下来自己可以为这个组织贡献一份力


# 2018-03-17

## @xxxx100

1.  已完成

    1.  终于弄会了GitHub，看了SVM。

2.  下一步计划

3.  随笔

## @hduyyg

1.  已完成

    1.  《机器学习实战》第七章 Adaboost元算法提高分类性能

        [Adaboost算法原理分析和实例+代码（简明易懂）](http://blog.csdn.net/guyuealian/article/details/70995333)

        这个链接里面讲得还不错，数据演示的还行。

    2.  <a href="https://zhuanlan.zhihu.com/p/29513760">聊一聊深度学习中的数据增强与实现</a>

        这篇链接讲了一些图片操作，例如旋转、缩放之类的，对我们现在的图片处理感觉很有用啊。

    3.  <a href="https://zhuanlan.zhihu.com/p/25184830">数据科学入门篇3：数据处理利器Pandas使用手册</a>

        干货整理的太好了，以后要使用pandas分析数据，可以直接先来这里查字典了。

2.  下一步计划

    1.  把《高等代数》、《数学分析》快速的过一遍
    2.  kaggle数字识别的一些基础操作，写一写

3.  随笔

## @rujinshi

1.  已完成

    1.  初次提交1329/1902

    2.  PCA数学原理 

2.  下一步计划

    1.  阅读sklearn中PCA文档，实现Demo。实现手写数字识别原始数据的降维。

    2.  尝试使用RF模型完成一次<预计结果会好>。

3.  随笔

    1. 其它也不能停呃。数据结构继续看。 


## @huangzijian

1.	已完成	
	
	1.	完成了[github相关操作](https://github.com/huangzijian888/knowledge/blob/master/doc/git%E6%93%8D%E4%BD%9C%E6%B5%81%E7%A8%8B.md)复习
	
	2. 看了线性代数  学习了矩阵的运算 逆矩阵相关知识
	
	3. 修改了昨天日报的格式 并补充了kaggle-learn上machine learnign的超链接(强烈推荐像我一样的新手阅读)

2. 下一步计划
	
	1. 查找国外优秀微积分、线性代数、概率论教材 为后续阅读作准备
	
	2. 查阅knn相关资料

3. 随笔
	
	1. 真心觉得老外写的书、教程非常nice 写的非常通俗易懂 写的让你有兴趣阅读


# 2018-03-16


## @huangzijian

1.	已完成
	
	1. 在kaggle learn上刷完了[machine learning Leve1](https://www.kaggle.com/learn/machine-learning) 初步了解了决策树、随机森林算法
	
	2. 接触了pandas、sklearn框架，并学习了几个简单函数

2. 下一步计划
	
	1. 学习二分类、数据降维知识
	
	2. 复习git知识

3. 随笔
	
	1. 很早以前就听说加入开源组织能够学到很多，如今我的亲身经历告诉我这是真的。自从昨天加入这个项目，我接触到了很多大牛，也学到了一些前辈们的经验。
	
	2. 终于能够静下心来阅读英文文档了。

## @hduyyg

1.  已完成
    1.  《高等代数》前三章
    2.  《机器学习实战》前两章
2.  下一步计划
3.  随笔

## @rujinshi

1.  已完成
    1.  复习《统计学习方法》第三章K近邻法。重新审视[个人CSDN博客](http://blog.csdn.net/rujin_shi/article/details/78766033)以前照着《机器学习实战》实现的代码。
    2.  阅读Sklearn中 [Nearest Neighbors官方文档¶](http://scikit-learn.org/stable/modules/neighbors.html#classification)。
    3.  借鉴别人代码，初次直接调包(KNeighborsClassifier、Cross-validation)实现手写数字识别。Score为0.947。但是运行时间接近1200s？有提高空间吗？

2.  下一步计划
    1.  研究一下数据集，对特征作进一步优化or抽取。对于方法参数作进一步调整，尝试提高Score值。
    2.  学习降维知识，尤其是针对图片的？

3.  随笔
    1.  昨天晚上Git出现了问题，不能PR，提示Time Out。Stackoverflow之找出原因并解决(因为使用了代理服务器)。
    2.  断断续续自我约束能力还是不行。



# 2018-03-15

## @hduyyg

1.  已完成

    1.v2版本的规划

2.  下一步计划

3.  随笔

## @wmpscc
1.  已完成
    1. 学习数据降维
    
2.  在数字识别上应用数据降维

3.  随笔

# 之前的V1版本日报

链接：https://github.com/hduyyg/kaggle-Digit-Recognizer
