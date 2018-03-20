# 2018-03-20

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


# 2018-03-17

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


# 2018-03-16


## @huangzijian

1.已完成

	1.在kaggle learn上刷完了machine learning Leve1 初步了解了决策树、随机森林算法
	2.接触了pandas、sklearn框架，并学习了几个简单函数

2.下一步计划

	1.学习二分类、数据降维知识
	2.复习git知识

3.随笔

	1.很早以前就听说加入开源组织能够学到很多，如今我的亲身经历告诉我这是真的。自从昨天加入这个项目，我接触到了很多大牛，也学到了一些前辈们的经验。
	2.终于能够静下心来阅读英文文档了。

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
