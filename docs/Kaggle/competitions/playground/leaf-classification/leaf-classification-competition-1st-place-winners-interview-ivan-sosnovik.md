# 树叶分类竞赛：Ivan Sosnovik 的冠军采访

> 原文：[Leaf Classification Competition: 1st Place Winner's Interview, Ivan Sosnovik](http://blog.kaggle.com/2017/03/24/leaf-classification-competition-1st-place-winners-interview-ivan-sosnovik/)
> 
> 译者：[飞龙](https://github.com/wizardforcel)
> 
> 自豪地采用[谷歌翻译](https://translate.google.cn)

你能看到随机森林的树叶吗？[树叶分类入门竞赛](https://www.kaggle.com/c/leaf-classification)于 2016 年 8 月至 2017 年 2 月在 Kaggle 上进行。Kagglers 面临着根据图像和预先提取的特征正确识别 99 种树叶的挑战。在这位获胜者的采访中，Kaggler [Ivan Sosnovik](https://www.kaggle.com/isosnovik) 分享了他的冠军方法。他解释了在这个特征工程竞赛中，他使用逻辑回归和随机森林算法比 XGBoost 或卷积神经网络更好运。

## 简介

我是莫斯科 Skoltech 的数据分析硕士生。 大约一年前，当我参加大学的第一门 ML 课程时，我加入了 Kaggle。 第一场比赛是 [What's Cooking](https://www.kaggle.com/c/whats-cooking)。 从那以后，我参加了几场 Kaggle 比赛，但没有那么多关注它。 理解 ML 方法的工作方式更像是一种练习。

![](https://i2.wp.com/blog.kaggle.com/wp-content/uploads/2017/03/Screen-Shot-2017-03-23-at-2.31.54-PM.png?resize=1184%2C311)

树叶分类的想法非常简单和具有挑战性。 看起来像我不需要堆叠这么多模型，解决方案可能是优雅的。 此外，数据总量仅为 100 多 MB，即使使用笔记本电脑也可以进行学习。 这是非常有希望的，因为大多数计算应该在我的 MacBook Air 上使用 1.3 GHz Intel Core i5 和 4GB RAM 完成。

之前我曾经处理过黑白图像。 我家附近有一片森林。 但是，在这次比赛中，它没有给我这么多的好处。

## 让我们看看技术

当我参加比赛时，发布了几个排名前20％的内核。 解决方案使用最初提取的特征和 Logistic 回归。 它的 logloss  约为 0.03818。通过调整参数，无法实现显着的改进。为了提高质量，必须进行特征工程。似乎没有人这样做，因为顶级解决方案的得分略高于我的。

## 特征工程

我先做了第一件事，并绘制了每个类的图像。

![](https://i2.wp.com/blog.kaggle.com/wp-content/uploads/2017/03/all_leaves.png?resize=1184%2C770)

原始图像具有不同的分辨率，旋转，纵横比，宽度和高度。 但是，类中每个参数的变化小于类之间的变化。 因此，可以即时构建一些信息特征。 他们是：

+   宽和高
+   纵横比：`width / height`
+   面积：`width * height`
+   是否横向：`int(width > height)`

另一个看似有用的非常有用的特征是图像像素的平均值。

我将这些特征添加到已经提取的特征中。 Logistic 回归改善了结果。 但是，大部分工作尚未完成。

所有上述特征都不代表图像的内容。

## PCA

尽管神经网络作为特征提取器取得了成功，但我仍然喜欢 PCA。它很简单，允许人们在`IR^N`中获得图像的有用表示。 首先，将图像重新调整为`50x50`。然后应用 PCA。 将成分添加到先前提取的特征集中。

![](https://i0.wp.com/blog.kaggle.com/wp-content/uploads/2017/03/eigenvalue.png?w=872)

成分数量各不相同。 最后，我使用了`N = 35`个主成分。 这种方法表明 logloss 约为 0.01511。

## Moments 和 hull

为了生成更多特征，我使用了 OpenCV。这里是如何获取图像的 Moments 和 hull 的精彩[教程](http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html)。 我还添加了几个特征的一些成对乘法。

最后的特征如下：

+   初始特征
+   宽高，比例，以及其他
+   PCA
+   Moments

Logistic 回归表明 loglos 约为 0.00686。

## 核心思想

所有上述都证明了良好的结果。 这样的结果适合于现实生活中的应用。 但是，它可以得到加强。

### 不确定性

大多数对象都有特定的决策：只有一个类别的 p 约为 1.0的类，其余的 p 小于 0.01。 但是，我在预测中发现了几个具有不确定性的对象，如下所示：

![](https://i1.wp.com/blog.kaggle.com/wp-content/uploads/2017/03/uncertain_case.png?w=564)

混淆类别的集合很小（15 个类别分成几个小组），所以我决定查看树叶的图片，并检查我是否可以对它们进行分类。 结果如下：

![](https://i1.wp.com/blog.kaggle.com/wp-content/uploads/2017/03/68_86.png?resize=1184%2C203)

![](https://i2.wp.com/blog.kaggle.com/wp-content/uploads/2017/03/22_24_29.png?resize=1184%2C298)

我必须承认，对于不同的亚种，Quercus'（橡树）的树叶看起来几乎相同。 我想，我可以区分桉树（Eucalyptus ）和山茱萸（Cornus），但亚种的分类对我来说似乎很复杂。


## 你能看到随机森林的树叶吗

我的解决方案的关键思想是创建另一个分类器，它将仅针对混淆类进行预测。 我试过的第一个是来自`sklearn`的`RandomForestClassifier`，它在调整超参数后得到了很好的结果。 随机森林使用逻辑回归的相同数据进行训练，但仅使用来自混淆类的对象。

如果逻辑回归给出了对象的不确定预测，则使用随机森林分类器的预测。 随机森林给出了 15 个类的概率，其余假设为绝对0。

最后的流水线如下：

![](https://i1.wp.com/blog.kaggle.com/wp-content/uploads/2017/03/pipeline.png?w=985)

### 阈值

排行榜得分是在整个数据集上计算的。 这就是为什么一些有风险的方法可以用于这场比赛。

提交由多分类 logloss 评估。

![](http://s0.wp.com/latex.php?zoom=1.5625&latex=+logloss+%3D+-+%5Cfrac%7B1%7D%7BN%7D+%5Csum_%7Bi%3D1%7D%5E%7BN%7D+%5Csum_%7Bj%3D1%7D%5E%7BM%7D+y_%7Bij%7D%5Clog%28p_%7Bij%7D%29%3Cbr+%2F%3E+&bg=ffffff&fg=000&s=0)

其中 N，M 是对象和类的数量，`p_{ij}`是预测，`y_ {ij}`是指标：如果对象`i`在类`j`中，则`y_ {ij} = 1`，否则它等于 0。 如果模型正确地选择了类，以下方法将减少整体 logloss，否则它将显着增加。

在阈值处理后，我得到了 logloss = 0.0 的分数。就是这样。所有标签都是正确的。

## 接下来

我已经尝试了几种方法，它们显示适当结果但未在最终流水线中使用。 此外，我有了一些想法，对于如何使解决方案更优雅。 在本节中，我将尝试讨论它们。

### XGBoost

dmlc 的 [XGBoost](https://github.com/dmlc/xgboost) 是一个很棒的工具。 我之前在几个比赛中使用过它，并决定在最初提取的特征上训练它。 它表现出与逻辑回归相同的分数甚至更差，但时间消耗更大。

### 提交的 blending

在我想出随机森林用作第二个分类器的想法之前，我尝试了不同的单模型方法。 因此我收集了很多提交。 一个微小的想法是将提交混合：使用预测的平均值或加权平均值。 结果也不是很好。

### 神经网络

神经网络是我试图实现的第一个想法之一。 卷积神经网络是很好的特征提取器，因此，它们可以用作第一级模型，甚至可以用作主分类器。 原始图像具有不同的分辨率。 我将它们重新调整为`50x50`。在我的笔记本电脑上进行 CNN 训练太费时间，而无法在合理的时间内选择合适的架构，所以经过几个小时的训练后我拒绝了这个想法。 我相信，CNN 可以为这个数据集提供准确的预测。
