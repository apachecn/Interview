# **Mercari 价格建议挑战赛**

`你能自动建议网上卖家的产品价格？`

## 比赛说明

可能很难知道有多少东西是真正值得的。小细节可能意味着定价的巨大差异。例如，这些毛衫中的一件成本为335美元，另一件成本为9.99美元。你能猜出哪一个是哪个？

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/mercari_comparison.png)

考虑到网上销售的产品有多少，产品定价的难度就更大了。服装具有强劲的季节性价格趋势，受品牌影响很大，而电子产品则根据产品规格波动。

日本最大的社区购物应用程序Mercari深知这个问题。他们想为卖家提供定价建议，但是这样做很困难，因为卖家可以在Mercari的市场上投入任何东西或任何东西。

在这场比赛中，Mercari挑战你建立一个算法，自动建议正确的产品价格。您将获得用户输入的产品文本说明，包括产品类别名称，品牌名称和项目条件等详细信息。

请注意，由于这些数据的公共性质，这个竞赛是一个“Kernels Only”竞赛。在挑战的第二阶段，文件只能通过内核获得，并且您将无法修改您的方法来响应新数据。在 [数据选项卡](https://www.kaggle.com/c/mercari-price-suggestion-challenge/data) 和 [内核常见问题页面](https://www.kaggle.com/c/mercari-price-suggestion-challenge#Kernels-FAQ) 阅读更多详细信息。

> 注意：[项目规范](/docs/kaggle-quickstart.md)

## 成员角色

| 角色 | 用户 | QQ | GitHub | 负责内容 | 进度 |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 发起人 | [片刻](http://cwiki.apachecn.org/display/~jiangzhonglian) | 529815144 |https://github.com/jiangzhonglian | 负责整个项目推进 | 5% |
| 负责人<br />1.目标定义 | [昵称](ApacheCN Cwiki地址) | xxx-可以选择匿名 | | 负责某个业务理解和指标确认 |  |
| 参与人<br />1.目标定义 | [昵称](ApacheCN Cwiki地址) | xxx-可以选择匿名 | | 负责某个业务理解和指标确认 |  |
| xx人<br />2.数据采集 | [昵称](ApacheCN Cwiki地址) | xxx-可以选择匿名 | | 负责 |  |
| 参与人<br />3.数据整理 | 佳乐 | 872520333| | 负责（助手） | |
| 参与人<br />3.数据整理 | 诺木人 |498744838| https://github.com/1mrliu| 负责（助手） | |
| xx人<br />4.构建模型 | [昵称](ApacheCN Cwiki地址) | xxx-可以选择匿名 | | 负责 |  |
| 参与人<br />4.构建模型 |/ | 610395649 |https://github.com/lai-bluejay | 负责（助手） |  |
| xx人<br />5.模型评估 | [昵称](ApacheCN Cwiki地址) | xxx-可以选择匿名 | | 负责 |  |
| 参与人<br />5.模型评估 | / | 610395649 |https://github.com/lai-bluejay | 负责（助手） |   |
| xx人<br />6.模型发布 | [昵称](ApacheCN Cwiki地址) | xxx-可以选择匿名 | | 负责 |  |

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/project_process.jpg)

## 1.目标定义

* 任务理解
* 指标确定

## 2.数据采集

* 建模抽样
* 质量把控

## 3.数据整理

* 数据探索
* 数据清洗
* 数据集成
* 数据变换
* 数据规约

## 4.构建模型

* 模式发现
* 构建模型
* 验证模型

## 5.模型评估

### 设定评估指标

> 1.绝对误差于相对误差

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160975410546.jpg)

> 2.平均绝对误差

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160975616578.jpg)

> 3.均方误差

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160975799651.jpg)

> 4.均方根误差

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160977834397.jpg)

> 5.平均绝对百分误差

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160979383193.jpg)

> 6.Kappa统计

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160982666965.jpg)

> 7.识别准确度（正确率）

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160988495710.jpg)

> 8.识别精确度（ P值 ）

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160989546724.jpg)

> 9.反馈率（ R值 ）

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160989672539.jpg)

> 10.非均衡问题（ F值 ）


> 11.ROC曲线/AUC面积

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160989939052.jpg)

> 12.混淆矩阵

![](/static/images/competitions/featured/mercari-price-suggestion-challenge/EvaluationCriteria/15160990207804.jpg)

* 多模型对比
* 模型优化



## 6.模型发布
    
* 模型部署
* 模型重构
