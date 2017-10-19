# [Kaggle](https://www.kaggle.com) 入门操作指南

## [注册](https://www.kaggle.com/?login=true)

1. 首先注册账号
2. 关联 GitHub 账号

![](../static/images/doc/login.jpg)

## [比赛 - competitions](https://www.kaggle.com/competitions)

* [选择 - All 和 Getting Started](https://www.kaggle.com/competitions?sortBy=deadline&group=all&page=1&pageSize=20&segment=gettingStarted)

![](../static/images/doc/All-GettingStarted.jpg)

* [选择 - Digit Recognizer（数字识别器）](https://www.kaggle.com/c/digit-recognizer)

![](../static/images/doc/DigitRecognizer.jpg)

* [阅读资料 - Digit Recognizer（数字识别器）](https://www.kaggle.com/c/digit-recognizer)

![](../static/images/doc/digit-recognizer.jpg)

**后面会持续更新**

## GitHub 使用指南

<a href=""> 此图片链接为 bilibili 视频地址: (视频图片下面为文本操作指南)
<img src="../static/images/doc/ApacheCN-GitHub入门操作-Fork到PullRequests.png">
</a>

> 一) fork apachecn/kaggle 项目

![](../static/images/doc/github-step-1-fork.jpg)

![](../static/images/doc/github-step-2-clone.jpg)


> 二) jiangzhonglian/kaggle 第一次初始化

可以使用 vscode 进行可视化操作

```
clone 自己的 repo 仓库  （这个是自己的地址， jiangzhonglian是我的，别弄错了）
$ git clone https://github.com/jiangzhonglian/kaggle.git

## 进入该仓库的文件夹
$ cd kaggle

## 查看该仓库远程 repo
$ git remote

## 添加 apachecn 的远程 repo 仓库（添加一次以后就不用使用了）
$ git remote add origin_online https://github.com/apachecn/kaggle.git
```

> 三) jiangzhonglian/kaggle 文件更新（修改文件后，第二次要进行提交）

```
# 用于 pull 保持和 apachecn 同步
$ git pull origin_online master

#上传到 自己的 repo 仓库
$ git push origin master
```

> 四) pull requests 到 apachecn/kaggle

![](../static/images/doc/github-step-3-PullRequests.jpg)

![](../static/images/doc/github-step-4-PullRequests.jpg)

![](../static/images/doc/github-step-5-PullRequests.jpg)
