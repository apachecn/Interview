**KNN:算法的通俗理解：**

一个村子有三个帮派，R帮，G帮，B帮，三个帮派势均力敌（也就是样本比较均衡）  
现在有一批逃难的人来到村子，所以需要将这些难民分配到这三个帮派当中。   
那怎么分配才好呢？  
那我们定一个衡量标准（距离）来判断，每个帮派的每个人都要对他们进行衡量（也就是计算新数据与样本数据集中每条数据的距离），衡量标准可能由以下因子(特征)组成，比如：个儿高不高啊，皮肤白不白啊，头发茂不茂盛啊等等。  
假如某个难民的身高`test_x1`，白的程度`test_x2`，头发茂不茂盛的程度`test_x3`

对于某个村民（R帮|G帮|B帮的某个人）他自己的身高`train_x1`,白的程度`train_x2`，头发茂盛程度`train_x2`。他自己的根据衡量标准得到答案就是`d = |train_x1-test_x1|+|train_x2-test_x2|+|train_x3-test_x3|`然后他感觉，咦，跟我很像来，那我就举这个牌子010（R:100,G:010,B:001假如对应关系是这个样子）

好了，每个村民都有个自己的一个距离。

**村长上场**！！！！！！！！  
哎呀，这么多人，这么多距离我可怎么比较啊，不要怕！！我们实行代表制度。那些手里距离比较大的，村长大人就默认他们投票热情不给力啊。好吧，那村长大人就挑几个热情比较高的：把距离从小到大排起来，按照顺序选取前10个做代表吧。 
 
**结果：**10个人当中有5个人的牌子是100,2个人的牌子是010，有3个人的牌子是001.
好，结果出来了！5>3>2根据民主表决，村长拍板决定：该难民属于R帮！！


**村长的困扰：**  
1：如果R帮G帮B帮的实力差别太大怎么办？假如R帮1000人，G帮100人，B帮300人。那么最后的结果不就不公平了么？如何解决。（所以此算法适合标称型数据）  
2：我去，你一个难民就出动我们整个村来投票，太劳师动众了吧，那你来个10000人还不得让我们村的村民累死啊。（计算复杂度高、空间复杂度高 ）  
3：衡量的标准如何确定？用哪个标准衡量比较好呢？（距离度量）  
4：村子这么多人选几个代表合适啊，会不会选k=10跟k=20最后结果不一样，选择不当会不会不太符合民意啊。（K值的选择）

KNN算法的优点：

1、思想简单，理论成熟，既可以用来做分类也可以用来做回归；
2、可用于非线性分类；
3、训练时间复杂度为O(n)；
4、准确度高，对数据没有假设，对outlier不敏感；

缺点：
1、计算量大；
2、样本不平衡问题（即有些类别的样本数量很多，而其它样本的数量很少）；
3、需要大量的内存；

其伪代码如下：
1. 计算已知类别数据集中的点与当前点之间的距离；
2. 按照距离递增次序排序；
3. 选择与当前距离最小的k个点；
4. 确定前k个点所在类别的出现概率
5. 返回前k个点出现频率最高的类别作为当前点的预测分类。


