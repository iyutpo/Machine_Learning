# Decision Tree

## Additive Models and Tree-based Models

### 1.1 Generalized Additive Models

虽然线性模型非常简单，且在许多数据分析中占很重要的地位，但是实际生活中的数据，很多都不是线性的。我们将要讨论一些拥有很强的灵活性的统计模型，这些模型常被用来识别和表征非线性回归效应。这些模型就叫做“广义可加模型”\(Generalized Additive Models\)。 （未完待续P295）

### 1.2 Tree-Based Models

基于树的模型通过将feature spae分割成若干个矩形，然后用一个简单的模型\(甚至可以是一个常数\)来拟合每一个小矩形。 最常见的基于树的模型叫做“回归分类树”\(CART, classification and regression tree\)。之后我们会提到另一种常见的基于树的模型C4.5。

假设现在我们有一个回归问题，因变量$Y$，自变量为$X_1$和$X\_2$，每一个都在单位区间内取值。在左上角的小图中，我们先在$X\_1=t\_1$处进行切分。然后在所得的$X\_1≤t\_1$中，在$X\_2=t\_2$处进行切分；然后在$X\_1&gt;t\_1$区域中，在$X\_1=t\_3$进行切分。最后，在区域$X\_1&gt;t\_3$中的$X\_2=t\_4$处进行切分。最终我们能得到5个子区域$R\_1, R\_2, R\_3, R\_4, R\_5$。对于该问题，用于预测目标变量$Y$的回归模型为： $\begin{equation}\hat{f}\(X\)=\sum_{m=1}^{5}c\_mI{\(X\_1, X\_2\)∈R\_m}\end{equation} \tag{1.1}$ 上式中，$c\_m$是常数，$R\_m$是第m个区域。

同样的树模型可以表示成右下的小图。 ![](https://i.imgur.com/rCWO7Pr.png) 决策树最大的好处就在于它的可解释性。feature space的切分完全可以用一棵决策树来描述。当自变量多于两个，我们通常比较难绘制feature space是如何被切分的\(如右上小图\)。我们可以通过右下小图来看一下在高维空间，决策树是如何切分feature space的。决策树根据自变量，将样本总体进行分层为high outcome和low outcome。

#### 1.2.1 回归树\(Regression Tree\)

我们现在来看如何建立一个回归树。假设我们的数据有$p$个自变量和一个因变量，共$N$个数据点。即$\(x_i, y\_i\), for \space i=1,2,...,N,$ with $x\_i=\(x_{i1}, x_{i2},..., x_{ip}\)$。我们的算法要能够确定： 1. 挑选哪个自变量进行分割 2. 在该自变量的什么地方进行分割 3. 应该使用树的什么结构

假设现在我们已将feature space切分为$M$个区域\(Region\) $R_1, R\_2,..., R\_M$，且我们将每个区域内的因变量用常数$c\_m$表示为如下形式： \begin{equation}f\(x\) = \sum_{m=1}^{M}c_mI\(x∈R\_m\) \tag{1.2}\end{equation} 如果我们采用“最小化 真实值与预测值之差 的平方和”\(i.e. $\sum{\(y\_i-f\(x\_i\)\)^2}$\)作为评价指标，那就很容易发现，最佳的$\hat{c}\_m$的值正好是$y\_i$在区域m（$R\_m$）的平均值（**因为平均值才能让 真实值与预测值之差 的平方和最小**）： $\begin{equation}\hat{c}\_m=ave\(y\_i\|x\_i∈R\_m\) \tag{1.3}\end{equation}$ 根据 最小平方和 的方法找最佳的二分点 通常在计算上是不可行的。因此我们要使用一种贪婪算法\(Greedy Algorithm\)。我们从所有的数据开始，考虑要进行切分的自变量$j$和要对自变量$j$进行切分的点$s$，然后定义一对半平面： $\begin{equation}R\_1\(j,s\)={X\|X\_j≤s} \space and \space R\_2\(j, s\)={X\|X\_j&gt;s} \tag{1.4}\end{equation}$ **然后我们要用遍历的方法找到能 最小化真实值与预测值之差 的平方和 的 $s$和$j$的值**。从而我们有如下的目标函数： \begin{equation}\min_{j,s}\[\min_{c\_1}\sum_{x_i∈R\_1\(j,s\)}\(y\_i-c\_1\)^2 \space + \min_{c_2}\sum_{x\_i∈R\_2\(j,s\)}\(y\_i-c\_2\)^2\] \tag{1.5}\end{equation} 不管如何选择$j$和$s$的值，式子\(1.5\)中括号里面的两个最小值都可以通过式子\(1.6\)来解决： \begin{equation}\hat{c}1=ave\(yi\|x\_i∈R\_1\(j,s\)\) \space and \space \hat{c}2=ave\(yi\|x\_i∈R\_2\(j,s\)\)\tag{1.6}\end{equation} 对于每一个待切分变量，对于且分点$s$的选择可以很快地通过遍历当前变量内的所有数据点来实现。因此最佳$\(j,s\)$值也确定。此时，我们就解决了：“在当前节点应该选择哪个自变量进行切分、且在什么地方进行切分”的问题。我们不断重复上面的过程，就能最终将整个的feature space（所有自变量）进行切分了。

**我们的决策树应该有多深？因为很明显，非常深的决策树会overfit，而决策树太浅会underfit。**

树的大小是我们要调试的超参数之一，它控制着模型复杂度。最佳的树的大小是根据数据本身来选择和确定的。 1. 一种选择树的大小的方法是：只有当**当前的切分方式\(包括切分的变量和切分位置\)对 真实值与预测值之差 的平方和 所造成的的减少量 超过一定大小**，我们才进行此次切分。但是这种方法 比较短见。因为当前的一次 较差的切分可能会在下一次处 得到较好的切分。 2. 比较理想的一种方法是：我们**先建立一棵较大的决策树$T\_0$，只有当达到某个最小节点大小\(比如5\)时才停止切分过程。然后这棵较大的树$T\_0$将会根据名为 “cost-complexity pruning”的方法进行“剪枝”\(Prune\)**。

我们定义一棵子树 $T\subset{T_0}$，该子树$T$可以是通过任何剪枝的方式从$T\_0$得到。也就是说，折叠\(collapse\)任意数量的内部\(非终端\)节点。我们将终结处的节点索引标记为$m$，那么节点$m$就表示第$m$个区域$R\_m$。我们用$\|T\|$表示在子树$T$中的节点数。令下面三个式子为$\(1.7\)$： \begin{equation}\begin{aligned} N\_m=\#{x\_i∈R\_m} \ \hat{c\_m}=\frac{1}{Nm}\sum_{x_i∈R\_m}y\_i \ Q\_m\(T\)=\frac{1}{N\_m}\sum_{x\_i∈R\_m}\(y\_i-\hat{c\_m}\)^2 \ \end{aligned} \tag{1.7}\end{equation}

然后我们定义"cost complexity criterion"为： \begin{equation}C_{\alpha}\(T\)=\sum_{m=1}^{\|T\|}N_mQ\_m\(T\)+\alpha\|T\| \tag{1.8}\end{equation} （中心思想是\*\*对于每个$\alpha$，子树$T_{\alpha}\subset{T_0}$要能够使得$C_{\alpha}\(T\)$最小**） 其中$\alpha$就是我们要调试的 超参数，它描述的是** 树的大小和树对于数据的拟合程度的取舍**\(The tradeoff between tree size and its goodness of fit to the data\)。** $\alpha$是$≥0$的。$\alpha$越大，树的大小越小**。当$\alpha=0$，则当前的树$T**_**0$就没被剪枝。 对于每个$\alpha$，我们都能发现，只有唯一一个最小的树$T**_**{\alpha}$能使得 $C**_**{\alpha}$最小。现在要找到这个$T**_**{\alpha}$，我们就要用到**weakest link pruning\*\*： 1. 我们依次折叠\(collapse or fold\)在$\sum_mN\_mQ\_m\(T\)$中产生最小每个节点增量的内部节点，然后继续折叠，直到产生单节点\(根\)树。 2. 上面的步骤给出了一个有限的子树序列，且我们能发现该序列中一定含有子树$T_{\alpha}$。对于${\alpha}$值的可以通过5或10叠交叉检验来估计：我们选择$\hat{\alpha}$值来最小化 交叉检验的 真实值与预测值之差 的平方和。最终，我们就能得到训练好的树$T\_{\hat{\alpha}}$。

#### 1.2.2 分类树\(Classification Tree\)

如果我们的因变量不是连续型变量而是分类型变量，$1, 2, ..., K$，那么我们的算法唯一需要进行改动的地方就是 “在当前节点处选择哪个自变量进行切分”和“在该自变量的哪个地方进行切分” 的判定条件\(criterion\)。

我们知道，**对于Regression Tree来说，这个判定条件\(criterion\)是“平方误差 节点不纯度”\(squared-error node impurity\)，即$\(1.7\)$中的$Q\_m\(T\)$**。但是这对Classification Tree来说并不适用。在节点$m$，我们用$N_m$表示在区域$R\_m$中的数据点的个数。令： \begin{equation}\hat{p_{mk}}=\frac{1}{N_m}\sum_{x_i∈R\_m}I\(y\_i=k\) \tag{1.9}\end{equation} 为 第$k$类的观测数据在节点$m$中的占比\(The proportion of class $k$ observations in node $m$。我们将落在节点$m$内的数据 分类为 $k\(m\)=argmax_{k}\hat{p_{mk}}$，也就是节点$m$中的多数类。不同的 $Q\_m\(T\)$ （也就是不纯度 impurity）的计算有： 1. Misclassification Error: $\frac{1}{N\_m}\sum_{i∈R_m}I\(y\_i≠k\(m\)\)=1-\hat{p}_{mk\(m\)}$ 2. Gini Index: $\sum_{k≠k'}\hat{p}_{mk}\hat{p}_{mk'}=\sum_{k=1}^{K}\hat{p}_{mk}\(1-\hat{p}_{mk}\)$ 3. Cross-Entropy or Deviance: $-\sum_{k=1}^{K}\hat{p}_{mk}\log\hat{p}\_{mk}$ $\tag{1.10}$

对于二分类问题，如果$p$是第二个类的占比，那么这三个指标分别为： 1. $1-\max{p, 1-p}$ 2. $2p\(1-p\)$ 3. $-p\log{p} - \(1-p\)\log{\(1-p\)}$ 我们来看下这三个指标与$p$的关系图： ![](https://i.imgur.com/O1xBwGm.png) 根据上面提到的$\(1.5\)$和$\(1.7\)$，我们发现我们需要将节点的不纯度 用 $N_{mL}$和$N_{mR}$进行加权。$N\_mL$和$N\_mR$分别是在分割点$m$的左右子节点的数据点的个数。

**1.2.2.1 Gini Index在Classification Tree的算例**

1. 假设我们有如下一个数据集，"Decision"是因变量

   ![](https://i.imgur.com/tS20E48.png)

   统计一下Outlook变量的值所对应的Decision的情况，如下图：

   ![](https://i.imgur.com/6KJCkOA.png)

计算第$k$类观测数据在节点$m$中的占比。我们假设"Decision = Yes"为第一类，即$k=1$；"Decision = No"为第二类，即$k=2$。同时假设我们在计算"Outlook = Sunny"，即$m=1$；$"Outlook = Overcast"$，即$m=2$；$"Outlook = Rain"$，即$m=3$。 1. 根据$\(1.9\)$，计算在节点m，在区域Rm中的数据点的个数Nm。我们看到，数据集中一共出现了5次"Sunny"，所以$N_m=5$。 2. 当$m =1$: \(1\). 计算$\hat{p}_{mk}$，$k=1$时，有\begin{equation} \hat{p_{11}}=\frac{1}{N\_1}\sum_{x_i∈R\_1}I\(y\_i=1\)=\frac{1}{N\_1}\sum_{i=1}^{N_1=5} \begin{cases} 1, if \space y\_i = 1 \(即"Decision = Yes"\).\ 0, if \space y\_i ≠ 1 \(即"Decision ≠ "Yes"\). \end{cases}\ = 2/5 （表示因变量"Decision"为"Yes"且自变量之一的"Outlook"为"Sunny"的数据点的个数\(2个\)，在所有"Outlook"为"Sunny"的数据点个数\(5个\)的占比，即2/5）\end{equation} \(2\). 计算$\hat{p}_{mk}$，$k=2$有\begin{equation} \hat{p_{12}}=\frac{1}{N\_1}\sum_{x_i∈R\_1}I\(y\_i=1\)=\sum_{i=1}^{N_1=5} \begin{cases} 1, if \space y\_i = 1 \(即"Decision = No"\).\ 0, if \space y\_i ≠ 1 \(即"Decision ≠ No"\). \end{cases}\ = 3/5 （表示因变量"Decision"为"No"且自变量之一的"Outlook"为"Sunny"的数据点的个数\(3个\)，在所有"Outlook"为"Sunny"的数据点个数\(5个\)的占比，即3/5）\end{equation} 3. 计算$Gini \space Index\(m=1\)$: $Gini\(Outlook=Sunny\) = \sum_{k≠k'}\hat{p}_{mk}\hat{p}_{mk'}=\sum_{k=1}^{K}\hat{p}_{mk}\(1-\hat{p}_{mk}\)=\sum_{k=1}^{K=2}\hat{p}_{1k}\(1-\hat{p}_{1k}\)\ =\hat{p}_{11}\times\hat{p}_{12}+\hat{p}_{12}\times\hat{p}_{11}=\(2/5 \times 3/5\) + \(3/5 \times 2/5\) = 12/25=0.48$ 4. 同理计算计算$Gini \space Index\(m=2\)$: $Gini\(Outlook=Overcast\) = \sum_{k≠k'}\hat{p}_{mk}\hat{p}_{mk'}=\sum_{k=1}^{K}\hat{p}_{mk}\(1-\hat{p}_{mk}\)=\sum_{k=1}^{K=2}\hat{p}_{2k}\(1-\hat{p}_{2k}\)\ =\hat{p}_{21}\times\hat{p}_{22}+\hat{p}_{22}\times\hat{p}_{21}=\(0/4 \times 4/4 + 4/4 \times 0/4\) = 0$ 5. 同理计算计算$Gini \space Index\(m=3\)$: $Gini\(Outlook=Rain\) = \sum_{k≠k'}\hat{p}_{mk}\hat{p}_{mk'}=\sum_{k=1}^{K}\hat{p}_{mk}\(1-\hat{p}_{mk}\)=\sum_{k=1}^{K=2}\hat{p}_{3k}\(1-\hat{p}_{3k}\)\ =\hat{p}_{31}\times\hat{p}_{32}+\hat{p}_{32}\times\hat{p}_{31}=\(3/5 \times 2/5 + 2/5 \times 3/5\) = 12/25 = 0.48$

**1.2.2.2 Cross-Entropy在Decision Tree中的算例**

第1步和第2步与上面的计算流程一样 1. 根据$\(1.9\)$，计算在节点$m$，在区域$R_m$中的数据点的个数$N\_m$。我们看到，数据集中一共出现了5次"Sunny"，所以$N\_m=5$。 2. 当$m =2$: \(1\). 计算$\hat{p}_{mk}$，$k=1$时，有\begin{equation} \hat{p_{11}}=\frac{1}{N\_1}\sum_{x_i∈R\_1}I\(y\_i=1\) \ =\frac{1}{N\_1}\sum_{i=1}^{N_1=5} \begin{cases} 1, if \space y\_i = 1 \(即"Decision = Yes"\).\ 0, if \space y\_i ≠ 1 \(即"Decision ≠ "Yes"\). \end{cases}\ = 2/5 （表示因变量"Decision"为"Yes"且自变量之一的"Outlook"为"Sunny"的数据点的个数\(2个\)，在所有"Outlook"为"Sunny"的数据点个数\(5个\)的占比，即2/5）\end{equation} \(2\). 计算$\hat{p}_{mk}$，$k=2$有\begin{equation} \hat{p_{12}}=\frac{1}{N\_1}\sum_{x_i∈R\_1}I\(y\_i=1\) \ =\sum_{i=1}^{N_1=5} \begin{cases} 1, if \space y\_i = 1 \(即"Decision = No"\).\ 0, if \space y\_i ≠ 1 \(即"Decision ≠ No"\). \end{cases}\ = 3/5 （表示因变量"Decision"为"No"且自变量之一的"Outlook"为"Sunny"的数据点的个数\(3个\)，在所有"Outlook"为"Sunny"的数据点个数\(5个\)的占比，即3/5）\end{equation} 3. 计算$Cross-Entropy \(m=1\)$： $Cross-Entropy\(Outlook=Sunny\) = -\sum_{k=1}^{K=2}{\hat{p}_{1k}\log{\hat{p}_{1k}}}\ =-\[\hat{p}_{11}\log{\hat{p}_{11}} + \hat{p}_{12}\log{\hat{p}_{12}}\]\ =-\[2/5 \times \log\(2/5\) + 3/5 \times \log\(3/5\)\]\ ≈0.971$ 4. 同理计算$Cross-Entropy \(m=2\)$： $Cross-Entropy\(Outlook=Overcast\) = -\sum_{k=1}^{K=2}{\hat{p}_{2k}\log{\hat{p}_{2k}}}\ =-\[\hat{p}_{21}\log{\hat{p}_{21}} + \hat{p}_{22}\log{\hat{p}_{22}}\]\ =-\[4/4 \times \log\(0/4\) + 0/4 \times \log\(4/4\)\] \leftarrow 这里的\log\(0/4\)会有"Math Error"，这是因为该节点是"Pure Node"，\此时我们要令 \log\(0/4\)的值为 0。 \ =0$ 5. 同理计算$Cross-Entropy \(m=3\)$： $Cross-Entropy\(Outlook=Rain\) = -\sum_{k=1}^{K=2}{\hat{p}_{3k}\log{\hat{p}_{3k}}}\ =-\[\hat{p}_{31}\log{\hat{p}_{31}} + \hat{p}_{32}\log{\hat{p}_{32}}\]\ =-\[3/5 \times \log\(2/5\) + 2/5 \times \log\(3/5\)\] \ ≈0.754$

另外，Cross-Entropy和Gini Index对于节点概率的变化更加敏感。 例如，在一个二分类问题中，每个类中有400个数据\(400,400\)，假设有一次分割形成了\(300,100\)和\(100,300\)的分割方式，另一个分割方式形成了\(200,400\)和\(200,0\)的分割方式。这两种分割所产生的Misclassification Rate都是0.25，但是第二种切分方式产生了一个"Pu re Node"。 但是Gini Index和Cross-Entropy在第二个切分\(Pure Node\)上的值会很低。所以我们一般会用Gini Index或Cross-Entropy作为criterion。 对于"Cost-Complexity Pruning"，可以使用这三种方法中的任何一种，但通常是Misclassification Rate。 对于Gini Index通常有两种解释方式： 1. **我们应当将观测点按照$\hat{p}**_**{mk}$的概率分类到 第$k$类中，而不是将观测点分类到该节点的多数类中。然后，当前节点的训练误差\(training error rate\)就是：$\sum**_**{k≠k'} \hat{p}**_**{mk}\hat{p}**_**{mk'}$，也就是Gini Index**。 2. 类似的，**如果我们将每个属于第$k$类的观测数据点编码为1，属于其他类的数据点编码为0，那么当前节点内的 0-1 值的方差就是 $\hat{p}**_**{mk}\(1-\hat{p}**_**{mk}\)$。然后将所有的类$k$相加求和，就得到了Gini Index**。

**1.2.2.3 Information Gain（信息增益）**

上一个小章节提到了，信息熵\(Information Entropy 或Cross-Entropy\)是用来衡量节点的purity（纯度）的指标。计算公式见$\(1.10\)$。我们令$Ent\(D\)=-\sum^{K}_{k=1}{\hat{p}_{mk}\log{\hat{p}\_{mk}}}$，$Ent\(D\)$越小，则当前样本集$D$的纯度（purity）越高。（这个公式也决定了信息熵的一个缺点：即信息熵对可取值数目多的特征有偏好（即该属性能取得值越多，信息熵，越偏向这个），因为特征可取的值越多，会导致“纯度”越大，即$Ent\(D\)$会很小，如果一个特征的离散个数与样本数相等，那么$Ent\(D\)$值会为0）

**Information Gain（信息增益）** 假设Categorical Feature（离散型自变量）$a$有$M$个可能的取值${a^1, a^2, ..., a^M}$如果用特征$a$来对数据集$D$进行划分，则会产生$M$个切分点，第$m$个切分点包含了数据集$D$中的所有在特征$a$上的取值为$a^m$的样本总数，记为$D^m$。因此可以根据上面的Information Entropy的计算公式计算。再考虑到不同且分店所包含的样本数量不同， 给且分店赋予权重$\frac{\|D^m\|}{\|D\|}$，即，样本数量越多的切分点的影响越大。因此，能计算出自变量$a$对于样本集$D$进行划分所获得的Information Gain： \begin{equation}Gain\(D,a\)=Ent\(D\)-\sum^{M}\_{m=1}{\frac{\|D^m\|}{\|D\|}Ent\(D^m\)} \tag{1.11}\end{equation}

一般来说，Information Gain越大，则使用自变量$a$对数据集划分所获得的“Purity Improvement”越大。所以Information Gain可以用于决策树切分时的自变量选择，其实就是选择Information Gain的最大属性。我们用上一章节中的\(Cross-Entropy\)的算例为例： 我们计算了“Outlook”这个自变量中，每个unique值的Cross-Entropy： 1. 计算$Cross-Entropy \(m=1\)$： $Cross-Entropy\(Outlook=Sunny\) = Ent\(D^1\) = -\sum_{k=1}^{K=2}{\hat{p}_{1k}\log{\hat{p}_{1k}}}\ =-\[\hat{p}_{11}\log{\hat{p}_{11}} + \hat{p}_{12}\log{\hat{p}_{12}}\]\ =-\[2/5 \times \log\(2/5\) + 3/5 \times \log\(3/5\)\]\ ≈0.971$ 2. 同理计算$Cross-Entropy \(m=2\)$： $Cross-Entropy\(Outlook=Overcast\) = Ent\(D^2\) = -\sum_{k=1}^{K=2}{\hat{p}_{2k}\log{\hat{p}_{2k}}}\ =-\[\hat{p}_{21}\log{\hat{p}_{21}} + \hat{p}_{22}\log{\hat{p}_{22}}\]\ =-\[4/4 \times \log\(0/4\) + 0/4 \times \log\(4/4\)\] \leftarrow 这里的\log\(0/4\)会有"Math Error"，这是因为该节点是"Pure Node"，\此时我们要令 \log\(0/4\)的值为 0。 \ =0$ 3. 同理计算$Cross-Entropy \(m=3\)$： $Cross-Entropy\(Outlook=Rain\) = Ent\(D^3\) = -\sum_{k=1}^{K=2}{\hat{p}_{3k}\log{\hat{p}_{3k}}}\ =-\[\hat{p}_{31}\log{\hat{p}_{31}} + \hat{p}_{32}\log{\hat{p}_{32}}\]\ =-\[3/5 \times \log\(2/5\) + 2/5 \times \log\(3/5\)\] \ ≈0.754$ 4. 为了计算Information Gain， 我们还要计算 $Ent\(D\)$，此时我们就不需要考虑$m$，只考虑$k$即可。其中，$k=1 \text{时}, \space \hat{p}_{k=1}=\frac{9}{14}$（注意$k=1$表示因变量为"Yes"）。$k=2 \text{时}, \space \hat{p}_{k=1}=\frac{5}{14}$： $Cross-Entropy\(Outlook\) = Ent\(D\) = -\sum_{k=1}^{K=2}{\hat{p}_{k}} \ = -\(\frac{9}{14}\log{\frac{9}{14}}+ \frac{5}{14}\log{\frac{5}{14}}\) ≈ 0.94$ 5. 有了这几个值，我们终于可以计算Information Gain了： $Gain\(Outlook\) = Ent\(D\) - \sum_{m=1}^{M=3}{\frac{\|D^m\|}{\|D\|}Ent\(D^m\)} = 0.94 - \(\frac{5}{14} \times 0.971 + \frac{4}{14} \times 0 + \frac{5}{14} \times 0.754\) ≈ 0.324$

1. 同理我们可以计算其他自变量的Information Gain:
   1. 对于Temp变量：我们令$Temp\(m=1\)为\text{Hot}$，$Temp\(m=2\)为\text{Mild}$，$Temp\(m=3\)为\text{Cool}$。

      $Cross-Entropy\(m=1\) = Cross-Entropy\(Temp=Hot\) = Ent\(D^1\) = \sum_{k=1}^{K=2}{\hat{p}_{1k}} \

      = -\[\hat{p}_{11}\log{\hat{p}_{11}} + \hat{p}_{12}\log{\hat{p}_{12}}\] \

      =-\[\frac{2}{4} \times \log{\frac{2}{4}} + \frac{2}{4} \times \log{\frac{2}{4}}\] \

      = 1$

      $Cross-Entropy\(m=2\) = Cross-Entropy\(Temp=Mild\) = Ent\(D^2\) = \sum_{k=1}^{K=2}{\hat{p}_{2k}} \

      = -\[\hat{p}_{21}\log{\hat{p}_{21}} + \hat{p}_{22}\log{\hat{p}_{22}}\] \

      =-\[\frac{4}{6} \times \log{\frac{4}{6}} + \frac{2}{6} \times \log{\frac{2}{6}}\] \

      ≈ 0.918$

      $Cross-Entropy\(m=3\) = Cross-Entropy\(Temp=Cool\) = Ent\(D^3\) = \sum_{k=1}^{K=2}{\hat{p}_{3k}} \

      = -\[\hat{p}_{31}\log{\hat{p}_{31}} + \hat{p}_{32}\log{\hat{p}_{32}}\] \

      =-\[\frac{3}{4} \times \log{\frac{3}{4}} + \frac{1}{4} \times \log{\frac{1}{4}}\] \

      ≈ 0.811$

      整个Temp的Cross-Entropy为：$Cross-Entropy\(Temp\) =  Ent\(D\) = -\sum_{k=1}^{K=2}{\hat{p}_{k}} \

      = -\(\frac{9}{14}\log{\frac{9}{14}}+ \frac{5}{14}\log{\frac{5}{14}}\) ≈ 0.94$

      所以Information Gain为： $Gain\(Temp\) = Ent\(D\) - \sum\_{m=1}^{M=3}{\frac{\|D^m\|}{\|D\|}Ent\(D^m\)} = 0.94 - \(\frac{5}{14} \times 0.971 + \frac{4}{14} \times 0 + \frac{5}{14} \times 0.754\) ≈ 0.0291$

   2. 对于变量：Humidity我们令$Humidity\(m=1\)为\text{High}$，$Humidity\(m=2\)为\text{Normal}$。

      $Cross-Entropy\(m=1\) = Cross-Entropy\(Humidity=High\) = Ent\(D^1\) = \sum_{k=1}^{K=2}{\hat{p}_{1k}} \

      = -\[\hat{p}_{11}\log{\hat{p}_{11}} + \hat{p}_{12}\log{\hat{p}_{12}}\] \

      =-\[\frac{3}{7} \times \log{\frac{3}{7}} + \frac{4}{7} \times \log{\frac{4}{7}}\] \

      = 0.985$

      $Cross-Entropy\(m=2\) = Cross-Entropy\(Humidity=Normal\) = Ent\(D^2\) = \sum_{k=1}^{K=2}{\hat{p}_{2k}} \

      = -\[\hat{p}_{21}\log{\hat{p}_{21}} + \hat{p}_{22}\log{\hat{p}_{22}}\] \

      =-\[\frac{6}{7} \times \log{\frac{6}{7}} + \frac{1}{7} \times \log{\frac{1}{7}}\] \

      ≈ 0.592$

      整个Humidity的Cross-Entropy为：$Cross-Entropy\(Humidity\) =  Ent\(D\) = -\sum_{k=1}^{K=2}{\hat{p}_{k}} \

      = -\(\frac{9}{14}\log{\frac{9}{14}}+ \frac{5}{14}\log{\frac{5}{14}}\) ≈ 0.94$

      所以Information Gain为： $Gain\(Humidity\) = Ent\(D\) - \sum\_{m=1}^{M=2}{\frac{\|D^m\|}{\|D\|}Ent\(D^m\)} = 0.94 - \(\frac{7}{14} \times 0.985 + \frac{7}{14} \times 0.592\) ≈ 0.1515$

   3. 对于变量：Wind我们令$Wind\(m=1\)为\text{Strong}$，$Wind\(m=2\)为\text{Weak}$。

      $Cross-Entropy\(m=1\) = Cross-Entropy\(Wind=Strong\) = Ent\(D^1\) = \sum_{k=1}^{K=2}{\hat{p}_{1k}} \

      = -\[\hat{p}_{11}\log{\hat{p}_{11}} + \hat{p}_{12}\log{\hat{p}_{12}}\] \

      =-\[\frac{3}{6} \times \log{\frac{3}{6}} + \frac{3}{6} \times \log{\frac{3}{6}}\] \

      = 1$

      $Cross-Entropy\(m=2\) = Cross-Entropy\(Wind=Weak\) = Ent\(D^2\) = \sum_{k=1}^{K=2}{\hat{p}_{2k}} \

      = -\[\hat{p}_{21}\log{\hat{p}_{21}} + \hat{p}_{22}\log{\hat{p}_{22}}\] \

      =-\[\frac{6}{8} \times \log{\frac{6}{8}} + \frac{2}{8} \times \log{\frac{2}{8}}\] \

      ≈ 0.811$

      整个Wind的Cross-Entropy为：$Cross-Entropy\(Wind\) =  Ent\(D\) = -\sum_{k=1}^{K=2}{\hat{p}_{k}} \

      = -\(\frac{9}{14}\log{\frac{9}{14}}+ \frac{5}{14}\log{\frac{5}{14}}\) ≈ 0.94$

      所以Information Gain为： $Gain\(Wind\) = Ent\(D\) - \sum\_{m=1}^{M=2}{\frac{\|D^m\|}{\|D\|}Ent\(D^m\)} = 0.94 - \(\frac{6}{14} \times 1 + \frac{8}{14} \times 0.811\) = 0.048$
2. 所以根据这几个自变量的Information Gain的计算，我们发现：
   * $Gain\(outlook\) = 0.324$
   * $Gain\(temp\) = 0.0291$
   * $Gain\(humidity\) = 0.1515$
   * $Gain\(wind\) = 0.048$

     所以我们选择$Outlook$这个自变量作为第一个节点处 被切分的自变量。

#### 1.2.3 其他注意问题

**1.2.3.1 Categorical Predictors（分类型 自变量）**

当切分一个含有$q$个无序值的分类型 自变量时，一共会有$2^{q-1}$个将这$q$个值 进行二分 的可能的切分方法，并且该值会随着$q$的值变得非常大。然而对于一个 $0-1$ 输出变量，该计算的计算量就会被简化。我们将预测变量的类 根据 落入 “输出变量为1” 的占比大小 进行排序。然后 假设该预测变量是有序预测变量，并对该有序变量进行 切分。

### 2.1 下面来看代码：

**注意，请先在这里下载iris数据，将下载的文件后缀改为'.csv'，并移动到当前文件夹下以方便Python读取。（注释虽多，但有助于理解代码）**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pprint import pprint


# Part 1: Load Iris data:
# The format of the data should be Pandas.DataFrame
df = pd.read_csv('iris.csv', names=['sepal_length', 'sepal_width',
                                    'petal_length', 'petal_width', 'label'])


# Part 2: Train and Test Split
# Next, we should split Iris dataset into training and testing sets.
# We'll define our own "split_train_test" function instead of using predefined API
def split_train_test(df, test_size):
    # df --> raw Iris data;
    # test_size --> the size of testing set. It can be float or int type
    # When test_size is float type ranging from 0 to 1, it means the the proportion of testing set selected from raw Iris data
    # When test_size is int type, it means the size of test set is "test_size"
    # Return --> A "train_df" and a "test_df", both are Pandas.DataFrame
    if isinstance(test_size, float):    # if test_size is float type
        test_size = round(test_size * len(df))    # e.g., if we have 150 records in Iris data set and test_size is 0.83, then round(150 * 0.83) =　round(124.5) = 125

    indices = df.index.tolist()     # retrieve indices from Iris DataFrame
    test_indices = random.sample(population=indices, k=test_size)   # Then, use "test_size" to perform random sampling from Iris DataFrame.

    test_df = df.loc[test_indices]  # the indices selected belong to testing set
    train_df = df.drop(test_indices)    # the indices not selected belong to training set
    return train_df, test_df

# Then we can set a random seed, run "split_train_test" and get "train_df" and "test_df"
random.seed(100)
train_df, test_df = split_train_test(df, 20)    # we select 20 records as testing set


# Part 3: Helper Functions
# Part 3.1: Helper Function -- "check_purity"
# If Iris dataset only has 1 class, then this data is called "Pure". If the number of classes is more then 1, then it's "Not Pure"
# Now, the first Helper Function is to check the purity of our dataset.
def check_purity(data):
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1: return True
    else: return False
# If the dataset is pure, return True; otherwise return False


# Part 3.2: Helper Function -- "classify_data"
# classify the data
def classify_data(data):
    label_column = data[:, -1]  #
    unique_classes, count_unique_classes = np.unique(label_column, return_counts=True)
    # previous line will find the number of classes of each independent variable, and the number of unique classes of each independent variable

    index = count_unique_classes.argmax()   # find index of the class that has the most records
    classification = unique_classes[index]

    return classification


# Part 3.3: Helper Function -- "find_potential_splits"
# In this function, we should traverse all records to find the potential splitting points
# What is splitting point? -- For a continuous variable, for example, if the unique values of "petal_length" are: "1.0, 1.1, 1.2, ..."
# Then, the potential splitting points will be the average value of each two nearby values: "1.05, 1.15, ..."
# We'll create a dictionary called "potential_splits" to store potential splitting points.
def find_potential_splits(data):
    potential_splits = {}
    i, n_columns = data.shape
    for index_column in range(n_columns - 1):
        potential_splits[index_column] = []
        values = data[:, index_column]
        unique_values = np.unique(values)

        for index in range(len(unique_values) - 1): # Why -1? -- Because the number of intervals of N numbers is N-1.
            current_value = unique_values[index]
            next_value = unique_values[index+1]
            potential_split = (current_value + next_value) / 2  # "potential_split" is one of the potential splits

            potential_splits[index_column].append(potential_split)

    return potential_splits     # Finally we return the dictionary that contains all potential splitting points
# To help you understand how "find_potential_splits" works, you can decomment the following 4 lines to see how this function works.
##########################################################################################################################################################
# potential_splits = find_potential_splits(train_df.values)
# sns.lmplot(data=train_df, x='petal_width', y='petal_length', hue='label',
#            fit_reg=False, height=5, aspect=1)   # If you're using Python 3.7+, use "height". If your Python version is lower than 3.6, use "size"
# plt.vlines(x=potential_splits[3], ymin=0.5, ymax=7.5)
# plt.show()
##########################################################################################################################################################

# Previous 4 lines showed the potential splitting points of "petal_width".
# The following 4 lines will show the potential splitting points of "petal_length":
##########################################################################################################################################################
# potential_splits = find_potential_splits(train_df.values)
# sns.lmplot(data=train_df, x='petal_width', y='petal_length', hue='label',
#            fit_reg=False, height=5, aspect=1)
# plt.hlines(y=potential_splits[2], xmin=0, xmax=2.5)
# plt.show()
##########################################################################################################################################################


# Part 4: Split data -- "split_data"
# Since we've found all potential splitting points, we can define a function to split our training data
# Remember, in decision tree, if we want to make a split at a node, there are 2 things we should decide:
# First is which feature (independent variable) should be selected to split. Say, this feature is f1.
# Second is, once the feature f1 is selected, which potential split should be selected
# We will talk about how these 2 things are determined later. For "split_data" function, we want it to return two datasets
# For dataset1, all its feature f1's values should be smaller than split value
# For dataset2, all its feature f1's values should be greater than split value
def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
    data_greater = data[split_column_values > split_value]
    data_smaller = data[split_column_values <= split_value]
    return data_greater, data_smaller
# If you're careful enough, you'll find that if "petal_width" < 0.8 (approximately), all records are "setosa".
# We'll utilize this insight to check if our functions work correctly: (you can decomment the following 9 lines to help you understand)
##########################################################################################################################################################
# split_column = 3
# split_value = 0.8   # we set split value to be 0.8. You'll see all records with "petal_width" < 0.8 are "setosa".
# data_greater, data_smaller = split_data(train_df.values, split_column, split_value)
# plot_df = pd.DataFrame(data_smaller, columns=df.columns)
# sns.lmplot(data=plot_df, x='petal_width', y='petal_length', hue='label',
#            fit_reg=False, height=5, aspect=1)
# plt.vlines(x=split_value, ymin=0.5, ymax=7.5)
# plt.xlim(0, 3)
# plt.show()
##########################################################################################################################################################


# Part 5: Find the smallest cross-entropy -- "compute_entropy", "compute_cross_entropy"
# We mentioned 2 things that should be determined every time we want to make a split in decision.
# Now is time to compute the cross entropy and find the smallest entropy.
# The split that minimizes the cross entropy is the best split.
def compute_entropy(data):
    label_column = data[:, -1]
    i, count = np.unique(label_column, return_counts=True)
    probabilities = count / count.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

def compute_cross_entropy(data_greater, data_smaller):
    count_data_points = len(data_smaller) + len(data_greater)
    probability_data_greater = len(data_greater) / count_data_points
    probability_data_smaller = len(data_smaller) / count_data_points
    cross_entropy = (probability_data_greater * compute_entropy(data_greater) +
                     probability_data_smaller * compute_entropy(data_smaller))
    return cross_entropy
# Take "petal_width" < 0.8 for example. "petal_width" will be the feature to be split and the splitting point is 0.8.
##########################################################################################################################################################
# To see an example of how cross entropy is calculated, decomment the following 4 lines:
# split_column = 3    # index 3 is "petal_width" column in our dataset.
# split_value = 0.8   # we set split value to be 0.8. You'll see all records with "petal_width" < 0.8 are "setosa".
# data_greater, data_smaller = split_data(train_df.values, split_column, split_value)
# print(compute_cross_entropy(data_greater, data_smaller))
##########################################################################################################################################################

# With previous functions, we can easily find the optimal splitting points by travering the dataset.
# The "feature-splitting point" pair that minimizes cross entropy will be the optimal feature-splitting point.
def find_optimal_split(data, potential_splits):
    cross_entropy = float('inf')   # we initialize a very large number and then replace it with smaller cross entropy value
    for column_index in potential_splits:      # traverse all features
        for split_value in potential_splits[column_index]:  # traverse all splitting points
            data_greater, data_smaller = split_data(data, split_column=column_index, split_value=split_value)
            current_cross_entropy = compute_cross_entropy(data_greater= data_greater, data_smaller=data_smaller)

            if current_cross_entropy <= cross_entropy:
                cross_entropy = current_cross_entropy
                optimal_split_column = column_index
                optimal_split_value = split_value
    return optimal_split_column, optimal_split_value
# Not surprisingly, one of the optimal split feature-splitting point should be "sepal_length-0.8" (or "3-0.8" because the index of sepal_length is 3)
# You can check the result using below 2 lines:
##########################################################################################################################################################
# potential_splits = find_potential_splits(train_df.values)
# print(find_optimal_split(train_df.values, potential_splits))
##########################################################################################################################################################


# Part 6: Decision Tree Algorithm -- "build_decision_tree"
# Now is time to build our decision tree algorithm.
def build_decision_tree(df, counter=0, min_sample=2, max_depth=5):
    # df --> Pandas.DataFrame type. Our dataset
    # current_depth --> current depth of tree
    # min_sample --> the minimum number of records of the dataset
    # max_depth --> the maximum depth of decision tree

    if counter == 0:        # when we are at the first layer (root layer), we should convert Pandas.DataFrame data to a numpy array
        global COLUMN_HEADERS
        COLUMN_HEADERS = df.columns
        data = df.values
    else:                   # Otherwise, if we are at the other layers, then no need for any conversion
        data = df

    # base case
    # If data is pure OR data size < min_samples OR we reached maximum depth, then we get the base case:
    if (check_purity(data)) or (len(data) < min_sample) or (counter == max_depth):
        classification = classify_data(data)    # then we should classify our data
        return classification
    else:                                       # Otherwise we keep growing our decision tree
        counter += 1

        potential_splits = find_potential_splits(data)
        split_column, split_value = find_optimal_split(data, potential_splits)
        data_greater, data_smaller = split_data(data, split_column=split_column, split_value=split_value)

        # instantiate subtree:
        feature_name = COLUMN_HEADERS[split_column]
        decision = "{} <= {}".format(feature_name, split_value)
        subtree = {decision: []}

        # find the answer (recursive part)
        no_answer = build_decision_tree(data_greater, counter, min_sample, max_depth)
        yes_answer = build_decision_tree(data_smaller, counter, min_sample, max_depth)

        if yes_answer == no_answer:
            subtree = yes_answer
        else:
            subtree[decision].append(yes_answer)
            subtree[decision].append(no_answer)

    return subtree
# To help you understand how this recursive function works, decomment the following code block.
##########################################################################################################################################################
# atree = build_decision_tree(train_df, max_depth=2)
# pprint(atree)
##########################################################################################################################################################


# Part 7: Make Classification -- "make_classification"
# We've already had "build_decision_tree" function to grow a decision tree.
# Now we're to make classification using the decision tree we built.
# In this part, "make_classification" function will use "build_decision_tree" function to make prediction and then return the "label" it predicts.
def make_classification(record, tree):
    # record --> an example data point we will use to testify our "build_decision_tree" function
    ''' A possible value of "record":
        sepal_length               6.5
        sepal_width                  3
        petal_length               5.5
        petal_width                1.8
        label           Iris-virginica
        Name: 116, dtype: object
    '''
    # tree --> a tree that is given by "build_decision_tree" function
    decision = list(tree.keys())[0]     # an example value of "decision" might be 'petal_width <= 0.8'
    column_name, comparison, split_value = decision.split()   # Once split, we'll get 3 values: "petal_width", "<=", and "0.8", respectively.

    # So, we can use column_name as an index to retrieve the corresponding value of 'petal_width' feature, which is 3.
    # Then if this value <= split_value, then the answer is True; otherwise False
    if record[column_name] <= float(split_value):
        answer = tree[decision][0]
    else:
        answer = tree[decision][1]

    # The base case:
    if not isinstance(answer, dict):    # if answer is not a dict type, which means we reached leaf node, then we return the answer
        return answer
    else:                               # otherwise we continue the recursive part
        remained_tree = answer
        return make_classification(record, remained_tree)

# Again, you can decomment the following code block to see an example.
##########################################################################################################################################################
# record = test_df.iloc[2]
# tree = build_decision_tree(train_df, max_depth=2)
# print('Predicted Label: ', make_classification(record, tree))
# print('Actual Label: ', record['label'])
##########################################################################################################################################################
# You can see the Predicted and Actual Labels are the same, meaning our "build_decision_tree" function works well


# Part 8: Calculate Accuracy -- "calculate_accuracy"
# This part is easy. For classification problem, accuracy = number of correctly classified records / number of total records
def calculate_accuracy(df, tree):
    df['classification'] = df.apply(make_classification, axis=1, args=(tree,))
    df['correctly_classified'] = df['classification'] == df['label']

    accuracy = df['correctly_classified'].mean()
    return accuracy
# Let's use our test_df to see the accuracy of our decision tree model:
##########################################################################################################################################################
# tree = build_decision_tree(train_df, max_depth=2)
# accuracy = calculate_accuracy(test_df, tree)
# print('Accuracy is: ', accuracy)
##########################################################################################################################################################
```

![downloads](https://img.shields.io/github/downloads/atom/atom/total.svg) ![build](https://img.shields.io/appveyor/ci/:user/:repo.svg) ![chat](https://img.shields.io/discord/:serverId.svg)

### Table of Contents

\[TOC\]

### Beginners Guide

If you are a total beginner to this, start here!

1. Visit hackmd.io
2. Click "Sign in"
3. Choose a way to sign in
4. Start writing note!

### User story

```text
Feature: Guess the word

  # The first example has two steps
  Scenario: Maker starts a game
    When the Maker starts a game
    Then the Maker waits for a Breaker to join

  # The second example has three steps
  Scenario: Breaker joins a game
    Given the Maker has started a game with the word "silky"
    When the Breaker joins the Maker's game
    Then the Breaker must guess a word with 5 characters
```

> I choose a lazy person to do a hard job. Because a lazy person will find an easy way to do it. \[name=Bill Gates\]

```text
Feature: Shopping Cart
  As a Shopper
  I want to put items in my shopping cart
  Because I want to manage items before I check out

  Scenario: User adds item to cart
    Given I'm a logged-in User
    When I go to the Item page
    And I click "Add item to cart"
    Then the quantity of items in my cart should go up
    And my subtotal should increment
    And the warehouse inventory should decrement
```

> Read more about Gherkin here: [https://docs.cucumber.io/gherkin/reference/](https://docs.cucumber.io/gherkin/reference/)

### User flows

```text
Alice->Bob: Hello Bob, how are you?
Note right of Bob: Bob thinks
Bob-->Alice: I am good thanks!
Note left of Alice: Alice responds
Alice->Bob: Where have you been?
```

> Read more about sequence-diagrams here: [http://bramp.github.io/js-sequence-diagrams/](http://bramp.github.io/js-sequence-diagrams/)

### Project Timeline

```text
gantt
    title A Gantt Diagram

    section Section
    A task           :a1, 2014-01-01, 30d
    Another task     :after a1  , 20d
    section Another
    Task in sec      :2014-01-12  , 12d
    anther task      : 24d
```

> Read more about mermaid here: [http://knsv.github.io/mermaid/](http://knsv.github.io/mermaid/)

### Appendix and FAQ

:::info **Find this document incomplete?** Leave a comment! :::

**tags: Templates Documentation**

