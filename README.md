#### [《Bi-Adapt: Few-shot Bimanual Adaptation for Novel Categories of 3D Objects via Semantic Correspondence》](https://arxiv.org/abs/2602.08425) 论文精读与思考笔记
<br><br><br><br><br>

# 问题形式化

## 1、总体设定

对于一个放置在地面上的三维物体，给定其在相机的部分扫描下的非完整观测点云 ![](https://latex.codecogs.com/svg.latex?O\in\mathbb{R}^{N\times{}3})（排除不可见的背面、遮挡处），并给定双臂操作任务类型

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?T\in\{\text{Unfolding,Opening,Closing,Uncapping,Capping}\},">
</p>

神经网络需要在这两个给定的条件下，提出左、右夹爪的动作 ![](https://latex.codecogs.com/svg.latex?u_1=\left(p_1,R_1\right),u_2=\left(p_2,R_2\right)) ，其中 ![](https://latex.codecogs.com/svg.latex?p_i\in{}O) 是夹爪与物体的接触点（只能取自于物体的非完整观测点云），![](https://latex.codecogs.com/svg.latex?R_i\in{}SO(3)) 是夹爪在接触时的姿态。<br>
对于任一夹爪

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?i\in\{1,2\},">
</p>

它在 ![](https://latex.codecogs.com/svg.latex?p_i\in{}O) 处接触，并沿 ![](https://latex.codecogs.com/svg.latex?R_i\in{}SO(3)) 相对坐标系中的特定方向（例如z轴方向）拉动，就完成了动作 ![](https://latex.codecogs.com/svg.latex?u_i=\left(p_i,R_i\right)) 。在此过程中，夹爪姿态 ![](https://latex.codecogs.com/svg.latex?R_i\in{}SO(3)) 保持不变，拉动方向保持不变。<br>
总之，整个Bi-Adapt双臂操作框架可以简单概括为一个输入 ![](https://latex.codecogs.com/svg.latex?O,T) 输出 ![](https://latex.codecogs.com/svg.latex?\left(u_1,u_2\right)) 的函数，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left(u_1,u_2\right)=\text{Bi-Adapt}\left(O,T\right).">
</p>

## 2、任务的成功标准

- ![](https://latex.codecogs.com/svg.latex?\text{Unfolding,\space{}Opening,\space{}Closing}) ：目标物体旋转关节的角度变化超过该关节旋转范围的 ![](https://latex.codecogs.com/svg.latex?0.10) 倍。
- ![](https://latex.codecogs.com/svg.latex?\text{Uncapping,\space{}Capping}) ：目标物体移动关节的两部分被分离或拉近超过 ![](https://latex.codecogs.com/svg.latex?0.05) 米，即两部分的距离变化超过 ![](https://latex.codecogs.com/svg.latex?0.05) 米。
- 所有任务：目标物体的关节作为一个整体，未发生剧烈位移（整体位移小于设定阈值），并且目标物体未倾倒、未翻转。<br><br>
只有达到以上标准，相应类型的任务才算执行成功。
<br><br><br><br><br><br><br><br><br><br>

# 神经网络模块 ![](https://latex.codecogs.com/svg.latex?\mathcal{M}_2) 的预训练

用于控制左夹爪的神经网络模块 ![](https://latex.codecogs.com/svg.latex?\mathcal{M}_1) 由动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1) 和动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_1) 组成，用于控制右夹爪的神经网络模块 ![](https://latex.codecogs.com/svg.latex?\mathcal{M}_2) 由动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_2) 和动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2) 组成。<br>
任务执行成功的样本是正样本，任务执行失败的样本是负样本。在预训练数据集的任意一个样本中，三维物体观测点云 ![](https://latex.codecogs.com/svg.latex?O\in\mathbb{R}^{N\times{}3}) 以及左、右夹爪动作 ![](https://latex.codecogs.com/svg.latex?u_1=\left(p_1,R_1\right),u_2=\left(p_2,R_2\right)) 都是已知量，尤其是正样本中的 ![](https://latex.codecogs.com/svg.latex?R_1,R_2\in{}SO(3)) ，它们将作为模型的似然估计对象。<br>
动作提议网络只在正样本中训练，正样本中的已知量 ![](https://latex.codecogs.com/svg.latex?R_1,R_2\in{}SO(3)) 将分别作为动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1,\mathcal{A}_2) 的预训练标签，或者说动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1,\mathcal{A}_2) 实现极大似然估计的对象。动作评分网络既在正样本中训练，也在负样本中训练，并且正、负样本数量配比设定为 ![](https://latex.codecogs.com/svg.latex?1:1) 。<br>

## 1、动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_2) 的预训练

用 PointNet++ (segmentation version) 神经网络 ![](https://latex.codecogs.com/svg.latex?\text{PN}(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?O\in\mathbb{R}^{N\times{}3}) 进行逐点特征提取，得到点级特征矩阵 ![](https://latex.codecogs.com/svg.latex?F\in\mathbb{R}^{N\times{}128}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?F=\text{PN}(O)\in\mathbb{R}^{N\times{}128}.">
</p>

用三个多层感知机 ![](https://latex.codecogs.com/svg.latex?\text{MLP}_1(\cdot),\text{MLP}_2(\cdot),\text{MLP}_3(\cdot)) 分别对 ![](https://latex.codecogs.com/svg.latex?p_1\in{}O,R_1\in{}SO(3),p_2\in{}O) 进行编码，分别得到语义向量 ![](https://latex.codecogs.com/svg.latex?f_{p_1},f_{R_1},f_{p_2}\in\mathbb{R}^{32}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}f_{p_1}&=\text{MLP}_1\left(p_1\right)\in\mathbb{R}^{32},\\{}f_{R_1}&=\text{MLP}_2\left(R_1\right)\in\mathbb{R}^{32},\\{}f_{p_2}&=\text{MLP}_3\left(p_2\right)\in\mathbb{R}^{32}.\end{align*}">
</p>

动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_2) 是以cVAE的方式构建的。具体而言，用编码器 ![](https://latex.codecogs.com/svg.latex?\text{Enc}_2\left(\cdot,\cdot,\cdot,\cdot,\cdot\right)) 对 ![](https://latex.codecogs.com/svg.latex?R_2\in{}SO(3),F\in\mathbb{R}^{N\times{}128},f_{p_1}\in\mathbb{R}^{32},f_{R_1}\in\mathbb{R}^{32},f_{p_2}\in\mathbb{R}^{32}) 进行编码，得到均值向量 ![](https://latex.codecogs.com/svg.latex?\mu_2\in\mathbb{R}^d) 和协方差矩阵 ![](https://latex.codecogs.com/svg.latex?\Sigma_2\in\mathbb{R}^{d\times{}d}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left(\mu_2,\Sigma_2\right)=\text{Enc}_2\left(R_2,F,f_{p_1},f_{R_1},f_{p_2}\right),">
</p>

它们组成潜变量分布 ![](https://latex.codecogs.com/svg.latex?\mathcal{N}\left(\mu_2,\Sigma_2\right)) 。从 ![](https://latex.codecogs.com/svg.latex?\mathcal{N}\left(\mu_2,\Sigma_2\right)) 中随机采样，得到潜变量 ![](https://latex.codecogs.com/svg.latex?z\in\mathbb{R}^d) 。<br>
用解码器 ![](https://latex.codecogs.com/svg.latex?\text{Dec}_2(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?z\in\mathbb{R}^d) 进行解码，得到预测夹爪姿态 ![](https://latex.codecogs.com/svg.latex?\widehat{R}_2\in{}SO(3)) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{R}_2=\text{Dec}_2(z)\in{}SO(3).">
</p>

动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_2) 的预训练损失函数即为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{A}_2}=\mathcal{L}_{geo}\left(\widehat{R}_2,R_2\right)+D_{KL}\left(\mathcal{N}\left(\mu_2,\Sigma_2\right)\,\|\;\mathcal{N}\left(0,I\right)\right),">
</p>

其中测地距离 ![](https://latex.codecogs.com/svg.latex?\mathcal{L}_{geo}\left(\cdot,\cdot\right)) 为自编码器重建损失，![](https://latex.codecogs.com/svg.latex?D_{KL}(\cdot)) 为潜空间正则化项。<br>
如果要强调cVAE潜变量 ![](https://latex.codecogs.com/svg.latex?z\in\mathbb{R}^d) 在预训练中“条件性生成”的本质，则潜空间正则化项可另记为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?D_{KL}\left(q\left(z\;|\;R_2,F,f_{p_1},f_{R_1},f_{p_2}\right)\,\|\;\mathcal{N}\left(0,I\right)\right).">
</p>

## 2、动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2) 的预训练

用多层感知机 ![](https://latex.codecogs.com/svg.latex?\text{MLP}_4(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?R_2\in{}SO(3)) 进行编码，得到语义向量 ![](https://latex.codecogs.com/svg.latex?f_{R_2}\in\mathbb{R}^{32}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f_{R_2}=\text{MLP}_4\left(R_2\right)\in\mathbb{R}^{32}.">
</p>

动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2\left(\cdot,\cdot,\cdot,\cdot\right)) 接受 ![](https://latex.codecogs.com/svg.latex?f_{p_1},f_{R_1},f_{p_2},f_{R_2}\in\mathbb{R}^{32}) 作为输入，输出一个位于 ![](https://latex.codecogs.com/svg.latex?\left(0,1\right)) 区间内的正实数，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\in\left(0,1\right),">
</p>

它表征了在左夹爪动作 ![](https://latex.codecogs.com/svg.latex?u_1=\left(p_1,R_1\right)) 先前被执行的条件下，右夹爪动作 ![](https://latex.codecogs.com/svg.latex?u_2=\left(p_2,R_2\right)) 被序贯执行后，任务的最终成功概率，是模型针对该预训练样本的正负性（二元离散量）所估计的似然（连续量）。<br>
在正样本中，动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2) 的预训练损失函数为 ![](https://latex.codecogs.com/svg.latex?\left(0,1\right)) 区间内单调递减的负对数似然，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{C}_2}^\text{positive}=-\log\left(\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\right)\in\left(0,+\infty\right),">
</p>

而在负样本中，动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2) 的预训练损失函数为失败概率 ![](https://latex.codecogs.com/svg.latex?1-\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)) 的负对数，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{C}_2}^\text{negative}=-\log\left(1-\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\right)\in\left(0,+\infty\right),">
</p>

它在 ![](https://latex.codecogs.com/svg.latex?\left(0,1\right)) 区间内单调递增。<br>
如果定义

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?r_i\in\{1,0\},">
</p>

其中 ![](https://latex.codecogs.com/svg.latex?r_i=1) 表示第 ![](https://latex.codecogs.com/svg.latex?i) 个样本为正样本，![](https://latex.codecogs.com/svg.latex?r_i=0) 表示第 ![](https://latex.codecogs.com/svg.latex?i) 个样本为负样本，那么动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2) 在第 ![](https://latex.codecogs.com/svg.latex?i) 个样本中的预训练损失函数可统一表示为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{C}_2}=-r_i\log\left(\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\right)-\left(1-r_i\right)\log\left(1-\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\right).">
</p>

## 3、动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2) 的条件概率模型本质

实际上，动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2) 是一个条件概率模型，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?P\left(r_i=1\;|\;f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)=\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)=\widehat{y}_i\in\left(0,1\right),">
</p>

它在整个预训练数据集中的伯努利似然即为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\prod_{i=1}^{N_\text{samples}}\widehat{y}_i^{r_i}\left(1-\widehat{y}_i\right)^{1-r_i},">
</p>

在第 ![](https://latex.codecogs.com/svg.latex?i) 个样本中的似然即为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{y}_i^{r_i}\left(1-\widehat{y}_i\right)^{1-r_i}=\left(\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\right)^{r_i}\cdot\left(1-\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\right)^{1-r_i},">
</p>

为了实现极大似然估计，就需要使其负对数

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?-\log\left(\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\right)^{r_i}-\log\left(1-\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\right)^{1-r_i}=-r_i\log\left(\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\right)-\left(1-r_i\right)\log\left(1-\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\right)">
</p>

作为动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2) 的预训练损失函数。
<br><br><br><br><br><br><br><br><br><br>

# 神经网络模块 ![](https://latex.codecogs.com/svg.latex?\mathcal{M}_1) 的预训练

在众多且各异的正样本中进行预训练后，动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_2) 拥有了在众多且各异的左夹爪动作 ![](https://latex.codecogs.com/svg.latex?u_1=\left(p_1,R_1\right)) 条件下，生成合适的右夹爪动作 ![](https://latex.codecogs.com/svg.latex?u_2=\left(p_2,R_2\right)) 的能力，也就是说，拥有了生成与各种左夹爪动作 ![](https://latex.codecogs.com/svg.latex?u_1=\left(p_1,R_1\right)) 均能展开有效协作的右夹爪动作 ![](https://latex.codecogs.com/svg.latex?u_2=\left(p_2,R_2\right)) 的能力。<br>
由此，双臂操作任务最终成败的责任便压在了动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1) 身上。在经过专门的预训练后，它需要提出便于动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_2) 配合的左夹爪动作 ![](https://latex.codecogs.com/svg.latex?u_1=\left(p_1,R_1\right)) ，从而有利于最终成功完成双臂操作任务。

## 1、动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1) 的预训练

用多层感知机 ![](https://latex.codecogs.com/svg.latex?\text{MLP}(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?p_1\in{}O) 进行编码，得到语义向量 ![](https://latex.codecogs.com/svg.latex?f_{p_1}^\prime\in\mathbb{R}^{32}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f_{p_1}^\prime=\text{MLP}\left(p_1\right)\in\mathbb{R}^{32}.">
</p>

动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1) 也是以cVAE的方式构建的。具体而言，用编码器 ![](https://latex.codecogs.com/svg.latex?\text{Enc}_1\left(\cdot,\cdot,\cdot\right)) 对 ![](https://latex.codecogs.com/svg.latex?R_1\in{}SO(3),F\in\mathbb{R}^{N\times{}128},f_{p_1}^\prime\in\mathbb{R}^{32}) 进行编码，得到均值向量 ![](https://latex.codecogs.com/svg.latex?\mu_1\in\mathbb{R}^d) 和协方差矩阵 ![](https://latex.codecogs.com/svg.latex?\Sigma_1\in\mathbb{R}^{d\times{}d}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left(\mu_1,\Sigma_1\right)=\text{Enc}_1\left(R_1,F,f_{p_1}^\prime\right),">
</p>

它们组成潜变量分布 ![](https://latex.codecogs.com/svg.latex?\mathcal{N}\left(\mu_1,\Sigma_1\right)) 。从 ![](https://latex.codecogs.com/svg.latex?\mathcal{N}\left(\mu_1,\Sigma_1\right)) 中随机采样，得到潜变量 ![](https://latex.codecogs.com/svg.latex?z^\prime\in\mathbb{R}^d) 。<br>
用解码器 ![](https://latex.codecogs.com/svg.latex?\text{Dec}_1(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?z^\prime\in\mathbb{R}^d) 进行解码，得到预测夹爪姿态 ![](https://latex.codecogs.com/svg.latex?\widehat{R}_1\in{}SO(3)) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{R}_1=\text{Dec}_1(z^\prime)\in{}SO(3).">
</p>

动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1) 的预训练损失函数即为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{A}_1}=\mathcal{L}_{geo}\left(\widehat{R}_1,R_1\right)+D_{KL}\left(\mathcal{N}\left(\mu_1,\Sigma_1\right)\,\|\;\mathcal{N}\left(0,I\right)\right),">
</p>

其中测地距离 ![](https://latex.codecogs.com/svg.latex?\mathcal{L}_{geo}\left(\cdot,\cdot\right)) 为自编码器重建损失，![](https://latex.codecogs.com/svg.latex?D_{KL}(\cdot)) 为潜空间正则化项。<br>
如果要强调cVAE潜变量 ![](https://latex.codecogs.com/svg.latex?z^\prime\in\mathbb{R}^d) 在预训练中“条件性生成”的本质，则潜空间正则化项可另记为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?D_{KL}\left(q\left(z^\prime\;|\;R_1,F,f_{p_1}^\prime\right)\,\|\;\mathcal{N}\left(0,I\right)\right).">
</p>

## 2、动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_1) 的预训练

用多层感知机 ![](https://latex.codecogs.com/svg.latex?\text{MLP}^\prime(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?R_1\in{}SO(3)) 进行编码，得到语义向量 ![](https://latex.codecogs.com/svg.latex?f_{R_1}^\prime\in\mathbb{R}^{32}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?f_{R_1}^\prime=\text{MLP}^\prime\left(R_1\right)\in\mathbb{R}^{32}.">
</p>

动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_1\left(\cdot,\cdot\right)) 接受 ![](https://latex.codecogs.com/svg.latex?f_{p_1}^\prime,f_{R_1}^\prime\in\mathbb{R}^{32}) 作为输入，输出一个位于 ![](https://latex.codecogs.com/svg.latex?\left(0,1\right)) 区间内的正实数，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{C}_1\left(f_{p_1}^\prime,f_{R_1}^\prime\right)\in\left(0,1\right),">
</p>

它表征了在左夹爪动作 ![](https://latex.codecogs.com/svg.latex?u_1=\left(p_1,R_1\right)) 被执行后，任务的最终成功概率（模型所估计的似然）。<br>
在正样本中，动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_1) 的预训练损失函数为 ![](https://latex.codecogs.com/svg.latex?\left(0,1\right)) 区间内单调递减的负对数似然，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{C}_1}^\text{positive}=-\log\left(\mathcal{C}_1\left(f_{p_1}^\prime,f_{R_1}^\prime\right)\right)\in\left(0,+\infty\right),">
</p>

而在负样本中，动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_1) 的预训练损失函数为失败概率 ![](https://latex.codecogs.com/svg.latex?1-\mathcal{C}_1\left(f_{p_1}^\prime,f_{R_1}^\prime\right)) 的负对数，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{C}_1}^\text{negative}=-\log\left(1-\mathcal{C}_1\left(f_{p_1}^\prime,f_{R_1}^\prime\right)\right)\in\left(0,+\infty\right),">
</p>

它在 ![](https://latex.codecogs.com/svg.latex?\left(0,1\right)) 区间内单调递增。<br>
动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_1) 在第 ![](https://latex.codecogs.com/svg.latex?i) 个样本中的预训练损失函数可统一表示为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{C}_1}=-r_i\log\left(\mathcal{C}_1\left(f_{p_1}^\prime,f_{R_1}^\prime\right)\right)-\left(1-r_i\right)\log\left(1-\mathcal{C}_1\left(f_{p_1}^\prime,f_{R_1}^\prime\right)\right).">
</p>

<br><br><br><br><br><br><br>

# 可供性迁移

针对一个从未在预训练正样本数据集中见过的目标物体，要求双臂机器人执行任务

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?T\in\{\text{Unfolding,Opening,Closing,Uncapping,Capping}\}.">
</p>

首先，在预训练正样本数据集中检索同一任务 ![](https://latex.codecogs.com/svg.latex?T) 下的多个源物体。这些源物体并不必然与目标物体属于同一类别。<br>
对于每一个源物体，其在被接触操作前的深度图 ![](https://latex.codecogs.com/svg.latex?D_s) 、RGB图像 ![](https://latex.codecogs.com/svg.latex?I_s\in\mathbb{R}^{H_s\times{}W_s\times{}3}) 以及左右夹爪待接触点 ![](https://latex.codecogs.com/svg.latex?p_{s_1}^{2D},p_{s_2}^{2D}) 已经被记录过。目标物体的深度图 ![](https://latex.codecogs.com/svg.latex?D_t) 、RGB图像 ![](https://latex.codecogs.com/svg.latex?I_t\in\mathbb{R}^{H_t\times{}W_t\times{}3}) 现在也是已知的。<br>
用DIFT基础模型分别对 ![](https://latex.codecogs.com/svg.latex?I_s\in\mathbb{R}^{H_s\times{}W_s\times{}3},I_t\in\mathbb{R}^{H_t\times{}W_t\times{}3}) 进行特征提取，得到像素级特征矩阵 ![](https://latex.codecogs.com/svg.latex?F_s\in\mathbb{R}^{H_s\times{}W_s\times{}d_{DIFT}},F_t\in\mathbb{R}^{H_t\times{}W_t\times{}d_{DIFT}}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}F_s&=\text{DIFT}\left(I_s\right)\in\mathbb{R}^{H_s\times{}W_s\times{}d_{DIFT}},\\F_t&=\text{DIFT}\left(I_t\right)\in\mathbb{R}^{H_t\times{}W_t\times{}d_{DIFT}}.\end{align*}">
</p>

从 ![](https://latex.codecogs.com/svg.latex?F_s\in\mathbb{R}^{H_s\times{}W_s\times{}d_{DIFT}}) 中抽取出左、右夹爪待接触点像素 ![](https://latex.codecogs.com/svg.latex?p_{s_1}^{2D},p_{s_2}^{2D}) 的特征向量 ![](https://latex.codecogs.com/svg.latex?f_{p_{s1}^{2D}},f_{p_{s2}^{2D}}\in\mathbb{R}^{d_{DIFT}}) 。后文用 ![](https://latex.codecogs.com/svg.latex?f_{p_s^{2D}}\in\mathbb{R}^{d_{DIFT}}) 任指其一。<br>
针对 ![](https://latex.codecogs.com/svg.latex?f_{p_s^{2D}}\in\mathbb{R}^{d_{DIFT}}) ，以及 ![](https://latex.codecogs.com/svg.latex?F_t=\text{DIFT}\left(I_t\right)\in\mathbb{R}^{H_t\times{}W_t\times{}d_{DIFT}}) 中的任一像素特征向量 ![](https://latex.codecogs.com/svg.latex?f_{p_{tj}^{2D}}\in\mathbb{R}^{d_{DIFT}}) ，计算两者的语义相似度，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?similarity_j=\cos\left\langle{}f_{p_s^{2D}},f_{p_{tj}^{2D}}\right\rangle,">
</p>

选择最高的语义相似度所对应的目标物体RGB图像像素 ![](https://latex.codecogs.com/svg.latex?p_t^{2D}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?p_t^{2D}=\underset{p_{tj}^{2D}}{\text{argmax}}\;\cos\left\langle{}f_{p_s^{2D}},f_{p_{tj}^{2D}}\right\rangle,">
</p>

借助于从深度图 ![](https://latex.codecogs.com/svg.latex?D_t) 生成目标物体观测点云 ![](https://latex.codecogs.com/svg.latex?O_t\in\mathbb{R}^{N\times{}3}) 这一过程，对 ![](https://latex.codecogs.com/svg.latex?p_t^{2D}) 像素进行反投影，得到 ![](https://latex.codecogs.com/svg.latex?p_t^{3D}\in{}O_t) 三维空间点，它就是相应夹爪即将在目标物体上接触的的待接触点。<br>
在Bi-Adapt双臂操作框架中，把可供性狭义地定义为物体即将被接触操作前的待接触点像素。源物体待接触点像素 ![](https://latex.codecogs.com/svg.latex?p_{s_1}^{2D},p_{s_2}^{2D}) 分别由基础模型映射到目标物体待接触点像素 ![](https://latex.codecogs.com/svg.latex?p_{t_1}^{2D},p_{t_2}^{2D}) 的过程，就是可供性迁移的过程。
<br><br><br><br><br><br><br><br><br><br>

# 少样本适应

所得的待接触点对 ![](https://latex.codecogs.com/svg.latex?p_{t_1}^{3D},p_{t_2}^{3D}\in{}O_t) 并不必然适合操作。正因如此，才需要检索同一任务 ![](https://latex.codecogs.com/svg.latex?T) 下的多个源物体，从而相应得出目标物体上的多组候选的待接触点对。但即便如此，由于这些源物体并不必然与目标物体属于同一类别，不同类别的物体又在物理属性和几何形状上存在显著差异，所以，多组候选的待接触点对，与仅经过预训练的动作提议网络所相应提议的多组夹爪姿态对所组成的多组候选的动作对，可能全都不成功，全部遭遇失败。<br>
为了提升神经网络在新类别物体上的性能，需要让神经网络运用自己的预训练知识进行前向推理，最后利用提议动作的执行结果对神经网络进行微调。<br>
神经网络对多组候选的待接触点对并行展开前向推理。在后文中，用 ![](https://latex.codecogs.com/svg.latex?p_{t_1}^{3D},p_{t_2}^{3D}\in{}O_t) 任指其中一组点对。

## 1、神经网络模块 ![](https://latex.codecogs.com/svg.latex?\mathcal{M}_1) 的前向推理

cVAE的推理本质上是轻量的，这主要得益于这一点：如果潜空间在预训练期间得到了良好的塑造（通过模型的全体可学习参数隐式体现），那么潜变量在推理期间的生成就可以具备不依赖于任何后验的无条件性，实现真正的随机多样且均合理的生成。具体而言，直接从标准正态分布 ![](https://latex.codecogs.com/svg.latex?\mathcal{N}\left(0,I\right)) 中随机采样，无条件地得到潜变量 ![](https://latex.codecogs.com/svg.latex?z^\prime\in\mathbb{R}^d) 。<br>
用经过预训练的解码器 ![](https://latex.codecogs.com/svg.latex?\text{Dec}_1(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?z^\prime\in\mathbb{R}^d) 进行解码，得到提议夹爪姿态 ![](https://latex.codecogs.com/svg.latex?\widehat{R}_1\in{}SO(3)) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{R}_1=\text{Dec}_1\left(z^\prime\right)\in{}SO(3).">
</p>

用经过预训练的两个多层感知机 ![](https://latex.codecogs.com/svg.latex?\text{MLP}(\cdot),\text{MLP}^\prime(\cdot)) 分别对 ![](https://latex.codecogs.com/svg.latex?p_{t_1}^{3D}\in{}O_t,\widehat{R}_1\in{}SO(3)) 进行编码，分别得到语义向量 ![](https://latex.codecogs.com/svg.latex?f_{p_1}^\prime,f_{R_1}^\prime\in\mathbb{R}^{32}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}f_{p_1}^\prime&=\text{MLP}\left(p_{t_1}^{3D}\right)\in\mathbb{R}^{32},\\f_{R_1}^\prime&=\text{MLP}^\prime\left(\widehat{R}_1\right)\in\mathbb{R}^{32}.\end{align*}">
</p>

经过预训练的动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_1\left(\cdot,\cdot\right)) 接受 ![](https://latex.codecogs.com/svg.latex?f_{p_1}^\prime,f_{R_1}^\prime\in\mathbb{R}^{32}) 作为输入，输出左夹爪提议动作 ![](https://latex.codecogs.com/svg.latex?\widehat{u}_1=\left(p_{t_1}^{3D},\widehat{R}_1\right)) 的成功概率，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{C}_1\left(f_{p_1}^\prime,f_{R_1}^\prime\right)\in\left(0,1\right).">
</p>

神经网络对多组候选的待接触点对并行展开前向推理，那么就会并行产生多组对应的提议动作对，进而并行预估各个左夹爪动作的成功概率、各个右夹爪动作的成功概率。<br>
选取最高的成功概率所对应的左夹爪提议动作，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\check{u}_1=\left(\check{p}_{t_1}^{3D},\check{R}_1\right)=\underset{\widehat{u}_1}{\text{argmax}}\;\mathcal{C}_1\left(f_{p_1}^\prime,f_{R_1}^\prime\right),">
</p>

将其作为左夹爪即将在目标物体上执行的动作。<br>
此外，将 ![](https://latex.codecogs.com/svg.latex?\check{p}_{t_1}^{3D}\in{}O_t,\check{R}_1\in{}SO(3)) 的语义向量分别记为 ![](https://latex.codecogs.com/svg.latex?\check{f}_{p_1}^\prime,\check{f}_{R_1}^\prime\in\mathbb{R}^{32}) ，它们满足

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{C}_1\left(\check{f}_{p_1}^\prime,\check{f}_{R_1}^\prime\right)=\max\;\mathcal{C}_1\left(f_{p_1}^\prime,f_{R_1}^\prime\right).">
</p>

## 2、神经网络模块 ![](https://latex.codecogs.com/svg.latex?\mathcal{M}_2) 的前向推理

除了需要额外接受左夹爪动作 ![](https://latex.codecogs.com/svg.latex?\check{u}_1=\left(\check{p}_{t_1}^{3D},\check{R}_1\right)) 的“条件化”以外，神经网络模块 ![](https://latex.codecogs.com/svg.latex?\mathcal{M}_2) 的前向推理过程与神经网络模块 ![](https://latex.codecogs.com/svg.latex?\mathcal{M}_1) 的前向推理过程几乎别无二致。通过在动作评分时额外考虑 ![](https://latex.codecogs.com/svg.latex?\check{p}_{t_1}^{3D},\check{R}_1) ，来间接实现“条件化”。与 ![](https://latex.codecogs.com/svg.latex?\check{u}_1=\left(\check{p}_{t_1}^{3D},\check{R}_1\right)) 配合不佳从而在双臂操作任务中多半失败的右夹爪提议动作，被动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2) 赋予较低的成功概率预估值。<br>
直接从标准正态分布 ![](https://latex.codecogs.com/svg.latex?\mathcal{N}\left(0,I\right)) 中随机采样，无条件地得到潜变量 ![](https://latex.codecogs.com/svg.latex?z\in\mathbb{R}^d) 。<br>
用经过预训练的解码器 ![](https://latex.codecogs.com/svg.latex?\text{Dec}_2(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?z\in\mathbb{R}^d) 进行解码，得到提议夹爪姿态 ![](https://latex.codecogs.com/svg.latex?\widehat{R}_2\in{}SO(3)) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{R}_2=\text{Dec}_2\left(z\right)\in{}SO(3).">
</p>

用经过预训练的四个多层感知机 ![](https://latex.codecogs.com/svg.latex?\text{MLP}_1(\cdot),\text{MLP}_2(\cdot),\text{MLP}_3(\cdot),\text{MLP}_4(\cdot)) 分别对 ![](https://latex.codecogs.com/svg.latex?\check{p}_{t_1}^{3D}\in{}O_t,\check{R}_1\in{}SO(3),p_{t_2}^{3D}\in{}O_t,\widehat{R}_2\in{}SO(3)) 进行编码，分别得到语义向量 ![](https://latex.codecogs.com/svg.latex?f_{p_1},f_{R_1},f_{p_2},f_{R_2}\in\mathbb{R}^{32}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{align*}f_{p_1}&=\text{MLP}_1\left(\check{p}_{t_1}^{3D}\right)\in\mathbb{R}^{32},\\f_{R_1}&=\text{MLP}_2\left(\check{R}_1\right)\in\mathbb{R}^{32},\\f_{p_2}&=\text{MLP}_3\left(p_{t_2}^{3D}\right)\in\mathbb{R}^{32},\\f_{R_2}&=\text{MLP}_4\left(\widehat{R}_2\right)\in\mathbb{R}^{32}.\end{align*}">
</p>

经过预训练的动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2\left(\cdot,\cdot,\cdot,\cdot\right)) 接受 ![](https://latex.codecogs.com/svg.latex?f_{p_1},f_{R_1},f_{p_2},f_{R_2}\in\mathbb{R}^{32}) 作为输入，输出右夹爪提议动作 ![](https://latex.codecogs.com/svg.latex?\widehat{u}_2=\left(p_{t_2}^{3D},\widehat{R}_2\right)) 的成功概率，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right)\in\left(0,1\right).">
</p>

选取最高的成功概率所对应的右夹爪提议动作，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\check{u}_2=\left(\check{p}_{t_2}^{3D},\check{R}_2\right)=\underset{\widehat{u}_2}{\text{argmax}}\;\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right),">
</p>

将其作为右夹爪即将在目标物体上执行的动作。<br>
此外，将 ![](https://latex.codecogs.com/svg.latex?\check{p}_{t_1}^{3D}\in{}O_t,\check{R}_1\in{}SO(3),\check{p}_{t_2}^{3D}\in{}O_t,\check{R}_2\in{}SO(3)) 的语义向量分别记为 ![](https://latex.codecogs.com/svg.latex?\check{f}_{p_1},\check{f}_{R_1},\check{f}_{p_2},\check{f}_{R_2}\in\mathbb{R}^{32}) ，它们满足

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{C}_2\left(\check{f}_{p_1},\check{f}_{R_1},\check{f}_{p_2},\check{f}_{R_2}\right)=\max\;\mathcal{C}_2\left(f_{p_1},f_{R_1},f_{p_2},f_{R_2}\right).">
</p>

## 3、动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_1,\mathcal{C}_2) 的微调

左夹爪动作 ![](https://latex.codecogs.com/svg.latex?\check{u}_1=\left(\check{p}_{t_1}^{3D},\check{R}_1\right)) 以及右夹爪动作 ![](https://latex.codecogs.com/svg.latex?\check{u}_2=\left(\check{p}_{t_2}^{3D},\check{R}_2\right)) 被同时在目标物体上执行。执行完毕后，获得任务成败结果，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?r\in\{1,0\}.">
</p>

![](https://latex.codecogs.com/svg.latex?r=1) 表示任务执行成功，![](https://latex.codecogs.com/svg.latex?r=0) 表示任务执行失败。<br>
动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_1) 的微调损失函数为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{C}_1}=-r\log\left(\mathcal{C}_1\left(\check{f}_{p_1}^\prime,\check{f}_{R_1}^\prime\right)\right)-\left(1-r\right)\log\left(1-\mathcal{C}_1\left(\check{f}_{p_1}^\prime,\check{f}_{R_1}^\prime\right)\right),">
</p>

动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_2) 的微调损失函数为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{C}_2}=-r\log\left(\mathcal{C}_2\left(\check{f}_{p_1},\check{f}_{R_1},\check{f}_{p_2},\check{f}_{R_2}\right)\right)-\left(1-r\right)\log\left(1-\mathcal{C}_2\left(\check{f}_{p_1},\check{f}_{R_1},\check{f}_{p_2},\check{f}_{R_2}\right)\right).">
</p>

## 4、动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_2) 的微调

如果本次新类别物体操作任务成功，即 ![](https://latex.codecogs.com/svg.latex?r=1) ，那么过程中产生的中间量将用于对动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1,\mathcal{A}_2) 进行微调。如果失败，即 ![](https://latex.codecogs.com/svg.latex?r=0) ，则不用于微调。以下对 ![](https://latex.codecogs.com/svg.latex?r=1) 的情况进行说明。<br>
用经过预训练的 PointNet++ (segmentation version) 神经网络 ![](https://latex.codecogs.com/svg.latex?\text{PN}(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?O_t\in\mathbb{R}^{N\times{}3}) 进行逐点特征提取，得到点级特征矩阵 ![](https://latex.codecogs.com/svg.latex?F\in\mathbb{R}^{N\times{}128}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?F=\text{PN}\left(O_t\right)\in\mathbb{R}^{N\times{}128}.">
</p>

用经过预训练的编码器 ![](https://latex.codecogs.com/svg.latex?\text{Enc}_2\left(\cdot,\cdot,\cdot,\cdot,\cdot\right)) 对 ![](https://latex.codecogs.com/svg.latex?\check{R}_2\in{}SO(3),F\in\mathbb{R}^{N\times{}128},\check{f}_{p_1}\in\mathbb{R}^{32},\check{f}_{R_1}\in\mathbb{R}^{32},\check{f}_{p_2}\in\mathbb{R}^{32}) 进行编码，得到均值向量 ![](https://latex.codecogs.com/svg.latex?\mu_2\in\mathbb{R}^d) 和协方差矩阵 ![](https://latex.codecogs.com/svg.latex?\Sigma_2\in\mathbb{R}^{d\times{}d}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left(\mu_2,\Sigma_2\right)=\text{Enc}_2\left(\check{R}_2,F,\check{f}_{p_1},\check{f}_{R_1},\check{f}_{p_2}\right),">
</p>

它们组成潜变量分布 ![](https://latex.codecogs.com/svg.latex?\mathcal{N}\left(\mu_2,\Sigma_2\right)) 。从 ![](https://latex.codecogs.com/svg.latex?\mathcal{N}\left(\mu_2,\Sigma_2\right)) 中随机采样，得到潜变量 ![](https://latex.codecogs.com/svg.latex?z\in\mathbb{R}^d) 。<br>
用经过预训练的解码器 ![](https://latex.codecogs.com/svg.latex?\text{Dec}_2(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?z\in\mathbb{R}^d) 进行解码，得到预测夹爪姿态 ![](https://latex.codecogs.com/svg.latex?\widehat{R}_2\in{}SO(3)) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{R}_2=\text{Dec}_2(z)\in{}SO(3).">
</p>

动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_2) 的微调损失函数即为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{A}_2}=\mathcal{L}_{geo}\left(\widehat{R}_2,\check{R}_2\right)+D_{KL}\left(\mathcal{N}\left(\mu_2,\Sigma_2\right)\,\|\;\mathcal{N}\left(0,I\right)\right),">
</p>

其中潜空间正则化项可另记为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?D_{KL}\left(q\left(z\;|\;\check{R}_2,F,\check{f}_{p_1},\check{f}_{R_1},\check{f}_{p_2}\right)\,\|\;\mathcal{N}\left(0,I\right)\right).">
</p>

## 5、动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1) 的微调

用经过预训练的编码器 ![](https://latex.codecogs.com/svg.latex?\text{Enc}_1\left(\cdot,\cdot,\cdot\right)) 对 ![](https://latex.codecogs.com/svg.latex?\check{R}_1\in{}SO(3),F\in\mathbb{R}^{N\times{}128},\check{f}_{p_1}^\prime\in\mathbb{R}^{32}) 进行编码，得到均值向量 ![](https://latex.codecogs.com/svg.latex?\mu_1\in\mathbb{R}^d) 和协方差矩阵 ![](https://latex.codecogs.com/svg.latex?\Sigma_1\in\mathbb{R}^{d\times{}d}) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\left(\mu_1,\Sigma_1\right)=\text{Enc}_1\left(\check{R}_1,F,\check{f}_{p_1}^\prime\right),">
</p>

它们组成潜变量分布 ![](https://latex.codecogs.com/svg.latex?\mathcal{N}\left(\mu_1,\Sigma_1\right)) 。从 ![](https://latex.codecogs.com/svg.latex?\mathcal{N}\left(\mu_1,\Sigma_1\right)) 中随机采样，得到潜变量 ![](https://latex.codecogs.com/svg.latex?z^\prime\in\mathbb{R}^d) 。<br>
用经过预训练的解码器 ![](https://latex.codecogs.com/svg.latex?\text{Dec}_1(\cdot)) 对 ![](https://latex.codecogs.com/svg.latex?z^\prime\in\mathbb{R}^d) 进行解码，得到预测夹爪姿态 ![](https://latex.codecogs.com/svg.latex?\widehat{R}_1\in{}SO(3)) ，即

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\widehat{R}_1=\text{Dec}_1\left(z^\prime\right)\in{}SO(3).">
</p>

动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1) 的微调损失函数即为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}_{\mathcal{A}_1}=\mathcal{L}_{geo}\left(\widehat{R}_1,\check{R}_1\right)+D_{KL}\left(\mathcal{N}\left(\mu_1,\Sigma_1\right)\,\|\;\mathcal{N}\left(0,I\right)\right),">
</p>

其中潜空间正则化项可另记为

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?D_{KL}\left(q\left(z^\prime\;|\;\check{R}_1,F,\check{f}_{p_1}^\prime\right)\,\|\;\mathcal{N}\left(0,I\right)\right).">
</p>

## 6、总结

动作评分网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{C}_1,\mathcal{C}_2) 以及动作提议网络 ![](https://latex.codecogs.com/svg.latex?\mathcal{A}_1,\mathcal{A}_2) 经过微调后，“感知模块（原论文中指神经网络模块 ![](https://latex.codecogs.com/svg.latex?\mathcal{M}_1,\mathcal{M}_2) 组成的整体）能够更好地判断所提出的动作是否会成功，并对其进行调整以适应新类别”。<br>
原论文作者团队指出：<br>
“只需在新类别的有限实例上进行少量交互，感知模块便能探索由基础模型选出的新类别语义显著区域，并更好地理解其与已学类别之间的几何与物理差异。经过少样本适应后，有关操作的知识便迁移到了新类别上。得益于同一类别内几何特征与物理属性的泛化能力，我们适应后的模块无需额外交互即可操作该类别中未见过的物体。”
