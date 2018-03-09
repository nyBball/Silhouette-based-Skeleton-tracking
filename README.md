# Silhouette-based-Skeleton-tracking
## 项目简介：
　　本项目的目的是利用回归方法在已构建的人体姿态数据库上训练得到从剪影和初始骨架到真实骨架的映射级联回归子，并基于单目视频进行测试，实现人体骨架的跟踪。该方法包含如下步骤：  
　　在训练阶段：a.定义特征描述符：从关节点随机方向引出射线直至与剪影边界相交，以射线长度量化剪影与初始骨架的差异；b.特征提取和分类：通过基于相关性的random fern方法提取出与回归目标有最大相关性的特征并分类，得到决定当前阶段骨架调整幅度的回归子；c.骨架调整：调整骨架适当的次数并输出级联回归子。  
　　在测试阶段：a.输入单目视频首帧的剪影和初始化骨架；b.根据训练得到的级联回归子逐步调整初始化骨架至最终骨架；c.在数据库中搜索与当前帧预测骨架最相似的五个骨架分别进行回归，取回归的均值骨架作为下一帧的初始化骨架进行下一帧的骨架预测。本方法可有效实时地实现骨架的跟踪，并避免跟踪过程中出现的误差积累现象。
## 项目构成：
- stdafx.h头文件包含了项目所需要使用的头文件，如OpenCV、Eigen等.其只使用了Opencv基本的Mat数据类型和基本的划线操作函数。类似，Eigen也只是使用了其矩阵数据结构。

- utils.h头文件主要定义了本项目所需要的基本数据结构，包括Vector2f（2维向量）、VectorNf（高维向量）、RandomDevice（产生符合线性分布或者高斯分布的随机数）、Delta（方便快速排序的比较）四个类。

- sample.h头文件定义了:

 	1)RFParam虚基类，其主要包含Param2Landmarks虚函数（3D关节点投影到剪影上的2D关节点）、DrawLandmarks虚函数（在剪影上标记出2D关节点);

	2)RFSampleVecNode结构体，包含随机投影时所需的射线角度和系数；

	3)RFSample模板虚基类。其主要包含Sampling（采样）虚函数、GetParamDelta(计算更新后和未更新骨架的差)函数、SetStatus（设置0、1状态）函数、GetStatus(根据0、1状态得到0-31的状态)、UpdateParam(更新骨架坐标)函数。

	4)RFBodyJoints类，其公共继承了RFParam虚基类；

	5)RFSample_BodyJoints类，其公共继承了RFSample模板虚基类。主要实例化了采样函数。  

- train.h头文件定义了：

	1)RFtrain模板虚基类，其主要实现训练的过程；

 	2)RFTrainBodyJoints类，其公共继承了RFtrain模板虚基类，主要实例 化虚基类中的SetSamples虚函数和GenerateRandomVector虚函数。
- test.h头文件只定义了RFtest模板类，实现测试的过程.

- main.cpp源文件，主要定义快速排序函数、搜寻相似骨架函数、并利用switch实现训练和测试过程的选择。


## 实验效果：
- 49481帧用于训练、4345帧用于测试，每帧的测试时间为19.43ms，满足实时性需求；
- 错误率统计：

	最大值：2.1147m;最小值：0m；平均值：0.2102m；统计直方图如图所示：
![](https://i.imgur.com/weaIlTY.png)


- 预测骨架真实骨架对比：

![](https://i.imgur.com/xbH2hrf.png)
![](https://i.imgur.com/T8RmrVs.png)
![](https://i.imgur.com/eQyfqAP.png)

## 流程图
![](https://i.imgur.com/QkK6ihU.png)

- 训练过程

![](https://i.imgur.com/s1X60KS.png)

- 测试过程

![](https://i.imgur.com/jVM11aM.png)
