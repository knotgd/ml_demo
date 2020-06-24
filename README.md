# 机器学习经典算法手写实现
## 目录
* [一、KNN算法](#一-KNN算法)
   * [1、算法概述](#1-1-算法概述)
   * [2、算法原理](#1-2-算法原理)
   * [3、优缺点](#1-3-优缺点)
   * [4、核心公式](#1-4-核心公式)
   * [5、手写数字识别](#1-5-基于KNN算法实现手写数字识别)
* [二、决策树](#一-KNN算法)
   * [1、算法概述](#1-1-算法概述)
   * [2、算法原理](#1-2-算法原理)
   * [3、优缺点](#1-3-优缺点)
   * [4、核心公式](#1-4-核心公式)
   * [5、手写数字识别](#1-5-基于KNN算法实现手写数字识别)
* [三、朴素贝叶斯](#一-KNN算法)
   * [1、算法概述](#1-1-算法概述)
   * [2、算法原理](#1-2-算法原理)
   * [3、优缺点](#1-3-优缺点)
   * [4、核心公式](#1-4-核心公式)
   * [5、手写数字识别](#1-5-基于KNN算法实现手写数字识别)
* [四、Logistic回归](#一-KNN算法)
   * [1、算法概述](#1-1-算法概述)
   * [2、算法原理](#1-2-算法原理)
   * [3、优缺点](#1-3-优缺点)
   * [4、核心公式](#1-4-核心公式)
   * [5、手写数字识别](#1-5-基于KNN算法实现手写数字识别)
* [五、支持向量机](#一-KNN算法)
   * [1、算法概述](#1-1-算法概述)
   * [2、算法原理](#1-2-算法原理)
   * [3、优缺点](#1-3-优缺点)
   * [4、核心公式](#1-4-核心公式)
   * [5、手写数字识别](#1-5-基于KNN算法实现手写数字识别)
* [六、集成学习](#一-KNN算法)
   * [1、算法概述](#1-1-算法概述)
   * [2、算法原理](#1-2-算法原理)
   * [3、优缺点](#1-3-优缺点)
   * [4、核心公式](#1-4-核心公式)
   * [5、手写数字识别](#1-5-基于KNN算法实现手写数字识别)
* [七、线性回归](#一-KNN算法)
   * [1、算法概述](#1-1-算法概述)
   * [2、算法原理](#1-2-算法原理)
   * [3、优缺点](#1-3-优缺点)
   * [4、核心公式](#1-4-核心公式)
   * [5、手写数字识别](#1-5-基于KNN算法实现手写数字识别)
* [八、树回归](#一-KNN算法)
   * [1、算法概述](#1-1-算法概述)
   * [2、算法原理](#1-2-算法原理)
   * [3、优缺点](#1-3-优缺点)
   * [4、核心公式](#1-4-核心公式)
   * [5、手写数字识别](#1-5-基于KNN算法实现手写数字识别)


## 一、KNN算法

- ### 1.1 算法概述

已知一个有标记的样本集合（训练集合），对于输入的测试样本不知道其分类，让其和所有训练样本做比较，选出最相似的N个样本的分类标签，这N个标签中占比最多的作为测试样本的分类。

- ### 1.2 算法原理

a. 计算已知类别数据集中的点与当前点(待分类)之间的距离

b.按照距离递增次序排序

c.选取与当前点距离最小的k个点

d.确定前k个点坐在类别出现的频率

e.返回前k个点出现频率最高的类别作为当前点的预测分类

- ### 1.3 优缺点

> 优点:精度高、对异常值不敏感、无数据输入假定

> 缺点:计算复杂度高、空间复杂度高

> 使用数据范围:数值型和标称型

- ### 1.4 核心公式

  欧式距离是最常见的距离度量，衡量的是多维空间中各个点之间的绝对距离公式如下：
  
  <img src="https://cdn.mathpix.com/snip/images/qp60VqxuICIsstz8_jZSt68Rn0sOuWVOz-sGRgYDJJw.original.fullsize.png" style="zoom: 25%;" />
  
- ### 1.5 基于KNN算法实现手写数字识别

## 二、决策树

- ### 2.1 算法概述

- ### 2.2 优缺点

- ### 2.3算法原理

- ### 2.4 核心公式

- ### 2.5 demo

## 朴素贝叶斯
- ### 2.1 算法概述

- ### 2.2 优缺点

- ### 2.3算法原理

- ### 2.4 核心公式

- ### 2.5 demo

## Logistic回归
- ### 2.1 算法概述

- ### 2.2 优缺点

- ### 2.3算法原理

- ### 2.4 核心公式

- ### 2.5 demo

## 支持向量机
- ### 2.1 算法概述

- ### 2.2 优缺点

- ### 2.3算法原理

- ### 2.4 核心公式

- ### 2.5 demo

## 集成学习
- ### 2.1 算法概述

- ### 2.2 优缺点

- ### 2.3算法原理

- ### 2.4 核心公式

- ### 2.5 demo
## AdaBoost
- 核心思想
能否使用弱分类器和多个实例来构建一个强分类器
- 算法原理
A、为训练集中每个样本赋予一个权重，这些权重构成向量D,这些权重初始是等值的
B、首先训练一个弱分类器，并计算错误率
C、再次在同一个训练集上训练分类器，此次将调整样本权重，第一次分对的样本权重将降低，第一次分错的权重会提高
D、adaBoost会为每个分类器分配一个权重alpha
- 算法示意图

## 线性回归
- ### 2.1 算法概述

- ### 2.2 优缺点

- ### 2.3算法原理

- ### 2.4 核心公式

- ### 2.5 demo
## 树回归
- ### 2.1 算法概述

- ### 2.2 优缺点

- ### 2.3算法原理

- ### 2.4 核心公式

- ### 2.5 demo
## K-均值聚类
- ### 2.1 算法概述

- ### 2.2 优缺点

- ### 2.3算法原理

- ### 2.4 核心公式

- ### 2.5 demo
## PCA降维
- ### 2.1 算法概述

- ### 2.2 优缺点

- ### 2.3算法原理

- ### 2.4 核心公式

- ### 2.5 demo