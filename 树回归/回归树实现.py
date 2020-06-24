# @Time : 2020/6/23
# @Author : 大太阳小白
# @Software: PyCharm
"""
回归树模型
叶节点是常数值
"""
from numpy import *


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 使用map映射，将字符型数据转换成浮点型
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    """
    根据特征和特征值切分数据
    :param dataSet:
    :param feature:
    :param value:
    :return:
    """
    # dataSet[:, feature] > value 将会比较特征值大于value的样本返回True，小于等于返回False，则生成一个0，1数组
    # nonzero 找出非零元素的索引
    # dataSet 通过非零样本索引找到相应元素，并返回该样本所有特征
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    """
    生成叶结点
    :param dataSet:
    :return:
    """
    return mean(dataSet[:, -1])


def regErr(dataSet):
    """
    误差估计函数，使用总体误差，即特征的均方误差*样本量
    :param dataSet:
    :return:
    """
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    核心函数，找到最佳二元切分
    :param dataSet: 数据集
    :param leafType: 创建叶结点函数
    :param errType:误差估计函数
    :param ops:控制参数，0：允许误差下降值，1：切分的最少样本数
    :return:
    """
    tolS = ops[0]
    tolN = ops[1]
    # 若样本数据的目标属性下的所有值都相同，则不需切分，其作为叶节点
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    # 计算样本误差
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    # 遍历所有特征
    for featIndex in range(n - 1):
        # 遍历特征下所有特征值
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            # 指定特征值进行切分
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                # 切分后样本少于用户指定的最少样本书时，则跳过该切分
                continue
            # 计算切分后的总体误差
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                # 保存最小误差分类
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果最佳切分误差下降值小于用户指定的值，则不用切分，其作为叶节点
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 若节点样本量小于最小样本数，则划分为叶节点
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    创建回归树
    :param dataSet:数据集
    :param leafType: 叶结点函数
    :param errType: 误差函数
    :param ops: 控制参数
    :return: 回归树
    """
    # 选择最佳分割
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        # 最佳分割是叶节点，返回叶节点
        return val
    # 若回归树不是单层
    retTree = {}
    # 节点最佳特征索引
    retTree['spInd'] = feat
    # 节点最佳特征分割值
    retTree['spVal'] = val
    # 分割
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 分割的后的数据递归生成叶结点
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    """
    判断输入变量是否是树
    :param obj:
    :return:
    """
    return type(obj).__name__ == 'dict'


def getMean(tree):
    """
    递归寻找叶节点（左节点和右节点都不是树），并计算它们的均值
    :param tree:
    :return:
    """
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    """
    后剪枝
    :param tree:
    :param testData:
    :return:
    """
    if shape(testData)[0] == 0:
        # 测试数据为空，则递归调用回归树，返回叶结点均值
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        # 若分支是树，则切分
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 若都是叶节点，则合并它们
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算未合并误差
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算未合并误差
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            # 若合并后误差减小，则进行合并
            print("进行合并")
            return treeMean
        else:
            return tree
    else:
        return tree


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    train_data = loadDataSet("data/bikeSpeedVsIq_train.txt")
    test_data = loadDataSet("data/bikeSpeedVsIq_test.txt")
    data_mat = mat(train_data)
    test_mat = mat(test_data)
    tree = createTree(data_mat, ops=(1, 20))
    y_predict = createForeCast(tree, test_mat[:, 0])
    corr = corrcoef(y_predict, test_mat[:, 1], rowvar=0)[0][1]
    print("计算预测结果和标注结果相关性", corr)
    # 对树进行预剪枝，即放宽对生成树的控制，允许误差加大，若总体误差小于10000则划分为叶结点
    tree = createTree(data_mat, ops=(10000, 20))
    y_predict = createForeCast(tree, test_mat[:, 0])
    corr = corrcoef(y_predict, test_mat[:, 1], rowvar=0)[0][1]
    print("计算预测结果和标注结果相关性-预剪枝", corr)
    # 对树进行预剪枝，若节点样本数小于50，则划分成叶节点
    tree = createTree(data_mat, ops=(1, 50))
    y_predict = createForeCast(tree, test_mat[:, 0])
    corr = corrcoef(y_predict, test_mat[:, 1], rowvar=0)[0][1]
    print("计算预测结果和标注结果相关性-预剪枝", corr)
    tree = createTree(data_mat, ops=(1, 5))
    y_predict = createForeCast(tree, test_mat[:, 0])
    source_corr = corrcoef(y_predict, test_mat[:, 1], rowvar=0)[0][1]
    prune_tree = prune(tree, test_mat[:, 0])
    y_predict = createForeCast(prune_tree, test_mat[:, 0])
    corr = corrcoef(y_predict, test_mat[:, 1], rowvar=0)[0][1]
    print("计算预测结果和标注结果相关性-后剪枝", corr, "原相关性得分", source_corr)
