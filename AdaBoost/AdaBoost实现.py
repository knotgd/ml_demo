# @Time : 2020/6/22 10:46 
# @Author : 大太阳小白
# @Software: PyCharm
from numpy import *


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    数据分类，对于二分类数据，指定某个特征，对特征值小于阈值时设为-1类，大于阈值设置1类
    :param dataMatrix:训练样本
    :param dimen:特征
    :param threshVal:阈值
    :param threshIneq:判断条件
    :return:
    """
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    构建树桩
    :param dataArr: 特征集
    :param classLabels: 标注集
    :param D: 样本权重
    :return:
    """
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    # 设置总步数
    numSteps = 10.0
    # 用来保存最优单层决策树
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf
    # 遍历特征
    for i in range(n):
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max();
        stepSize = (rangeMax - rangeMin) / numSteps
        #  通过计算特征值最大、最小值，得到每部大小
        for j in range(-1, int(numSteps) + 1):
            # 遍历判别号
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                # 通过步长划分数据集，并计算准确率
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)
                errArr = mat(ones((m, 1)))
                # 分类对的设置成0，错的设置成1
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # 保存最优错误率下的单层决策树
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    构建以单层决策树为基学习器的adaboost
    :param dataArr: 数据特征
    :param classLabels: 数据标签
    :param numIt: 迭代次数
    :return:
    """
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化样本权重
    D = mat(ones((m, 1)) / m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        # 构建一个最优单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # 计算该学习器的alpha值
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        # 存储该学习器
        weakClassArr.append(bestStump)
        # 计算新的权重向量D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # 计算总的错误率
        aggClassEst += alpha * classEst
        # 符号函数 输出aggClassEst结果
        output_label = sign(aggClassEst)
        # 判断若输出结果和标注不相等则元素值为1，否则为0
        aggErrors = multiply(output_label != mat(classLabels).T, ones((m, 1)))
        # 计算错误率
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    分类
    :param datToClass:待分类数据
    :param classifierArr:一组分类器
    :return:
    """
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return sign(aggClassEst)


if __name__ == '__main__':
    train_features, train_labels = loadDataSet("data/horseColicTraining2.txt")
    test_features, test_labels = loadDataSet("data/horseColicTest2.txt")
    class_arr,_ = adaBoostTrainDS(train_features,train_labels)
    result = adaClassify(test_features,class_arr)
    for i in range(len(test_labels)):
        print("预测值：{} 实际值：{}".format(result[i],test_labels[i]))

