# @Time : 2020/6/17 10:34 
# @Author : 大太阳小白
# @Software: PyCharm
import numpy as np
import numpy as np
from numpy import *


def selectJrand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn);
    labelMat = mat(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        # 判断一堆阿尔法是否更新
        alphaPairsChanged = 0
        for i in range(m):
            # w = 连加 alphai * yi * xi
            # 超平面 = wT * x + b
            # fxi = multiply(multiply(alphas,labelMat),dataMatrix[i]).T * dataMatrix + b
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])  # if checks if an example violates KKT conditions
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[
                                                                                                            j,
                                                                                                            :] * dataMatrix[
                                                                                                                 j, :].T
                if eta >= 0:
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # update i by the same amount as j
                    # the update is in the oppostie direction
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
    return b, alphas


def loadDataSet(fileName):
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet("testSet.txt")
    b, alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    print(shape(b))
    print(shape(alphas))