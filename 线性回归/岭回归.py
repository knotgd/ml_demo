import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    """
       加载数据，并转换成特征和标记矩阵
       :param file_path:
       :return:
       """
    file = open(file_path)
    lines = file.readlines()
    features = []
    labels = []
    for c_line in lines:
        item = [1]
        c_data = c_line.strip().split('\t')
        item.extend([float(c_data_item) for c_data_item in c_data])
        features.append(item[:-1])
        labels.append(item[-1])
    return np.mat(features), np.mat(labels).T


def ridge_regres(features, labels, lam=0.2):
    """
    岭回归最小二乘求解
    :param features:
    :param labels:
    :param lam:
    :return:
    """
    xtx = features.T * features
    denom = xtx + np.eye(np.shape(features)[1]) * lam
    ws = denom.I * (features.T * labels)
    return ws


if __name__ == '__main__':
    feature, label = load_data('data/abalone.txt')
    num_test_pts = 30
    weights = np.zeros((num_test_pts, np.shape(feature)[1]))
    for i in range(num_test_pts):
        w = ridge_regres(feature, label, np.exp(i - 10))
        weights[i, :] = w.T
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(weights))
    plt.show()
# 当lam非常小时候，系数与普通回归样，lam变大时所有系数将向0减小，可以再某个位置部分系数等于0时进行lam取值
# 次方法可以将系数缩减，若对输入进行标准化，系数会变小
