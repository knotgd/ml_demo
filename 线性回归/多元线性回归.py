import numpy as np
import matplotlib.pyplot as plt


def least_square(feature, label):
    """
    小二乘法求解多元线性回归
    :param feature:
    :param label:
    :return:
    """
    w = (feature.T * feature).I * feature.T * label
    return w


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
        c_data = c_line.strip().split('\t')
        features.append([float(c_data[0]),float(c_data[1])])
        labels.append(float(c_data[-1]))
    return np.mat(features), np.mat(labels).T


if __name__ == '__main__':
    feature, label = load_data('data/ex0.txt')
    weight = least_square(feature, label)
    print(weight)
    plt.figure(figsize=(10, 6))
    x = feature.T.tolist()[1]
    y = label.T.tolist()[0]
    plt.plot(x, y, 'o')
    plt.plot(x, [float(i * weight) for i in feature], 'r--')
    plt.show()
