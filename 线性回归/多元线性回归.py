import numpy as np
import pandas as pd
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
    data = pd.read_csv(file_path)
    feature = data[u'中国平安']
    label = data[u'沪深300']
    feature = np.mat([np.ones((1, feature.shape[0])).tolist()[0], feature.values.tolist()]).T
    label = np.mat([label.values.tolist()]).T
    return feature, label


if __name__ == '__main__':
    feature, label = load_data('data/data.csv')
    weight = least_square(feature, label)
    print(weight)
    plt.figure(figsize=(10, 6))
    x = feature.T.tolist()[1]
    y = label.T.tolist()[0]
    plt.plot(x, y, 'o')
    plt.plot(x, [float(i * weight) for i in feature], 'r--')
    plt.show()
