import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def least_square(feature, label):
    """
    最小二乘法求解一元线性回归
    :param feature: 特征，m行1列
    :param label: 标注，m行1列
    :return:
    """
    m = np.shape(feature)[0]
    sp1 = 0
    sp2 = 0
    sp3 = (np.sum(feature, 0) ** 2) / m
    avg_x = float(np.mean(feature, 0))
    for index in range(len(feature)):
        sp1 += label[index] * (feature[index] - avg_x)
        sp2 += feature[index] ** 2
    w = sp1 / (sp2 - sp3)
    b = sum([label[i] - w * feature[i] for i in range(len(feature))]) / m
    return w, b


def load_data(file_path):
    """
    加载数据，并转换成特征和标记矩阵
    :param file_path:
    :return:
    """
    data = pd.read_csv(file_path)
    feature = data[u'中国平安']
    label = data[u'沪深300']
    return np.mat(feature.values.tolist()).T, np.mat([label.values.tolist()]).T

if __name__ == '__main__':
    feature, label = load_data('data/data.csv')
    w, b = least_square(feature, label)
    print(float(w), float(b))
    plt.figure(figsize=(10, 6))
    x = feature.T.tolist()[0]
    y = label.T.tolist()[0]
    plt.plot(x, y, 'o')
    plt.plot(x, [i * float(w) + float(b) for i in x], 'r--')
    plt.show()
