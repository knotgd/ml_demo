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
        c_data = c_line.strip().split('\t')
        features.append([float(c_data[0]), float(c_data[1])])
        labels.append(float(c_data[-1]))
    return np.mat(features), np.mat(labels).T


def lwlr(test_point, features, labels, k=0.1):
    """
    局部加权核心逻辑，每个样本都要计算权重
    :param test_point:
    :param features:
    :param labels:
    :param k:
    :return:
    """
    m = np.shape(features)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff_mat = test_point - features[j]
        weights[j, j] = np.exp(diff_mat * diff_mat.T / (-2 * k ** 2))
    xTx = features.T * (weights * features)
    ws = xTx.I * (features.T * (weights * labels))
    return test_point * ws


if __name__ == '__main__':
    feature, label = load_data('data/ex0.txt')
    k = 0.01
    prediction = [lwlr(feature_item, feature, label, k) for feature_item in feature]
    plt.figure(figsize=(10, 6))
    x = feature.T.tolist()[1]
    y = label.T.tolist()[0]
    plt.plot(x, y, 'o')
    # 展示需要对输入特征排序，因为训练的权重会对输入特征附近的值敏感，也可以通过样本点与待测点距离做排序
    xrtind = feature[:, 1].argsort(0)
    xsort = feature[xrtind][:, 0]
    p_v = np.array(prediction)[xrtind].T.tolist()[0][0][0]
    plt.plot(xsort.T.tolist()[1], p_v)
    plt.show()
#存在问题：1、增加了计算量，每个样本都需要素有训练数据，当k足够小时候权重大部分数都是0
# 2、存在过拟合问题，不是均方误差越小越好，需要在测试集上评估
# 3、测试集上需要所有的训练样本数据参与