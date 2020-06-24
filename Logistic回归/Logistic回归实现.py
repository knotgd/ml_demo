# @Time : 2020/6/16 10:42 
# @Author : 大太阳小白
# @Software: PyCharm
"""
优点：计算代价不高，易于理解和实现
缺点：容易欠拟合，分类精度可能不高
适用数据类型：数值型和标称型
"""
import numpy as np


def sigmoid(inx):
    """
    :param inx:
    :return:
    """
    return 1.0 / (1 + np.exp(-inx))


def grad_ascent(feature, label,lam = 0.001,max_iter = 5000):
    """

    :param feature:
    :param label:
    :return:
    """
    feature_mat = np.mat(feature)
    label_mat = np.mat(label).T
    m, n = np.shape(feature_mat)
    weights = np.ones((n, 1))
    for k in range(max_iter):
        h = sigmoid(feature_mat * weights)
        error = label_mat - h
        weights = weights + lam * feature_mat.T * error
    return weights


def load_data():
    """

    :return:
    """
    train_file = open("data/horseColicTraining.txt")
    test_file = open("data/horseColicTest.txt")
    train_data = np.array([line.strip().split("\t") for line in train_file.readlines()])
    test_data = np.array([line.strip().split("\t") for line in test_file.readlines()])
    train_feature = train_data[:, :-1].astype(np.float64)
    train_label = train_data[:, -1].astype(np.float64).astype(np.int32)
    test_feature = test_data[:, :-1].astype(np.float64)
    test_label = test_data[:, -1].astype(np.float64).astype(np.int32)
    return train_feature, train_label, test_feature, test_label


def predict(test_feature, weights):
    """

    :param test_feature:
    :param weights:
    :return:
    """
    h = sigmoid(np.sum(np.mat(test_feature) * weights))
    return 1 if h > 0.5 else 0


if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels = load_data()
    weight = grad_ascent(train_features, train_labels)
    sum_truth = 0
    for index, test_v in enumerate(test_features):
        pre = predict(test_v, weight)
        print("预测值：", pre, "真实值：", test_labels[index], "Ture" if pre == test_labels[index] else "False")
        sum_truth += 1 if pre == test_labels[index] else 0
    print("准确率：", (sum_truth / len(test_labels)))
