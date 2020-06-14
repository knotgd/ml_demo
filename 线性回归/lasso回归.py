# @Time : 2020/6/14 10:57 
# @Author : 大太阳小白
# @Software: PyCharm
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


def ress_error(y_array, y_hat_array):
    return ((y_array - y_hat_array) ** 2).sum()


def stage_wise(features, labels, eps=0.01, num_it=100):
    label_mean = np.mean(labels, axis=0)
    labels -= label_mean
    feature_mean = np.mean(features, 0)
    feature_var = np.var(features, 0)
    features = ((features - feature_mean) / feature_var)
    features[np.isnan(features)] = 1
    m, n = np.shape(features)
    return_mat = np.zeros((num_it, n))
    ws = np.zeros((n, 1))
    ws_test = ws.copy()
    ws_max = ws.copy()
    lowest_error = np.inf
    for i in range(num_it):
        for j in range(n):
            for sign in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += eps * sign
                y_test = features * ws_test
                ress_e = ress_error(labels.A, y_test.A)
                if ress_e < lowest_error:
                    lowest_error = ress_e
                    ws_max = ws_test
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat


if __name__ == '__main__':
    feature, label = load_data('data/abalone.txt')
    weights = stage_wise(feature, label, eps=0.001, num_it=5000)
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(weights))
    plt.show()
