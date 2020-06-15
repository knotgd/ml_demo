# @Time : 2020/6/15 14:42 
# @Author : 大太阳小白
# @Software: PyCharm
import os
import numpy as np
import operator


def load_data_set(file_path):
    """

    :param file_path:
    :return:
    """
    file_list = os.listdir(file_path)
    num_lines = len(file_list)
    features =  np.zeros((num_lines,1024))
    labels = []
    for index in range(num_lines):
        current_file = file_list[index]
        img_vec_file = open('{}/{}'.format(file_path, current_file), 'r')
        img_vec_data = [list(line.strip()) for line in img_vec_file.readlines()]
        img_vec_data = np.array(img_vec_data).reshape(1, len(img_vec_data) * len(img_vec_data)).astype(np.int32)
        label = current_file.split('_')[0]
        features[index]=img_vec_data
        labels.append(label)
    return features, labels


def classify0(input_vec, train_featues, train_labels, k):
    num_data = train_featues.shape[0]
    diff_mat = np.tile(input_vec, (num_data, 1)) - train_featues
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = train_labels[sorted_dist_indicies[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_class_count)
    return sorted_class_count[0][0]


if __name__ == '__main__':
    train_vec, train_label = load_data_set('data/trainingDigits')
    test_vec, test_label = load_data_set('data/testDigits')
    random_test_index = np.random.choice(np.arange(len(test_label)))
    print('随机抽取测试数据,标注结果为', test_label[random_test_index])
    p = classify0(test_vec[random_test_index], train_vec, train_label, 9)
    print('预测结果为', p)
