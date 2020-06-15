# @Time : 2020/6/14 16:31 
# @Author : 大太阳小白
# @Software: PyCharm
import numpy as np
import operator


def calc_shannon_ent(data_set):
    """
    计算香农熵
    特点：分类越多，它的香农熵越大
    :param data_set:
    :return:
    """
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label in label_counts:
            label_counts[current_label] += 1
        else:
            label_counts[current_label] = 1
    shannon_ent = 0
    for value in label_counts.values():
        p = value / num_entries
        shannon_ent -= p * np.log2(p)
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    按照指定特征划分数据集
    :param data_set: 待划分数据集
    :param axis: 指定特征列索引
    :param vlaue: 指定特征值
    :return:
    """
    ret_data_set = []
    for item in data_set:
        if item[axis] == value:
            temp = []
            temp.extend(item[:axis])
            temp.extend(item[axis + 1:])
            ret_data_set.append(temp)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """

    :param data_set:
    :return:
    """
    num_feature = len(data_set[0]) - 1
    total_num = len(data_set)
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = 0
    best_feature = -1
    for i in range(num_feature):
        feature_values = set([item[i] for item in data_set])
        new_entropy = 0
        for feature_value in feature_values:
            sub_data_set = split_data_set(data_set, i, feature_value)
            prob = len(sub_data_set) / total_num
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """
    返回占比最多的类别
    :param class_list:
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sort_class_count[0][0]


def create_tree(data_set, labels):
    """

    :param data_set:
    :param labels:
    :return:
    """
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del (labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """

    :param input_tree:
    :param feat_labels:
    :param test_vec:
    :return:
    """
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


if __name__ == '__main__':
    import util
    fr = open('data/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    label = ['age', 'prescript', 'astigmatic', 'tear_rate']
    tree = create_tree(lenses, label.copy())
    util.storeTree(str(tree),'tree_model')
    pre = classify(tree, label.copy(), ["young", "hyper", "yes", "normal"])
    print(pre)
