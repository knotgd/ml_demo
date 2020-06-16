# @Time : 2020/6/15 16:02 
# @Author : 大太阳小白
# @Software: PyCharm
"""
有点：在数据较少的情况下仍然有效，可以处理多类别问题
缺点：对于输入数据的准备方式较为敏感
使用数据类型：标称型数据
"""
import os
import numpy as np
import re


def load_data():
    """
    加载目录下的文件
    :param path:
    :return:
    """
    ham_path = "data/ham"
    spam_path = "data/spam"
    ham_list = []
    spam_list = []
    for path in [ham_path, spam_path]:
        current_list = os.listdir(path)
        for item in current_list:
            current_path = "{}/{}".format(path, item)
            content = open(current_path, "rb").read()
            # 用于非匹配字母，数字或下划线字符进行分割
            list_tokens = re.split('\W', str(content))
            words = [tok.lower() for tok in list_tokens if len(tok) > 2]
            if "ham" in path:
                ham_list.append(words)
            else:
                spam_list.append(words)
    return ham_list, spam_list


def create_vecab_list(*kwargs):
    """
    使用训练集创建词向量
    :param ham_list:
    :param spam_list:
    :return:
    """
    tokens = set([])
    for data in kwargs:
        print(data)
        for item in data:
            tokens = tokens | set(item)
    return list(tokens)


def word_to_vec(token_list, input_set):
    """
    对输入的分词集合转换成词向量
    :param word_vec:
    :param input_set:
    :return:
    """
    vec = np.zeros((len(input_set), len(token_list)))
    for index, input_item in enumerate(input_set):
        for word in input_item:
            if word in token_list:
                vec[index, token_list.index(word)] += 1
    return vec


def train_nb0(train_vec_mat):
    """
    计算类别下的条件概率
    :param train_vec_mat:
    :param train_category:
    :return:
    """
    num_docs = len(train_vec_mat)
    num_words = len(train_vec_mat[0])
    # 降低概率连乘，其中一项为0影响
    p = np.ones(num_words)
    p_sum = 2
    for index in range(num_docs):
        p += train_vec_mat[index]
        p_sum += sum(train_vec_mat[index])
    return p / p_sum


def classify_nb(test_vec, ham_vec, spam_vec, ham_class_p):
    """

    :param test_vec:
    :param ham_vec:
    :param spam_vec:
    :param ham_class_p:
    :return:
    """
    ham_p = np.sum(test_vec * ham_vec) + np.log(ham_class_p)
    spam_p = np.sum(test_vec * spam_vec) + np.log(1 - ham_class_p)
    if ham_p > spam_p:
        return "ham"
    else:
        return "spam"


if __name__ == '__main__':
    # 加载所有邮件，返回正常邮件列表和垃圾邮件列表
    ham_list, spam_list = load_data()
    # 定义一个文档索引量
    all_index = np.arange(25)
    # 随机抽取20个索引作为训练集的索引
    train_index = np.random.choice(all_index, 20, False)
    # 去差集得到测试集索引
    test_index = list(set(all_index) - set(train_index))
    train_ham_list = [ham_list[index] for index in train_index]
    train_spam_list = [spam_list[index] for index in train_index]
    token_list = create_vecab_list(train_ham_list, train_spam_list)
    ham_tokens = word_to_vec(token_list, train_ham_list)
    spam_tokens = word_to_vec(token_list, train_spam_list)
    train_ham_vec = train_nb0(ham_tokens)
    train_spam_vec = train_nb0(spam_tokens)
    ham_p = len(ham_list) / (len(ham_list) + len(spam_list))
    # 测试
    for test_clazz in ["ham", "spam"]:
        for index in test_index:
            if test_clazz == "ham":
                test_doc = ham_list[index]
                p = ham_p
            else:
                test_doc = spam_list[index]
                p = 1 - ham_p
            test_vec = word_to_vec(token_list, [test_doc])
            p = classify_nb(test_vec[0], train_ham_vec, train_spam_vec, p)
            print('测试真实类别', test_clazz, "预测类别", p)
