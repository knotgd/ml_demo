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
    # 降低概率连乘，其中一项为0影响
    vec = np.ones((len(input_set), len(token_list)))
    for index, input_item in enumerate(input_set):
        for word in input_item:
            if word in token_list:
                vec[index, token_list.index(word)] += 1
    return vec


def tarin_nb0():
    pass


if __name__ == '__main__':
    ham_list, spam_list = load_data()
    token_list = create_vecab_list(ham_list, spam_list)
    vec = word_to_vec(token_list, ham_list)
    print(vec)
    # all_index = np.arange(25)
    # train_index = np.random.choice(all_index, 20, False)
    # test_index = list(set(all_index) - set(train_index))
