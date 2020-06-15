# @Time : 2020/6/15 11:40 
# @Author : 大太阳小白
# @Software: PyCharm
"""
模型保存和加载工具
"""
import pickle


def store_tree(input_tree, filename):
    fw = open(filename, 'wb')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)
