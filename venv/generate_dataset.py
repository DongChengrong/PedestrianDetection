import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def load_data_set(logger):
    '''
    导入数据集
    :param logger: 日志信息打印模块
    :return pos: 正样本文件名的列表
    :return neg: 负样本文件名的列表
    :return test: 测试数据集文件名的列表。
    '''
    path = os.getcwd()
    logger.info('Checking data path!')
    logger.info('Current path is:{}'.format(path))

    # 提取正样本
    pos_dir = os.path.join(path, 'train_data/Positive')
    if os.path.exists(pos_dir):
        logger.info('Positive data path is:{}'.format(pos_dir))
        pos = os.listdir(pos_dir)
        logger.info('Positive samples number:{}'.format(len(pos)))

    # 提取负样本
    neg_dir = os.path.join(path, 'train_data/Negative')
    if os.path.exists(neg_dir):
        logger.info('Negative data path is:{}'.format(neg_dir))
        neg = os.listdir(neg_dir)
        logger.info('Negative samples number:{}'.format(len(neg)))
    # 提取测试集
    '''
    test_dir = os.path.join(path, 'train_data/TestData')
        if os.path.exists(test_dir):
        logger.info('Test data path is:{}'.format(test_dir))
        test = os.listdir(test_dir)
        logger.info('Test samples number:{}'.format(len(test)))
    '''

    return pos, neg

def generate(logger):
    '''
        生成训练及测试数据集
        :param logger: 日志信息打印模块
        :return pos: 正样本文件名的列表
        :return neg: 负样本文件名的列表
        :return test: 测试数据集文件名的列表。
    '''
    pos, neg = load_data_set(logger)
    pos_train, pos_test, neg_train, neg_test = train_test_split(pos, neg, test_size=0.1, random_state=42)
    test = pos_test + neg_test
    return pos, neg, test
