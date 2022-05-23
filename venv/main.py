# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press 双击 Shift to search everywhere for classes, files, tool windows, actions, and settings.

# *_*coding:utf-8 *_*

import os
import sys
from typing import List
from generate_dataset import generate

import cv2
import logging
import numpy as np

from Smote import Smote


def logger_init():
    '''
    自定义python的日志信息打印配置
    :return logger: 日志信息打印模块
    '''

    # 获取logger实例，如果参数为空则返回root logger
    logger = logging.getLogger("PedestranDetect")

    # 指定logger输出格式
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

    # 文件日志
    # file_handler = logging.FileHandler("test.log")
    # file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter  # 也可以直接给formatter赋值

    # 为logger添加的日志处理器
    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 指定日志的最低输出级别，默认为WARN级别
    logger.setLevel(logging.INFO)

    return logger

def load_train_samples(pos, neg):
    '''
    合并正样本pos和负样本pos，创建训练数据集和对应的标签集
    :param pos: 正样本文件名列表
    :param neg: 负样本文件名列表
    :return samples: 合并后的训练样本文件名列表
    :return labels: 对应训练样本的标签列表
    '''

    path = os.getcwd()
    pos_dir = os.path.join(path, 'train_data/Positive')
    neg_dir = os.path.join(path, 'train_data/Negative')

    print(pos_dir)
    samples = []
    labels = []
    for f in pos:
        file_path = os.path.join(pos_dir, str(f))
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(1.)

    for f in neg:
        file_path = os.path.join(neg_dir, str(f))
        if os.path.exists(file_path):
            samples.append(file_path)
            labels.append(-1.)

    # labels 要转换成numpy数组，类型为np.int32
    labels = np.int32(labels)
    labels_len = len(pos) + len(neg)
    labels = np.resize(labels, (labels_len, 1))

    return samples, labels

def extract_hog(samples, logger):
    '''
    从训练数据集中提取HOG特征，并返回
    :param samples: 训练数据集
    :param logger: 日志信息打印模块
    :return train: 从训练数据集中提取的HOG特征
    '''
    train = []
    logger.info('Extracting HOG Descriptors...')
    num = 0.
    total = len(samples)
    for f in samples:
        num += 1.
        logger.info('Processing {} {:2.1f}%'.format(f, num/total*100))
        hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
        # hog = cv2.HOGDescriptor()
        img = cv2.imread(f, -1)
        img = cv2.resize(img, (64,128))
        descriptors = hog.compute(img)
        logger.info('hog feature descriptor size: {}'.format(descriptors.shape))    # (3780, 1)
        train.append(descriptors)

    train = np.float32(train)
    train = np.resize(train, (total, 3780))

    return train

def get_svm_detector(svm):
    '''
    导出可以用于cv2.HOGDescriptor()的SVM检测器，实质上是训练好的SVM的支持向量和rho参数组成的列表
    :param svm: 训练好的SVM分类器
    :return: SVM的支持向量和rho参数组成的列表，可用作cv2.HOGDescriptor()的SVM检测器
    '''
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)

def train_svm(train, labels, logger):
    '''
    训练SVM分类器
    :param train: 训练数据集
    :param labels: 对应训练集的标签
    :param logger: 日志信息打印模块
    :return: SVM检测器（注意：opencv的hogdescriptor中的svm不能直接用opencv的svm模型，而是要导出对应格式的数组）
    '''
    logger.info('Configuring SVM classifier.')
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)

    logger.info('Starting training svm.')
    svm.train(train, cv2.ml.ROW_SAMPLE, labels)
    logger.info('Training done.')

    pwd = os.getcwd()
    model_path = os.path.join(pwd, 'svm.xml')
    svm.save(model_path)
    logger.info('Trained SVM classifier is saved as: {}'.format(model_path))

    return get_svm_detector(svm)

def test_hog_detect(test, svm_detector, logger):
    '''
    导入测试集，测试结果
    :param test: 测试数据集
    :param svm_detector: 用于HOGDescriptor的SVM检测器
    :param logger: 日志信息打印模块
    :return: 无
    '''
    test_dir = "C:/Users/qazxs/PycharmProjects/PedestrianDetection/venv/train_data/"
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(svm_detector)
    # opencv自带的训练好了的分类器
    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cv2.namedWindow('Detect')
    test_class = []
    print(len(test))
    for f in test:
        #根据文件扩展名得到完整的测试文件路径
        #print(f.split('.')[1])
        if f.split('.')[1] == "ppm":
            file_path = test_dir + "Positive/" + f
        else:
            file_path = test_dir + "Negative/" + f
        logger.info('Processing {}'.format(file_path))
        img = cv2.imread(file_path)
        rects, _ = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)
        if rects != ():
            test_class.append(1)
        else:
            test_class.append(0)
        '''
        for (x,y,w,h) in rects:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.imshow('Detect', img)
        '''
    cv2.destroyAllWindows()
    return test_class


if __name__ == '__main__':

    logger = logger_init()
    pos, neg, test = generate(logger=logger)
    samples, labels = load_train_samples(pos, neg)
    train = extract_hog(samples, logger=logger)
    logger.info('Size of feature vectors of samples: {}'.format(train.shape))
    logger.info('Size of labels of samples: {}'.format(labels.shape))
    svm_detector = train_svm(train, labels, logger=logger)
    test_class = test_hog_detect(test, svm_detector, logger)
    print(test_class)
