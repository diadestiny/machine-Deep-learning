#!/usr/bin/python
# -*- coding:utf-8 -*-

# 每个图片8x8  识别数字：0,1,2,3,4,5,6,7,8,9

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split

digits = load_digits()
# print(digits.data.shape)
# import  pylab as pl
# pl.gray()
# pl.matshow(digits.images[0])
# pl.show()

X = digits.data
y = digits.target
X -= X.min()  # normalize the values to bring them into the range 0-1
X /= X.max()

# input layer 64=8*8  output layer 10=(0~9) hidden layer 一般多于input layer
nn = NeuralNetwork([64, 100, 10], 'logistic')
X_train, X_test, y_train, y_test = train_test_split(X, y)
# 转化为numpy的输入要求: 8 --> 0 0 0 0 0 0 0 1 0
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)
print("start fitting")
nn.fit(X_train, labels_train, epochs=5000)
predictions = []
# 测试集的行数X_test.shape[0]
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))

# 位于对角线的是预测成功的次数,一行中别的位置对应预测成该位置对应数字的错误次数
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

