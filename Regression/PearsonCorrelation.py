import numpy as np
# from astropy.units import  Ybarn
import math


def computeCorrelation(x, y):
    xBar = np.mean(x)
    ybar = np.mean(y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(x)):  # 多少实例
        diffxxBar = x[i] - xBar
        diffyyBar = y[i] - ybar
        SSR += (diffxxBar * diffyyBar)
        varX += diffxxBar ** 2  # 求平方然后累计起来
        varY += diffyyBar ** 2  # 求平方然后累计起来

    SST = math.sqrt(varX * varY)
    return SSR / SST


def polyfit(x, y, degree):
    result = {}  # 定义一个字典
    coeffs = np.polyfit(x, y, degree)  # 直接求出b0 b1 b2 b3 ..的估计值
    result["polynomial"] = coeffs.tolist()

    p = np.poly1d(coeffs)  # 返回预测值
    yhat = p(x)  # 传入x 返回预测值
    ybar = np.sum(y) / len(y)  # 求均值
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    result["determination"] = ssreg / sstot

    return result


testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]

print("r:", computeCorrelation(testX, testY))
print("r**2:", (computeCorrelation(testX, testY) ** 2))

print("result: ", polyfit(testX, testY, 1))
print("r**2:", polyfit(testX, testY, 1)["determination"])  # degree=1  一次
print(polyfit(testX, testY, 1)["polynomial"])  # 打印除斜率和截距
