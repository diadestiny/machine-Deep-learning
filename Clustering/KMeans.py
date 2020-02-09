#! encoding:utf-8
import numpy as np


def kmeans(X, k, maxIt):  # X为矩阵 k：所需划分的块数 MaxIt：迭代最多的次数
    numPoints, numDim = X.shape  # numPoints,numDim该矩阵的行列数
    dataSet = np.zeros((numPoints, numDim + 1))  # dataSet 比 X 多出来一列 其代表类别
    dataSet[:, :-1] = X  # 除了最后一列，赋值为X

    centroids = dataSet[np.random.randint(numPoints, size=k), :]  # centroids 代表中心点（随机）
    # centroids = dataSet[0:2, :]#挑选前两个
    centroids[:, -1] = range(1, k + 1)  # 2  附上类别
    iterations = 0  # 迭代更新次数
    oldCentroids = None  # 旧的类别
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):  # 4
        print("iteration: \n", iterations)
        print("dataSet: \n", dataSet)
        print("centroids: \n", centroids)
        oldCentroids = np.copy(centroids)  # 复制矩阵   此处不能直接用等于  因为如果是等于  则oldCentroids会随着Centroids变而变
        iterations += 1
        updateLabels(dataSet, centroids)  # 更新类别
        centroids = getCentroids(dataSet, k)  # 获取新的中心点

    return dataSet


def shouldStop(oldCentroids, centroids, iterations, maxIt):  # 终止条件
    if iterations > maxIt:  # 1、迭代次数大于规定的次数
        return True
    return np.array_equal(oldCentroids, centroids)  # 2、两个矩阵的数值是一样的


def updateLabels(dataSet, centroids):
    numPoints, numDim = dataSet.shape

    for i in range(numPoints):
        dataSet[i, -1] = getLabelFromClosestCentroid(dataSet[i, :-1], centroids)


def getCentroids(dataSet, k):
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k + 1):
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]  # 此处很精妙
        # 新的中心点的坐标值
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)  # 求均值  axis=0对 行 求均值 即是一行
        # 新的中心点的标签
        result[i - 1, -1] = i
    return result


def getLabelFromClosestCentroid(setRow, centroids):
    lable = centroids[0, -1]
    mindis = np.linalg.norm(setRow - centroids[0, :-1])  # numpy 里面求欧式距离

    for i in range(1, centroids.shape[0]):  # 求离中心点近的
        dis = np.linalg.norm(setRow - centroids[i, :-1])
        if dis < mindis:
            mindis = dis
            lable = centroids[i, -1]
    return lable


# (weightIndex,PhFeature)无标签(对应的y)
x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 5])
TestX = np.vstack((x1, x2, x3, x4))

data = kmeans(TestX, 2, 10)
print("final result")
print(data)
