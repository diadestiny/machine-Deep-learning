from sklearn import neighbors
from sklearn import datasets

# knn算法
knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()

print(iris)
# iris.data代表数据集, iris.target代表每个实例对应的对象是哪一类
knn.fit(iris.data, iris.target)

predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print("predictedLabel:"+str(predictedLabel))
