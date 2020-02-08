from numpy import genfromtxt
from sklearn import linear_model
import numpy as np

data = genfromtxt(r"Delivery_Dummy.csv", delimiter=",")

x = data[:, :-1]
y = data[:, -1]
print(x)
print(y)

mlr = linear_model.LinearRegression()

mlr.fit(x, y)

print(mlr)
print("coef:")
print(mlr.coef_)
print("intercept")
print(mlr.intercept_)

xPredict = [90, 2, 0, 0, 1]
yPredict = mlr.predict(np.array(xPredict).reshape(1, -1))

print("predict:")
print(yPredict)
