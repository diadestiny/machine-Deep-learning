from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
import numpy as np
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'AllElectronics.csv', 'rt')
reader = csv.reader(allElectronicsData)
headers = next(reader)
# print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

for i in featureList:
    print(i)

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print(vec.get_feature_names())
print("X："+str(dummyX))

print("labelList: " + str(labelList))

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("Y："+str(dummyY))

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))

# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
# newRowX[0] = 1
# newRowX[2] = 0
print("newRowX: " + str(newRowX))

new_array = np.array(newRowX).reshape(1, -1)

predictedY = clf.predict(new_array)
print("predictedY: " + str(predictedY))
