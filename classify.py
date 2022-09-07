# Loading datasets
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()

# Intialing features and labels

f1 = iris.data
l1 = iris.target

# printing description,features and labels of data
# print(features[0], labels[0])
# print(iris.DESCR)  

#  0 - Iris-Setosa
#  1 - Iris-Versicolour
#  2 - Iris-Virginica

clf = KNeighborsClassifier()
clf.fit(f1, l1)

preds = clf.predict([[4.1 ,2.5 ,0.4 ,0.02]])
print(preds) 