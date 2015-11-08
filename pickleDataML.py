import sys
import numpy as np
import sklearn
from sklearn import svm, cluster, grid_search, cross_validation, neighbors

def loadData(dataset):
	X = np.genfromtxt(dataset + '_X.dat')
	y = np.genfromtxt(dataset + '_y.dat')
	return (X,y)

(X,y) = loadData(sys.argv[1])

print(len(X))

# repeat 20 times: split into cross-validation and testing sets
svmSum = 0
knnSum = 0
for i in range(20):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
	
	# evaluate svm
	svmParams = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	svr = svm.SVC()
	svmclf = grid_search.GridSearchCV(svr, svmParams, cv=5)
	svmclf.fit(X_train, y_train)
	svmSum += svmclf.score(X_test, y_test)

	# overfishing: tuple index out of range, reason because of ???
	# evaluate knn
	"""
	knnParams = {'n_neighbors':[4, 5], 'weights':('uniform','distance')}
	knn = neighbors.KNeighborsClassifier(metric='mahalanobis')
	knnclf = grid_search.GridSearchCV(knn, knnParams, cv=5)
	knnclf.fit(X_train, y_train)
	knnSum += knnclf.score(X_test, y_test)
	"""


print "svm accuracy is: " + str(svmSum / 20.0)
print "knn accuracy is: " + str(knnSum / 20.0)