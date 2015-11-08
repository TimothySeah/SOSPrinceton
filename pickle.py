#!/usr/bin/python2

import numpy as np
from sklearn import mixture
from sklearn import svm
from sklearn import preprocessing
from sklearn import multiclass
from sklearn import metrics
from sklearn import cross_validation
from matplotlib.pyplot import *

def loadData(dataset):
	
	X = np.genfromtxt(dataset + '_X.dat')
	y = np.genfromtxt(dataset + '_y.dat')
	return (X,y)

def generatePublicVector(X):
	mu = np.mean(X,axis=0)
	sigma = np.cov(X.T)
	Z = gaussianMixtureSamples([mu],[sigma],samples=3*X.shape[1])
	return Z

def gaussianMixtureSamples(centroids,ccov,mc=None,samples=1):
	cc = centroids
	D = len(cc[0])
	# Check if inputs are ok:
	K = len(cc)
	if mc is None: # Default equally likely clusters
		mc = np.ones(K) / K
	if len(ccov) != K:
		raise ValueError, "centroids and ccov must contain the same number" +"of elements."
	if len(mc) != K:
		raise ValueError, "centroids and mc must contain the same number" +"of elements."

    # Check if the mixing coefficients sum to one:
	EPS = 1E-15
	if np.abs(1-np.sum(mc)) > EPS:
		raise ValueError, "The sum of mc must be 1.0"

    # Cluster selection
	cs_mc = np.cumsum(mc)
	cs_mc = np.concatenate(([0], cs_mc))
	sel_idx = np.random.rand(samples)

    # Draw samples
	res = np.zeros((samples, D))
	for k in range(K):
		idx = (sel_idx >= cs_mc[k]) * (sel_idx < cs_mc[k+1])
		ksamples = np.sum(idx)
		drawn_samples = np.random.multivariate_normal(cc[k], ccov[k], ksamples)
		res[idx,:] = drawn_samples
	return res

def generateUniformSample(low,high):
	sample = np.zeros(len(low))
	for i in range(len(low)):
		foo = np.random.uniform(low[i],high[i])
		sample[i] = foo
	return sample


def generateUserPrivacyParameters(Z, dim, userData, alpha=0.3, dist='Gaussian'):
	M = Z.shape[1]
	sigmaZ = np.cov(Z.T)
	epsilon = np.random.multivariate_normal(np.zeros(M),alpha*sigmaZ)
	Ru = np.array([]) 
	## note that other matrix here is the transposed version of the ones in the paper, except Ru. Ru has the same dim as in the paper.
	if dist == 'Gaussian':
		Ru = np.random.normal(0,1,(dim,M))
	else:
		Ru = np.random.uniform(0,1,(dim,M))
	
	Zu = np.inner((Z+epsilon),Ru)
	Xu = np.inner((userData+epsilon),Ru)
	return (Ru,epsilon,Zu,Xu)


def regression(Zu,Zc):
	invQ = np.linalg.pinv(Zu)
	theta = np.inner(invQ,Zc.T)
	return theta

def reconstruction(Xu,theta):
	Xhat = np.inner(Xu,theta.T)
	return Xhat

def reconstructionError(X,Xhat):
	# 2-norm
	diff = Xhat-X
	foo = np.inner(diff,diff)
	re_2norm = np.sqrt(np.diag(foo))
	
	(re_rmse,re_R2) = rmseAndR2(Xhat,X)
	return (re_2norm,re_rmse,re_R2)
	

def rmseAndR2(Xhat,Xtrue):
	xhat = Xhat.T
	xtrue = Xtrue.T
	sum_y2 = 0
	sum_yp = 0
	sum_p2 = 0
	sum_y = 0
	n = 0

	for i in range(len(xtrue)):
		sum_y2 += xtrue[i] * xtrue[i]
		sum_yp += xtrue[i] * xhat[i]
		sum_p2 += xhat[i] * xhat[i]
		sum_y += xtrue[i]
		n += 1
	
	R2 = 1 - ((sum_y2 - 2*sum_yp + sum_p2)/(sum_y2 - (sum_y*sum_y)/n))
	rmse = np.sqrt((sum_y2 - 2*sum_yp + sum_p2)/n)
	
	return (rmse,R2)

def ovrSVM(X,y,svmKernel):
	labelValues = range(int(min(y)),int(max(y))+1)
	y2 = preprocessing.label_binarize(y,classes=labelValues)
	clf = multiclass.OneVsRestClassifier(svm.SVC(kernel=svmKernel, probability=True))
	clf.fit(X,y2)
	return clf


def performance(prediction, target):
	acc = metrics.accuracy_score(target, prediction, normalize=True)
	return acc
	

def randomSplit(X,y,user,svmKernel='rbf',perturb = True, dim = 10, noiseInt = 0.3,RDist = 'Gaussian'):
	
	accuracy = np.array([])
	re = np.array([])

	for i in range(20):
		# leave 20% out for testing
		skf = cross_validation.StratifiedKFold(user,n_folds=5,shuffle=True)

		for cv_i,test_i in skf:
			train_user = user[cv_i]
			train_X = X[cv_i]
			train_y = y[cv_i]
			
			if perturb:
					
				# 1. the cloud creates public vectors
				Z = generatePublicVector(train_X)
				
				# 2. users perturb the pub vectors and the training vectors
				(Ru, epsilon, Zu, Xu) = generateUserPrivacyParameters(Z,dim,train_X,alpha=noiseInt,dist = RDist)
				
				# 3. regression by the cloud
				theta = regression(Zu, Z)
				
				# 4. the cloud reconstructs the training vectors
				Xhat = reconstruction(Xu, theta)
				
			else:
				Xhat = train_X
			
			
			# do training here
			clf = ovrSVM(Xhat,train_y,svmKernel)
			
			
			
			test_user = user[test_i]
			test_X = X[test_i]
			test_y = y[test_i]
			# do testing here
			prediction = clf.predict(test_X)
			labelValues = range(int(min(y)),int(max(y))+1)
			test_y2 = preprocessing.label_binarize(test_y,classes=labelValues)		
			
			
			# record performance
			foo = performance(prediction, test_y2)
			accuracy = np.append(accuracy, foo)
			(twoNorm,rmse,r2) = reconstructionError(train_X,Xhat)
			re = np.append(re,twoNorm)
			
			break #use only one test set and then re-shuffle
	
	mean_acc = np.mean(accuracy)
	mean_re = np.mean(re)
	return (mean_acc, mean_re)
		


def main():
	(X, y) = loadData('wine')
	user = np.zeros(y.shape)
	
	# no privacy case
	(baseline_acc, baseline_re) = randomSplit(X,y,user,perturb=False)
	
	# with diff dimensions
	dimensions = (10,7,5,2)
	accuracy1 = np.array([])
	re1 = np.array([])
	for d in dimensions:
		(foo, bar) = randomSplit(X,y,user,perturb = True, dim = d)
		accuracy1 = np.append(accuracy1,foo)
		re1 = np.append(re1, bar)
	
	plot(dimensions,(1-baseline_acc+accuracy1)*100)
	xlim(10,2)
	gca().yaxis.grid(True)
	title('Reduction in accuracy vs dimension')
	ylabel('% accuracy')
	xlabel('dimension')
	show()
	
	plot(dimensions,re1)
	xlim(10,2)
	gca().yaxis.grid(True)
	title('Reconstruction error vs dimension')
	ylabel('Reconstruction error')
	xlabel('dimension')
	show()
     
    # with diff noise intensity
	alphas = (0.1, 0.3, 0.5, 0.75, 1)
	accuracy2 = np.array([])
	re2 = np.array([])
	for a in alphas:
		(foo, bar) = randomSplit(X,y,user,perturb = True, dim = 10, noiseInt=a)
		accuracy2 = np.append(accuracy2,foo)
		re2 = np.append(re2, bar)
	
	plot(alphas,(1-baseline_acc+accuracy2)*100)
	gca().yaxis.grid(True)
	title('Reduction in accuracy vs noise intensity')
	ylabel('% accuracy')
	xlabel('noise intensity')
	show()
	
	plot(alphas,re2)
	gca().yaxis.grid(True)
	title('Reconstruction error vs noise intensity')
	ylabel('Reconstruction error')
	xlabel('noise intensity')	
	show()
	
	
	
if __name__ == "__main__":
	main()

	