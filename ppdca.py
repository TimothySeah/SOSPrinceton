#!/usr/bin/python2

import numpy as np
from sklearn import mixture
from sklearn import svm
from sklearn import preprocessing
from sklearn import multiclass
from sklearn import metrics
from sklearn import cross_validation
from matplotlib.pyplot import *
from DCA import *

def loadData(dataset):
	
	X = np.genfromtxt(dataset + '_X.dat')
	y = np.genfromtxt(dataset + '_y.dat')
	return (X,y)

def generatePublicVector(X,y):
	classIdx = {}
	
	for i in range(len(X)):
		l = y[i]
		x = X[i]
		if l in classIdx:
			foo = classIdx[l]
			foo = np.append(foo, i)
			classIdx[l] = foo
		else:
			foo = np.array([i])
			classIdx[l] = foo
	bar = {}
	for l in classIdx:
		x = X[classIdx[l]]
		
		mu = np.mean(x,axis=0)
		sigma = np.cov(x.T)
		bar[l] = gaussianMixtureSamples([mu],[sigma],samples=3*x.shape[1])
	Z = np.array([])
	yZ = np.array([])
	for l in bar:
		if len(Z) == 0:
			Z = bar[l]
		else:
			Z = np.vstack((Z,bar[l]))
		foo = l*np.ones(len(bar[l]))
		yZ = np.append(yZ,foo)
	return (Z,yZ)

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


def regression(Zu,Zc):
	invQ = np.linalg.pinv(Zu)
	theta = np.inner(invQ,Zc.T)
	return theta


def generateUserPrivacyParameters(X, y,dim, rho=None, rho_p=None, alpha=0.3):
	M = X.shape[1]
	sigmaX = np.cov(X.T)
	epsilon = np.random.multivariate_normal(np.zeros(M),alpha*sigmaX)
	dca = DCA(rho=rho, rho_p=rho_p, n_components=dim)
	dca.fit(X,y)
	Xu = dca.transform(X)

	return (Xu,epsilon,dca)

def reconstructionAttack(X,y,Xu,dim):
	(Z,yZ) = generatePublicVector(X,y)
	rhoZ = np.random.normal(0,1)
	rho_pZ = np.random.normal(-1,1)
	dca = DCA(rho=rhoZ,rho_p=rho_pZ,n_components = dim)
	dca.fit(Z,yZ)
	invW = np.linalg.pinv(dca.components)
	Xhat = np.inner(Xu,invW)
	return Xhat
	

def reconstruction(Xu,theta):
	Xhat = np.inner(Xu,theta.T)
	return Xhat

def alphaSpear(dca):
	re = dca.reconstruction_error()
	return re

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
	clf = multiclass.OneVsRestClassifier(svm.SVC(C = 100000,kernel=svmKernel, probability=True))
	clf.fit(X,y2)
	return clf


def performance(prediction, target):
	acc = metrics.accuracy_score(target, prediction, normalize=True)
	return acc



def randomSplit(X,y,user,svmKernel='rbf',perturb = True, dim = 10, rho = None, rho_p = None, noiseInt = 0.1):
	
	accuracy = np.array([])
	re = np.array([])
	re_alpha = np.array([])

	for i in range(20):
		# leave 20% out for testing
		skf = cross_validation.StratifiedKFold(user,n_folds=5,shuffle=True)

		for cv_i,test_i in skf:
			train_user = user[cv_i]
			train_X = X[cv_i]
			train_y = y[cv_i]
			
			if perturb:
				(Xu,epsilon,dca) = generateUserPrivacyParameters(train_X,train_y,dim,rho,rho_p,noiseInt)
				
				
			else:
				Xu = train_X
				dim = X.shape[1]
			
			# do training here
			clf = ovrSVM(Xu,train_y,svmKernel)
			
			
			
			test_user = user[test_i]
			test_X = X[test_i]
			test_y = y[test_i]
			# do testing here
			if perturb:
				prediction = clf.predict(dca.transform(test_X))
			else:
				prediction = clf.predict(test_X)
			labelValues = range(int(min(y)),int(max(y))+1)
			test_y2 = preprocessing.label_binarize(test_y,classes=labelValues)		
			
			
			# record performance
			foo = performance(prediction, test_y2)
			accuracy = np.append(accuracy, foo)
			
			
			# for reconstruction error, assume reconstruction attack
			Xhat = reconstructionAttack(train_X,train_y,Xu,dim)
			(twoNorm,rmse,r2) = reconstructionError(train_X,Xhat)
			if perturb:
				re_dca = alphaSpear(dca)
			else:
				re_dca = 0
			re = np.append(re,twoNorm)
			re_alpha = np.append(re_alpha,re_dca)
			
			break #use only one test set and then re-shuffle
	
	mean_acc = np.mean(accuracy)
	mean_re = np.mean(re)
	mean_re_alpha = np.mean(re_alpha)
	return (mean_acc, mean_re, mean_re_alpha)
		


def main():
	(X, y) = loadData('vehicle')
	user = np.zeros(y.shape)
	
	# no privacy case
	(baseline_acc, baseline_re1, baseline_re_alpha) = randomSplit(X,y,user,perturb=False)
	
	# with diff dimensions
	dimensions = (17,15,12,10,8,6,4,2)
	accuracy1 = np.array([])
	re1 = np.array([])
	re_alpha = np.array([])
	for d in dimensions:
		(foo, bar, third) = randomSplit(X,y,user,perturb = True, dim = d,rho=0.001,rho_p = -0.001, noiseInt = 0.1)
		accuracy1 = np.append(accuracy1,foo)
		re1 = np.append(re1, bar)
		re_alpha = np.append(re_alpha,third)
	
	plot(dimensions,accuracy1)
	plot(dimensions,baseline_acc*np.ones(len(dimensions)),'--')
	xlim(10,2)
	gca().yaxis.grid(True)
	title('Accuracy vs Dimension')
	ylabel('accuracy')
	xlabel('dimension')
	show()
	
	plot(dimensions,np.absolute(re1))
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

	