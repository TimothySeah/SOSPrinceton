import numpy as np

class DCA:
	def __init__(self, rho=None, rho_p=None, n_components=None):
		self.n_components = n_components
		self.rho = rho
		self.rho_p = rho_p
	
	def Smatrices(self, X,y):
		Sw = np.zeros((X.shape[1],X.shape[1]))
		Sb = np.zeros((X.shape[1],X.shape[1]))
		
		mu = np.zeros(X.shape[1])
		classMu = {}
		classN = {}
		for i in range(len(X)):
			l = y[i]
			x = X[i]
			if l in classMu:
				classMu[l] = classMu[l] + x
				classN[l] = classN[l] + 1.0
			else:
				classMu[l] = x
				classN[l] = 1.0
			mu += x
			
		mu /= len(X)
		
		for l in classMu:
			classMu[l] = classMu[l] / classN[l]
			foo = classMu[l] - mu
			Sb += classN[l]*np.outer(foo,foo)
		
		for i in range(len(X)):
			x = X[i]
			l = y[i]
			foo = x - classMu[l]
			bar = np.outer(foo,foo)
			Sw += bar
		
		return (Sw,Sb)
	
	def fit(self, X, y):
		
		(self.Sw, self.Sb) = self.Smatrices(X,y)
		(w,v) = np.linalg.eig(self.Sw)
		
		if self.rho == None:
			self.rho = 0.02*max(w)
		if self.rho_p == None:
			self.rho_p = 0.1*self.rho
		
		pSw = self.Sw + self.rho*np.eye(self.Sw.shape[0])
		pSbar = self.Sb + self.Sw + (self.rho_p+self.rho)*np.eye(self.Sw.shape[0])
		pSwInv = np.linalg.inv(pSw)
		
		Ddca = np.inner(pSwInv,pSbar.T)
		(lmda,V) = np.linalg.eig(Ddca) 
		## columns of V are the eigen vectors  
		
		# sort eigen vectors
		idx = np.argsort(lmda)
		ordered_idx = idx[::-1]
		
		# get principle components in each row
		U = V.T[ordered_idx]
		
		# scale to satisfy the unity constraint
		alpha2 = np.inner(np.inner(U, pSw.T),U)
		alpha2 = np.diag(np.diag(alpha2))
		alpha = np.sqrt(alpha2)
		invAlpha = np.linalg.inv(alpha)
		Wdca = np.inner(invAlpha.T,U.T)
		
		self.eigVal = lmda[ordered_idx]
		self.allComponents = Wdca
		if self.n_components:
			self.components = Wdca[0:self.n_components]
		else:
			self.components = Wdca
	
		
	def transform(self, X):
		X_trans = np.inner(self.components,X)
		return X_trans.T
	
	def reconstruction_error(self):
		
		if self.n_components:
			Vmaj = self.allComponents[0:self.n_components]
			Vmin = self.allComponents[self.n_components:]
			Lmaj = self.eigVal[0:self.n_components]
			Lmin = self.eigVal[self.n_components:]
					
			pSwInv = np.linalg.inv(self.Sw + self.rho*np.eye(self.Sw.shape[0]))
		
			Sw2 = np.inner(pSwInv,pSwInv.T)
			foo = np.inner(np.inner(Vmin,Sw2.T),Vmin)
			bar = np.inner(foo,np.diag(Lmin).T)
			re = np.trace(bar)
		else:
			re = 0.0
		

		return re
		
		