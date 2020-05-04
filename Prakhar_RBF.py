import pandas as pd
import numpy as np

def rbf(X, sigma = 0.3):
	X = np.array(X)
	m,n = X.shape
	K = np.zeros(m,m)
	for i in range(0,m):
		for j in range(0,m):
			temp = X[i] - X[j]
			K[i][j] = np.exp(temp.dot(temp.T)/(2*(sigma**2)))
	K = K.append(np.ones(m).reshape(m,1),axis=1)
	return K


def gaussianSVM(X_train, y_train,sigma=0.3, reg, L_rate, epsilon = 1.0e-3):
	X_train = np.array(X_train)
	m,n = X_train.shape
	K_train = rbf(X_train, sigma)			# rbf kernel
	y_train = np.array(y_train)
	y_train.reshape(m,1)
	theta = np.zeros(m+1,1)
	cost = 1000
	while(cost > term):
		der = np.zeros(m+1,1)
		
		# Train Theta
		for i in range(m):
			z = K_train[i].dot(theta)
			if y_train[i]*(max(0,1-z)==0) + (1-y_train[i])*(max(0,1+z)==0) :     # y=1 and max(0,1-z)=0 or y=0 and max(0,1+z) = 0
				der += theta                                                     # then derivative only of regularization term as cost term = 0
			else :																 # else both term derivated
				der += theta - reg*(y_train[i]*K_train[i].T - (1-y_train[i])*K_train[i].T)    
		
		# Update Theta
		theta = theta - L_rate*der/m

		# FInd cost for updated Theta
		Z = K_train.dot(theta)
		cost = 0
		cost += (y_train.T).dot(np.maximum(np.zeros(m,1),1-Z)) + ((1-y_train).T).dot(np.maximum(np.zeros(m,1),1+Z))
		cost += (theta.T).dot(theta)
	
	# Return trained Theta
	return theta

def predict(X_test, theta, sigma = 0.3):

	X_test = np.array(X_test)
	m,n = X_test.shape
	K_test = rbf(X_test, sigma)
	y_test = K_test*theta
	
	for i in range(m):
		if y_test[i] >= 0 :
			y_test[i] = 1
		else :
			y_test[i] = 0

	return y_test




