import pandas as pd
import numpy as np

def rbf(X, sigma = 0.3):
	X = np.array(X)
	m,n = X.shape
	K = np.zeros((m,m))
	for i in range(0,m):
		for j in range(0,m):
			temp = X[i] - X[j]
			temp = temp[None]
			dist = temp.dot(temp.T)
			K[i,j] = np.exp(-1*dist/(2*(sigma**2)))
	K = np.c_[np.ones((m,1)) , K]
	return K


def gaussianSVM(X_train, y_train, reg, L_rate, sigma=0.3, epsilon = 1):
    X_train = np.array(X_train)
    m,n = X_train.shape
    y_train = np.array(y_train)
    y_train = y_train[None]
    y_train = y_train.T
    y_train[y_train == 0] = -1
    K_train = rbf(X_train, sigma)                  # rbf kernel
    theta = np.zeros((m+1,1))
    cost = 1000
    for i in range(100):
        der = np.zeros((m+1,1))
        

       # Train Theta
        for j in range(m) :
            k = K_train[j]
            k = k[None]
            z = k.dot(theta)
            if (y_train[j]*z < 1) :
                der = der + theta - reg*y_train[j]*k.T
            else :
                der = der + theta
        print ('der ={}'.format(der))

        # Update Theta
        theta = theta - L_rate*der/m
        print ('theta={}'.format(theta))

       
        # FInd cost for updated Theta
        Z = K_train.dot(theta)
        cost = 0
        cost = cost + reg * np.sum(np.maximum(np.zeros((m,1)), 1-np.multiply(y_train.T,Z)))
        cost = cost + (theta.T).dot(theta)
        print ( "Cost = ",cost)
   

    # Return trained Theta
    return theta

def predict(X_test, theta, sigma = 0.3):

	X_test = np.array(X_test)
	m,n = X_test.shape
	K_test = rbf(X_test, sigma)
	y_test = K_test.dot(theta)
	
	for i in range(m):
		if y_test[i] >= 0 :
			y_test[i] = 1
		else :
			y_test[i] = 0

	return y_test




