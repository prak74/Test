import numpy as np
import pandas as pd

def LogisticRegression(X,Y,LearningRate,Termination,Reg=0):


    X=np.array(X)
    Y=np.array(Y)
    m,n=X.shape
    Y.reshape(m,1)
    theta=np.zeros(n)
    thata=theta.reshape(n,1)
    cost=10**5
    # LearningRate=0.01

    while(cost>Termination):

        temp=X.dot(theta)-Y
        hx=(LearningRate*(X.T).dot(temp))/m
        theta[0]=theta[0]-hx[0]
        theta[1:]=theta[1:]*(1-Reg/m)-hx
        cost_prev=cost

        cost=((temp.T).dot(temp)+Reg*(theta.T).dot(theta))/(2*m)
        print(cost)
        if(cost-cost_prev<Termination):
            return theta
        if(cost_prev>cost):
            LearningRate=LearningRate/2

    return theta

def predict(X_train,theta):
    m,n=X_train.shape
    X_train=np.array(X_train)
    theta=np.array(theta)
    theta.reshape(n,1)
    return X_train.dot(theta)






