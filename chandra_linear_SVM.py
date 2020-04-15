

if(__name__=="__main__"):

    import numpy as np
    import pandas as pd


    def Train_SVM(X_train,Y_train,L_rate,term,reg):
        X_train=np.array(X_train)
        m,n=X_train.shape
        X_train=X_train.append(np.ones(m).reshape(m,1),axis=1) ##assuming ones are not explicitly added by the user
        Y_train=np.array(Y_train)
        Y_train.reshape(n,1)
        theta=np.zeros(n+1,1)
        cost = 1000
        while(cost > term):
            dw=np.zeros(n+1,1)
            for i in range(m):
                margin=1-y[i]*(X_train[i].dot(theta))
                if(max(margin,0)==0):
                    dw+=theta
                else:
                    dw+=theta-reg*y[i]*(X_train[i].T)
            theta=theta-L_rate*dw/m
            cost=0
            for i in range(m):
                cost+=max(0,1-y[i][0]*(X_train[i].dot(theta)))
            cost+=(theta.T).dot(theta)/2
            cost/=m;

        return theta






    def predict(X_test,theta,bias=+1):
        X_test=np.array(X_test)
        Y_test=X_test*theta
        if(bias==1):
            Y_test[x>=0,1]
            Y_test[x<0,-1]
        else:
            Y_test[x>0,1]
            Y_test[x<=0,-1]


