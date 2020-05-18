import pandas as pd
import numpy as np

def rbf(X, X_train, sigma = 0.3):
	X = np.array(X)
	m = np.size(X,0)
	n = np.size(X_train,0)
	K = np.zeros((m,n))
	for i in range(0,m):
		for j in range(0,n):
			temp = X[i] - X_train[j]
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
    K_train = rbf(X_train, X_train, sigma)                  # rbf kernel
    theta = np.zeros((m+1,1))
    #cost = 1000
    for i in range(300):
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
        #print ('der ={}'.format(der))
        der[0] = 0
        # Update Theta
        theta = theta - L_rate*der/m
        #print ('theta={}'.format(theta))

       
        # FInd cost for updated Theta
        Z = K_train.dot(theta)
        cost = 0
        cost = cost + reg * np.sum(np.maximum(np.zeros((m,1)), 1-np.multiply(y_train.T,Z)))
        cost = cost + (theta.T).dot(theta) - theta[0]**2
        if i%50 == 0:
        	print ( "Cost after iter ",i," = ",cost)

   

    # Return trained Theta
    return theta

def predict(X_test, X_train, theta, sigma = 0.3):

	X_test = np.array(X_test)
	m,n = X_test.shape
	K_test = rbf(X_test, X_train, sigma)
	y_test = K_test.dot(theta)
	
	for i in range(m):
		if y_test[i] >= 0 :
			y_test[i] = 1
		else :
			y_test[i] = 0

	return y_test


import numpy.matlib
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

db=pd.read_csv("FlightDelays.csv")
train_db=db.sample(frac=0.7, random_state=0)
test_db=db.drop(train_db.index)

train_db=train_db.drop(['FL_DATE'],axis=1)
test_db=test_db.drop(['FL_DATE'],axis=1)
train_db=train_db.drop(['FL_NUM'],axis=1)
test_db=test_db.drop(['FL_NUM'],axis=1)
train_db=train_db.drop(['TAIL_NUM'],axis=1)
test_db=test_db.drop(['TAIL_NUM'],axis=1)

data=[train_db, test_db]
carriers={"OH":0,"DH":1,"DL":2,"MQ":3,"UA":4,"US":5,"RU":6,"CO":7 }
answer={"ontime":0,"delayed":1}
destinations={"JFK":0,"LGA":1,"EWR":2}
origins={"DCA":0, "BWI":1, "IAD":2}

for dataset in data:
    
    # Mapping
    dataset["CARRIER"]=dataset["CARRIER"].map(carriers)
    dataset["Flight Status"]=dataset["Flight Status"].map(answer)
    dataset["DEST"]=dataset["DEST"].map(destinations)
    dataset["ORIGIN"]=dataset["ORIGIN"].map(origins)
    
    mean = dataset.mean(axis = 0)
    
    dataset["CARRIER"]=(dataset["CARRIER"] - mean['CARRIER'])/7    
    dataset["CRS_DEP_TIME"]=(dataset["CRS_DEP_TIME"] - mean['CRS_DEP_TIME'])/1400
    dataset["DEP_TIME"]=(dataset["DEP_TIME"] - mean['DEP_TIME'])/1400
    dataset["DISTANCE"]=(dataset["DISTANCE"] - mean['DISTANCE'])/30
    dataset["DAY_WEEK"]=(dataset["DAY_WEEK"] - mean['DAY_WEEK'])/7
    dataset["DAY_OF_MONTH"]=(dataset["DAY_OF_MONTH"] - mean['DAY_OF_MONTH'])/30
    dataset["DIFFERENCE"]=dataset["CRS_DEP_TIME"]-dataset["DEP_TIME"]
    dataset["BIAS"]=1

X_train=train_db.drop(['Flight Status'],axis=1)
y_train=train_db['Flight Status']
X_test=test_db.drop(['Flight Status'],axis=1)
y_test=test_db['Flight Status']



X_train_mat=X_train.to_numpy()
y_train_mat=y_train.to_numpy()
X_test_mat=X_test.to_numpy()
y_test_mat=y_test.to_numpy()



theta = gaussianSVM(X_train_mat, y_train_mat, 0.01, 1)
pred = predict(X_test_mat, X_train_mat, theta)
print(confusion_matrix(pred,y_test_mat))

svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train_mat,y_train_mat)
pred = svclassifier.predict(X_test_mat)
print(confusion_matrix(pred,y_test_mat))
