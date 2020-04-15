import numpy as np
import numpy.matlib
import pandas as pd
from sklearn import svm

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
    dataset["CARRIER"]=dataset["CARRIER"].map(carriers)
    dataset["CARRIER"]=dataset["CARRIER"]/7
    dataset["Flight Status"]=dataset["Flight Status"].map(answer)
    dataset["DEST"]=dataset["DEST"].map(destinations)
    dataset["ORIGIN"]=dataset["ORIGIN"].map(origins)
    dataset["CRS_DEP_TIME"]=dataset["CRS_DEP_TIME"]/1400
    dataset["DEP_TIME"]=dataset["DEP_TIME"]/1400
    dataset["DISTANCE"]=(dataset["DISTANCE"]-200)/30
    dataset["DAY_WEEK"]=dataset["DAY_WEEK"]/7
    dataset["DAY_OF_MONTH"]=dataset["DAY_OF_MONTH"]/30
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

theta=2*(np.matlib.rand(11,1)-0.5)
r=0.1 #vary and check
for j in range(100):
    for i in range(X_train_mat.shape[0]):
        x=X_train_mat[i,:]
        x=x.reshape(1,11)
        z=np.matmul(x,theta)
        if z>=0:
            a=1
        else:
            a=0
        if not a==y_train_mat[i]:
            if a==0:
                theta=theta+r*x.transpose()
            else:
                theta=theta-r*x.transpose()
cnt11=0
cnt10=0
cnt01=0
cnt00=0

for i in range((y_test_mat.shape[0])):
    z_pred=np.matmul(X_test_mat[i,:].reshape(1,11),theta)
    if z_pred>=0:
        y_pred=1
    else:
        y_pred=0
    if y_pred==y_test_mat[i] and y_pred==1:
        cnt11+=1
    elif y_pred==y_test_mat[i] and y_pred==0:
        cnt00+=1
    elif y_pred==1:
        cnt10+=1
    else:
        cnt01+=1
        
print(cnt11,"   ",cnt10)
print(cnt01,"   ",cnt00)
a=cnt11/(cnt11+cnt10)
b=cnt11/(cnt11+cnt01)
f1=2*a*b/(a+b)
print(f1)
