import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
data_url="https://raw.githubusercontent.com/codeLAlit/MLGYM/master/FlightDelays.csv"
data=pd.read_csv(data_url)
y=np.array((data['Flight Status']=='ontime')*1)
datanew=data.copy();
a=data.CRS_DEP_TIME-data.DEP_TIME
TIMEDIFFo=pd.DataFrame({'TIMEDIFF':a})
datanew=pd.concat([datanew,TIMEDIFFo],axis=1)

#                                                         One Hot Encoding
datanew=pd.concat([datanew,pd.get_dummies(datanew.ORIGIN,prefix='ORIGIN'),
pd.get_dummies(datanew.DEST,prefix='DEST'),
pd.get_dummies(datanew.CARRIER,prefix='CARRIER'),
pd.get_dummies(datanew.DAY_WEEK,prefix='WEEK'),
pd.get_dummies(datanew.DAY_OF_MONTH,prefix='MONTH')],axis=1)
datanew['Flight Status']=np.array((data['Flight Status']=='ontime')*1)
first_10_TAIL_NUM=[x for x in datanew.TAIL_NUM.value_counts().head(10).index]
for label in first_10_TAIL_NUM:
 datanew[label]=np.where(datanew.TAIL_NUM==label,1,0)

datanew=datanew.drop(['CRS_DEP_TIME','DEP_TIME','FL_DATE','FL_NUM','TAIL_NUM','CARRIER','DAY_OF_MONTH','DAY_WEEK','DEST','ORIGIN'],axis=1)

#                                                    incresing weather's weight
newweather={0:0,
1:100000
}
datanew.Weather=datanew.Weather.map(newweather)

#print(data.FL_NUM.value_counts().head(21))
#print(data.TAIL_NUM.value_counts().head(10))


Xrand=datanew.sample(frac=1)
scaler=preprocessing.StandardScaler()
columnnames=['DISTANCE','TIMEDIFF']
Xrand[columnnames] = scaler.fit_transform(Xrand[columnnames]) 
y_train=Xrand["Flight Status"][:1761]
y_test=Xrand["Flight Status"][1761:2201] 
X_train=Xrand[:1761]
X_test=Xrand[1761:2201]
X_test=X_test.drop('Flight Status',axis=1)
X_train=X_train.drop('Flight Status',axis=1)

#                                         Logistic Regression
logreg=LogisticRegression(max_iter=100)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
acc_log=round(logreg.score(X_train,y_train)*100,5)
print('\t\t\t\tFor Logistic Regression\nscore=',acc_log)
print(classification_report(y_test,y_pred))

#                                        Stochastic Gradient Descent
sgd = SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)
y_pred2 = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
print('\t\t\t\tFor Stochastic Gradient Descent\nscore=',acc_sgd)
print(classification_report(y_test,y_pred2))

#                                       Support Vector Classification
linear_svc = LinearSVC(max_iter=2000,C=70)
linear_svc.fit(X_train, y_train)
y_pred3 = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
print('\t\t\t\tFor Support Vector Classification\nscore=',acc_linear_svc)
print(classification_report(y_test,y_pred3))

#                                       Random Forest Classifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred4 = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) *
100, 2)
print('\t\t\t\tFor Random Forest Classifier\nscore=',acc_random_forest)
print(classification_report(y_test,y_pred4))

sns.barplot(x=['LR','SGD','SVC','RFC'],y=[acc_log,acc_sgd,acc_linear_svc,acc_random_forest])
plt.xlabel('Algorithms')
plt.ylabel('Fitting accuracy')
plt.show()