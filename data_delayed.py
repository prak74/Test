#data shows only the entries where flights were delayed

import pandas as pd 
import os
import numpy as np

df=pd.read_csv("FlightDelays.csv")
df.drop(columns=['FL_DATE','FL_NUM','DISTANCE','TAIL_NUM'],inplace=True)

df['ORIGIN']=df['ORIGIN'].astype('category')
df['ORIGIN']=df['ORIGIN'].cat.codes

df['DEST']=df['DEST'].astype('category')
df['DEST']=df['DEST'].cat.codes

df['CARRIER']=df['CARRIER'].astype('category')
df['CARRIER']=df['CARRIER'].cat.codes
df['DEP_DIFF']=np.absolute(df['CRS_DEP_TIME']-df['DEP_TIME'])

df['Flight Status']=df['Flight Status'].astype('category')
df['Flight Status']=df['Flight Status'].cat.codes
df.drop(columns=['CRS_DEP_TIME','DEP_TIME'],inplace=True)


#ontime is 1 and delayed is 0
#DEP_DIFF brings more clarity to the data
#if DEP_DIFF is>60 in almost all cases flight is delayed except some outliers
cols=['CARRIER','DEST','ORIGIN','DAY_WEEK','DAY_OF_MONTH','Flight Status','DEP_DIFF','Weather']
df=df[cols]
for i,value in enumerate(df['DEP_DIFF']):
	if value>1200:
		df.iat[i,6]=2400-value   

test_data=df.sample(frac=0.3,random_state=1)
df=df.drop(test_data.index)
print(df['Flight Status'].value_counts())
#reshaping dataframe to keel weather in the end
#notice if weather is one then filght is definitely delayed
dl=df.loc[df['Weather']==0]
#dl stores the data in which flight was delayed due to reasin other than bad weather
print(dl['Flight Status'].value_counts())
print(dl['Weather'].value_counts())
dl.drop(columns=['Weather'],inplace=True)

if os.path.exists("new_file.csv"):
	os.remove("new_file.csv")

#now to choose training and test data. Only training on data with weather=0
#choosing delayed and not delayed filg
dl.to_csv("new_file.csv")

test1=dl.loc[dl['Flight Status']==1]
test2=dl.loc[dl['Flight Status']==0]

test12=test1.sample(frac=0.8,random_state=0)
test22=test2.sample(frac=0.8,random_state=0)

train_data=pd.concat([test12,test22])
print(train_data.head(),train_data.tail())
train_outcome=list(train_data['Flight Status'])
train_data.drop(columns=['Flight Status'],inplace=True)
train_data=np.array(train_data)
training_input=[]
for x in list(train_data):
	training_input.append(x.reshape(6,1))
training_data=list(zip(training_input,train_outcome))


print(test_data['Weather'].value_counts())
test_outcome=list(test_data['Flight Status'])
test_data.drop(columns=['Flight Status'],inplace=True)
test_data=np.array(test_data)
test_input=[]
for x in list(test_data):
	test_input.append(x.reshape(7,1))
test_data=list(zip(test_input,test_outcome))

print(training_data[0],test_data[0])
print(type(training_data[0]),type(test_data[0]))
print(type(training_data[0][0]),type(test_data[0][0]))
print(type(training_data[0][1]),type(train_data[0][1]))