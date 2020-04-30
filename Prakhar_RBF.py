import pandas as pd
import numpy as np
from sklearn.svm import train_test_split, SVC, classification_report, confusion_matrix

# Assuming pre-processed data


def GaussianSVM(X,y,test_size=0.20)
	
	# Splitting data into train and test group (default test size is 20% of the data)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)
	
	# Train the Gaussian kernel SVM
	svclassifier = SVC(kernel = 'rbf')
	svclassifier.fit(X_train, y_train)

	# Prediction based on trained SVM
	y_predict = svclassifier.predict(X_test)

	# Print out the results
	print (confusion_matrix(y_test,y_predict))
	print (classification_report(y_test, y_predict))

	return