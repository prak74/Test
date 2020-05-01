import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Koustav_LR

train_df = pd.read_csv("FlightDelays.csv")

X_train = train_df.drop("Flight Status", axis=1)
Y_train = train_df["Flight Status"]

def get_attributes(n, Model):

	params = Model.train(X_train, Y_train, num_iter = n)
	Y_predict = Model.predict(X_train,params)
	acc = Model.accuracy(Y_predict, Y_train)
	f1score, precision, recall = Model.f1score(Y_predict, Y_train)
	return acc, f1score, precision, recall

def accuracy_plots(Model) :
	x = list(range(50,10001,50))
	y1 = []
	y2 = []
	y3 = []
	y4 = []
	for i in x :
		acc, f1score, precision, recall = get_attributes(i, Model)
		y1.append(acc)
		y2.append(f1score)
		y3.append(precision)
		y4.append(recall)
	plt.figure(1)
	plt.plot(x, y1, 'g-', label = "Accuracy")
	plt.plot(x, y2, 'b-', label = "F1score")
	plt.xlabel("iterations")
	plt.title("Learning Rate :" + str(0.001))
	plt.legend()
	plt.figure(2)
	plt.plot(x, y3, 'r-', label = "Precision")
	plt.plot(x, y4, 'b-', label = "Recall")
	plt.xlabel("iterations")
	plt.title("Learning Rate :" + str(0.001))
	plt.legend()
	plt.show()

accuracy_plots(Koustav_LR)
