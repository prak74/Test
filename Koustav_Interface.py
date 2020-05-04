import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Koustav_LR
import Koustav_NN

train_df = pd.read_csv("FlightDelays.csv")

X_train = train_df.drop("Flight Status", axis=1).to_numpy()
Y_train = train_df["Flight Status"].to_numpy().reshape(X_train.shape[0], 1)

def get_attributes(n, Model):
	attributes, params = Model.train(X_train.copy(), Y_train.copy(), num_epoc = n)
	return attributes

def accuracy_plots() :
	x = list(range(0,10000,50))
	y1 = []
	y2 = []
	y3 = []
	y4 = []
	for i in x :
		(train_acc, f1score, precision, recall) = get_attributes(i, Neural_Network)
		y1.append(train_acc)
		y2.append(f1score)
		y3.append(precision)
		y4.append(recall)
	plt.figure(1)
	plt.plot(x, y1, 'g-', label = "Train Accuracy")
	plt.plot(x, y2, 'b-', label = "F1score")
	plt.xlabel("epoc")
	plt.title("Learning Rate :" + str(0.001))
	plt.legend()
	plt.figure(2)
	plt.plot(x, y3, 'r-', label = "Precision")
	plt.plot(x, y4, 'b-', label = "Recall")
	plt.xlabel("epoc")
	plt.title("Learning Rate :" + str(0.001))
	plt.legend()
	plt.show()

accuracy_plots()
