import numpy as np

# X is assumed to be dataframe(m,n)
# m,n is the usual notation
# Y is assumed to be dataframe(m,1)

def initialize(dim):
	w = np.zeros((dim,1))
	b = 0
	return w,b

def sigmoid(z):
	s = 1/(np.exp(-z)+1)
	return s

def propagate(w,b,X,Y):
	m = X.shape[0]
	Z = np.dot(X,w) + b
	A = sigmoid(Z)
	cost = np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))*(-1)/m
	dw = np.dot(X.T,A-Y)/m
	db = np.sum(A-Y)/m
	grad = {"dw": dw,"db": db}
	return grad,cost

def optimize(w,b,X,Y,num_iter,learning_rate):
	dw = np.zeros((X.shape[1],1))
	db = 0
	cost = 0
	for i in range(num_iter):
		grad, cost = propagate(w,b,X,Y)
		dw = grad["dw"]
		db = grad["db"]
		w = w - learning_rate*dw
		b = b - learning_rate*db
	params = {"w": w, "b": b}
	return params

def train(X_train, Y_train, num_iter=9000, learning_rate=0.0005):
	X_train = X_train.to_numpy()
	Y_train = Y_train.to_numpy().reshape((X_train.shape[0],1))
	w,b = initialize(X_train.shape[1])
	params = optimize(w,b,X_train,Y_train,num_iter,learning_rate)
	return params
	
def predict(X_test, params):
	X = X_test.to_numpy()
	m = X.shape[0]
	Y_predict = sigmoid(np.dot(X,params["w"])+params["b"])
	Y_predict = (Y_predict > 0.5).astype(int)
	return Y_predict

def accuracy(Y_predict, Y_test):
	Y_test = Y_test.to_numpy().reshape((Y_test.shape[0],1))
	c = 0
	m = Y_test.shape[0]
	c = m-(Y_predict^Y_test).astype(int).sum()
	return c/m

def f1score(Y_predict, Y_test):
	Y_test = Y_test.to_numpy().reshape((Y_test.shape[0],1))
	tp = (Y_predict & Y_test).astype(int).sum()
	fn = (~Y_predict & Y_test).astype(int).sum()
	fp = (Y_predict & ~Y_test).astype(int).sum()
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	f1 = 2*precision*recall/(precision+recall)
	return f1