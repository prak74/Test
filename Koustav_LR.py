import numpy as np

# X_train is input as numpy array(m_train,n)
# Y_train is input as numpy array(m_train, classes) except for binary classification it is (m_train,1)
# X_test is input as numpy array(m_test, n)
# Y_test is input as numpy array(m_test, classes) except for binary classification it is (m_test,1)

np.seterr(divide = 'ignore')

def initialize(dim):
	w = np.zeros((dim,1))
	b = 0
	return w,b

def sigmoid(z):
	z = np.float128(z)
	s = 1/(np.exp(-z)+1)
	return s

def propagate(w,b,X,Y):
	m = X.shape[0]
	Z = np.dot(X,w) + b
	A = sigmoid(Z)
	dw = np.dot(X.T,A-Y)/m
	db = np.sum(A-Y)/m
	grad = {"dw": dw,"db": db}
	return grad

def optimize(w,b,X,Y,num_epoc,learning_rate):
	dw = np.zeros((X.shape[1],1))
	db = 0
	for i in range(num_epoc):
		grad = propagate(w,b,X,Y)
		dw = grad["dw"]
		db = grad["db"]
		w = w - learning_rate*dw
		b = b - learning_rate*db
	params = {"w": w, "b": b}
	return params

def train(X_train, Y_train, num_epoc=9000, learning_rate=0.001):
	w,b = initialize(X_train.shape[1])
	params = optimize(w,b,X_train,Y_train,num_epoc,learning_rate)
	Y_predict = predict(X_train, params)
	train_acc = accuracy(Y_predict, Y_train)
	f1, precision, recall = f1score(Y_predict, Y_train)
	return (train_acc, f1, precision, recall), params
	
def predict(X_test, params):
	Y_predict = sigmoid(np.dot(X_test,params["w"])+params["b"])
	Y_predict = (Y_predict >= 0.5).astype(int)
	return Y_predict

def accuracy(Y_predict, Y_test):
	c = 0
	m = Y_predict.shape[0]
	c = m-(Y_predict^Y_test).astype(int).sum()
	return c/m

def f1score(Y_predict, Y_test):
	Y = Y_test
	tp = (Y_predict & Y).astype(int).sum()
	fn = (~Y_predict & Y).astype(int).sum()
	fp = (Y_predict & ~Y).astype(int).sum()
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	f1 = 2*precision*recall/(precision+recall)
	return f1, precision, recall