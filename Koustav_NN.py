import numpy as np

# X_train is input as numpy array(m_train,n)
# Y_train is input as numpy array(m_train, classes) except for binary classification it is (m_train,1)
# X_test is input as numpy array(m_test, n)
# Y_test is input as numpy array(m_test, classes) except for binary classification it is (m_test,1)
# all operations using X as matrix(n,m) & Y as matrix(1,m)
# sigmoid activation

def sigmoid(Z):
	Z = np.float128(Z)
	A = 1/(1+np.exp(-Z))
	return A, Z

def sigmoid_backward(dA, Z):
	A, Z = sigmoid(Z)
	dZ = dA*A*(1-A)
	return dZ

def initialize_parameters(layers_dims):
	np.random.seed(1)
	parameters = {}
	L = len(layers_dims)
	for l in range(1,L):
		parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(1/layers_dims[l-1]) # standardising the random initialization
		parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
	return parameters

def random_mini_batches(X, Y, mini_batch_size, seed):

	np.random.seed(seed)
	m = X.shape[1]      
	mini_batches = []

	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation].reshape((1,m))

	num_complete_minibatches = int(m/mini_batch_size) # number of mini batches of size mini_batch_size in partitionin
	for k in range(0, num_complete_minibatches):

		mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	if m % mini_batch_size != 0:

		mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size : m]
		mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size : m]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches

def linear_forward(A_prev, W, b):
	Z = np.dot(W,A_prev) + b
	cache = (A_prev, W, b)
	return Z, cache

def linear_activation_forward(A_prev, W, b, activation):

	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)     
	elif activation == "tanh":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = tanh(Z)
	cache = (linear_cache, activation_cache)
	return A, cache

def forward_propagation(X, parameters):

	caches = []
	A = X
	L = len(parameters) // 2
	for l in range(1, L):
		A_prev = A 
		A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], "sigmoid")
		caches.append(cache)
	AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], "sigmoid")
	caches.append(cache)       
	return AL, caches

def compute_cost(AL, Y, parameters, lambd):
	
	cost = - np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))
	r_cost = 0
	L = len(parameters)//2 
	for l in range(1,L+1):
		r_cost = r_cost + np.sum(parameters["W"+str(l)]**2)
	cost = cost + lambd*r_cost/2
	return cost

def linear_backward(dZ, cache, lambd):

	A_prev, W, b = cache
	m = A_prev.shape[1]
	dW = np.dot(dZ,A_prev.T)/m + lambd*W/m
	db = np.sum(dZ,axis=1,keepdims=True)/m
	dA_prev = np.dot(W.T,dZ)
	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation, lambd):

	linear_cache, activation_cache = cache
	if activation == "tanh":
		dZ = tanh_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
	return dA_prev, dW, db

def backward_propagation(AL, Y, caches, lambd):

	grads = {}
	L = len(caches)
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)    
	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	current_cache = caches[L-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)
	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "sigmoid", lambd)
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp
	return grads

def update_parameters(parameters, grads, learning_rate):
	L = len(parameters) // 2
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
	return parameters

def optimize(X, Y, layers_dims, num_epoc, minibatch_size, learning_rate, lambd):

	parameters = initialize_parameters(layers_dims)
	seed = 0
	m = X.shape[1]

	for i in range(num_epoc):

		seed = seed + 1
		minibatches = random_mini_batches(X, Y, minibatch_size,seed)
		cost_total = 0

		for minibatch in minibatches:
			(minibatch_X, minibatch_Y) = minibatch
			AL, caches = forward_propagation(minibatch_X, parameters)
			cost_total += compute_cost(AL, minibatch_Y, parameters, lambd)
			grads = backward_propagation(AL, minibatch_Y, caches, lambd)
			parameters = update_parameters(parameters, grads, learning_rate)
		cost = cost_total/m

	return parameters

def train(X_train, Y_train, layers_dims=[10], num_epoc=4500, minibatch_size = 64, learning_rate=0.005, lambd=0):  # layers_dims = list of no. of hidden units of each hidden layer
	X = X_train.T
	Y = Y_train.T
	layers_dims = [X.shape[0]] + layers_dims + [Y.shape[0]]
	parameters = optimize(X, Y, layers_dims, num_epoc, minibatch_size, learning_rate, lambd)
	classes = Y.shape[0]
	if classes>1 :
		AL, caches = forward_propagation(X, parameters)
		Y_temp = (AL >= 0.5).astype(int)
		f1, precision, recall = f1score(Y_temp, Y)
		Y_predict = predict(X_train, parameters)
		train_acc = accuracy(Y_predict, Y_train)
		return (train_acc, f1, precision, recall), parameters
	else :
		Y_predict = predict(X_train, parameters)
		f1, precision, recall = f1score(Y_predict, Y_train)
		train_acc = accuracy(Y_predict, Y_train)
		return (train_acc, f1, precision, recall), parameters

def predict(X_test, parameters):
	X = X_test.T
	AL, caches = forward_propagation(X, parameters)
	if AL.shape[0]>1:
		Y_predict = np.argmax(AL, axis = 0).reshape((X_test.shape[0],1))
		return Y_predict
	else :
		Y_predict = (AL >= 0.5).astype(int)
		return Y_predict.T

def accuracy(Y_predict, Y_test):
	c = 0
	m = Y_predict.shape[0]
	for i in range(0,m):
		if Y_predict[i,0]==Y_test[i,0]:
			c+=1
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
