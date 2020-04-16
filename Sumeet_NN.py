import numpy as np
import random
from scipy.special import expit

class Network(object):

	def __init__(self,sizes):
		self.sizes=sizes
		self.num_layers=len(sizes)
		self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		self.biases=[np.random.randn(y,1) for y in sizes[1:]]

	def gradient_descent(self,training_data,epochs,mini_batch_size,eta,lmbda,test_data=None):
		training_data=list(training_data)
		n=len(training_data)
		if test_data:
			length=len(list(test_data))
		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update(mini_batch,eta,lmbda)
			if test_data:
				print("Epoch {} Accuracy:{}/{}".format(j+1,self.evaluate(test_data),length))
			else:
				#if test data is not provided
				print("Epoch {} is complete".format(j+1))
			
		

	def update(self,mini_batch,eta,lmbda):
		m=len(mini_batch)
		mini_batch=list(mini_batch)
		nabla_b=[np.zeros(b.shape) for b in self.biases]
		nabla_w=[np.zeros(w.shape) for w in self.weights]
		
		for x,y in mini_batch:
			#returns the change is all weights and biases
			dnb,dnw=self.backprop(x,y)
			nabla_b=[nb+dnb1 for nb,dnb1 in zip(nabla_b,dnb)]
			nabla_w=[nw+dnw1 for nw,dnw1 in zip(nabla_w,dnw)]
		self.biases=[b*(1-(lmbda*eta/m))-(eta/m)*nb for b,nb in zip(self.biases,nabla_b)]
		self.weights=[w*(1-(lmbda*eta/m))-(eta/m)*nw for w,nw in zip(self.weights,nabla_w)]

	def backprop(self,x,y):
		nabla_b=[np.zeros(b.shape) for b in self.biases]
		nabla_w=[np.zeros(w.shape) for w in self.weights]
		activation=x
		activations=[x]
		
		for w,b in zip(self.weights,self.biases):
			z=np.dot(w,activation)+b
			activation=expit(z)
			activations.append(activation)
        
        #using cross entropy cost, therefore delta is not dependent on sigmooid_prime(z)
		delta=activations[-1]-y 
		nabla_b[-1]=delta
		nabla_w[-1]=np.dot(delta,activations[-2].transpose())

		for l in range(1,self.num_layers-1):
			delta=np.dot(self.weights[-l].transpose(),delta)*(activations[-l-1])*(1-activations[-l-1])
			nabla_b[-l-1]=delta
			nabla_w[-l-1]=np.dot(delta,activations[-l-2].transpose())

		return (nabla_b,nabla_w)

	def feedforward(self,x):
		for w,b in zip(self.weights,self.biases):
			x=expit(np.dot(w,x)+b)
		x=float(x)
		if x>=0.5:
			return 1.0
		else:
			return 0.0

	def evaluate(self,test_data):
		test_results=[]
		for (x,y) in test_data:
			if x[-1]==1:
				test_results.append((1,1))
			else:
				test_results.append((self.feedforward(x[:-1]),y))
		#delayed is positive ie 0
		true_positive=0
		false_positive=0
		false_negative=0
		for x,y in test_results:
			if x==y:
				true_positive+=1
			elif x==1 and y==0:
				false_positive+=1
			elif x==0 and y==1:
				false_negative+=1
		recall=true_positive/(true_positive+false_positive)
		precision=true_positive/(true_positive+false_negative)
		print('RECALL={}'.format(recall))
		print('PRECISION={}'.format(precision))
		print('F1SCORE={}'.format(2*recall*precision/(recall+precision)))
		return sum(int(x==y) for x,y in test_results)

import data_delayed
network=Network([6,10,1])
network.gradient_descent(data_delayed.training_data,30,10,0.05,0.005,data_delayed.test_data)
								  


	 

