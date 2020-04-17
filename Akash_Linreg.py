import numpy as np
from sklearn import preprocessing

def cost_fn(x,y,theta,lamb=0):
    y=np.array(y)
    m=np.size(x,0)
    n=np.size(x,1)
    one=np.ones((m,1),dtype=int)
    x=np.concatenate((one,x),axis=1)
   
    theta_null=theta.copy()
    theta_null[0]=0
 
    inter=np.matmul(x,theta)
    inter=inter-y
    J=np.matmul(inter.transpose(),inter)/(2*m)+((lamb/(2*m))*np.dot(theta_null.transpose(),theta_null))
    return J

def grad_des(x,y,theta,lamb=0,alpha=0.00009,maxiter=150000,max_error=10**-4):
    y=np.array(y)
    m=np.size(x,0)
    n=np.size(x,1)
    x2=x.copy()
    if(m<10000):
     one=np.ones((m,1),dtype=int)
     x=np.concatenate((one,x),axis=1)
     theta=np.matmul(np.linalg.inv(np.matmul(x.transpose(),x)),np.matmul(x.transpose(),y))
     return theta
    scaler=preprocessing.StandardScaler()
    x= scaler.fit_transform(x)
    y=scaler.fit_transform(y)    
    y=np.array(y)
    m=np.size(x,0)
    n=np.size(x,1)    
    x2=x.copy()    
    one=np.ones((m,1),dtype=int)
    x=np.concatenate((one,x),axis=1)
    theta_null=theta.copy()
    theta_null[0]=0        
    inter=np.matmul(x,theta)-y    
    inter2=np.matmul(x.transpose(),inter)
    i=0
    J2=0
    J1=cost_fn(x2,y,theta,lamb=0)
    Difference=100
    while(True):
     theta=theta-(alpha/m)*(inter2+lamb*(theta_null))
     J2=cost_fn(x2,y,theta,lamb=0)
     Difference=J1-J2     
     if((Difference<=max_error)|(i>maxiter)):
      break
     i=i+1 
     J1=J2
     return theta

def linearreg(x,y,ini_theta,lamb=0,alpha=0.009,maxiter=150000,max_error=10**-4):
    theta_final=grad_des(x,y,ini_theta,lamb,alpha,maxiter,max_error)
    return theta_final
