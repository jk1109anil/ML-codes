#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:49:55 2019

@author: jayakrishnan
"""

import numpy as np
#from sklearn import metrics
#from sklearn.preprocessing import StandardScaler
#from sklearn.utils.extmath import softmax
import matplotlib.pyplot as plt

class layer:
    def __init__(self, weights, b, activate, slope,learning_rate):
        self.weights = weights
        self.b = b
        self.size = np.shape(weights)
        self.delta_weights = np.zeros(self.size)
        self.delta_weights_old = np.zeros(self.size)
        self.delta_b = np.zeros([self.size[1],1])
        self.delta_b_old = np.zeros([self.size[1],1])
        
        self.r = np.zeros(self.size)  # adaGrad
        self.r_2 = np.zeros(self.size)
        self.rb = np.zeros([self.size[1],1])
        self.rb_2 = np.zeros([self.size[1],1])
        self.epsilon = 0.01
        
        self.L = 100 # RMSProp
        self.g_array = np.zeros((self.size[0],self.size[1],self.L+1))
        self.g_array_b = np.zeros((self.size[1],1,self.L+1))
        self.rho = 0.9
        
        # Ada delta
        self.W_array = np.zeros((self.size[0],self.size[1],self.L+1))
        self.b_array = np.zeros((self.size[1],1,self.L+1))
        self.u_2 = np.zeros(self.size)
        self.ub_2 = np.zeros([self.size[1],1])
        self.u = np.zeros(self.size)
        self.ub = np.zeros([self.size[1],1])
        
        self.a = np.zeros([self.size[1],1])
        self.s = np.zeros([self.size[1],1])
        self.A = np.zeros([self.size[1],1])
        self.delta = np.zeros([self.size[1],1])
        
        # adam
        self.U = np.zeros(self.size)
        self.Ub = np.zeros([self.size[1],1])
        self.rho1 = 0.9
        self.V = np.zeros(self.size)
        self.Vb = np.zeros([self.size[1],1])
        self.V_2 = np.zeros(self.size)
        self.Vb_2 = np.zeros([self.size[1],1])
        self.rho2 = 0.999
        
        self.activation = activate
        self.slope = slope
        self.learning_rate = learning_rate
        
        #  "delta" , "Gen delta" , "adaGrad" , "RMSProp" , "Ada delta" ,"adam"
        self.rule = "delta" #learning rule
        self.alpha = 0.9
        
    def activation_function(self):
        if self.activation=="logistic":
            self.s = 1/(1+np.exp(-self.slope*self.a))
        elif self.activation=="tanh":
            self.s = (np.exp(self.slope*self.a)-np.exp(-self.slope*self.a))/(np.exp(self.slope*self.a)+np.exp(-self.slope*self.a))
        elif self.activation=="linear":
            self.s = self.a
        elif self.activation=="ReLU":
            self.s = np.zeros(np.shape(self.a))
            self.s[self.a >= 0] = self.a[self.a >= 0]
        elif self.activation=="ELU":
            self.s = np.exp(self.slope*self.a) - 1
            self.s[self.a >= 0] = self.a[self.a >= 0]
        elif self.activation=="softplus":
            self.s = np.log(np.exp(self.slope*self.a) + 1)
                    
    def derivative(self):
        if self.activation=="logistic":
            self.A = self.slope*(self.s)*(1-self.s)
        elif self.activation=="linear":
            self.A = self.slope*np.ones(np.size(self.s))
        elif self.activation=="tanh":
            self.A = self.slope*(1-self.s)*(1+self.s)
        elif self.activation=="ReLU":
            self.s = np.zeros(np.shape(self.a))
            self.s[self.a >= 0] = 1
        elif self.activation=="ELU":
            self.s = self.slope*(self.s + 1)
            self.s[self.a >= 0] = 1        
        elif self.activation=="softplus":
            self.s = self.slope*1.0/(np.exp(-self.slope*self.a) + 1)
     
    def Weight_update(self):
        if(self.rule == "delta"):
            self.weights = self.weights + self.delta_weights
            self.b = self.b + self.delta_b
        elif(self.rule == "Gen delta"):
            self.delta_weights += self.alpha*self.delta_weights_old # weights
            self.weights = self.weights + self.delta_weights
            self.delta_weights_old = self.delta_weights
            
            self.delta_b += self.alpha*self.delta_b_old # b
            self.b = self.b + self.delta_b 
            self.delta_b_old = self.delta_b
        elif(self.rule == "adaGrad"):
            self.delta_weights = self.delta_weights/self.learning_rate # weights
            self.delta_b = self.delta_b/self.learning_rate
            self.r_2 = self.r_2 + self.delta_weights**2
            self.r = np.sqrt(np.abs(self.r_2))
            self.delta_weights = -self.learning_rate*self.delta_weights/(self.epsilon + self.r)

            self.delta_b = self.delta_b/self.learning_rate  # b
            self.rb_2 = self.rb_2 + self.delta_b**2
            self.rb = np.sqrt(np.abs(self.rb_2))
            self.delta_b = self.learning_rate*self.delta_b/(self.epsilon + self.rb)

            self.weights = self.weights + self.delta_weights
            self.b = self.b + self.delta_b
        elif(self.rule == "RMSProp"):
            self.delta_weights = self.delta_weights/self.learning_rate
            self.delta_b = self.delta_b/self.learning_rate            
            for i in range(self.L):
                self.g_array[:,:,i] = 1*self.g_array[:,:,i+1]
                self.g_array_b[:,:,i] = 1*self.g_array_b[:,:,i+1]
                self.r_2 += self.rho*self.g_array[:,:,i]/self.L
                self.rb_2 += self.rho*self.g_array_b[:,:,i]/self.L
            self.g_array[:,:,self.L] = 1*self.delta_weights**2
            self.g_array_b[:,:,self.L] = 1*self.delta_b**2
            self.r_2 += (1-self.rho)*self.g_array[:,:,self.L]
            self.rb_2 += (1-self.rho)*self.g_array_b[:,:,self.L]
            
            self.r =  np.sqrt(np.abs(self.r_2))
            self.rb =  np.sqrt(np.abs(self.rb_2))
            
            self.weights += self.learning_rate*self.delta_weights/(self.epsilon + self.r)
            self.delta_b += self.learning_rate*self.delta_b/(self.epsilon + self.rb)

            self.r_2 = 0*self.r_2   # reinitialising to zero
            self.rb_2 = 0*self.rb_2
        elif(self.rule == "Ada delta"):
            self.delta_weights = self.delta_weights/self.learning_rate
            self.delta_b = self.delta_b/self.learning_rate
            for i in range(self.L):
                self.g_array[:,:,i] = 1*self.g_array[:,:,i+1]
                self.g_array_b[:,:,i] = 1*self.g_array_b[:,:,i+1]
                
                self.r_2 += self.rho*self.g_array[:,:,i]/self.L
                self.rb_2 += self.rho*self.g_array_b[:,:,i]/self.L
                self.u_2 += self.rho*self.W_array[:,:,i]/self.L
                self.ub_2 += self.rho*self.b_array[:,:,i]/self.L

                self.W_array[:,:,i] = 1*self.W_array[:,:,i+1]
                self.b_array[:,:,i] = 1*self.b_array[:,:,i+1]
                        
            self.g_array[:,:,self.L] = 1*self.delta_weights**2
            self.g_array_b[:,:,self.L] = 1*self.delta_b**2
            
            self.u_2 += (1-self.rho)*self.W_array[:,:,self.L]
            self.ub_2 += (1-self.rho)*self.b_array[:,:,self.L]
                            
            self.r_2 += (1-self.rho)*self.g_array[:,:,self.L]
            self.rb_2 += (1-self.rho)*self.g_array_b[:,:,self.L]
            
            self.r =  np.sqrt(np.abs(self.r_2))      
            self.rb =  np.sqrt(np.abs(self.rb_2))
            self.u =  np.sqrt(np.abs(self.r_2))
            self.ub =  np.sqrt(np.abs(self.rb_2))
            
            self.delta_weights = (self.epsilon + self.u)*self.delta_weights/(self.epsilon + self.r)
            self.delta_b = (self.epsilon + self.ub)*self.delta_b/(self.epsilon + self.rb)
                        
            self.W_array[:,:,self.L] = 1*self.delta_weights**2
            self.b_array[:,:,self.L] = 1*self.delta_b**2
            
            self.weights += self.delta_weights
            self.b += self.delta_b
            
            self.r_2 = 0*self.r_2     # reinitialising to zero
            self.rb_2 = 0*self.rb_2
            self.u_2 = 0*self.u_2
            self.ub_2 = 0*self.ub_2
        elif(self.rule == "adam"):
            self.U = self.rho1*self.U + (1-self.rho1)*self.delta_weights
            self.Ub = self.rho1*self.Ub + (1-self.rho1)*self.delta_b
            self.V_2 = self.rho2*self.V_2 + (1-self.rho2)*self.delta_weights
            self.Vb_2 = self.rho2*self.Vb_2 + (1-self.rho2)*self.delta_b
            
            self.V = np.sqrt(np.abs(self.V_2)/self.learning_rate)
            self.Vb = np.sqrt(np.abs(self.Vb_2)/self.learning_rate)
            
            self.weights += self.delta_weights/(self.epsilon + self.V)
            self.b += self.delta_b/(self.epsilon + self.Vb)
           
    def zero_delta_weights(self):
        self.delta_weights = np.zeros(self.size)
        self.delta_b = np.zeros([self.size[1],1])
        
def forward(h,x):         # forward path
    no_layers = len(h)
    h[no_layers-1].a = np.matmul(np.transpose(h[no_layers-1].weights),
     x.reshape(-1,1) ) + h[no_layers-1].b    # activation value for input layer
    h[no_layers-1].activation_function()            
    for j in range(no_layers-1):  # for hidden layers
        k = no_layers-2-j
        # activation value
        h[k].a = np.matmul(np.transpose(h[k].weights), h[k+1].s) + h[k].b
        h[k].activation_function()
    return h

no_layers = 3
input_size = 13
output_size = 1

learning_mode = "pattern"
epochs = 0

f = open('housing-data.txt', "r")
mat =  np.genfromtxt(f)     # features and labels
N = np.shape(mat)[0]
mat = np.take(mat,np.random.permutation(mat.shape[0]),axis=0,out=mat)

features = mat[:,:-1]
label = mat[:,13]

train_size = int(0.7*N)

tr = np.zeros(train_size) 
label_tr = label[:train_size]  #Training data

features_v = features[int(0.7*N):int(0.8*N),:] # validation set
label_v = label[int(0.7*N):int(0.8*N)]
v = np.size(label_v)
val = np.zeros([v,1])

features_t = features[int(0.8*N):,:]   # test set
label_t = label[int(0.8*N):]
t = np.size(label_t)

# 201, 113, 768, 237
np.random.seed(237)

n_epochs =7500

n_h1 = 15
n_h2 = 15
slope = 1e-3
Weights_output = np.random.randn(n_h1,1)/((n_h1)**0.5)  # initial values to weights
b_output = np.random.randn(1,1)
Weights_1 = np.random.randn(n_h2,n_h1)/((n_h2)**0.5)
b_1 = np.random.randn(n_h1,1)
Weights_2 = np.random.randn(13,n_h2)/((13)**0.5)
b_2 = np.random.randn(n_h2,1)

learning_rate = 0.1

h = []     # layers
h.append(layer(Weights_output, b_output,"linear",1,learning_rate))
h.append(layer(Weights_1, b_1,"logistic",slope,learning_rate))
h.append(layer(Weights_2, b_2,"logistic",slope,learning_rate))

error_ratio = 0.8
error = np.linalg.norm(label_v)
train_error = np.zeros(n_epochs) # holds error values of each epoch
val_error = np.zeros(n_epochs)

f=open('weights_for_5_batch_last.csv','ab')  ####  test   ######

#while (error_ratio < 1 and epochs <= 10):
while (epochs < n_epochs): 
    for i in range(train_size):  #  training  for each epoch
        # forward path
        h = forward(h,features[i,:])
        
        #  start of BP
        h[0].derivative()  # BP for output layer
        h[0].delta = np.multiply( (label[i]-h[0].s), h[0].A)
        if(learning_mode == "pattern"): 
            h[0].delta_weights = h[0].learning_rate*np.matmul( h[1].s,np.transpose(h[0].delta) )
            h[0].delta_b = h[0].delta*h[0].learning_rate
            h[0].Weight_update()
        elif(learning_mode == "batch"):
            h[0].delta_weights += h[0].learning_rate*np.matmul( h[1].s,np.transpose(h[0].delta) )/train_size
            h[0].delta_b += h[0].delta*h[0].learning_rate
            
        for j in range(no_layers-2):  # BP for layers other than 1st hidden and output
            k = j+1
            h[k].derivative()
            h[k].delta = np.multiply( np.matmul(h[k-1].weights,h[k-1].delta) , h[k].A)
            if(learning_mode == "pattern"):
                h[k].delta_weights = h[k].learning_rate*np.matmul( h[k+1].s,np.transpose(h[k].delta) )
                h[k].delta_b = h[k].delta*h[k].learning_rate
                h[k].Weight_update()
            elif(learning_mode == "batch"):                
                h[k].delta_weights += h[k].learning_rate*np.matmul( h[k+1].s,np.transpose(h[k].delta) )/train_size
                h[k].delta_b += h[k].delta*h[k].learning_rate

        k = no_layers-1      # BP of 1st hidden layer
        h[k].derivative()
        h[k].delta = np.multiply( np.matmul(h[k-1].weights,h[k-1].delta) , h[k].A)
        if(learning_mode == "pattern"):
            h[k].delta_weights = h[k].learning_rate*np.matmul(features[i,:].reshape(-1,1),np.transpose(h[k].delta) )
            h[k].delta_b = h[k].delta*h[k].learning_rate
            h[k].Weight_update()
        elif(learning_mode == "batch"):
            h[k].delta_weights += h[k].learning_rate*np.matmul(features[i,:].reshape(-1,1),np.transpose(h[k].delta) )/train_size
            h[k].delta_b += h[k].delta*h[k].learning_rate 
    
    if(learning_mode == "batch"):
        for j in range(no_layers):
            h[j].Weight_update()
            h[j].zero_delta_weights()
    # end of BP step
    
    # Validation for pattern
    error_old = error
    error = 0
         
    for i in range(v):  #  validation set
        h = forward(h,features_v[i])
        error = error + (h[0].s-label_v[i])**2
        val[i] = 1*h[0].s        
    error = np.sqrt(error/v)
    val_error[epochs] = error
    error_ratio = error/error_old
    error = 0
    
    for i in range(train_size):  #  train set
        h = forward(h,features[i])
        error = error + (h[0].s-label[i])**2
        tr[1] = 1*h[0].s
    error = np.sqrt(error/train_size)
    train_error[epochs] = error
    print epochs    ####    test
    epochs = epochs + 1

best_n_epochs_val = np.argmin(val_error)


plt.scatter(val, label_v) #  X,Y
#plt.scatter(tr, label_tr)
#plt.plot(np.arange(0,train_size,1))
plt.plot(np.arange(0,50,1))
plt.show()

print(val_error[epochs-1])
plt.plot(val_error)
plt.plot(train_error,'r')
#plt.plot(np.arange(epochs),val_error,'b')
low=np.argmin(val_error)
plt.show()

print(low,train_error[low],val_error[low])
print(val_error[-1],slope,learning_rate,learning_mode,h[0].rule)