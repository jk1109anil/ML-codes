#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 22:39:16 2019

@author: jayakrishnan
"""
import tensorflow as tf
from tensorflow.train import AdamOptimizer as opt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

t0 = time()

pa_dir = r'D:\ML\DL\Assignment_2\SingleLabelImageFeatures'

norm = True
if (norm):
    features = np.loadtxt(r'D:\ML\DL\Assignment_2\SingleLabelImageFeatures\features_norm_single_label.csv', delimiter = ',')
else:
    features = np.loadtxt(r'D:\ML\DL\Assignment_2\SingleLabelImageFeatures\features_single_label.csv', delimiter = ',')

labels = np.loadtxt(r'D:\ML\DL\PA1\labels_single_label.csv', delimiter = ',')

np.random.seed(201)

features_unshuffled = features
label_unshuffled = features

mat = np.append(features,labels,axis=1)
mat = np.take(mat,np.random.permutation(mat.shape[0]),axis=0,out=mat)

features = mat[:,:-5]
label = features

N = np.shape(features)[0]

X = tf.placeholder(tf.float32, [None, 828])
y = tf.placeholder(tf.float32, [None, 828])

n_epochs = 1700
loss_fn = 'ce'
learning_mode = "batch"
learning_rate = 3e-3
loss_fn = 'ce'

#no_layers = 2
n_op = 828
n_ip = 828

slope = 1
#'sigmoid', 'tanh', 'relu', 'elu', 'softplus'
activation = "tanh"

arch = [600,750,600]
bn_index = 1
Weights = []
b = []

for i in range(len(arch)):
    if i == 0:
        Weights.append(tf.Variable(tf.random_normal([n_ip, arch[i]], stddev=(n_ip)**-0.5), name='Weights_'+str(i)))
    else:
        Weights.append(tf.Variable(tf.random_normal([arch[i-1], arch[i]], stddev=(arch[i-1])**-0.5), name='Weights_'+str(i)))
    
    b.append(tf.Variable(tf.random_normal([arch[i]]), name='b_'+str(i)))


Weights_output = tf.Variable(tf.random_normal([arch[(len(arch)-1)], n_op], stddev=(arch[(len(arch)-1)])**-0.5), name='Weights_output')
b_output = tf.Variable(tf.random_normal([n_op]), name='b_output')

s = []

for i in range(len(arch)):
    if i==0:
        s.append(slope*tf.add(tf.matmul(X, Weights[i]), b[i]))
    else:
        s.append(tf.add(tf.matmul(s[i-1], Weights[i]), b[i]))
    
    if i!=bn_index:
        if activation=='sigmoid':
            s[i] = tf.nn.sigmoid(s[i])
        elif activation=='tanh':
            s[i] = tf.nn.tanh(s[i])
        elif activation=='relu':
            s[i] = tf.nn.relu(s[i])
        elif activation=='elu':
            s[i] = tf.nn.elu(s[i])
        elif activation=='softplus':
            s[i] = tf.nn.softplus(s[i])

s_out = tf.add(tf.matmul(s[(len(arch)-1)], Weights_output), b_output)

error = tf.reduce_mean( tf.reduce_sum(((y-s_out)**2),1) )

optimiser = opt(learning_rate=learning_rate, beta1 = 0.9, beta2 = 0.999).minimize(error)

init = tf.global_variables_initializer()

train_SSE = []

epochs = 0
err=1000.0
if(learning_mode == 'batch'):
    with tf.Session() as sess:
        sess.run(init)
        while(err>25.0):
            opt_info, err = sess.run([optimiser, error], feed_dict={X: features, y: label})
            print("Epoch: ", (epochs + 1), "SSE =", "{:.4f}".format(err))
            train_SSE.append(err)
            epochs = epochs+1
        Z = sess.run([s[bn_index]], feed_dict={X: features_unshuffled, y: label_unshuffled})

elif(learning_mode == 'pattern'):
    with tf.Session() as sess:
        sess.run(init)
        for epochs in range(n_epochs):
            epoch_error = 0
            for i in range(N):
                opt_info, err = sess.run([optimiser, error], feed_dict={X: [features[i]], y: [label[i]]})
                epoch_error = epoch_error + err
            
            epoch_error = epoch_error/N
            print("Epoch: ", (epochs + 1), "SSE =" + "{:.4f}".format(err))
            train_SSE.append(epoch_error)
        Z = sess.run([s[bn_index]], feed_dict={X: features_unshuffled, y: label_unshuffled})

plt.plot(np.arange(epochs),train_SSE,'r')
plt.show()

Z = Z[0]
np.savetxt(pa_dir + '\AANN_features_' + str(err) + '_' + str(norm) + '_' + str(learning_rate) + '_' + str(len(arch)) + '_' + str(arch[bn_index]) + '_' + activation + '.csv', Z, delimiter=",")

Z_norm = scale(Z,axis=0)
np.savetxt(pa_dir + '\AANN_norm_features_' + str(err) + '_' + str(norm) + '_' + str(learning_rate) + '_' + str(len(arch)) + '_' + str(arch[bn_index]) + '_' + activation + '.csv', Z_norm, delimiter=",")
