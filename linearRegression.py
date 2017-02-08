#!/usr/bin/python
# -*-coding:utf-8-*-
'''
Created on  Feb. 8, 2017

@author: Jun Chai
'''



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def getData(number_of_points, a, b):
    x_data = []
    y_data = []
    for i in range(number_of_points):
        x1 = np.random.normal(0.0, 0.5)
        y1 = a * x1 + b + np.random.normal(0.0, 0.1)
        x_data.append([x1])
        y_data.append([y1])
    return x_data, y_data


def linearRegression(x_data, y_data, epoch=100, rate=0.01):
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    Y = W * x_data + b
    cost_function = tf.reduce_mean(tf.square(Y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(rate)
    train = optimizer.minimize(cost_function)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in xrange(epoch):
        sess.run(train)
    '''''
            print (step, sess.run(W), sess.run(b))
            print (step,sess.run(cost_function))
            plt.plot(x_data,y_data,'ro',label='train Data')
            plt.plot(x_data,sess.run(W)*x_data+sess.run(b),label='regression')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(-2,2)
            plt.ylim(0,1.2)
            plt.legend()
            plt.show()
    '''''
    print 'W is', sess.run(W)
    print 'b is', sess.run(b)
    print 'loss is', sess.run(cost_function)
    w = sess.run(W)
    b = sess.run(b)
    return w, b


def predict(x_test, y_test, w, b):
    pred = tf.add(tf.mul(x_test, w), b)
    loss = tf.reduce_mean(tf.square(pred - y_test))
    sess = tf.Session()
    loss = sess.run(loss)
    return loss


if __name__ == "__main__":
    xTrain, yTrain = getData(500, 0.1, 0.3)
    xTest, yTest = getData(50, 0.1, 0.3)
    w, b = linearRegression(xTrain, yTrain, 8, 0.05)
    loss = predict(xTest, yTest, w, b)
    print loss
