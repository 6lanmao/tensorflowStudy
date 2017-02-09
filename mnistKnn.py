#!/usr/bin/python
# -*-coding:utf-8-*-
'''
Created on  Feb. 8, 2017

@author: Jun Chai
'''
import numpy as np
import tensorflow as tf
import input_data

def getData(batch_size):
    mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)
    pixels,labels=mnist.train.next_batch(batch_size)
    #print labels
    return pixels, labels


def knnClassify(train_pixels,train_labels,test_pixels,test_labels):
    train_pixel_tensor=tf.placeholder('float',[None,784])
    test_pixel_tensor=tf.placeholder('float',[784])
    distance=tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor,tf.neg(test_pixel_tensor))),reduction_indices=1)
    pred=tf.arg_min(distance,0)
    accuracy=0
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range((len(test_labels))):
            nn_index=sess.run(pred,feed_dict={train_pixel_tensor:train_pixels, test_pixel_tensor:test_pixels[i,:]})
            print "Test ", i, "Predicted Class: ",  np.argmax(train_labels[nn_index]), \
                                                    "True Class: ", np.argmax(test_labels[i])
            if np.argmax(train_labels[nn_index])==np.argmax(test_labels[i]):
                accuracy +=1./len(test_labels)
        print 'accuracy', accuracy


if __name__ == "__main__":
    train_pixels,train_labels=getData(1000)
    test_piels,test_labels=getData(100)
    knnClassify(train_pixels,train_labels,test_piels,test_labels)

