# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 12:12:57 2016

@author: rob
"""

import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


#  Extracts form the implementation by https://github.com/hardmaru/write-rnn-tensorflow
def lstm_cell(lstm_size, keep_prob):
  lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
  drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
  return drop


def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
  """ 2D normal distribution
  input
  - x,mu: input vectors
  - s1,s2: standard deviances over x1 and x2
  - rho: correlation coefficient in x1-x2 plane
  """
  # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
  norm1 = tf.subtract(x1, mu1)
  norm2 = tf.subtract(x2, mu2)
  s1s2 = tf.multiply(s1, s2)
  z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2.0*tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
  negRho = 1-tf.square(rho)
  result = tf.exp(tf.div(-1.0*z,2.0*negRho))
  denom = 2*np.pi*tf.multiply(s1s2, tf.sqrt(negRho))
  px1x2 = tf.div(result, denom)
  return px1x2

def tf_1d_normal(x3,mu3,s3):
  """ 3D normal distribution Under assumption that x3 is uncorrelated with x1 and x2
  input
  - x,mu: input vectors
  - s1,s2,s3: standard deviances over x1 and x2 and x3
  - rho: correlation coefficient in x1-x2 plane
  """
  norm3 = tf.subtract(x3, mu3)
  z = tf.square(tf.div(norm3, s3))
  result = tf.exp(tf.div(-z,2))
  denom = 2.0*np.pi*s3
  px3 = tf.div(result, denom)  #probability in x3 dimension
  return px3

#def plot_traj_MDN(sess,val_dict,batch,sl_plot = 5, ind = -1):
#  """Plots the trajectory. At given time-stamp, it plots the probability distributions
#  of where the next point will be
#  input:
#  - sess: the TF session
#  - val_dict: a dictionary with which to evaluate the model
#  - batch: the batch X_val[some_indices] that you feed into val_dict.
#    we could also pick this from val-dict, but this workflow is cleaner
#  - sl_plot: the time-stamp where you'd like to visualize
#  - ind: some index into the batch. if -1, we'll pick a random one"""
#  try:
#    result = sess.run([model.mu1,model.mu2,model.mu3,model.s1,model.s2,model.s3,model.rho],feed_dict=val_dict)
#  except:
#    print('We cannot fetch all variables for the MDN')
#  batch_size,crd,seq_len = batch.shape
#  assert ind < batch_size, 'Your index is outside batch'
#  assert sl_plot < seq_len, 'Your sequence index is outside sequence'
#  if ind == -1: ind = np.random.randint(0,batch_size)
#  delta = 0.025  #Grid size to evaluate the PDF
#  width = 1.0  # how far to evaluate the pdf?
#
#  fig = plt.figure()
#  ax = fig.add_subplot(2,2,1,projection='3d')
#  ax.plot(batch[ind,0,:], batch[ind,1,:], batch[ind,2,:],'r')
#  ax.scatter(batch[ind,0,sl_plot], batch[ind,1,sl_plot], batch[ind,2,sl_plot])
#  ax.set_xlabel('x coordinate')
#  ax.set_ylabel('y coordinate')
#  ax.set_zlabel('z coordinate')
#
#  mean1 = result[0][ind,0,sl_plot]
#  mean2 = result[1][ind,0,sl_plot]
#  mean3 = result[2][ind,0,sl_plot]
#  sigma1 = result[3][ind,0,sl_plot]
#  sigma2 = result[4][ind,0,sl_plot]
#  sigma3 = result[5][ind,0,sl_plot]
#  sigma12 = result[6][ind,0,sl_plot]*sigma1*sigma2
#
#  ax = fig.add_subplot(2,2,2)
#
#  x1 = np.arange(-width, width, delta)
#  x2 = np.arange(-width, width, delta)
#  X1, X2 = np.meshgrid(x1, x2)
#  Z = mlab.bivariate_normal(X1, X2, sigma1, sigma2, mean1, mean2,sigma12)
#  CS = ax.contour(X1, X2, Z)
#  plt.clabel(CS, inline=1, fontsize=10)
#  ax.set_xlabel('x coordinate')
#  ax.set_ylabel('y coordinate')
#
#  ax = fig.add_subplot(2,2,3)
#  x3 = np.arange(-width, width, delta)
#  X1, X3 = np.meshgrid(x1, x3)
#  Z = mlab.bivariate_normal(X1, X3, sigma1, sigma3, mean1, mean3)
#  CS = ax.contour(X1, X3, Z)
#  plt.clabel(CS, inline=1, fontsize=10)
#  ax.set_xlabel('x coordinate')
#  ax.set_ylabel('Z coordinate')
#
#  ax = fig.add_subplot(2,2,4)
#  X2, X3 = np.meshgrid(x2, x3)
#  Z = mlab.bivariate_normal(X2, X3, sigma2, sigma3, mean2, mean3)
#  CS = ax.contour(X2, X3, Z)
#  plt.clabel(CS, inline=1, fontsize=10)
#  ax.set_xlabel('y coordinate')
#  ax.set_ylabel('Z coordinate')

def plot_traj_MDN_mult(model,sess,val_dict,batch,sl_plot = 5, ind = -1):
  """Plots the trajectory. At given time-stamp, it plots the probability distributions
  of where the next point will be
  THIS IS FOR MULTIPLE MIXTURES
  input:
  - sess: the TF session
  - val_dict: a dictionary with which to evaluate the model
  - batch: the batch X_val[some_indices] that you feed into val_dict.
    we could also pick this from val-dict, but this workflow is cleaner
  - sl_plot: the time-stamp where you'd like to visualize
  - ind: some index into the batch. if -1, we'll pick a random one"""
  result = sess.run([model.mu1,model.mu2,model.mu3,model.s1,model.s2,model.s3,model.rho,model.theta],feed_dict=val_dict)
  batch_size,crd,seq_len = batch.shape
  assert ind < batch_size, 'Your index is outside batch'
  assert sl_plot < seq_len, 'Your sequence index is outside sequence'
  if ind == -1: ind = np.random.randint(0,batch_size)
  delta = 0.025  #Grid size to evaluate the PDF
  width = 1.0  # how far to evaluate the pdf?

  fig = plt.figure()
  ax = fig.add_subplot(2,2,1,projection='3d')
  ax.plot(batch[ind,0,:], batch[ind,1,:], batch[ind,2,:],'r')
  ax.scatter(batch[ind,0,sl_plot], batch[ind,1,sl_plot], batch[ind,2,sl_plot])
  ax.set_xlabel('x coordinate')
  ax.set_ylabel('y coordinate')
  ax.set_zlabel('z coordinate')


  # lower-case x1,x2,x3 are indezing the grid
  # upper-case X1,X2,X3 are coordinates in the mesh
  x1 = np.arange(-width, width+0.1, delta)
  x2 = np.arange(-width, width+0.2, delta)
  x3 = np.arange(-width, width+0.3, delta)
  X1,X2,X3 = np.meshgrid(x1,x2,x3,indexing='ij')
  XX = np.stack((X1,X2,X3),axis=3)

  PP = []

  mixtures = result[0].shape[1]
  for m in range(mixtures):
    mean = np.zeros((3))
    mean[0] = result[0][ind,m,sl_plot]
    mean[1] = result[1][ind,m,sl_plot]
    mean[2] = result[2][ind,m,sl_plot]
    cov = np.zeros((3,3))
    sigma1 = result[3][ind,m,sl_plot]
    sigma2 = result[4][ind,m,sl_plot]
    sigma3 = result[5][ind,m,sl_plot]
    sigma12 = result[6][ind,m,sl_plot]*sigma1*sigma2
    cov[0,0] = np.square(sigma1)
    cov[1,1] = np.square(sigma2)
    cov[2,2] = np.square(sigma3)
    cov[1,2] = sigma12
    cov[2,1] = sigma12
    rv = multivariate_normal(mean,cov)
    P = rv.pdf(XX)  #P is now in [x1,x2,x3]
    PP.append(P)
  # PP is now a list
  PP = np.stack(PP,axis=3)
  # PP is now in [x1,x2,x3,mixtures]
  #Multiply with the mixture
  theta_local = result[7][ind,:,sl_plot]
  ZZ = np.dot(PP,theta_local)
  #ZZ is now in [x1,x2,x3]

  print('The theta variables %s'%theta_local)


  #Every Z is a marginalization of ZZ.
  # summing over axis 2, gives the pdf over x1,x2
  # summing over axis 1, gives the pdf over x1,x3
  # summing over axis 0, gives the pdf over x2,x3
  ax = fig.add_subplot(2,2,2)
  X1, X2 = np.meshgrid(x1, x2)
  Z = np.sum(ZZ,axis=2)
  CS = ax.contour(X1, X2, Z.T)
  plt.clabel(CS, inline=1, fontsize=10)
  ax.set_xlabel('x coordinate')
  ax.set_ylabel('y coordinate')

  ax = fig.add_subplot(2,2,3)
  X1, X3 = np.meshgrid(x1, x3)
  Z = np.sum(ZZ,axis=1)
  CS = ax.contour(X1, X3, Z.T)
  plt.clabel(CS, inline=1, fontsize=10)
  ax.set_xlabel('x coordinate')
  ax.set_ylabel('Z coordinate')

  ax = fig.add_subplot(2,2,4)
  X2, X3 = np.meshgrid(x2, x3)
  Z = np.sum(ZZ,axis=0)
  CS = ax.contour(X2, X3, Z.T)
  plt.clabel(CS, inline=1, fontsize=10)
  ax.set_xlabel('y coordinate')
  ax.set_ylabel('Z coordinate')






# Piece of code doesn;t work yet
#def tf_3d_normal(x_in, mu_in, s1, s2, s3, rho12, rho13, rho23):
#
#      x = tf.sub(x_in,mu_in)
#      V11 = tf.pow(s1,2)
#      V12 = tf.mul(tf.mul(rho12,s1),s2)
#      V13 = tf.mul(tf.mul(rho13,s1),s3)
#      V22 = tf.pow(s2,2)
#      V23 = tf.mul(tf.mul(rho23,s2),s3)
#      V22 = tf.pow(s3,2)
#
#      cov = tf.pack([tf.pack([V11,V12.V13]),tf.pack([V12,V22,V23]),tf.pack([V13,V23,V33])])
#      quad = tf.matmul(x,tf.matmul(tf.inv(cov),tf.transpose(x)))
#      expo = tf.exp(tf.mul(tf.constant([-0.5]),quad))
#      den = tf.mul(tf.pow((tf.constant([2*3.1415])),tf.constant([3.0/2.0])),
#      determinant = tf.mul(V11,tf.mul(V22,V33)) - tf.mul(V11,tf.pow(V23,2)) + tf.mul(V12,tf.sub(tf.mul(tf.mul(tf.constant([2]),V13),V23),tf.mul(V12,V33))) - tf.mul(V22,tf.pow(V13,2))
#
#
#						w = tf.pow(x1,2)
#
##      s1s2 = tf.mul(s1, s2)
##      z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2*tf.div(tf.mul(rho, tf.mul(norm1, norm2)), s1s2)
##      negRho = 1-tf.square(rho)
##      result = tf.exp(tf.div(-z,2*negRho))
##      denom = 2*np.pi*tf.mul(s1s2, tf.sqrt(negRho))
##      result = tf.div(result, denom)
#      return result