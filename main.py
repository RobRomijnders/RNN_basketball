# -*- coding: utf-8 -*-
"""
This code comes with the paper
"Applying Deep Learning to Basketball Trajectories"
By Rajiv Shah and Rob Romijnders

Reach us at
rshah@pobox.com
romijndersrob@gmail.com


MAKE SURE to unpack your data and set the "direc" variable to
the directory where your data is stored.

"""

#The folder where your dataset is. Note that is must end with a '/'
direc = 'data/'

plot = False                     #Set True if you wish plots and visualizations

from sklearn import metrics
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from util_basket import *
from util_MDN import *
import sklearn
from dataloader import *
from model import *


"""Hyperparameters"""
config = {}
config['MDN'] = MDN = False       #Set to falso for only the classification network
config['num_layers'] = 2         #Number of layers for the LSTM
config['hidden_size'] = 64     #Hidden size of the LSTM
config['max_grad_norm'] = 1      #Clip the gradients during training
config['batch_size'] = batch_size = 64
config['sl'] = sl = 12           #Sequence length to extract data
config['mixtures'] = 3           #Number of mixtures for the MDN
config['learning_rate'] = .005   #Initial learning rate


ratio = 0.8                      #Ratio for train-val split
plot_every = 100                 #How often do you want terminal output for the performances
max_iterations = 20000             #Maximum number of training iterations
dropout = 0.7                    #Dropout rate in the fully connected layer

db = 5                            #distance to basket to stop trajectories


"""Load the data"""
#The name of the dataset. Note that it must end with '.csv'
csv_file = 'seq_all.csv'
#Load an instance
center = np.array([5.25, 25.0, 10.0])   #Center of the basket for the dataset
dl = DataLoad(direc,csv_file, center)
#Munge the data. Arguments see the class

dl.munge_data(11,sl,db)
#Center the data
dl.center_data(center)
dl.split_train_test(ratio = 0.8)
data_dict = dl.data
if plot:
  dl.plot_traj_2d(20,'at %.0f feet from basket'%db)

X_train = np.transpose(data_dict['X_train'],[0,2,1])
y_train = data_dict['y_train']
X_val = np.transpose(data_dict['X_val'],[0,2,1])
y_val = data_dict['y_val']

N,crd,_ = X_train.shape
Nval = X_val.shape[0]

config['crd'] = crd            #Number of coordinates. usually three (X,Y,Z) and time (game_clock)

#How many epochs ill we train
epochs = np.floor(batch_size*max_iterations / N)
print('Train with approximately %d epochs' %(epochs))

model = Model(config)

#A numpy array to collect performances
perf_collect = np.zeros((7,int(np.floor(max_iterations /plot_every))))

sess = tf.Session()

#Initial settings for early stopping
auc_ma = 0.0
auc_best = 0.0

if True:
#  writer = tf.train.SummaryWriter("/home/rob/Dropbox/ml_projects/MDN/log_tb", sess.graph)

  sess.run(tf.initialize_all_variables())

  step = 0      # Step is a counter for filling the numpy array perf_collect
  i=0
  early_stop = False
  while i < max_iterations and not early_stop:
    batch_ind = np.random.choice(N,batch_size,replace=False)
    if i%plot_every == 0:
      ### Check training performance ###
      if MDN:
        fetch = [model.accuracy,model.cost,model.cost_seq]
      else:
        fetch = [model.accuracy,model.cost]
      result = sess.run(fetch,feed_dict = { model.x: X_train[batch_ind], model.y_: y_train[batch_ind], model.keep_prob: 1.0})
      perf_collect[0,step] = result[0]
      perf_collect[1,step] = cost_train = result[1]
      if MDN:
        perf_collect[4,step] = cost_train_seq = result[2]
      else:
        cost_train_seq = 0.0

      ### Check validation performance ###
      batch_ind_val = np.random.choice(Nval,batch_size,replace=False)
      if MDN:
        fetch = [model.accuracy,model.cost,model.merged,model.h_c,model.cost_seq]
      else:
        fetch = [model.accuracy,model.cost,model.merged,model.h_c]

      result = sess.run(fetch, feed_dict={ model.x: X_val[batch_ind_val], model.y_: y_val[batch_ind_val], model.keep_prob: 1.0})
      acc = result[0]
      perf_collect[2,step] = acc
      perf_collect[3,step] = cost_val = result[1]
      if MDN:
        perf_collect[5,step] = cost_val_seq = result[4]
      else:
        cost_val_seq = 0.0

      #Perform early stopping according to AUC score on validation set
      sm_out = result[3]
      #Pick of the column in sm_out is arbitrary. If you see consistently AUC's under 0.5, then switch columns
      AUC = sklearn.metrics.roc_auc_score(y_val[batch_ind_val],sm_out[:,1])
      perf_collect[6,step] = AUC
      ma_range = 5 #How many iterations to average over for AUCS
      if step > ma_range:
        auc_ma = np.mean(perf_collect[6,step-ma_range+1:step+1])
      elif step > 1 and step <= ma_range:
        auc_ma = np.mean(perf_collect[6,:step+1])

      if auc_best < AUC: auc_best = AUC
      # if auc_ma < 0.8*auc_best: early_stop = True
      #Write information to TensorBoard
      summary_str = result[2]
#      writer.add_summary(summary_str, i)
#      writer.flush()
      print("At %6s / %6s val acc %5.3f and AUC is %5.3f(%5.3f) trainloss %5.3f / %5.3f(%5.3f)" % (i,max_iterations, acc, AUC,auc_ma,cost_train,cost_train_seq,cost_val_seq ))
      print "At {}, the training cost is {}, the valid cost is {}".format(i, perf_collect[1,step], perf_collect[3,step])
      print("The best AUC is %6s")%auc_best
      step +=1
    sess.run(model.train_step,feed_dict={model.x:X_train[batch_ind], model.y_: y_train[batch_ind], model.keep_prob: dropout})
    i += 1
  #In the next line we also fetch the softmax outputs
  batch_ind_val = np.random.choice(Nval,batch_size,replace=False)
  result = sess.run([model.accuracy,model.numel], feed_dict={ model.x: X_val[batch_ind_val], model.y_: y_val[batch_ind_val], model.keep_prob: 1.0})
  acc_test = result[0]
  print('The network has %s trainable parameters'%(result[1]))
if plot:
  """Sample from the MDN"""
  if MDN:
    val_dict = { model.x: X_val[batch_ind_val], model.y_: y_val[batch_ind_val], model.keep_prob: 1.0}
    batch = X_val[batch_ind_val]
    plot_traj_MDN_mult(model,sess,val_dict,batch)

    sl_pre = 5  # the number of true sequences you feed
    seq_pre = X_val[3]
    seq_samp = model.sample(sess,seq_pre,sl_pre,bias=2.0)

  """Plot the performances"""
  plt.figure()
  plt.plot(perf_collect[0],label= 'Train accuracy')
  plt.plot(perf_collect[2],label = 'Valid accuracy')
  plt.legend()
  plt.show()


  plt.figure()
  plt.plot(perf_collect[1],label= 'Train class cost')
  plt.plot(perf_collect[3],label='Valid class cost')
  if MDN:
    plt.plot(perf_collect[4], label='Train seq cost')
    plt.plot(perf_collect[5],label='Valid seq cost')
  plt.legend()
  plt.show()

  """Export to Shiny for visualization"""
  export_shiny=False
  export_lab = []   #The logs to export containing the sl_pre and bias
  if export_shiny:
    block = 0
    seq_exp = seq_pre.copy()
    for pre in np.arange(5,10):
      for b in np.arange(0,4,0.5):
        seq_samp = model.sample(sess,seq_pre,pre,b)
        seq_exp = np.concatenate((seq_exp,seq_samp),axis=0)
        block += 1
        print('Block %.0f has sl_pre %.0f and bias %.3f'%(block,pre,b))
        export_lab.append(np.array([block,pre,b]))
        plt.close('all')
    np.savetxt(direc+'traj_shiny.csv',seq_exp,delimiter=',')
    export_lab = np.stack(export_lab,axis=1).T    #
    np.savetxt(direc+'traj_shiny_log.csv',export_lab,delimiter=',')
