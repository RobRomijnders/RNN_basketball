# -*- coding: utf-8 -*-
"""
Created on Sun May  8 17:35:13 2016

@author: rob
"""

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plot_basket(data,labels, extra_title=' '):
    """Utility function to plot XYZ trajectories of the basketbal data
    Input:
    - data: takes any 3D array, the first three indices of the second dimension must correspond to X, Y and Z
    - labels: the correct (binary) labels for the sequences. Same size as first dimension of data
    - extra_title: string to append to the title
    Output
    - Plots =)
    #Credits for this code go to
# http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    mpl.rcParams['legend.fontsize'] = 10
    """



    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    N,crd,sl = data.shape
    if crd>sl:
      data = np.transpose(data,[0,2,1])
      print('We transpose dimension 1 and 2')
    
    P = 100     #How many lines you want?
    ind = np.random.permutation(N)
    
    for p in ind[:P]:
        if labels[p] == 1:
            ax.plot(data[p,0,:], data[p,1,:], data[p,2,:],'r', label='miss')
        elif labels[p] == 0:
            ax.plot(data[p,0,:], data[p,1,:], data[p,2,:],'b', label='hit')
   
        elif labels[p] == 2:
            ax.plot(data[p,0,:], data[p,1,:], data[p,2,:],'c', label='lc hit')
   
        elif labels[p] == 3:
            ax.plot(data[p,0,:], data[p,1,:], data[p,2,:],'m', label='lc miss')
												
    #Next lines serve to only plot one legend per label
    handles, labels = ax.get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)
    ax.legend(newHandles, newLabels)											
    plt.title('Blue=hit, red=miss '+extra_title)
    plt.show()
    
def shuffle_basket(X,order,ind):
    """Function in order to calculate variable importance during session
    The function takes in
    - X, the data. Expected a 3D Tensor of N x crd x sl
     - N = number of samples
     - crd = how many coordinates we have
     - sl = sequence length
    - order, the kind of variable we check
     - 'crd' we will random shuffle different coordinates
     - 'sl' we will random shuffle different sequence indices
    - ind specifies the exact index along order to shuffle
    """
    X = X.copy()
    if order == 'crd':
        extract = X[:,ind,:].copy()   #Extracted 2D tensor N x sl
        np.random.shuffle(extract)
        X[:,ind,:] = extract
    elif order == 'sl':
        extract = X[:,:,ind].copy()   #Extracted 2D tensor N x sl
        np.random.shuffle(extract)
        X[:,:,ind] = extract        
    return X

def plot_vi(vi_crd, vi_sl):
    # Four axes, returned as a 2-d array
    D = vi_crd.shape[1]
    #Make color array to color baseline different
    color = ['b'] * (D)
    color[D-1] = 'r'
    
    ## TODO For now, the axes are defined with numerics, in future we might add
    # a better heuristic here
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].bar(xrange(D),vi_crd[0],color=color)
    axarr[0, 0].set_title('Accuracy - crd')
    axarr[0, 0].axis([0,D,0.6,1.0])
    axarr[0, 1].bar(xrange(D),vi_crd[1],color=color)
    axarr[0, 1].set_title('Cost - crd')
    axarr[0, 1].axis([0,D,0.3,1.0])
    if vi_sl is not None:
        # Four axes, returned as a 2-d array
        D = vi_sl.shape[1]
        #Make color array to color baseline different
        color = ['b'] * (D)
        color[D-1] = 'r'    
        axarr[1, 0].bar(xrange(D),vi_sl[0],color=color)
        axarr[1, 0].set_title('Accuracy - sl')
        axarr[1, 0].axis([0,D,0.6,1.0])
        axarr[1, 1].bar(xrange(D),vi_sl[1],color=color)
        axarr[1, 1].set_title('Cost - sl')
        axarr[1, 1].axis([0,D,0.3,1.0])

#    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
#    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
#    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

def conf_ind(conf_correct,y_val,am,which_conf):
    """Computes indices per class for which we have highest confidence
    Input arguments
    - conf_correct: Confidence at the correct target. That is the probability that the Softmax classifiers outputs at thus sample
    - y_val the true labels
    - am: how many indices per class you want?
    - which_conf: string label for which confidence you want?
    -- hc for high confidence
    -- lc for low confidence
    returns
    - conf_pos: indices at pos class for which we have high confidence
    - hc_ned: indices at nex class for which we have high confidence
    
    Note that this function only works for binary classes
    """
    N = len(y_val)  #How many samples we got?  
    
				
    if which_conf == 'hc':				
      conf_ind = np.argsort(conf_correct)[::-1]   #indices for which we have the highest confidence. 
      # descending order, becase we are interested in high confidence
      count = 0
    elif which_conf == 'lc':
      conf_ind = np.argsort(conf_correct)
      count = 0
	#We only want indices for which the confidence is at least 0.5
      while conf_correct[conf_ind[count]] < 0.5:
        count +=1
    y_val.astype(int)    #change so that we can use it for indexing
    
    #Set up some variables
    conf_pos = np.zeros((am))
    ipos = 0
    conf_neg = np.zeros((am))
    ineg = 0
    
    
    while count<N: #while we're still in the array
        if ipos<am and y_val[conf_ind[count]] == 0:
            #positive sample
            conf_pos[ipos] = conf_ind[count]
            ipos +=1
        if ineg<am and y_val[conf_ind[count]] == 1:
            conf_neg[ineg] = conf_ind[count]
            ineg +=1
        count +=1
    return conf_pos.astype(int), conf_neg.astype(int)
    

    
def plot_grad(data,labels,grad):
    """ Plots the gradient over the input trajectory wrt the cross-entropy cost
    Input:
    - data: the 3D Tensor containing the original trajectories. The first three indices
    of the second dimension must correspond to X Y and Z
    - labels: the corresponding target labels
    - grad: The gradient over the trajectory. Due to Tensorflow's default for 4D Tensors,
    also this input can be four dimensional
    Output
    - Plots =)
    """
    if len(grad.shape) == 4: #remove the fourth dimension. SHould be empty
      grad = np.squeeze(grad,axis=3)    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    N = data.shape[0]    
    
    P = np.min((100,N))     #How many lines you want?
    ind = np.random.permutation(N)
    
    for p in ind[:P]:
        if labels[p] == 1:
            ax.plot(data[p,0,:], data[p,1,:], data[p,2,:],'r', label='miss')
            length_quiver = np.linalg.norm(grad[p,:3,:],axis=0)
            ax.quiver(data[p,0,:], data[p,1,:], data[p,2,:],grad[p,0,:],grad[p,1,:],grad[p,2,:],length = 1.0, pivot = 'middle')
        else:
            ax.plot(data[p,0,:], data[p,1,:], data[p,2,:],'b', label='hit')
            ax.quiver(data[p,0,:], data[p,1,:], data[p,2,:],grad[p,0,:],grad[p,1,:],grad[p,2,:],length = 1.0, pivot = 'middle')

    plt.title('Blue=hit, red=miss')
    print('Red lines are misses, blue lines are hits')
    plt.show()
    
def abs_to_off(data):
  """Converts absolute data to offset data
  Every first vector in the sequence is a zero vector
  Input
  - data in size [N by coordinates by sequence_length]
  Output
  - offset data in size [N by coordinates by sequence_length]"""
  assert len(data.shape) == 3, 'abs_to_off() expects three dimensional matrices'
		
  off = np.zeros_like(data)
  sl = data.shape[2]
  off[:,:,1:] = data[:,:,1:] - data[:,:,:sl-1]
  return off  
  