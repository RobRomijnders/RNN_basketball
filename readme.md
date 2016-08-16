# LSTM with MDN for basketball trajectories

This is the code repository for our paper, [Applying Deep Learning to Basketball Trajectories](http://www.large-scale-sports-analytics.org/Large-Scale-Sports-Analytics/Submissions_files/paperID07.pdf). We also have written a short interactive [summary of the paper](http://tinyurl.com/traj-rnn) or find the full paper on [arXiv](https://arxiv.org/abs/1608.03793).  

This repo contains the data and tensorflow models we used in the paper. To run this code, it is necessary to have installed:  
- Tensorflow > 0.8
- Numpy
- Sklearn

## To create a model
* Unpack the seq_all.csv.tar.gz
* Run __main.py__ (This python script was designed to be run within an IDE, but will function as a standalone script.)
    * Within main.py there are a number of configuration settings that can be modified. These include settings for the model architecture, sequence length, distance to the basket, and performance measures. There is also an option to turn plotting on and see output at different model stages.

## The files
* main.py is the main file
* dataloader.py contains a class to load the data
    * util_*.py are two files with utility functions



