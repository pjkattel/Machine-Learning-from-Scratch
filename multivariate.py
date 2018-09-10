## Multivariate Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la   #importing linear algebra

# Read data matrix X and labels t from text file.
def read_data(file_name):
#  YOUR CODE here:
  data_set=np.loadtxt(file_name)
  X = data_set[:,:-1] 
  t = data_set[:,data_set.shape[1]-1]
  

  return X, t


# Implement normal equations to compute w = [w0, w1, ..., w_M].
def train(X, t):
#  YOUR CODE here:
  X=np.column_stack((np.ones(X.shape[0]),X))
  w = np.dot(np.dot(la.inv(np.dot(X.T,X)),X.T),t) 

  return w


# Compute RMSE on dataset (X, t).
def compute_rmse(X, t, w):
#  YOUR CODE here:
  X=np.column_stack((np.ones(X.shape[0]),X))
  RMSEMUL= np.sqrt((np.sum(((np.sum(w.T*X,axis=1))-t)**2))/(X.shape[0]))
  return RMSEMUL


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, t, w):
#  YOUR CODE here:
  X=np.column_stack((np.ones(X.shape[0]),X))
  COST=np.sum((np.sum(w.T*X,axis=1)-t)**2)/(2*X.shape[0]) 
  return COST


##======================= Main program =======================##
parser = argparse.ArgumentParser('Multivariate Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/multivariate',
                    help='Directory for the multivariate houses dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")

#  YOUR CODE here: add the bias feature to each training and test example,
#                  create new design matrices X1train and X1test.
X1train = Xtrain
X1test = Xtest



# Train model on training examples.
w = train(X1train, ttrain)

# Print model parameters.
print('Params: ', w, '\n')

# Print cost and RMSE on training data.
print('Training RMSEMUL: %0.2f.' % compute_rmse(X1train, ttrain, w))
print('Training cost: %0.2f.' % compute_cost(X1train, ttrain, w))

# Print cost and RMSE on test data.
print('Test RMSEMUL: %0.2f.' % compute_rmse(X1test, ttest, w))
print('Test cost: %0.2f.' % compute_cost(X1test, ttest, w))


#  YOUR CODE here: plot the training and test examples with different symbols,
#                  plot the linear approximation on the same graph.


