## polyfit Exercise

import argparse
import sys

import numpy as np
from matplotlib import pyplot as plt
import numpy.linalg as la

# Read data matrix X and labels t from text file.
def read_data(file_name):
#  YOUR CODE here:
  data_set=np.loadtxt(file_name)
  X1 = data_set[:,0]
  t = data_set[:, 1]

  return X1, t


# Implement normal equations to compute w = [w0, w1, ..., w_M].
def train(X1, M, t):
#  YOUR CODE here:
  X2=X1.ravel()
  X=(np.vander(X2, M+1,increasing=True))
  w =np.dot(np.dot(la.inv(np.dot(X.T,X)),X.T),t)
  return w


def train_regu(X, M,lamda, t):
#  YOUR CODE here:
  X=X.ravel()
  X=(np.vander(X, M+1,increasing=True))
  I=np.eye(M+1)
  N=len(t)
#  w =np.dot(np.dot(la.inv(np.dot(lamda*I+X.T,X)),X.T),t)
  w =np.dot(np.dot(la.inv(lamda*I*N+ np.dot(X.T,X)),X.T),t)
  return w




# Compute RMSE on dataset (X, t).
def compute_rmse(X,M ,t, w):
#  YOUR CODE here:\
  X=X.ravel()
  X=(np.vander(X, M+1,increasing=True))
  return np.sqrt(np.sum((np.sum(w.T*X,axis=1)-t)**2)/X.shape[0])


# Compute objective function (cost) on dataset (X, t).
def compute_cost(X, M, t, w):
#  YOUR CODE here:
  X=X.ravel()
  X=(np.vander(X, M+1,increasing=True))
  return np.sum((np.sum(w.T*X,axis=1)-t)**2)/(2*X.shape[0])


##======================= Main program =======================##
parser = argparse.ArgumentParser('Poly fit Exercise.')
parser.add_argument('-i', '--input_data_dir',
                    type=str,
                    default='../data/polyfit',
                    help='Directory for the polyift dataset.')
FLAGS, unparsed = parser.parse_known_args()

# Read the training and test data.
Xtrain, ttrain = read_data(FLAGS.input_data_dir + "/train.txt")
Xtest, ttest = read_data(FLAGS.input_data_dir + "/test.txt")
Xdevel, tdevel = read_data(FLAGS.input_data_dir + "/devel.txt")
#  YOUR CODE here: add the bias feature to each training and test example,
#                  create new design matrices X1train and X1test.
X1train =Xtrain
X1test = Xtest


M=9
train_rmse=[]
test_rmse=[]
model=[]
for i in range(M+1):
# Train model on training examples.
  w = train(X1train,i, ttrain)
  model=np.append(i,model)
# Print model parameters.
#  print('M =', i, '\n')
#  print('Params: ', w, '\n')
# Print RMSE on training data.
  a=compute_rmse(X1train, i, ttrain, w)
  train_rmse=np.append(a, train_rmse)
 # train_rmse=train_rmse.append(a)
#  print('Training  for M=%2i RMSE:  %0.2f.' %(i,a ))

# Print RMSE on test data.
  b=compute_rmse(X1test,i, ttest, w)
  test_rmse=np.append(b,test_rmse)
#  print('Test for M=%2i RMSE: %0.2f.' %( i,b))
  

plt.xlabel('M')
plt.ylabel('RMSE')
plt.plot( model,train_rmse , 'bo-', label= 'train_rmse')
plt.plot( model,test_rmse , 'go-', label= 'test_rmse')
plt.legend()
plt.savefig('train-test-rmse-without-regu.png')
plt.close()




M=9
train_rmse=[]
devel_rmse=[]
model=[]

for i in range(0,51,5):
  lamda=np.exp(i-50*1.0)
  w = train_regu(X1train,M,lamda, ttrain)
  model=np.append(i-50,model)
  a=compute_rmse(X1train, M, ttrain, w)
  train_rmse=np.append(a, train_rmse)
  print(i-50,a)
  b=compute_rmse(Xdevel,M, tdevel, w)
  devel_rmse=np.append(b,devel_rmse)
  

plt.xlabel(' ln lamda')
plt.ylabel('RMSE')
plt.plot( model,train_rmse , 'bo-', label= 'train_rmse')
plt.plot( model,devel_rmse , 'go-', label= 'devel_rmse')
plt.legend()
plt.savefig('train-devel-rmse-with-regu.png')
plt.close()



w = train(X1train,M, ttrain)
b=compute_rmse(X1test,M, ttest, w)
print('Test without for M=%2i RMSE: %0.2f.' %( M,b))
lamda=np.exp(-10)
w = train_regu(X1train,M,lamda, ttrain)
b=compute_rmse(X1test,M, ttest, w)
print('Test with for M=%2i RMSE: %0.2f.' %( M,b))









