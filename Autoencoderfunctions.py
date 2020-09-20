import numpy as np

# basic helper functions for a 2 layered autoencoder

def sigmoid(s):
  return 1/(1+np.exp(-s))
  
def encoder(X):
  out1 = sigmoid(np.dot(W['encoder_l1'],X)+b['encoder_l1'])
  out2 = sigmoid(np.dot(W['encoder_l2'],out1)+b['encoder_l2'])
  return out1, out2
  
def decoder(X):
  out1 = sigmoid(np.dot(W['decoder_l1'],X)+b['decoder_l1'])
  out2 = sigmoid(np.dot(W['decoder_l2'],out1)+b['decoder_l2'])
  return out1, out2
  
def NMSE(ypred, ytrue):
  N = np.shape(ypred)[0]
  MSE=np.sum((ypred - ytrue)**2)/N
  mu = np.sum(ytrue)/N
  NMSE = (N-1)*MSE*np.sum((ypred-mu)**2)
  return NMSE
  
def L1reg(lambda):
  l1reg = np.sum(np.abs(W['encoder_l1']))
  l1reg += np.sum(np.abs(W['encoder_l2']))
  l1reg += np.sum(np.abs(W['decoder_l1']))
  l1reg += np.sum(np.abs(W['decoder_l2']))
  l1reg *= lambda
  return l1reg

# DERIVATIVES
def dNMSE(ypred, ytrue):
  N = np.shape(ypred)[0]
  dMSE=2*np.sum((ypred - ytrue))/N
  mu = np.sum(ytrue)/N
  dNMSE = (N-1)*dMSE*np.sum((ypred-mu)**2)
  return dNMSE

def dsigmoid(s):
  return sigmoid(s)*(1-sigmoid(s))

def abs(x):
  return x/np.abs(x)
  
def dW(X):
  grad ={}
  out1,out2 = encoder(X)
  out3,out4 = decoder(out2)
  
  dNMSE_dout4 = dNMSE*dsigmoid(out4) #|N,input|
  dloss_dw4 = (out3.transpose()).dot(dNMSE_dout4) #|h1,N| x |N,input| = |h1,input|
  dloss_db4 = np.sum(dNMSE_dout4,0) #|1,input|
  
  dNMSE_dout3 = dNMSE_dout4.dot(W4.transpose())*dsigmoid(out3) # |N,input|x|input,h1|x|N,h1| =|N,h1|
  dloss_dw3 = (out2.transpose()).dot(dNMSE_dout3) #|h2,N| x |N,h1| = |h2,h1|
  dloss_db3 = np.sum(dNMSE_dout3,0) #|1,h1|
  
  dNMSE_dout2 = dNMSE_dout3.dot(W3.transpose())*dsigmoid(out2) # |N,h1|x|h1,h2|x|N,h2| =|N,h2|
  dloss_dw2 = (out1.transpose()).dot(dNMSE_dout2) #|h1,N| x |N,h2| = |h1,h2|
  dloss_db2 = np.sum(dNMSE_dout2,0) #|1,h2| 
  
  dNMSE_dout1 = dNMSE_dout2.dot(W2.transpose())*dsigmoid(out1) # |N,h2|x|h2,h1|x|N,h1| =|N,h1|
  dloss_dw1 = (X.transpose()).dot(dNMSE_dout1) #|input,N| x |N,h1| = |input,h1|
  dloss_db1 = np.sum(dNMSE_dout1,0) #|1,h1| 
  
  grad['dw1'] = dloss_dw1
  grad['dw2'] = dloss_dw2
  grad['dw3'] = dloss_dw3
  grad['dw4'] = dloss_dw4
  
  grad['db1'] = dloss_db1
  grad['db2'] = dloss_db2
  grad['db3'] = dloss_db3
  grad['db4'] = dloss_db4
  
  return grad
  

  
  


  
  
  
  

