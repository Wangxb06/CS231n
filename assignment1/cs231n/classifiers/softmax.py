import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  outval = np.dot(X,W)
  num = X.shape[0]
  for item in range(num):
    proval = outval[item,y[item]]
    expsum = np.sum(np.exp(outval[item,:]))
    pro = np.exp(proval)/expsum
    loss += -np.log(pro)

    for ii  in  range(W.shape[1]) :
        if ii == y[item]:
            dW[:, ii] += X[item, :].T * (1 - pro)
        else:
            dW[:,ii] += -X[item, :].T * np.exp(outval[item,ii])/expsum

  loss = loss/num
  loss += reg/2.0 * np.sum(np.sum(np.square(W), axis=1))
  dW = -dW/num + reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  outval = np.dot(X, W)  #N*C
  num = X.shape[0]
  item = range(num)
  expsum = np.sum(np.exp(outval),axis=1,keepdims=True) #N
  pro = np.exp(outval) / expsum  #N*C
  loss = -np.sum(np.log(pro[item,y]))/num+reg/2.0 * np.sum(np.sum(np.square(W), axis=1))
  pro4dw = -pro
  pro4dw[item,y] = 1+pro4dw[item,y]
  dW = -np.dot(X.T,pro4dw)/num + reg*W  #D*C
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

class softmaxclassfier(object):
    def __init__(self,learningrate=0.01,reg=0,itertimes=1000):
        self.learningrate = learningrate
        self.reg = reg
        self.itertimes = itertimes
    def train(self,X,Y,X_val, y_val):
        num_train, dim = X.shape
        num_classes = np.max(Y) + 1  # assume y takes values 0...K-1 where K is number of classes
        W = np.random.randn(dim, num_classes) * 0.0001
        for iter in range(self.itertimes):
            loss_naive, grad_naive = softmax_loss_vectorized(W, X, Y, self.reg)
            W -= self.learningrate * grad_naive
            if iter % 50 == 0:
                self.W = W
                y_test_pred = self.predict(X_val)
                acc = np.mean(y_val == y_test_pred, dtype=np.float32)
                print("reg = %f, learningrate = %f , after %d time iterations, loss = %f ,acc=%f" %
                      (self.reg, self.learningrate, iter, loss_naive,acc))
        self.W = W
        y_test_pred = self.predict(X_val)
        acc = np.mean(y_val == y_test_pred,dtype=np.float32)
        y_val_pred = self.predict(X)
        acc_train = np.mean(Y == y_val_pred,dtype=np.float32)
        return loss_naive,acc,acc_train

    def predict(self,X):
        outval = np.dot(X, self.W)  # N*C
        return outval.argmax(axis=1)