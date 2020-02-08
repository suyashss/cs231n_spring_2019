from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]
    for i in range(num_train):
        score = np.squeeze(np.transpose(W).dot(np.reshape(X[i,:],(-1,1))))
	#dW[:,y[i]] -= X[i,:]
	scale_factor = np.max(score)
	score -= scale_factor
	expterm = np.exp(score)
	den = np.sum(expterm)
	for j in range(C):
		dW[:,j] +=  expterm[j]*X[i,:]/den
		if j == y[i]:
			dW[:,j] -= X[i,:]
	loss += (np.log(den) - score[y[i]])
    loss /= num_train
    dW /= num_train
    loss += (reg*np.sum(np.square(W)))
    dW += (reg*2*W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]
    score = X.dot(W) # N x C
    score -= -np.max(score,axis=1,keepdims=True)
    expterm = np.exp(score) # N x C
	
    loss -= np.sum(score[np.arange(num_train),y]) 
    den = np.sum(expterm,axis=1,keepdims=True) # N x 1
    loss += np.sum(np.log(den))

    expterm /= den
    dW = np.transpose(X).dot(expterm) # D x C
    dummy = np.zeros((num_train,C))
    dummy[np.arange(num_train),y] = 1
    dW -= np.transpose(X).dot(dummy)

    loss /= num_train
    dW /= num_train

    loss += (reg*np.sum(np.square(W)))
    dW += (reg*2*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
