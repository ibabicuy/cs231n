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
    num_classes = W.shape[1]
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        scores_exp = np.exp(scores)
        loss -= np.log(scores_exp[y[i]]/np.sum(scores_exp))
        for j in xrange(num_classes):
            p_j = scores_exp[j]/np.sum(scores_exp)
            dW[:,j] += (p_j - (j == y[i])) * X[i]
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    dW /= num_train 
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
  
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
  
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    scores -= np.max(scores, axis=-1)[:, np.newaxis]
    scores_exp = np.exp(scores)
    sum_exp = np.sum(scores_exp, axis=1)
    corr_score = scores_exp[np.arange(num_train), y]
    loss = np.sum(np.log( corr_score / sum_exp))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W*W)
    mask = np.zeros((num_train, num_classes))
    mask[np.arange(num_train), y] = -1
    dW = np.dot(X.T,(scores_exp/sum_exp[:, np.newaxis] + mask))
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
  
    return loss, dW

