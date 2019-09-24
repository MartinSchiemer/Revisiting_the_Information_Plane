"""
Author: Martin Schiemer
KDE estimator taken from:
https://github.com/artemyk/ibsgd/blob/master/kde.py
and adapted to numpy instead of keras backend
"""

import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
import scipy.special as sp


def get_dists_np(X):
    x2 = (X**2).sum(axis=1)[:,None]
    dists = x2 + x2.T - 2*X.dot(X.T)
    return dists

def get_shape(x):
    dims = x.shape[1] 
    N    = x.shape[0]
    return dims, N

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    # if there is an empty list for this class
    if len(x) == 0:
        return [0]
    else:
        N, dims = np.shape(x)
        dists = get_dists_np(x)

        dists2 = dists / (2*var)
        normconst = (dims/2.0)*np.log(2*np.pi*var)
        lprobs = sp.logsumexp(-dists2, axis=1) - np.log(N) - normconst
        h = -np.mean(lprobs)
        return [dims/2 + h,]

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I 
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    # if there is an empty list for this class
    if len(x) == 0:
        return [0]
    else:
        N, dims = np.shape(x)
        val = entropy_estimator_kl(x,4*var)
        return val + np.log(0.25)*dims/2

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)
