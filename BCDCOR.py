import numpy as np
from math import *
import sklearn
from sklearn.metrics.pairwise import  euclidean_distances

def BCDCOR(x, y):
    '''
    This function calculates the distance correlation coefficient, Which is 
    modified from the bcdcor() and DCOR() function from R package energy. 
    ----------------------------------------------------------------------
    Parameters
    -----------
    x: The variable $x$ with it's n observations. 
    TYPE: numpy array.
    shape: (observations, dimensions).
    ----
    y: The variable $y$ with it's n observations. 
    TYPE: numpy array.
    shape: (observations, dimensions).
    ----------------------------------------------------------------------
    Returns
    -----------
    The the distance correlation coefficient between x and y. 
    
    You can refer to the formula in 
    https://doi.org/10.1016/j.jmva.2013.02.012.
    ----------------------------------------------------------------------
    '''
    x = euclidean_distances(x,x)
    y = euclidean_distances(y,y)
    
    n = x.shape[0]
    AA = Astar(x)
    BB = Astar(y)
    s1 = AA*AA
    s2 = BB*BB
    s3 = AA*BB
    XY = np.sum(s3) - (n/(n-2))*np.sum(np.diag(s3))
    XX = np.sum(s1) - (n/(n-2))*np.sum(np.diag(s1))
    YY = np.sum(s2) - (n/(n-2))*np.sum(np.diag(s2))
    
    return XY/sqrt(XX*YY)


def Astar(d):
    '''
    This function calculates the A^{*} in formula 2.8 in 
    https://doi.org/10.1016/j.jmva.2013.02.012.

    '''
    n = d.shape[0]
    m = np.mean(d, axis=0)
    M = np.mean(d)
    A = d - m.reshape(n,1)-m.reshape(1,n) + M - d/n
    np.fill_diagonal(A,  m-M) 
    
    return (n/(n-1))*A