#This function return the p value of Distance Correlation

from BCDCOR import *
from scipy import stats
from math import *

def dcorT_p(x,y):
    '''
    This function calculates the p value distance correlation coefficient, 
    since the staistic obeys a t-distribution.
    
    You can refer to the formula in 
    https://doi.org/10.1016/j.jmva.2013.02.012.
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
    The the p value of the distance correlation coefficient between x and y. 
    ----------------------------------------------------------------------
    '''
    bcR = BCDCOR(x,y)#the distance correlation coefficient between x and y
    n = x.shape[0]
    M = n*(n-3)/2
    df = M-1
    tstat = sqrt(M-1) * (bcR/sqrt(1-bcR**2))
    p = 1 - stats.t.cdf(tstat ,df=df)
    
    return p

