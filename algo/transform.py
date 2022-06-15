import numpy as np
import pandas as pd

def lift(y,h1,h2):
    """
    y: pd.Series containing univariate log_ret series
    h1: window size
    h2: offset
    Return: m x h1 Matrix containing sliding window of the time series, where m is the max number of slices
    
    Example: transform(df.log_ret.values,35,7) will output a matrix with rows containing a week of hourly returns, offset by one day
    """

    n = len(y)
    m = 1 + (n - h1) // h2 
    ind = np.arange(h1)[None,:] + h2*np.arange(m)[:,None]
    arr = y.values
    return arr[ind]

#def transform_kmeans_labels(labels,h1,h2,k):
#    
#    """
#    Input: the label of sliding windows from a k-means type algorithm
#    k: no. of clusters
#    Note it works only if h2 divides h1
#    Return: the frequency of label for original time series with complete votes, k columns
#    """
#    m = len(labels) - h1//h2 + 1
#    count = np.array([np.sum(labels[np.arange(h1//h2)[None,:] + np.arange(m)[:,None]] == i,axis = 1) for i in range(k)])
#    #[[0,1,2,...,h1//h2-1],[1,2,3,...h1//h2],...[m,m+1,...m+h1//h2-1]]
#    count = np.tile(count.T,h2).reshape(-1,k) # count is the same for the same h2 rows
#    return np.concatenate([np.zeros((h1-h2,k)),count,np.zeros((h1-h2,k))]) # Incomplete votes will be filled with 0
