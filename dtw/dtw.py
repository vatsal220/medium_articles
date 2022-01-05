import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def dtw(x,y):
    '''
    This function will calculate the cost matrix of two time series using the
    DTW formulation.
    
    params:
        x (Numpy Array) : The first time series
        y (Numpy Array) : The second time series
        
    returns:
        The cost matrix associated to the two time series.
        
    example:
        x = np.array([0,2,0,1,0, -1, 1])
        y = np.array([0,1,-1,-0,2,-1,0])
        dtw(x,y)
    '''
    n, m = len(x), len(y)
    
    # generate & initialize cost matrix
    cost_mat = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            if i == 0 and j == 0:
                cost_mat[i,j] = 0
            else:
                cost_mat[i,j] = np.inf

    # Fill the cost matrix 
    for i in range(1, n+1):
        for j in range(1, m+1):
            c = abs(x[i-1] - y[j-1])
            _min = min([cost_mat[i-1,j-1], cost_mat[i-1,j], cost_mat[i, j-1]])
            cost_mat[i,j] = c + _min
    return cost_mat    

def main():
    x = np.array([0.5, 21, 34, 5, 2, 4, 2, 0])
    y = np.array([5, 42, 23, 4, 2, 4, 5, 3, 2, 1, 30, 0,0])
    res = dtw(x,y)
    
if __name__ == '__main__':
    main()