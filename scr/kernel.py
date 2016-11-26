#!/usr/bin/env python

import numpy as np 
from scipy.spatial import cKDTree


class KernelRegression:
    """
    This class aims to estimate unkown point's value using kernel methods.
    
    Parameters
    -----------
    x:  np.array
        input data points.
    
    y:  np.array
        value
    
    n_neighbors: int
        Number of nearset neighbor points used to estimate a given point's
        value.
    
    kernel: string
        Type of kernel for estimation. Possible values:

        - 'uniform': uniform kernel. All points in each neighborhood are
          weighted equally.
        - 'gaussian': use gaussian as weights to each neighbor.
        - 'inverse': use inverse function to each neighbor.

    bandwidth: float
        value of bandwidth
    


    Examples
    -----------
    to be continued...
    """
    
    def __init__(self, n_neighbors=4, kernel_type='gaussian', bandwidth=0.01):
        self.n_neighbors = n_neighbors
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.params = {'n_neighbors':self.n_neighbors,
                       'kernel_type': self.kernel_type,
                       'bandwidth' : self.bandwidth}



    def fit(self, X, y):
        self.tree = cKDTree(X)
        self.values = y



    def kernel(self, distance):
        if(self.kernel_type == 'uniform'):
            return 1.0/2 * (distance/self.bandwidth < 1.0)
        elif(self.kernel_type == 'gaussian'):
            return 1.0/np.sqrt(2*np.pi) * np.exp(-1.0/2 * (distance/self.bandwidth)**2)
        else:
            raise Exception("Invalid kernel type!", self.kernel_type)



    def predict(self, X):
        """Predict the values for the provided data
        
        Parameters
        -----------
        X: array-like, shape(n_query, n_features)

        Returns
        -----------
        y: array of shape [n_samples]
        """
        
        result = []
        for x_ in X:
            denominator = 0.0
            numerator = 0.0
            dist, neighbor_index = self.tree.query([x_],k=self.n_neighbors)
            for i in range(len(dist[0])):
                denominator = denominator + self.kernel(dist[0][i])
                numerator = numerator + self.kernel(dist[0][i]) * self.values[neighbor_index[0][i]]
            
            if(numerator == 0.0):
                result.append(0.0)
            else:
                result.append(numerator*1.0/denominator)

        return np.array(result)

    
    def loss(self, X, y):
        """Return the train loss

        """
        y_predict = self.predict(X)
        return sum((y - y_predict)**2)/len(y_predict)


    def score(self, X, y):
        """Return the negative of loss as score.
        This method is used as interface for grid search in scikit-learn.
        """

        return -1 * self.loss(X,y)


    def get_params(self, deep=True):
        """Get params for this estimator

        Returns
        ----------
        params: dict of string to any.

        """

        result = {}
        result['bandwidth'] = self.bandwidth
        result['kernel_type'] = self.kernel_type
        result['n_neighbors'] = self.n_neighbors
        return result


    def set_params(self, params):
        """Set params of this estimator
        
        Parameter
        -----------
        params: dict
                map from string to any

        """

        valid_params = self.get_params()
        for key, value in params.iteritems():
            if(key in valid_params.keys()):
                setattr(self, key, value)

        return self







"""
class InverseKernelRegression(KernelRegression):

    #Specifies kernel regression when kernel is inverse functions.


    def __init__(self, x, y, n_neighbors=4, kernel='inverse', bandwidth=0.01,
                 order=2)

"""


