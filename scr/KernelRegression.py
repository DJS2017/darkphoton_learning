#!/usr/bin/env python

import numpy as np 
from scipy.spatial import cKDTree


class KernelRegression:
	"""
    This class aims to estimate unkown point's value using kernel methods.
    parameter:
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

	def __init__(self, x, y, n_neighbors=4, kernel='exp', bandwidth=0.01):






class InverseKernelRegression(KernelRegression):
    """
    Specifies kernel regression when kernel is inverse functions.
    """

    def __init__(self, x, y, n_neighbors=4, kernel='inverse', bandwidth=0.01,
                 order=2)




