#!/usr/bin/env python

import numpy as np 
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
import copy

class GridSearchCV:
    """
    This class uses cross-validation to optimize hyper-parameters
    of model by dooing grid search.

    Parameters
    -----------
    estimator: any model class
                  should contain 'fit', 
                  'score', 'get_params' and 'set_params' method


    params:  dict, map string to any
             Dictionary of parameters as keys, and list of descrete
             values as values


    cv: positive int
        number of cross-validations

    """

    def __init__(self, estimator, params, cv=20):
        self.estimator = estimator
        self.params = params

        # optimal params and estimators we will return
        #self.optimal_params = 0
        #self.optimal_estimator = 0

        # cross validation
        self.cv = cv
        self.kfold = KFold(n_splits=self.cv)



    def fit_param(self, X, y, param):
    	"""
    	For a certain param, return the score and corresponding estimator
    	"""

        # verify the given param is a valid param in estimator
        valid_params = self.estimator.get_params()
        valid_keys = valid_params.keys()
        for key in param.iterkeys():
        	if key not in valid_keys:
        		raise Exception("Invalid params!", key)

        # set param as estimator's parameters and calculate its score
        # deep copy estimator instance and assign params
        estimator = copy.deepcopy(self.estimator)
        estimator.set_params(param)

        score_sum = 0.0
        count = 0
        for train_idx, test_idx in self.kfold.split(X):
            count = count + 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            estimator.fit(X_train, y_train)
            score = estimator.score(X_test, y_test)
            score_sum = score_sum + score

        return score_sum / count, estimator



    def fit(self, X, y):
        """Loop all possible params provided and find the one
           that can maximize score.

        """
        max_score = (-1) * 10E8
        max_estimator = copy.deepcopy(self.estimator)
        
        for param in list(ParameterGrid(self.params)):
            score, estimator = self.fit_param(X, y, param)
            if(score > max_score):
                max_score = score
                max_estimator = estimator
                max_params = param

        self.optimal_params = max_params
        self.optimal_estimator = max_estimator
        return self





        








        

