#!/usr/bin/env python

# This script aims to reconstruct features for a event,
# our method is to choose the candidate with highest score.
# The command-line parameters are:
# input_file_name, output_file_name, key

# data processing packages
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame

# model packages
from sklearn.externals import joblib

# load model
rf = joblib.load('../models/gbdt100.pkl')

# obtain dataset
input_file_name = sys.argv[1]
output_file_name = sys.argv[2]
key = sys.argv[3]
signal_test = pd.read_hdf(input_file_name, key)
signal_test_without_na = signal_test.dropna()
signal_test_with_na = signal_test[signal_test.isnull().any(axis=1)]

feature_name = signal_test.columns.tolist()
feature_name.remove('eid')
feature_name.remove('truth_matching')
feature_name.remove('upsmcmass')
feature_name.remove('A1mcmass')
feature_name.remove('A2mcmass')
feature_name.remove('A3mcmass')

X_test = signal_test_without_na[feature_name]
y_test = signal_test_without_na['truth_matching']
print X_test.shape, y_test.shape

weight_test = np.zeros(y_test.shape)
weight_test = weight_test + 1
weight_test[y_test==1] = y_test.shape[0] / sum(y_test)

# calculate cr_score for without_na data points
signal_test_without_na['cr_score'] = pd.Series(rf.predict_proba(signal_test_without_na[feature_name])[:,1], index=signal_test_without_na.index)
signal_test_with_na['cr_score'] = 0

# construct event for without_na data points,
# pay attention: we do not need to construct event for with_na
# points, as they just have no candidates
event_list = []
for eid, candidates in signal_test_without_na.groupby('eid'):
    event_list.append(candidates.ix[candidates['cr_score'].argmax()])
events_test = DataFrame(event_list, columns=signal_test_without_na.columns)

# concat without_na and with_na data points together
signal_output = pd.concat([events_test, signal_test_with_na], ignore_index=True)
signal_output.to_hdf(output_file_name, key)
