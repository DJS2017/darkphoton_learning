#!/usr/bin/env python

# This script aims to split dataset into separate parts
# for exploration, train, valid, test and backup.
# The command-line parameter is:
#     filename, key

# The file should has columns called 'eid', to indicate
# which event each candidate belongs to

import pandas as pd 
from pandas import DataFrame
import numpy as np 
from sklearn.model_selection import train_test_split
import sys

# import data
signal = pd.read_hdf(sys.argv[1], sys.argv[2])

# separate datasets to explore, train, valid, test, backup
index_list = np.unique(signal.eid)
index_list, index_explore = train_test_split(index_list, test_size = 100000, random_state=42)
index_list, index_train = train_test_split(index_list, test_size = 800000, random_state=42)
index_list, index_valid = train_test_split(index_list, test_size = 200000, random_state=42)
index_backup, index_test = train_test_split(index_list, test_size = 200000, random_state=42)

signal_explore = signal[signal.eid.isin(index_explore)]
signal_train = signal[signal.eid.isin(index_train)]
signal_valid = signal[signal.eid.isin(index_valid)]
signal_test = signal[signal.eid.isin(index_test)]
signal_backup = signal[signal.eid.isin(index_backup)]

signal_explore.to_hdf('fullsignal_explore.hdf', sys.argv[2])
signal_train.to_hdf('fullsignal_train.hdf', sys.argv[2])
signal_valid.to_hdf('fullsignal_valid.hdf', sys.argv[2])
signal_test.to_hdf('fullsignal_test.hdf', sys.argv[2])
signal_backup.to_hdf('fullsignal_backup.hdf', sys.argv[2])
