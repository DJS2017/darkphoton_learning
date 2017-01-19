#!/usr/bin/env python

# This script merges several components of data
# into a single file
# input parameters: 
# file1_name, file2_name, ..., fileN_name, newfile_name, key

import sys
import pandas as pd 
from pandas import DataFrame

if(len(sys.argv) < 4):
    raise IOEroor

# read file name and key
newfile_name = sys.argv[-2]
key = sys.argv[-1]
single_file_list = []
for i in range(1, len(sys.argv)-2):
    temp = pd.read_hdf(sys.argv[i], key)
    single_file_list.append(temp)

# concat files
signal = pd.concat(single_file_list, ignore_index=True)
signal.to_hdf(newfile_name, key)