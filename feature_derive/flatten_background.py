#!/usr/bin/env python

# This script aims to flatten derived features 
# of signal data

import numpy as np
import csv
import pandas as pd
from pandas import DataFrame, Series

# load data
massdiff = pd.read_hdf('massdiff_background.hdf','massdiff')
recoilmass = pd.read_hdf('recoilmass_background.hdf','recoilmass')

# join data
frame = pd.merge(recoilmass,massdiff,left_index=True,right_index=True)

print('merged shape is:', frame.shape)

#flatten
datalist = []
for event_id in frame.index:
#    print event_id
    event = frame.loc[event_id]
    nups = len(event.massdiff)
    for candidate in range(nups):
        temp = [event_id, event.massdiff[candidate], event.recoilmass[candidate]]
        datalist.append(temp)

feature = pd.DataFrame(datalist,columns=('eid','massdiff','recoilmass'))
feature = feature.set_index('eid')

# save to HDF5 file
feature.to_hdf('feature_background.hdf','feature')







