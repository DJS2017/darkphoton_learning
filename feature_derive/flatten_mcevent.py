#!/usr/bin/env python

# This script aims to flatten derived features 
# of signal data

import numpy as np
import csv
import pandas as pd
from pandas import DataFrame, Series

# load data
massdiff = pd.read_hdf('massdiff_2l.hdf','massdiff')
rphoton = pd.read_hdf('rphoton_2l.hdf','rphoton')
nlfrombrem = pd.read_hdf('nlfrombrem_2l.hdf','nlfrombrem')
recoilmass = pd.read_hdf('recoilmass_2l.hdf','recoilmass')
v0mass_avg = pd.read_hdf('v0mass_avg_2l.hdf','v0mass_avg')
upsmass = pd.read_hdf('upsmass_2l.hdf','upsmass')
bestphoton = pd.read_hdf('bestphoton_2l.hdf','bestphoton')
true_match = pd.read_hdf('true_match_2l.hdf','true_match')
mcmass = pd.read_hdf('mcmass_2l.hdf','mcmass')
rphoton_costh = pd.read_hdf('rphoton_costh_2l.hdf','rphoton_costh')

# join data
frame = pd.merge(rphoton,massdiff,left_index=True,right_index=True)
frame = pd.merge(frame,nlfrombrem,left_index=True,right_index=True)
frame = pd.merge(frame,true_match,left_index=True,right_index=True)
frame = pd.merge(frame, recoilmass,left_index=True,right_index=True)
frame = pd.merge(frame, v0mass_avg,left_index=True,right_index=True)
frame = pd.merge(frame, upsmass,left_index=True,right_index=True)
frame = pd.merge(frame, bestphoton,left_index=True,right_index=True)
frame = pd.merge(frame, mcmass,left_index=True,right_index=True)
frame = pd.merge(frame, rphoton_costh,left_index=True,right_index=True)
print('merged shape is:', frame.shape)

#flatten
datalist = []
for event_id in frame.index:
#    print event_id
    event = frame.loc[event_id]
    nups = len(event.massdiff)
    for candidate in range(nups):
        temp = [event_id, event.massdiff[candidate], event.rphoton_px[candidate], event.rphoton_py[candidate], event.rphoton_pz[candidate], event.rphoton_e[candidate], event.rphoton_costh[candidate], event.recoilmass[candidate], event.v0mass_avg[candidate], event.upsmass[candidate], event.bestphoton[candidate], event.true_matching[candidate], event.upsmcidx[candidate], event.upsvmcmass[0],event.upsvmcmass[1],event.upsvmcmass[2],event.upsvmcmass[3]]
        datalist.append(temp)

feature = pd.DataFrame(datalist,columns=('eid','massdiff','rphoton_px','rphoton_py','rphoton_pz','rphoton_e','rphoton_costh','recoilmass','v0mass_avg','upsmass','bestphoton','true_match','upsmcidx','upsmcmass','v0mcmass1','v0mcmass2','v0mcmass3'))
feature = feature.set_index('eid')

# save to HDF5 file
feature.to_hdf('feature_2l.hdf','feature')







