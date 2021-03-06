#!/usr/bin/env python

# This script applies SignalFeatureCompute to
# calculate features from signal data

import sys
sys.path.append('/home/yunxuanli/DarkMatter/darkphoton_learning/feature_derive/featuretable')
import numpy as np
import pandas as pd
import SignalFeatureCompute as fc



frame0 = fc.update_mcmass('mcevent_2l')
frame0.to_hdf('mcmass_2l.hdf','mcmass')

print 'massdiff, nlfrombrem, rphoton, recoilmass'

print 'massdiff'
frame1 = fc.update_massdiff('mcevent_2l')
frame1.to_hdf('massdiff_2l.hdf','massdiff')

print 'nlfrombrem'
frame2 = fc.update_nlFromBrem('mcevent_2l')
frame2.to_hdf('nlfrombrem_2l.hdf','nlfrombrem')

print 'rphoton'
frame3 = fc.update_rphoton('mcevent_2l')
frame3.to_hdf('rphoton_2l.hdf','rphoton')

print 'recoilmass'
frame4 = fc.update_recoilmass('rphoton_2l.hdf')
frame4.to_hdf('recoilmass_2l.hdf','recoilmass')


print 'rphoton_costh, v0mass_avg, upsmass, bestphoton'

print 'rphoton_costh'
frame1 = fc.update_rphoton_costh('rphoton_2l.hdf')
frame1.to_hdf('rphoton_costh_2l.hdf','rphoton_costh')

#print 'v0mass_avg'
#frame2 = fc.update_v0mass_avg('mcevent_2l')
#frame2.to_hdf('v0mass_avg_2l.hdf','v0mass_avg')

print 'upsmass'
frame3 = fc.update_upsmass('mcevent_2l')
frame3.to_hdf('upsmass_2l.hdf','upsmass')

print 'bestphoton'
frame4 = fc.update_bestphoton('rphoton_2l.hdf','rphoton_costh_2l.hdf','nlfrombrem_2l.hdf','mcevent_2l')
frame4.to_hdf('bestphoton_2l.hdf','bestphoton')
"""
