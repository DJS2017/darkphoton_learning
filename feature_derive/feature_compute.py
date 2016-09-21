# This file includes functions needed for feature process. 
# These functions are used to calculate high-level features from row data,
# and update in database

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import psycopg2
import csv


def massdiff(record):
    """
    # This function aims to calculate each candidate's massdiff given a collection of candidates.
    # The input record should have at least columns with name 'nups','upsd1idx','upsd2idx','upsd3idx','v0mass','massdiff'
    # Input: pandas.core.series.Series
    # Output: numpy.array
              each entry in the array is the massdiff of the corresponding candidate.

    """

    n = record['nups']
    mass_diff = np.zeros(n)

    for i in range(n):

        upsd1idx = record['upsd1idx'][i]
        upsd2idx = record['upsd2idx'][i]
        upsd3idx = record['upsd3idx'][i]
        
        mass1 = record['v0mass'][upsd1idx]
        mass2 = record['v0mass'][upsd2idx]
        mass3 = record['v0mass'][upsd3idx]
        
        massdiff1 = abs(mass1-mass2)
        massdiff2 = abs(mass2-mass3)
        massdiff3 = abs(mass3-mass1)
        
        if(massdiff1 > massdiff2):
            max2_massdiff = massdiff1
        else:
            max2_massdiff = massdiff2
        
        if(massdiff3 > max2_massdiff):
            mass_diff[i] = massdiff3
        else:
            mass_diff[i] = max2_massdiff
    
    return mass_diff



def update_massdiff(rawtable):
    """
    This function uses massdiff() to update each record's massdiff column in database.
    This is a candidate feature.

    parameters:
    -------------
    -rawtable:		name of raw table 	(string)
    """


    # connect to database, and obtain data
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()

    cur_nl.execute("SELECT eid,nups,upsd1idx,upsd2idx,upsd3idx,v0mass FROM %s WHERE nups>0" % rawtable)
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'upsd1idx':data_mc[:,2],
            'upsd2idx':data_mc[:,3],
            'upsd3idx':data_mc[:,4],
            'v0mass':data_mc[:,5],
            'massdiff':Series(data_mc.shape[0]*[np.zeros(1)])}
    frame = DataFrame(data)


    # deal with massdiff
    for i in range(frame.shape[0]):
        frame['massdiff'][i] = massdiff(frame.loc[i])

    return frame[['eid','massdiff']].set_index('eid')




def update_nlFromBrem(rawtable):
    """
    This function constructs
            nlFromBrem float[nnl]
    which indicates whether the corresponding photon is from Brem effect.
    This function is useful in extraenergy(), as nlFromBrem is actually the 
    mother index of photons generated from electrons.
    """
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute("SELECT eid,nele,nnl,eled2idx FROM %s WHERE nnl>0" % rawtable)
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nele':data_mc[:,1],
            'nnl':data_mc[:,2],
            'eled2idx':data_mc[:,3],
            'nlfrombrem':Series(data_mc.shape[0]*[np.zeros(1)])}

    frame = DataFrame(data)

    for i in range(frame.shape[0]):

        result = frame.loc[i]
        nele = result['nele']
        nnl = result['nnl']
        
        brem = np.zeros(nnl) - 1

        for j in range(nele):
            if(result['eled2idx'][j] != -1):
                brem[result['eled2idx'][j]] = j

        result['nlfrombrem'] = brem

    return frame[['eid','nlfrombrem']].set_index('eid')



