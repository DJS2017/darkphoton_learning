#!/usr/bin/env python

# This script aims to compute Signal MC's features, and store
# them in a dataframe.

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import psycopg2
import psycopg2.extras

def pidMap(event, lund, index):
    if(abs(lund) == 211):
        return event['piselectorsmap'][event['pitrkidx'][index]]
    elif(abs(lund) == 11):
        return event['eselectorsmap'][event['eletrkidx'][index]]
    elif(abs(lund) == 13):
        return event['muselectorsmap'][event['mutrkidx'][index]]


if __name__ == '__main__':
	# connection to database
    print 'connect to database ...'
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli",host='positron02.hep.caltech.edu')
    cur = conn.cursor(cursor_factory = psycopg2.extras.DictCursor)
    print 'connect to database successfully!'
    
    sector = '2l'
    print 'SQL query execution ...'
    sql = 'SELECT * from %s'
    cur.execute(sql % 'mcevent_' + sector)
    print 'query successfully!'

    # derive features
    #features: upsmass, Amass1,Amass2,Amass3, massdiff, recoil(mass, energy, px, py, pz, costh), extraenergy, 
    #          piMap, muMap, eMap, PID, truth_matching, 
    table = []

    event = cur.fetchone()
    while(event != None):
        nups = event['nups']
        if(nups > 0):
            for candidate_id in range(nups):
                # MC: upsmcmass, A1mcmass, A2mcmass, A3mcmass
                upsmcmass = event['mcmass'][0]
                A1mcmass = event['mcmass'][1]
                A2mcmass = event['mcmass'][2]
                A3mcmass = event['mcmass'][3]

                ## upsmass
                upsmass = event['upsmass'][candidate_id]
                ups_p3 = event['upsp3'][candidate_id]
                ups_costh = event['upscosth'][candidate_id]
                ups_phi = event['upsphi'][candidate_id]
                ups_e = event['upsenergy'][candidate_id]
                ups_px = ups_p3 * np.sqrt(1 - ups_costh**2) * np.cos(ups_phi)
                ups_py = ups_p3 * np.sqrt(1 - ups_costh**2) * np.sin(ups_phi)
                ups_pz = ups_p3 * ups_costh
                
                ## dark photon
                A1 = event['upsd1idx'][candidate_id]
                A2 = event['upsd2idx'][candidate_id]
                A3 = event['upsd3idx'][candidate_id]
                #dark photon mass
                A1mass = event['v0mass'][A1]
                A2mass = event['v0mass'][A2]
                A3mass = event['v0mass'][A3]
                #dark photon mass difference
                massdiff = max(max(abs(A1mass-A2mass), abs(A2mass-A3mass)), abs(A3mass-A1mass))
                
                # recoil
                recoil_px = event['eepx'] - ups_px
                recoil_py = event['eepy'] - ups_py
                recoil_pz = event['eepz'] - ups_pz
                recoil_e = event['eee'] - ups_e
                recoil_costh = recoil_pz / np.sqrt(recoil_px**2 + recoil_py**2 + recoil_pz**2)
                recoil_mass2 = recoil_e**2 - (recoil_px**2 + recoil_py**2 + recoil_pz**2)
                
                # PID
                A1_lepton_lund = event['v0d1lund'][A1]
                A2_lepton_lund = event['v0d1lund'][A2]
                A3_lepton_lund = event['v0d1lund'][A3]
                
                A1_lepton1_idx = event['v0d1idx'][A1]
                A1_lepton2_idx = event['v0d2idx'][A1]
                A2_lepton1_idx = event['v0d1idx'][A2]
                A2_lepton2_idx = event['v0d2idx'][A2]
                A3_lepton1_idx = event['v0d1idx'][A3]
                A3_lepton2_idx = event['v0d2idx'][A3]
                
                A1_lepton1_pid = pidMap(event, A1_lepton_lund, A1_lepton1_idx)
                A1_lepton2_pid = pidMap(event, A1_lepton_lund, A1_lepton2_idx)
                
                A2_lepton1_pid = pidMap(event, A2_lepton_lund, A2_lepton1_idx)
                A2_lepton2_pid = pidMap(event, A2_lepton_lund, A2_lepton2_idx)
                
                A3_lepton1_pid = pidMap(event, A3_lepton_lund, A3_lepton1_idx)
                A3_lepton2_pid = pidMap(event, A3_lepton_lund, A3_lepton2_idx)
                
                # truth-matching
                isTrueA1 = (event['v0mcidx'][A1]>-1) and abs(A1_lepton_lund)==abs(event['mclund'][event['dauidx'][event['v0mcidx'][A1]]])
                isTrueA2 = (event['v0mcidx'][A2]>-1) and abs(A2_lepton_lund)==abs(event['mclund'][event['dauidx'][event['v0mcidx'][A2]]])
                isTrueA3 = (event['v0mcidx'][A3]>-1) and abs(A3_lepton_lund)==abs(event['mclund'][event['dauidx'][event['v0mcidx'][A3]]])
                isTrueUps = isTrueA1 and isTrueA2 and isTrueA3
                truth_matching = isTrueUps
            
                temp = [str(event['eid']) + sector,
                        upsmcmass, A1mcmass, A2mcmass, A3mcmass,
                        upsmass, 
                        A1mass, A2mass, A3mass, massdiff,
                        recoil_px, recoil_py, recoil_pz, recoil_e, recoil_costh, recoil_mass2,
                        A1_lepton1_pid, A1_lepton2_pid, A2_lepton1_pid, A2_lepton2_pid, A3_lepton1_pid, A3_lepton2_pid,
                        truth_matching]
                table.append(temp)
        else:
            upsmcmass = event['mcmass'][0]
            A1mcmass = event['mcmass'][1]
            A2mcmass = event['mcmass'][2]
            A3mcmass = event['mcmass'][3]

            temp = [str(event['eid']) + sector,
                    upsmcmass, A1mcmass, A2mcmass, A3mcmass,
                    float('nan'),
                    float('nan'), float('nan'), float('nan'), float('nan'),
                    float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                    float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                    0]
            table.append(temp)
        
        event = cur.fetchone()


    # insert into a dataframe, and store
    df = DataFrame(table, columns=['eid', 
                'upsmcmass', 'A1mcmass', 'A2mcmass', 'A3mcmass',
                'upsmass',
                'A1mass', 'A2mass', 'A3mass', 'massdiff',
                'recoil_px', 'recoil_py', 'recoil_pz', 'recoil_e', 'recoil_costh', 'recoil_mass2',
                'A1_lepton1_pid', 'A1_lepton2_pid', 'A2_lepton1_pid', 'A2_lepton2_pid', 'A3_lepton1_pid', 'A3_lepton2_pid',
                'truth_matching'])
    df.to_hdf('signaltable_'+sector+'.hdf','signal')