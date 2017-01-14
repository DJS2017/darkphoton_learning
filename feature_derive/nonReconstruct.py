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
    
    print 'SQL query execution ...'
    sql = 'SELECT * from %s where nups=0'
    cur.execute(sql % 'mcevent_2l')
    print 'query successfully!'

    # derive features
    #features: upsmass, Amass1,Amass2,Amass3, massdiff, recoil(mass, energy, px, py, pz, costh), extraenergy, 
    #          piMap, muMap, eMap, PID, truth_matching, 
    table = []

    event = cur.fetchone()
    while(event != None):
        # MC: upsmcmass, A1mcmass, A2mcmass, A3mcmass
        upsmcmass = event['mcmass'][0]
        A1mcmass = event['mcmass'][1]
        A2mcmass = event['mcmass'][2]
        A3mcmass = event['mcmass'][3]
        
        temp = [str(event['eid']),
                upsmcmass, A1mcmass, A2mcmass, A3mcmass,
                0]
            table.append(temp)
        
        event = cur.fetchone()


    # insert into a dataframe, and store
    df = DataFrame(table, columns=['eid', 
                'upsmcmass', 'A1mcmass', 'A2mcmass', 'A3mcmass',
                'truth_matching'])
    df.to_hdf('nonReconstructtablestable.hdf','nonReconstruct')