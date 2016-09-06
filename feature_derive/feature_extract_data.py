# This file is to extract processed data from database, and collect to table style, so that can be easily applied to
# machine learning & data mining algorithms.

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import psycopg2
import csv

def data2pandas(path):

    plt.switch_backend('agg')

    # connect to database, and obtain data
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()

    # obtain structure data from database
    cur_nl.execute('''SELECT eid,nups,cr_score,massdiff,upschi2,recoilmass,rphoton_costh,extraenergy,pid,upsmass,upsmasserr,upsenergy,upsp3,upsphi,upspreFitVtxx FROM mcevent WHERE nups>0 and nnl>0''')
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'cr_score':data_mc[:,2],
            'massdiff':data_mc[:,3],
            'upschi2':data_mc[:,4],
            'recoilmass':data_mc[:,5],
            'rphoton_costh':data_mc[:,6],
            'extraenergy':data_mc[:,7],
            'pid':data_mc[:,8],
            'upsmass':data_mc[:,9],
            'upsmasserr':data_mc[:,10],
            'upsenergy':data_mc[:,11],
            'upsp3':data_mc[:,12],
            'upsphi':data_mc[:,13],
	'upspreFitVtxx':data_mc[:,14]}

    df = DataFrame(data)

    # convert structure data to table
    candidate_feature_name = ['massdiff','recoilmass','rphoton_costh','upschi2','upsenergy','upsmass','upsmasserr','upsp3','upsphi','pid','cr_score','upspreFitVtxx']
    event_feature_name = ['extraenergy']
    data = []
    for i in range(df.shape[0]):
        result = df.loc[i]
        for j in range(result['nups']):
            data_array = np.append(np.array(result[candidate_feature_name].tolist())[:,j],np.array(result[event_feature_name].tolist()))
            data.append(data_array.tolist())

    frame = DataFrame(data, columns=candidate_feature_name+event_feature_name)
    frame.to_hdf(path,'df')



if __name__ == '__main__':
    import feature_extract_data
    #feature_evaluate.recoilmass("recoilmass.png")
    feature_extract_data.data2pandas("data_pandas.h5")
