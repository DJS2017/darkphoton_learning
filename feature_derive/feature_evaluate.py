# This file can be used to evalute each feature's efficiency for signal/background calssification.

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import psycopg2
import csv

def massdiff(path):

    plt.switch_backend('agg')

    # connect to database, and obtain data
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()

    cur_nl.execute('''SELECT eid,nups,cr_score,massdiff FROM mcevent''')
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'cr_score':data_mc[:,2],
            'massdiff':data_mc[:,3]}
    frame = DataFrame(data)

    mass_signal = []
    mass_background = []
    for i in range(frame.shape[0]):
        result = frame.loc[i]

        for j in range(result['nups']):
            if(result['cr_score'][j] == True):
                mass_signal.append(result['massdiff'][j])
            else:
                mass_background.append(result['massdiff'][j])

    plt.figure(1,figsize=(8,4))
    plt.hist(mass_background, bins=100, range=[-1,1], color='b')
    plt.hist(mass_signal, bins=100, range=[-1,1], color='r')
    plt.savefig(path)


def recoilmass(path):
    plt.switch_backend('agg')

    # connect to database, and obtain data
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()

    cur_nl.execute('''SELECT eid,nups,cr_score,recoilmass FROM mcevent''')
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'cr_score':data_mc[:,2],
            'recoilmass':data_mc[:,3]}
    frame = DataFrame(data)

    recoilmass_signal = []
    recoilmass_background = []
    for i in range(frame.shape[0]):
        result = frame.loc[i]

        for j in range(result['nups']):
            if(result['cr_score'][j] == True):
                recoilmass_signal.append(result['recoilmass'][j])
            else:
                recoilmass_background.append(result['recoilmass'][j])

    plt.figure(1,figsize=(8,4))
    plt.hist(recoilmass_background, bins=100, range=[-5,5], color='b')
    plt.hist(recoilmass_signal, bins=100, range=[-5,5], color='r')
    plt.savefig(path)


def isrphoton(path):
    plt.switch_backend('agg')
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute('''SELECT eid,nups,cr_score,isrphoton_angle FROM mcevent WHERE nups>0 AND nnl>0''')
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'cr_score':data_mc[:,2],
            'isrphoton_angle':data_mc[:,3]}
    frame = DataFrame(data)

    print("finish inputting data")
    isrphoton_angle_signal = []
    isrphoton_angle_background = []
    for i in range(frame.shape[0]):
        print("i is ") + str(i)
        result = frame.loc[i]

        for j in range(result['nups']):
            if(result['cr_score'][j] == True):
                isrphoton_angle_signal.append(result['isrphoton_angle'][j])
            else:
                isrphoton_angle_background.append(result['isrphoton_angle'][j])

    plt.figure(1,(15,4))
    plt.hist(isrphoton_angle_background, bins=100, range=[-1.1,1.1], color='b')
    plt.hist(isrphoton_angle_signal, bins=100, range=[-1.1,1.1], color='r')
    plt.savefig(path)



def nlenergy(path):

    plt.switch_backend('agg')

    # connect to database, and obtain data
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()

    cur_nl.execute('''SELECT eid,nnl,nlenergy FROM mcevent''')
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nnl':data_mc[:,1],
            'nlenergy':data_mc[:,2]}
    frame = DataFrame(data)

    nle = []
    for i in range(frame.shape[0]):
        result = frame.loc[i]

        for j in range(result['nnl']):
            nle.append(result['nlenergy'][j])


    plt.figure(1,figsize=(8,4))
    plt.hist(nle, bins=100, range=[-1,10], color='b')
    plt.savefig(path)



def extraenergy(path):
    plt.switch_backend('agg')

    #connect to database, and obtain data
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    
    cur_nl.execute('''SELECT eid,nups,extraenergy,cr_score FROM mcevent''')
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'extraenergy':data_mc[:,2],
            'cr_score':data_mc[:,3]}
    frame = DataFrame(data)


    extraenergy_signal_list = []
    extraenergy_background_list = []
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        nups = result['nups']

        for j in range(nups):
            if(result['cr_score'][j] == True):
                extraenergy_signal_list.append(frame.loc[i,'extraenergy'])
            else:
                extraenergy_background_list.append(frame.loc[i,'extraenergy'])


    plt.figure(1,figsize=(8,4))
    plt.hist(extraenergy_background_list, bins=100, range=[-1,10], color='b')
    plt.hist(extraenergy_signal_list, bins=100, range=[-1,10], color='r')
    plt.savefig(path)


if __name__ == '__main__':
    import feature_evaluate
    #feature_evaluate.recoilmass("recoilmass.png")
    #feature_evaluate.isrphoton("isrphoton.png")
    #feature_evaluate.nlenergy("nlenergy.png")
    feature_evaluate.extraenergy("extraenergy.png")
