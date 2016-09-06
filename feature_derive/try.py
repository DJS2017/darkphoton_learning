import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import psycopg2
import csv


conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
cur_nl = conn.cursor()
cur_nl.execute("SELECT eid,eepx,eepy,eepz,eee,nups,upsp3,upscosth,upsphi,upsenergy,recoilmass FROM mcevent WHERE nups>0")
rows_nl = cur_nl.fetchall()
data_mc = np.array(rows_nl, dtype=object)
data = {'eid':data_mc[:,0],
        'eepx':data_mc[:,1],
        'eepy':data_mc[:,2],
        'eepz':data_mc[:,3],
        'eee':data_mc[:,4],
        'nups':data_mc[:,5],
        'upsp3':data_mc[:,6],
        'upscosth':data_mc[:,7],
        'upsphi':data_mc[:,8],
        'upsenergy':data_mc[:,9],
        'recoilmass':data_mc[:,10]}
frame = DataFrame(data)


plt.figure(1,(15,4))
plt.hist(frame['eee'],bins=50,range=[-10,10])
plt.hist(frame['upsenergy'],bins=50,range=[-10,10])
plt.hist(frame['']
