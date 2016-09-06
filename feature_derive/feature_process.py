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




def update_massdiff():
    """
    This function uses massdiff() to update each record's massdiff column in database.
    This is a candidate feature.
    """
    # connect to database, and obtain data
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()

    cur_nl.execute('''SELECT eid,nups,upsd1idx,upsd2idx,upsd3idx,v0mass FROM mcevent WHERE nups>0''')
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

    # update massdiff
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        cur_nl.execute("UPDATE mcevent SET massdiff=%s WHERE eid=%s",(result['massdiff'].tolist(),result['eid']))

    conn.commit()




def update_nlFromBrem():
    """
    This function constructs
            nlFromBrem float[nnl]
    which indicates whether the corresponding photon is from Brem effect.
    This function is useful in extraenergy(), as nlFromBrem is actually the 
    mother index of photons generated from electrons.
    """
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute("SELECT eid,nele,nnl,eled2idx FROM mcevent WHERE nnl>0")
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

    # update to database
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        cur_nl.execute("UPDATE mcevent SET nlfrombrem=%s WHERE eid=%s",(result['nlfrombrem'].tolist(),result['eid']))

    conn.commit()




def update_rphoton():
    """
    This function updates 4-momentum of reconstructed photon, which is Pee - Pups.
    """
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()

    cur_nl.execute("SELECT eid,eepx,eepy,eepz,eee,nups,upsp3,upscosth,upsphi,upsenergy FROM mcevent WHERE nups>0")
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
            'rphoton_px':Series(data_mc.shape[0]*[np.zeros(1)]),
            'rphoton_py':Series(data_mc.shape[0]*[np.zeros(1)]),
            'rphoton_pz':Series(data_mc.shape[0]*[np.zeros(1)]),
            'rphoton_e':Series(data_mc.shape[0]*[np.zeros(1)])}
    frame = DataFrame(data)

    for i in range(frame.shape[0]):

        result = frame.loc[i]
        n = result['nups']

        dp_px = np.zeros(n)
        dp_py = np.zeros(n)
        dp_pz = np.zeros(n)
        dp_e = np.zeros(n)

        for j in range(n):
            ups_p3 = result['upsp3'][j]
            ups_costh = result['upscosth'][j]
            ups_phi = result['upsphi'][j]
            ups_energy = result['upsenergy'][j]
            
            ups_px = ups_p3 * np.sqrt(1 - ups_costh**2) * np.cos(ups_phi)
            ups_py = ups_p3 * np.sqrt(1 - ups_costh**2) * np.sin(ups_phi)
            ups_pz = ups_p3 * ups_costh
            
            dp_px[j] = result['eepx'] - ups_px
            dp_py[j] = result['eepy'] - ups_py
            dp_pz[j] = result['eepz'] - ups_pz
            dp_e[j]  = result['eee'] - ups_energy

        result['rphoton_px'] = dp_px
        result['rphoton_py'] = dp_py
        result['rphoton_pz'] = dp_pz
        result['rphoton_e']  = dp_e

    # update to database
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        cur_nl.execute("UPDATE mcevent SET rphoton_px=%s,rphoton_py=%s,rphoton_pz=%s,rphoton_e=%s WHERE eid=%s",(result['rphoton_px'].tolist(),result['rphoton_py'].tolist(),result['rphoton_pz'].tolist(),result['rphoton_e'].tolist(),result['eid']))

    conn.commit()




def update_recoilmass():
    """
    This function aims to preprocess recoilmass, which is the mass of reconstructed photon
    These two are all candidate features.
    """
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute("SELECT eid,nups,rphoton_px,rphoton_py,rphoton_pz,rphoton_e FROM mcevent WHERE nups>0")
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'rphoton_px':data_mc[:,2],
            'rphoton_py':data_mc[:,3],
            'rphoton_pz':data_mc[:,4],
            'rphoton_e':data_mc[:,5],
            'recoilmass':Series(data_mc.shape[0]*[np.zeros(1)])}
    frame = DataFrame(data)

    # calculate recoilmass
    for i in range(frame.shape[0]):
        
        result = frame.loc[i]
        n = result['nups']
        recoilmass = np.zeros(n)

        for j in range(n):
            recoilmass[j] = result['rphoton_e'][j]**2 - (result['rphoton_px'][j]**2 + result['rphoton_py'][j]**2 + result['rphoton_pz'][j]**2)
        
        result['recoilmass'] = recoilmass

    # update to database
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        cur_nl.execute("UPDATE mcevent SET recoilmass=%s WHERE eid=%s",(result['recoilmass'].tolist(),result['eid']))
    conn.commit()




def update_rphoton_costh():
    """
    This function deals with reconstructed photon's costh as a feature
    """
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute("SELECT eid,nups,rphoton_px,rphoton_py,rphoton_pz,rphoton_e FROM mcevent WHERE nups>0")
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'rphoton_px':data_mc[:,2],
            'rphoton_py':data_mc[:,3],
            'rphoton_pz':data_mc[:,4],
            'rphoton_e':data_mc[:,5],
            'rphoton_costh':Series(data_mc.shape[0]*[np.zeros(1)])}
    frame = DataFrame(data)

    # calculate rphoton_costh
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        n = result['nups']
        rphoton_costh = np.zeros(n)

        for j in range(n):
            dp_r = np.sqrt(result['rphoton_px'][j]**2 + result['rphoton_py'][j]**2 + result['rphoton_pz'][j]**2)
            rphoton_costh[j] = result['rphoton_pz'][j] / dp_r

        result['rphoton_costh'] = rphoton_costh

    # update to database
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        cur_nl.execute("UPDATE mcevent SET rphoton_costh=%s WHERE eid=%s",(result['rphoton_costh'].tolist(),result['eid']))
    conn.commit()




def update_bestphoton():
    """
    This function aims to calculate best matched photon with reconstructed photon.
    It is useful for computing event features.
            bestphoton real[nnl]
    """
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute("SELECT eid,nups,rphoton_px,rphoton_py,rphoton_pz,rphoton_e,rphoton_costh,nlfrombrem,nnl,nlp3,nlcosth,nlphi FROM mcevent WHERE nups>0")
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'rphoton_px':data_mc[:,2],
            'rphoton_py':data_mc[:,3],
            'rphoton_pz':data_mc[:,4],
            'rphoton_e':data_mc[:,5],
            'rphoton_costh':data_mc[:,6],
            'nlfrombrem':data_mc[:,7],
            'nnl':data_mc[:,8],
            'nlp3':data_mc[:,9],
            'nlcosth':data_mc[:,10],
            'nlphi':data_mc[:,11],
            'bestphoton':Series(data_mc.shape[0]*[np.zeros(1)])}
    frame = DataFrame(data)

    for i in range(frame.shape[0]):
       
        result = frame.loc[i]
        nups = result['nups']
        nnl = result['nnl']

        bestphoton = np.zeros(nnl)

        for j in range(nups):
            
            if(nnl == 0):
                break

            # reconstructed photon should be in calorimeter
            rphoton_px = result['rphoton_px'][j]
            rphoton_py = result['rphoton_py'][j]
            rphoton_pz = result['rphoton_pz'][j]
            rphoton_e  = result['rphoton_e'][j]
            rphoton_costh = result['rphoton_costh'][j]

            if(rphoton_costh < -0.8 or rphoton_costh > 0.96):
                break

            for k in range(nnl):
                # best photon should not be generated by brem effect
                if(result['nlfrombrem'][k] > -1):
                    continue

                nl_p3 = result['nlp3'][k]
                nl_costh = result['nlcosth'][k]
                nl_phi = result['nlphi'][k]

                nl_px = nl_p3 * np.sqrt(1 - nl_costh**2) * np.cos(nl_phi) 
                nl_py = nl_p3 * np.sqrt(1 - nl_costh**2) * np.sin(nl_phi)
                nl_pz = nl_p3 * nl_costh                

                cosangle = (rphoton_px*nl_px + rphoton_py*nl_py + rphoton_pz*nl_pz) / np.sqrt((rphoton_px**2 + rphoton_py**2 + rphoton_pz**2) * (nl_px**2 + nl_py**2 + nl_pz**2))
                angle = np.arccos(cosangle)
                deltaE = abs(rphoton_e - nl_p3)

                if(angle < 0.1 and deltaE/nl_p3 < 0.1):
                    bestphoton[k] = 1
                    break

        result['bestphoton'] = bestphoton

    # update database
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        cur_nl.execute("UPDATE mcevent SET bestphoton=%s WHERE eid=%s",(result['bestphoton'].tolist(),result['eid']))
    conn.commit()



def update_extraenergy():
    """
    extra energy is an event feature.
    This function uses result of bestphoton to calculate extra energy for each event.
    """
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute("SELECT eid,nnl,nlp3,nlfrombrem,bestphoton FROM mcevent WHERE nups>0")
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nnl':data_mc[:,1],
            'nlp3':data_mc[:,2],
            'nlfrombrem':data_mc[:,3],
            'bestphoton':data_mc[:,4],
            'extraenergy':0}
    frame = DataFrame(data)

    for i in range(frame.shape[0]):
        #result = frame.loc[i]
        #nnl = result['nnl']
        nnl = frame.loc[i,'nnl']

        if(nnl == 0):
            continue
        for j in range(nnl):
            if(frame.loc[i,'nlfrombrem'][j] > -1):
                continue
            if(frame.loc[i,'bestphoton'][j] != 0):
                continue
            extraenergy_update = frame.loc[i,'extraenergy'] + frame.loc[i,'nlp3'][j]
            frame.loc[i,'extraenergy'] = extraenergy_update


    # update database
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        cur_nl.execute("UPDATE mcevent SET extraenergy=%s WHERE eid=%s",(result['extraenergy'],result['eid']))
    conn.commit()



def update_code():
    """
    This function updates code for each candidate.
    There are 10 possible kinds of codes for each candidate, and based on the 
    order of importance, they are:
        3e0u0pi:    300     0
        2e1u0pi:    210     1
        1e2u0pi:    120     2
        0e3u0pi:    030     3
        2e0u1pi:    201     4
        1e1u1pi:    111     5
        0e2u1pi:    021     6
        1e0u2pi:    102     7
        0e1u2pi:    012     8
        0e0u3pi:    003     9

    For each event, we will use an int array to store candidates' code.
            code int[nups]
    """
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute("SELECT eid,nups,upsd1idx,upsd2idx,upsd3idx,v0d1lund FROM mcevent WHERE nups>0")
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'upsd1idx':data_mc[:,2],
            'upsd2idx':data_mc[:,3],
            'upsd3idx':data_mc[:,4],
            'v0d1lund':data_mc[:,5],
            'code':Series(data_mc.shape[0]*[np.zeros(1, dtype=int)])}
    frame = DataFrame(data)

    for i in range(frame.shape[0]):
        result = frame.loc[i]
        nups = result['nups']
        code = np.zeros(nups, dtype=int)

        for j in range(nups):
            nElectron = 0
            nMuon     = 0
            nPion     = 0

            iv1 = result['upsd1idx'][j]
            iv2 = result['upsd2idx'][j]
            iv3 = result['upsd3idx'][j]


            if(abs(result['v0d1lund'][iv1]) == 211):
                nPion = nPion + 1
            if(abs(result['v0d1lund'][iv1]) == 11):
                nElectron = nElectron + 1
            if(abs(result['v0d1lund'][iv1]) == 13):
                nMuon = nMuon + 1


            if(abs(result['v0d1lund'][iv2]) == 211):
                nPion = nPion + 1
            if(abs(result['v0d1lund'][iv2]) == 11):
                nElectron = nElectron + 1
            if(abs(result['v0d1lund'][iv2]) == 13):
                nMuon = nMuon + 1


            if(abs(result['v0d1lund'][iv3]) == 211):
                nPion = nPion + 1
            if(abs(result['v0d1lund'][iv3]) == 11):
                nElectron = nElectron + 1
            if(abs(result['v0d1lund'][iv3]) == 13):
                nMuon = nMuon + 1



            if(nElectron == 3):
                code[j] = 0
                continue
            if(nMuon==1 and nElectron==2):
                code[j] = 1
                continue
            if(nMuon==2 and nElectron==1):
                code[j] = 2
                continue
            if(nMuon == 3):
                code[j] = 3
                continue
            if(nElectron==2 and nPion==1):
                code[j] = 4
                continue
            if(nElectron==1 and nMuon==1 and nPion==1):
                code[j] = 5
                continue
            if(nMuon==2 and nPion==1):
                code[j] = 6
                continue
            if(nElectron==1 and nPion==2):
                code[j] = 7
                continue
            if(nMuon ==1 and nPion==2):
                code[j] = 8
                continue
            if(nPion==3):
                code[j] = 9
                continue
            else:
                code[j] = -1
                continue

        frame.loc[i,'code'] = code


    # update database
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        cur_nl.execute("UPDATE mcevent SET code=%s WHERE eid=%s",(result['code'].tolist(),result['eid']))
    conn.commit()


def update_pid():
    """
    This function aims to update
        pid bool[nups]
    as a candidate feature
    """
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute("SELECT eid,nups,ntl,muselectorsmap,eselectorsmap,piselectorsmap,code FROM mcevent WHERE nups>0 and ntl>0")
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'ntl':data_mc[:,2],
            'muselectorsmap':data_mc[:,3],
            'eselectorsmap':data_mc[:,4],
            'piselectorsmap':data_mc[:,5],
            'code':data_mc[:,6],
            'pid':Series(data_mc.shape[0]*[np.zeros(1,dtype=bool)])}
    frame = DataFrame(data)

    for i in range(frame.shape[0]):
        result = frame.loc[i]
        nups = result['nups']
        ntl = result['ntl']
        pid = np.zeros(nups,dtype=bool)

        for j in range(nups):
            nMuPid16 = 0
            nMuPid17 = 0
            nMuPid18 = 0
            nMuPid19 = 0
            nElPid6 = 0
            nElPid7 = 0
            nElPid8 = 0
            nElPid9 = 0
            nPi = 0

            for k in range(ntl):
                if((result['muselectorsmap'][k] >> 16 & 0x1)):
                    nMuPid16 = nMuPid16 + 1
                if((result['muselectorsmap'][k] >> 17 & 0x1)):
                    nMuPid17 = nMuPid17 + 1
                if((result['muselectorsmap'][k] >> 18 & 0x1)):
                    nMuPid18 = nMuPid18 + 1
                if((result['muselectorsmap'][k] >> 19 & 0x1)):
                    nMuPid19 = nMuPid19 + 1
                if((result['eselectorsmap'][k] >> 6 & 0x1)):
                    nElPid6 = nElPid6 + 1
                if((result['eselectorsmap'][k] >> 7 & 0x1)):
                    nElPid7 = nElPid7 + 1
                if((result['eselectorsmap'][k] >> 8 & 0x1)):
                    nElPid8 = nElPid8 + 1
                if((result['eselectorsmap'][k] >> 9 & 0x1)):
                    nElPid9 = nElPid9 + 1
                if((result['piselectorsmap'][k] >> 14 & 0x1)):
                    nPi = nPi + 1                


            if(result['code'][j] == 0):
                pid[j] = (nElPid9>=4);

            if(result['code'][j] == 1):
                pid[j] = (nElPid7>=3 and nMuPid16>=1);

            if(result['code'][j] == 2):
                pid[j] = (nElPid7>=1 and nMuPid17>=2);

            if(result['code'][j] == 3):
                pid[j] = (nMuPid18>=3 and nMuPid16>=4);

            if(result['code'][j] == 4):
                pid[j] = (nElPid6>=3);

            if(result['code'][j] == 5):
                pid[j] = (nElPid9>=1 and nMuPid17>=1);

            if(result['code'][j] == 6):
                pid[j] = (nMuPid16>=3 and nMuPid18>=2);

            if(result['code'][j] == 7):
                pid[j] = (nElPid9==1);

            if(result['code'][j] == 8):
                pid[j] = (nMuPid17==1);

            if(result['code'][j] == 9):
                pid[j] = True;


        result['pid'] = pid


    # update database
    for i in range(frame.shape[0]):
        result = frame.loc[i]
        cur_nl.execute("UPDATE mcevent SET pid=%s WHERE eid=%s",(result['pid'].tolist(),result['eid']))
    conn.commit()




if __name__ == '__main__':

    import feature_process

    #feature_process.update_massdiff();
    #feature_process.update_nlFromBrem();
    #feature_process.update_rphoton();
    #feature_process.update_recoilmass();
    #feature_process.update_rphoton_costh();
    #feature_process.update_bestphoton();
    #feature_process.update_extraenergy();
    #feature_process.update_code();
    feature_process.update_pid();
