# This file includes functions needed for feature process. 
# These functions are used to calculate high-level features from row data,
# and update in database
#
#After computing each feature, we will use Dataframe.join function to join features.
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import psycopg2
import csv




def true_match(rawtable):
    """
    This function aims to do truth-matching for each events.
    This is identical to labelling data.

    parameters:
    -------------
    -rawtable:      name of raw table   (string)
    """
    

    # connect to database, and obtain data
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()

    cur_nl.execute("SELECT eid,nups,upsd1idx,upsd2idx,upsd3idx,v0MCIdx,v0d1Lund,mcLund,dauIdx FROM %s WHERE nups>0" % rawtable)
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'upsd1idx':data_mc[:,2],
            'upsd2idx':data_mc[:,3],
            'upsd3idx':data_mc[:,4],
            'v0MCIdx':data_mc[:,5],
            'v0d1Lund':data_mc[:,6],
            'mcLund':data_mc[:,7],
            'dauIdx':data_mc[:,8],
            'true_matching':Series(data_mc.shape[0]*[np.zeros(1)])}
    frame = DataFrame(data)

    for event_id in range(frame.shape[0]):
        result = frame.iloc[event_id]
        nups = result.nups
        matching = np.zeros(nups)	#initialization true_matching list for a given event

        for candidate_id in range(nups):
            iv1 = result.upsd1idx[candidate_id]
            iv2 = result.upsd2idx[candidate_id]
            iv3 = result.upsd3idx[candidate_id]
            isTrueV1 = (result.v0MCIdx[iv1]>-1) and abs(result.v0d1Lund[iv1])==abs(result.mcLund[result.dauIdx[result.v0MCIdx[iv1]]])
            isTrueV2 = (result.v0MCIdx[iv2]>-1) and abs(result.v0d1Lund[iv2])==abs(result.mcLund[result.dauIdx[result.v0MCIdx[iv2]]])
            isTrueV3 = (result.v0MCIdx[iv3]>-1) and abs(result.v0d1Lund[iv3])==abs(result.mcLund[result.dauIdx[result.v0MCIdx[iv3]]])
            isTrueUps = isTrueV1 and isTrueV2 and isTrueV3
            matching[candidate_id] = isTrueUps

        result['true_matching'] = matching


    return frame[['eid','true_matching']].set_index('eid')
        




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
        frame.iloc[i]['massdiff'] = massdiff(frame.iloc[i])

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

        result = frame.iloc[i]
        nele = result['nele']
        nnl = result['nnl']
        
        brem = np.zeros(nnl) - 1

        for j in range(nele):
            if(result['eled2idx'][j] != -1):
                brem[result['eled2idx'][j]] = j

        result['nlfrombrem'] = brem

    return frame[['eid','nlfrombrem']].set_index('eid')






def update_rphoton(rawtable):
    """
    This function updates 4-momentum of reconstructed photon, which is Pee - Pups.
    """
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()

    cur_nl.execute("SELECT eid,eepx,eepy,eepz,eee,nups,upsp3,upscosth,upsphi,upsenergy FROM %s WHERE nups>0" % rawtable)
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

        result = frame.iloc[i]
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


    return frame[['eid','rphoton_px','rphoton_py','rphoton_pz','rphoton_e']].set_index('eid')






def update_recoilmass(rphoton):
    """
    This function aims to preprocess recoilmass, which is the mass of reconstructed photon
    These two are all candidate features.

    parameters:
    -------------
    rphoton: path of hdf file in which rphoton stores.      (string)

    """
    
    #cur_nl.execute("SELECT eid,nups,rphoton_px,rphoton_py,rphoton_pz,rphoton_e FROM %s WHERE nups>0" % rawtable)
    frame = pd.read_hdf(rphoton,'rphoton')

    # calculate recoilmass
    for i in range(frame.shape[0]):
        
        result = frame.iloc[i]
        n = len(result['rphoton_px'])
        recoilmass = np.zeros(n)

        for j in range(n):
            recoilmass[j] = result['rphoton_e'][j]**2 - (result['rphoton_px'][j]**2 + result['rphoton_py'][j]**2 + result['rphoton_pz'][j]**2)
        
        result['recoilmass'] = recoilmass


    return frame[['eid','recoilmass']].set_index('eid')






def update_rphoton_costh(rphoton):
    """
    This function deals with reconstructed photon's costh as a feature

    parameters:
    ------------
    rphoton: path of hdf file in which rphoton stores.      (string)

    """
    
    #cur_nl.execute("SELECT eid,nups,rphoton_px,rphoton_py,rphoton_pz,rphoton_e FROM mcevent WHERE nups>0")
    frame = pd.read_hdf(rphoton,'rphoton')

    # calculate rphoton_costh
    for i in range(frame.shape[0]):
        result = frame.iloc[i]
        n = len(result['rphoton_px'])
        rphoton_costh = np.zeros(n)

        for j in range(n):
            dp_r = np.sqrt(result['rphoton_px'][j]**2 + result['rphoton_py'][j]**2 + result['rphoton_pz'][j]**2)
            rphoton_costh[j] = result['rphoton_pz'][j] / dp_r

        result['rphoton_costh'] = rphoton_costh


    return frame[['eid','rphoton_costh']].set_index('eid')








def update_bestphoton(rphoton_path, rphoton_costh_path, nlfrombrem_path, rawtable):
    """
    This function aims to calculate best matched photon with reconstructed photon.
    It is useful for computing event features.
            bestphoton real[nnl]
    """

    # read data from hdf files.
    rphoton_hdf = pd.read_hdf(rphoton_path,'rphoton')
    rphoton_costh_hdf = pd.read_hdf(rphoton_costh_path,'rphoton_costh')
    nlfrombrem_hdf = pd.read_hdf(nlfrombrem_path,'nlfrombrem')


    # read raw data from postgresql database.
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute("SELECT eid,nups,nnl,nlp3,nlcosth,nlphi FROM %s WHERE nups>0" % rawtable)
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nups':data_mc[:,1],
            'nnl':data_mc[:,2],
            'nlp3':data_mc[:,3],
            'nlcosth':data_mc[:,4],
            'nlphi':data_mc[:,5],
            'bestphoton':Series(data_mc.shape[0]*[np.zeros(1)])}
    frame = DataFrame(data)


    # calculate bestphoton
    for i in range(frame.shape[0]):
       
        result = frame.iloc[i]
        eid = result['eid']
        nups = result['nups']
        nnl = result['nnl']

        rphoton_px_list = rphoton_hdf.loc[eid]['rphoton_px']
        rphoton_py_list = rphoton_hdf.loc[eid]['rphoton_py']
        rphoton_pz_list = rphoton_hdf.loc[eid]['rphoton_pz']
        rphoton_e_list = rphoton_hdf.loc[eid]['rphoton_e']
        rphoton_costh_list = rphoton_costh_hdf.loc[eid]['rphoton_costh']

        bestphoton = np.zeros(nnl)

        for j in range(nups):
            
            if(nnl == 0):
                break

            # reconstructed photon should be in calorimeter
            rphoton_px = rphoton_px_list[j]
            rphoton_py = rphoton_py_list[j]
            rphoton_pz = rphoton_pz_list[j]
            rphoton_e  = rphoton_e_list[j]
            rphoton_costh = rphoton_costh_list[j]

            if(rphoton_costh < -0.8 or rphoton_costh > 0.96):
                break

            nlfrombrem_list = nlfrombrem_hdf.loc[eid]['nlfrombrem']
            for k in range(nnl):
                # best photon should not be generated by brem effect
                if(nlfrombrem_list[k] > -1):
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

    conn.close()
    
    return frame[['eid','bestphoton']].set_index('eid')




def update_extraenergy(nlfrombrem_path, bestphoton_path, rawtable):
    """
    extra energy is an event feature.
    This function uses result of bestphoton to calculate extra energy for each event.
    """

    # read from hdf files.
    nlfrombrem_hdf = pd.read_hdf(nlfrombrem_path,'nlfrombrem')
    bestphoton_hdf = pd.read_hdf(bestphoton_path,'bestphoton')


    # read from psql database.
    conn = psycopg2.connect(database="darkphoton",user="yunxuanli")
    cur_nl = conn.cursor()
    cur_nl.execute("SELECT eid,nnl,nlp3 FROM %s WHERE nups>0" % rawtable)
    rows_nl = cur_nl.fetchall()
    data_mc = np.array(rows_nl, dtype=object)
    data = {'eid':data_mc[:,0],
            'nnl':data_mc[:,1],
            'nlp3':data_mc[:,2],
            'extraenergy':0}
    frame = DataFrame(data)

    for i in range(frame.shape[0]):
        #result = frame.loc[i]
        #nnl = result['nnl']
        nnl = frame.iloc[i]['nnl']
        eid = frame.iloc[i]['eid']

        if(nnl == 0):
            continue
        for j in range(nnl):
            if(nlfrombrem_hdf.loc[eid]['nlfrombrem'][j] > -1): 
                continue
            if(bestphoton_hdf.loc[eid]['bestphoton'][j] != 0):
                continue
            extraenergy_update = frame.iloc[i,'extraenergy'] + frame.iloc[i,'nlp3'][j]
            frame.iloc[i,'extraenergy'] = extraenergy_update

    conn.close()
    
    return frame[['eid','extraenergy']].set_index('eid')

