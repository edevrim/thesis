#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:38:53 2020

@author: salihemredevrim
"""

import pandas as pd
import numpy as np
import gzip
from scipy.spatial.distance import pdist 
from scipy.spatial.distance import squareform 
import matplotlib.pyplot as plt
from sklearn.neighbors import DistanceMetric

#%%
#data prep 

def tsp_prep(tsp_id, tsp_name, distance_type1):
    
    tsp_name = tsp_name
    distance_type = distance_type1
    
    if distance_type == 'EUC_2D': 
        distance_type = 'euclidean'
    elif distance_type == 'GEO':
        distance_type = 'geo'
        
    tsp1 = pd.DataFrame(index=range(1))
    tsp1['str'] = 'update'
    tsp1['counter'] = 0
    counter = 0 
    
    with gzip.open(tsp_name,'r') as f: 
        for line in f:
            counter = counter + 1
            tsp0 = pd.DataFrame(index=range(1))
            tsp0['str'] = str(line)
            tsp0['counter'] = counter
            tsp1 = tsp1.append(tsp0, ignore_index=True)
       
    slice1 = int(tsp1[tsp1['str'].str.contains("SECTION")]['counter'])          
    slice2 = int(tsp1[tsp1['str'].str.contains("EOF")]['counter'])           
           
    tsp11 = tsp1.iloc[slice1+1:slice2]     
    tsp11['del'], tsp11['node'] = tsp11['str'].str.split("'", 1).str    
    tsp11['node'] = tsp11['node'].str.strip()
    tsp11['node'], tsp11['coordinates'] = tsp11['node'].str.split(' ', 1).str 
    tsp11['coordinates'] = tsp11['coordinates'].str.strip()
    tsp11['coor1'], tsp11['coor2'] = tsp11['coordinates'].str.split(' ', 1).str   
    tsp11['coor2'] = tsp11['coor2'].str.strip()
    tsp11['coor2'] = tsp11['coor2'].map(lambda x: str(x)[:-3]).astype(float)
    tsp11['coor1'] = tsp11['coor1'].astype(float)
    tsp11['node'] = tsp11['node'].astype(int)
    tsp11['name'] = tsp_name
    tsp = tsp11[['name', 'node', 'coor1', 'coor2']].reset_index(drop=True)    
    
    if distance_type == 'euclidean':
        #to array 
        arr1 = tsp[['coor1', 'coor2']].values
        dist1 = pdist(arr1, distance_type)
        tsp_distance = squareform(dist1)
        #upper = np.triu(tsp_distance)
        
    elif distance_type == 'geo':
        
        tsp['coor1'] = np.radians(tsp['coor1'])
        tsp['coor2'] = np.radians(tsp['coor2'])
        dist = DistanceMetric.get_metric('haversine')
        tsp_distance = dist.pairwise(tsp[['coor1','coor2']].to_numpy())*6373 #km
    
    #Statistics 
    #number of nodes (vertices)
    n_nodes = len(tsp)
    
    #Mean, median distance, 0's exluded
    avg_dist = tsp_distance[tsp_distance > 0].mean()
    median_dist = np.median(tsp_distance[tsp_distance > 0])
    std_dist = np.std(tsp_distance[tsp_distance > 0])
    avg_med_ratio = avg_dist / median_dist
    
    #plot
    num_bins = 10
    n, bins, patches = plt.hist(tsp_distance[tsp_distance>0], num_bins, facecolor='blue', alpha=0.5)
    plt.show()
    
    # return Xth percentile value (10 to 90)
    p_min = min(tsp_distance[tsp_distance>0])
    p_max = max(tsp_distance[tsp_distance>0])
    p10 = np.percentile(tsp_distance[tsp_distance>0], 10)
    p25 = np.percentile(tsp_distance[tsp_distance>0], 25)
    p50 = np.percentile(tsp_distance[tsp_distance>0], 50)
    p75 = np.percentile(tsp_distance[tsp_distance>0], 75)
    p90 = np.percentile(tsp_distance[tsp_distance>0], 90)
    p95 = np.percentile(tsp_distance[tsp_distance>0], 95)
    
    #average number of nodes having connections less than p10, p25, p50 
    df = pd.DataFrame(tsp_distance)
    df10 = df[df <= p10].count() - 1 
    df25 = df[df <= p25].count() - 1 
    df50 = df[df <= p50].count() - 1 
    
    cnt_p10_avg = df10.mean()
    cnt_p10_median = df10.median()
    cnt_p10_std = df10.std()
    
    cnt_p25_avg = df25.mean()
    cnt_p25_median = df25.median()
    cnt_p25_std = df25.std()

    cnt_p50_avg = df50.mean()
    cnt_p50_median = df50.median()
    cnt_p50_std = df50.std()
    
    dict1 = {'n_nodes' : n_nodes,
             'avg_dist' : avg_dist, 
             'median_dist' : median_dist,
             'std_dist' : std_dist,
             'avg_med_ratio' : avg_med_ratio, 
             'p_min' : p_min, 
             'p_max' : p_max, 
             'p10' : p10,
             'p25' : p25,
             'p50' : p50,
             'p75' : p75,
             'p90' : p90, 
             'p95' : p95,
             'cnt_p10_avg' : cnt_p10_avg, 
             'cnt_p10_median' : cnt_p10_median, 
             'cnt_p10_std' : cnt_p10_std, 
             'cnt_p25_avg' : cnt_p25_avg, 
             'cnt_p25_median' : cnt_p25_median, 
             'cnt_p25_std' : cnt_p25_std, 
             'cnt_p50_avg' : cnt_p50_avg, 
             'cnt_p50_median' : cnt_p50_median, 
             'cnt_p50_std' : cnt_p50_std}
    
    dict1 = pd.DataFrame(dict1, index=[0])
    dict1['tsp_name'] = tsp_name
    tsp_distance = np.round(tsp_distance, 1)
    tsp_distance = pd.DataFrame(tsp_distance)
    
    #save
    writer = pd.ExcelWriter(tsp_id+'_data.xlsx', engine='xlsxwriter');
    dict1.to_excel(writer, sheet_name= 'stats');
    tsp.to_excel(writer, sheet_name= 'tsp');
    tsp_distance.to_excel(writer, sheet_name= 'distance_matrix');
    writer.save();
   
    return tsp, tsp_distance, dict1

#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP1', 'a280.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP2', 'ali535.tsp.gz', 'GEO')
tsp, tsp_distance, dict1 = tsp_prep('TSP3', 'att48.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP4', 'att532.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP7', 'berlin52.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP8', 'bier127.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP10', 'brd14051.tsp.gz', 'EUC_2D') #didnt work
tsp, tsp_distance, dict1 = tsp_prep('TSP11', 'burma14.tsp.gz', 'GEO')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP12', 'ch130.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP13', 'ch150.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP14', 'd198.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP15', 'd493.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP16', 'd657.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP17', 'd1291.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP18', 'd1655.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP19', 'd2103.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP20', 'd15112.tsp.gz', 'EUC_2D') #didnt work 
tsp, tsp_distance, dict1 = tsp_prep('TSP21', 'd18512.tsp.gz', 'EUC_2D') # didnt work

#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP22', 'eil51.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP23', 'eil76.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP24', 'eil101.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP25', 'tsp225.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP26', 'u159.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP27', 'u574.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP28', 'u724.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP29', 'u1060.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP30', 'u1432.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP31', 'u1817.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP32', 'u2152.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP33', 'u2319.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP34', 'ulysses16.tsp.gz', 'GEO')
tsp, tsp_distance, dict1 = tsp_prep('TSP35', 'ulysses22.tsp.gz', 'GEO')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP36', 'usa13509.tsp.gz', 'EUC_2D') # didnt work
tsp, tsp_distance, dict1 = tsp_prep('TSP37', 'vm1084.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP38', 'vm1748.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP39', 'fl417.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP40', 'fl1400.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP41', 'fl1577.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP42', 'fl3795.tsp.gz', 'EUC_2D')

#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP45', 'gil262.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP50', 'gr96.tsp.gz', 'GEO')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP52', 'gr137.tsp.gz', 'GEO')
tsp, tsp_distance, dict1 = tsp_prep('TSP53', 'gr202.tsp.gz', 'GEO')
tsp, tsp_distance, dict1 = tsp_prep('TSP54', 'gr229.tsp.gz', 'GEO')
tsp, tsp_distance, dict1 = tsp_prep('TSP55', 'gr431.tsp.gz', 'GEO')
tsp, tsp_distance, dict1 = tsp_prep('TSP56', 'gr666.tsp.gz', 'GEO')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP58', 'kroA100.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP59', 'kroA150.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP60', 'kroA200.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP61', 'kroB100.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP62', 'kroB150.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP63', 'kroB200.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP64', 'kroC100.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP65', 'kroD100.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP66', 'kroE100.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP67', 'lin105.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP68', 'lin318.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP69', 'nrw1379.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP70', 'p654.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP72', 'pcb1173.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP73', 'pcb3038.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP74', 'pla7397.tsp.gz', 'EUC_2D') # didnt work
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP75', 'pr76.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP76', 'pr107.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP77', 'pr124.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP78', 'pr136.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP79', 'pr144.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP80', 'pr152.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP81', 'pr226.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP82', 'pr264.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP83', 'pr299.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP84', 'pr439.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP85', 'pr1002.tsp.gz', 'EUC_2D') #didnt work 
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP86', 'pr2392.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP87', 'rat99.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP88', 'rat195.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP89', 'rat575.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP90', 'rat783.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP91', 'rd100.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP92', 'rd400.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP93', 'rl1304.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP94', 'rl1323.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP95', 'rl1889.tsp.gz', 'EUC_2D')
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP96', 'rl5915.tsp.gz', 'EUC_2D') #didnt work
tsp, tsp_distance, dict1 = tsp_prep('TSP97', 'rl11849.tsp.gz', 'EUC_2D') #didnt work
#%%
tsp, tsp_distance, dict1 = tsp_prep('TSP100', 'st70.tsp.gz', 'EUC_2D')
tsp, tsp_distance, dict1 = tsp_prep('TSP102', 'ts225.tsp.gz', 'EUC_2D')

#%%

def tsp_prep_given_dist(structured_tsp_name, tsp_name, tsp_id, full_flg):
    
    xd1= pd.read_excel(structured_tsp_name, index=False)    
    xd1 = xd1.iloc[:,1:]
    xd2 = xd1.reindex(sorted(xd1.columns), axis=1)
    xd2 = xd2.fillna(0)
    
    if full_flg == 1:
        tsp_distance = xd2
        tsp_distance = tsp_distance.fillna(0)
    else:
        tsp_distance = xd2 + xd2.T - np.diag(np.diag(xd2)) 
        tsp_distance = tsp_distance.fillna(0)

    #Statistics 
    #number of nodes (vertices)
    n_nodes = len(tsp_distance)
    tsp_distance = np.array(tsp_distance)
    #Mean, median distance, 0's exluded
    avg_dist = tsp_distance[tsp_distance > 0].mean()
    median_dist = np.median(tsp_distance[tsp_distance > 0])
    std_dist = np.std(tsp_distance[tsp_distance > 0])
    avg_med_ratio = avg_dist / median_dist
    
    #plot
    num_bins = 10
    n, bins, patches = plt.hist(tsp_distance[tsp_distance>0], num_bins, facecolor='blue', alpha=0.5)
    plt.show()
    
    # return Xth percentile value (10 to 90)
    p_min = min(tsp_distance[tsp_distance>0])
    p_max = max(tsp_distance[tsp_distance>0])
    p10 = np.percentile(tsp_distance[tsp_distance>0], 10)
    p25 = np.percentile(tsp_distance[tsp_distance>0], 25)
    p50 = np.percentile(tsp_distance[tsp_distance>0], 50)
    p75 = np.percentile(tsp_distance[tsp_distance>0], 75)
    p90 = np.percentile(tsp_distance[tsp_distance>0], 90)
    p95 = np.percentile(tsp_distance[tsp_distance>0], 95)
    
    #average number of nodes having connections less than p10, p25, p50 
    df = pd.DataFrame(tsp_distance)
    df10 = df[df <= p10].count() - 1 
    df25 = df[df <= p25].count() - 1 
    df50 = df[df <= p50].count() - 1 
    
    cnt_p10_avg = df10.mean()
    cnt_p10_median = df10.median()
    cnt_p10_std = df10.std()
    
    cnt_p25_avg = df25.mean()
    cnt_p25_median = df25.median()
    cnt_p25_std = df25.std()

    cnt_p50_avg = df50.mean()
    cnt_p50_median = df50.median()
    cnt_p50_std = df50.std()
    
    dict1 = {'n_nodes' : n_nodes,
             'avg_dist' : avg_dist, 
             'median_dist' : median_dist,
             'std_dist' : std_dist,
             'avg_med_ratio' : avg_med_ratio, 
             'p_min' : p_min, 
             'p_max' : p_max, 
             'p10' : p10,
             'p25' : p25,
             'p50' : p50,
             'p75' : p75,
             'p90' : p90, 
             'p95' : p95,
             'cnt_p10_avg' : cnt_p10_avg, 
             'cnt_p10_median' : cnt_p10_median, 
             'cnt_p10_std' : cnt_p10_std, 
             'cnt_p25_avg' : cnt_p25_avg, 
             'cnt_p25_median' : cnt_p25_median, 
             'cnt_p25_std' : cnt_p25_std, 
             'cnt_p50_avg' : cnt_p50_avg, 
             'cnt_p50_median' : cnt_p50_median, 
             'cnt_p50_std' : cnt_p50_std}
    
    dict1 = pd.DataFrame(dict1, index=[0])
    dict1['tsp_name'] = tsp_name
    tsp_distance = np.round(tsp_distance, 1)
    tsp_distance = pd.DataFrame(tsp_distance)
    
    #save
    writer = pd.ExcelWriter(tsp_id+'_data.xlsx', engine='xlsxwriter');
    dict1.to_excel(writer, sheet_name= 'stats');
    tsp_distance.to_excel(writer, sheet_name= 'distance_matrix');
    writer.save();
    
    return tsp_distance, dict1
#%%
tsp_distance, dict1 = tsp_prep_given_dist('TSP5_data.xlsx', 'bayg29.tsp.gz', 'TSP5', 0)
tsp_distance, dict1 = tsp_prep_given_dist('TSP6_data.xlsx', 'bays29.tsp.gz', 'TSP6', 1)
tsp_distance, dict1 = tsp_prep_given_dist('TSP9_data.xlsx', 'brazil58.tsp.gz', 'TSP9', 0)
tsp_distance, dict1 = tsp_prep_given_dist('TSP44_data.xlsx', 'fri26.tsp.gz', 'TSP44', 0)
tsp_distance, dict1 = tsp_prep_given_dist('TSP47_data.xlsx', 'gr21.tsp.gz', 'TSP47', 0)
tsp_distance, dict1 = tsp_prep_given_dist('TSP49_data.xlsx', 'gr48.tsp.gz', 'TSP49', 0)
tsp_distance, dict1 = tsp_prep_given_dist('TSP46_data.xlsx', 'gr17.tsp.gz', 'TSP46', 0)
tsp_distance, dict1 = tsp_prep_given_dist('TSP48_data.xlsx', 'gr24.tsp.gz', 'TSP48', 0)
tsp_distance, dict1 = tsp_prep_given_dist('TSP57_data.xlsx', 'hk48.tsp.gz', 'TSP57', 0)
tsp_distance, dict1 = tsp_prep_given_dist('TSP71_data.xlsx', 'pa561.tsp.gz', 'TSP71', 0)
tsp_distance, dict1 = tsp_prep_given_dist('TSP101_data.xlsx', 'swiss42.tsp.gz', 'TSP101', 1)