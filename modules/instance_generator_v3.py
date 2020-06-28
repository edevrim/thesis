#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:25:58 2020

@author: salihemredevrim
"""
import random
import numpy as np 
import pandas as pd 
import networkx as nx
import matplotlib.pyplot as plt

# PART 1: RANDOM DISTRIBUTIONS 
#%% UNIFORM DIST (TRAIN-TEST)
def uniform_dist():
    
    number_of_nodes = random.randint(30, 120)
    
    df = pd.DataFrame(np.random.rand(number_of_nodes, 2), columns=['x', 'y'])
    print(number_of_nodes, 'uniform')
        
    #normalize coordinates
    df_max = df.max()
    df['x'] = df['x']/df_max[0] if df_max[0] > 0 else 1
    df['y'] = df['y']/df_max[1] if df_max[1] > 0 else 1
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    return number_of_nodes, df, 0

#%% NORMAL DIST (TRAIN-TEST)
def normal_dist():
    
    number_of_nodes = random.randint(30, 120)
    
    df = pd.DataFrame(np.random.normal(50, size=(number_of_nodes, 2)), columns=['x', 'y'])
    print(number_of_nodes, 'normal')
        
    #normalize coordinates
    df_max = df.max()
    df['x'] = df['x']/df_max[0] if df_max[0] > 0 else 1
    df['y'] = df['y']/df_max[1] if df_max[1] > 0 else 1
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    return number_of_nodes, df, 1 

#%% EXPONENTIAL DIST (TRAIN-TEST)
def expo_dist():
    
    number_of_nodes = random.randint(30, 120)
    
    df = pd.DataFrame(np.random.exponential(scale=1.0, size=(number_of_nodes, 2)), columns=['x', 'y'])
    print(number_of_nodes, 'exponential')
        
    #normalize coordinates
    df_max = df.max()
    df['x'] = df['x']/df_max[0] if df_max[0] > 0 else 1
    df['y'] = df['y']/df_max[1] if df_max[1] > 0 else 1
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    return number_of_nodes, df, 2 

#%% RUN
for k in range(35):
    
    number_of_nodes, df, flag1 =  uniform_dist()
    df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
    
for k in range(15):
    
    number_of_nodes, df, flag1 =  uniform_dist()
    df.to_csv("TestTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 


for k in range(35):
    
    number_of_nodes, df, flag1 =  normal_dist()
    df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
    
for k in range(15):
    
    number_of_nodes, df, flag1 =  normal_dist()
    df.to_csv("TestTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
    
for k in range(35):
    
    number_of_nodes, df, flag1 =  expo_dist()
    df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
    
for k in range(15):
    
    number_of_nodes, df, flag1 =  expo_dist()
    df.to_csv("TestTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True)     

#%% PART 2: MORE STRUCTURES

# BALANCED TREE (TRAIN-TEST)
def balanced_tree():
    random1 = random.randint(2, 25)
    
    if random1 <= 5:
        random2 = random.randint(1, 4)
    elif random1 <= 8 :
        random2 = random.randint(1, 3)
    else:
        random2 = 2        
        
    graph = nx.balanced_tree(random1, random2)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'balanced_tree')
    
    return number_of_nodes, df, 3 

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 =  balanced_tree()
    
    if number_of_nodes <= 625:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 =  balanced_tree()
    
    if number_of_nodes not in number_of_nodes_list and number_of_nodes <= 625:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        j = j + 1 
        
#%% BARBELL TREE (TRAIN-TEST)
def barbell():
    graph = nx.barbell_graph(random.randint(3, 50), random.randint(3, 6), create_using=None)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'barbell')
    
    return number_of_nodes, df, 4     

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 =  barbell()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 =  barbell()
    
    if number_of_nodes not in number_of_nodes_list and number_of_nodes <= 625:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        j = j + 1     
        
#%% COMPLETE_MULTIPARTITE (TRAIN-TEST)
def complete_multipartite():
    random1 = random.randint(5, 100)
    graph = nx.complete_multipartite_graph(random1 , int(random1/2))
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'complete_multipartite')
    
    return number_of_nodes, df, 5        

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = complete_multipartite()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 =  complete_multipartite()
    
    if number_of_nodes not in number_of_nodes_list and number_of_nodes <= 625:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        j = j + 1    
        
#%% CIRCULAR LADDER (TRAIN-TEST)
def circular_ladder():
    graph = nx.circular_ladder_graph(random.randint(10, 50), create_using=None)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'circular_ladder')
    
    return number_of_nodes, df, 6

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = circular_ladder()
    
    if 1==1:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 =  circular_ladder()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        j = j + 1          

#%% dorogovtsev_goltsev_mendes_graph (TRAINING ONLY)
def dorogovtsev_goltsev_mendes():
    graph = nx.dorogovtsev_goltsev_mendes_graph(random.randint(2, 6))
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'dorogovtsev_goltsev_mendes')
    
    return number_of_nodes, df, 19

#%%
number_of_nodes_list = []
k = 0 
counter1 = 0

while k < 4 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = dorogovtsev_goltsev_mendes()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1       

#%% HYPER CUBE (TRAINING ONLY)
def hypercube():
    #n -> 2^(n-1)
    graph = nx.hypercube_graph(random.randint(3, 9))
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'hypercube')
    
    return number_of_nodes, df, 20  

#%%
number_of_nodes_list = []
k = 0 
counter1 = 0

while k < 5 and counter1 <= 25: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = hypercube()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1  
        
#%% LADDER GRAPH (TRAIN-TEST)
def ladder():
    graph = nx.ladder_graph(random.randint(5, 100))
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'ladder')
    
    return number_of_nodes, df, 7

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = ladder()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 =  ladder()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        j = j + 1          

#%% STAR GRAPH (TRAIN-TEST)
def star_graph():
    graph = nx.star_graph(random.randint(5, 80))
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'star')
    
    return number_of_nodes, df, 8  

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = star_graph()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 =  star_graph()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        j = j + 1          

#%% CAVEMEN (TRAIN-TEST)
def cave_men():
    graph = nx.caveman_graph(random.randint(10, 30), random.randint(2, 5))
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'cavemen')
    
    return number_of_nodes, df, 9  

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = cave_men()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 = cave_men()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        j = j + 1          


#%% RANDOM LOBSTER (TRAIN-TEST)
def r_lobster():
    graph = nx.random_lobster(random.randint(20, 50), 0.5, 0.25)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'random_lobster')
    
    return number_of_nodes, df, 10  

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 =  r_lobster()
    
    if 1==1:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        k = k + 1 
        
while j < 15 and counter1 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 = r_lobster()
    
    if 1==1:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        j = j + 1          

#%% LOLLIPOP (TRAIN-TEST)
def lolli():
    graph = nx.lollipop_graph(random.randint(5, 50), random.randint(3, 10), create_using=None)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'lollipop')
    
    return number_of_nodes, df, 11  

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = lolli()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 = lolli()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        j = j + 1   
        
#%% MINI GRAPHS
# TRAIN-ONLY
def chvatal():
    #n of in cluster and stick
        
    graph = nx.chvatal_graph()
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'chvatal')
    
    return number_of_nodes, df  

#%% TRAIN-ONLY
def n_bull():
    #x number of bulls (between 5-10)
    x = random.randint(5, 12)
    df_all = pd.DataFrame()    
    
    for j in range(x):
        graph = nx.bull_graph()
        nodePos = nx.spring_layout(graph)
        df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
        df = df[['x', 'y']].reset_index(drop=True)
        number_of_nodes = len(df)
        df_all = df_all.append(df)
    
    df = df_all.reset_index(drop=True)    
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes*(x+1), 'n_bull')
    
    return number_of_nodes*(x+1), df, 21  

#%%
number_of_nodes_list = []
k = 0 
counter1 = 0

while k < 5 and counter1 <= 25: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = n_bull()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1  
        
#%% TRAIN-ONLY
def make_it_coord(graph):
    
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print('number of nodes:', number_of_nodes)
    
    return number_of_nodes, df  

#%% MINI GRAPHS
graph = nx.cubical_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(0)+".csv" ,index=True) 

graph = nx.desargues_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(1)+".csv" ,index=True) 

graph = nx.diamond_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(2)+".csv" ,index=True) 

graph = nx.dodecahedral_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(3)+".csv" ,index=True) 

graph = nx.frucht_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(4)+".csv" ,index=True) 

graph = nx.heawood_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(5)+".csv" ,index=True) 

graph = nx.hoffman_singleton_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(6)+".csv" ,index=True) 

graph = nx.house_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(7)+".csv" ,index=True) 

graph = nx.house_x_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(8)+".csv" ,index=True) 

graph = nx.icosahedral_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(9)+".csv" ,index=True) 

graph = nx.krackhardt_kite_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(10)+".csv" ,index=True) 

graph = nx.moebius_kantor_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(11)+".csv" ,index=True) 

graph = nx.octahedral_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(12)+".csv" ,index=True) 

graph = nx.pappus_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(13)+".csv" ,index=True) 

graph = nx.petersen_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(14)+".csv" ,index=True) 

graph = nx.sedgewick_maze_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(15)+".csv" ,index=True) 

graph = nx.tetrahedral_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(16)+".csv" ,index=True) 

graph = nx.truncated_cube_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(17)+".csv" ,index=True) 

graph = nx.truncated_tetrahedron_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(18)+".csv" ,index=True) 

graph = nx.tutte_graph()
number_of_nodes, df = make_it_coord(graph)
df.to_csv("TrainTSP111_"+str(19)+".csv" ,index=True)

#%% OTHER GRAPH TYPES (tsp list 6)

# GRID (TRAIN - TEST)
def grid():
    random1 = random.randint(1, 3)
    random2 = random.randint(2, 8)
    random3 = random.randint(2, 12)
            
    graph = nx.grid_graph(dim=[random1, random2, random3])
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'grid')
    
    return number_of_nodes, df, 12 

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = grid()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 = grid()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        j = j + 1   
        
#%% HEXAGONAL LATTICE (TRAIN - TEST)
def hexa_lattice():
    random1 = random.randint(2, 3)
    random2 = random.randint(2, 15)
            
    graph = nx.hexagonal_lattice_graph(random1, random2, periodic=False, with_positions=True, create_using=None)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'hexagonal lattice')
    
    return number_of_nodes, df, 13 

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = hexa_lattice()
    
    if 1==1:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 = hexa_lattice()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        j = j + 1   
        
#%% TURAN (TRAIN - TEST)
def turan():
    random1 = random.randint(8, 30)
    random2 = random.randint(4, 8)
            
    graph = nx.turan_graph(random1, random2)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'turan')
    
    return number_of_nodes, df, 14 

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = turan()
    
    if 1==1:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 = turan()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        j = j + 1   
        
#%% RELAXED_CAVEMEN (TRAIN - TEST)
def relaxed_cavemen():
    random1 = random.randint(2, 15)
    random2 = random.randint(3, 9)
    random3 = random.randint(10, 30)/100
            
    graph = nx.relaxed_caveman_graph(random1, random2, random3)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'relaxed_cavemen')
    
    return number_of_nodes, df, 15 

#%% 
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = relaxed_cavemen()
    
    if 1==1:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 = relaxed_cavemen()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        j = j + 1   
        
#%% RING OF CLIQUES (TRAIN - TEST)
def ring_of_cliques():
    random1 = random.randint(2, 20)
    random2 = random.randint(3, 10)
            
    graph = nx.ring_of_cliques(random1, random2)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'ring of cliques')
    
    return number_of_nodes, df, 16 

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = ring_of_cliques()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 = ring_of_cliques()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        j = j + 1   
        
#%% WINDMILL (TRAIN - TEST)
def windmill():
    random1 = random.randint(3, 20)
    random2 = random.randint(2, 6)
            
    graph = nx.windmill_graph(random1, random2)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'windmill')
    
    return number_of_nodes, df, 17 

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = windmill()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 = windmill()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        j = j + 1  
        
#%% RANDOM TREE (TRAIN - TEST)
def random_tree():
    random1 = random.randint(10, 50)
            
    graph = nx.random_tree(random1, seed=None)
    nodePos = nx.spring_layout(graph)
    df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
    df = df[['x', 'y']].reset_index(drop=True)
    number_of_nodes = len(df)
        
    #plot graph 
    plt.scatter(df.x, df.y)
    plt.show()
    
    print(number_of_nodes, 'random_tree')
    
    return number_of_nodes, df, 18 

#%%
number_of_nodes_list = []
k = 0 
j = 0
counter1 = 0
counter2 = 0

while k < 35 and counter1 <= 100: 
    counter1 = counter1 + 1

    number_of_nodes, df, flag1 = random_tree()
    
    if 1==1:
        print('save me as train')
        df.to_csv("TrainTSP"+str(flag1)+'_'+str(k)+".csv" ,index=True) 
        number_of_nodes_list.append(number_of_nodes)
        k = k + 1 
        
while j < 15 and counter2 <= 100: 
    counter2 = counter2+ 1

    number_of_nodes, df, flag1 = random_tree()
    
    if number_of_nodes not in number_of_nodes_list:
        print('save me as test')
        df.to_csv("TestTSP"+str(flag1)+'_'+str(j)+".csv" ,index=True) 
        j = j + 1  
        
# #%% SOFT RANDOM GEOMETRIC
# def soft_random_tree():
#     random1 = random.randint(15, 40)
            
#     graph = nx.soft_random_geometric_graph(random1, 0.15)
#     nodePos = nx.spring_layout(graph)
#     df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
#     df = df[['x', 'y']].reset_index(drop=True)
#     number_of_nodes = len(df)
        
#     #plot graph 
#     plt.scatter(df.x, df.y)
#     plt.show()
    
#     print(number_of_nodes, 'soft geometric random tree')
    
#     return number_of_nodes, df 

# #%% MYCIELSKI
# def mycielski():
#     random1 = random.randint(2, 9)
            
#     graph = nx.mycielski_graph(random1)
#     nodePos = nx.spring_layout(graph)
#     df = pd.DataFrame(nodePos).T.rename(columns={0:'x', 1:'y'}) 
#     df = df[['x', 'y']].reset_index(drop=True)
#     number_of_nodes = len(df)
        
#     #plot graph 
#     plt.scatter(df.x, df.y)
#     plt.show()
    
#     print(number_of_nodes, 'mycielski')
    
#     return number_of_nodes, df 

