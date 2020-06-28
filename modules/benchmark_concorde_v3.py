#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 12:18:25 2020

@author: salihemredevrim
"""

import pandas as pd
from concorde.tsp import TSPSolver
import time
import networkx as nx
import matplotlib.pyplot as plt

#%% CONCORDE
def concorde(tsp_instance, verbose1):
    
    if verbose1 == 1: 
        print('concorde has started', time.time())
    
    t = time.time()          
    coordinates_ = pd.DataFrame(tsp_instance.coordinates_dict).T
    # Instantiate solver
    solver = TSPSolver.from_data(
        coordinates_[0],
        coordinates_[1],
        norm="EUC_2D")
        
    tour_data = solver.solve(time_bound = 60.0, verbose = verbose1, random_seed = 42) 
    
    time_diff = time.time() - t 
    
    if verbose1 == 1: 
        print('concorde has finished', time.time())
              
        print(tour_data.found_tour)
        
        # Plot tour
        graph11 = nx.Graph()
        graph11.add_nodes_from(tsp_instance.node_list)
        tour_ = tour_data[0]
    
        for k in range(len(tour_)-1):
            graph11.add_edge(tour_[k], tour_[k+1], weight = 1)

        print('Concorde solution:', tour_data.optimal_value)
        #set coordinates and plot
        nx.set_node_attributes(graph11, tsp_instance.coordinates_dict, 'pos')    
        nx.draw(graph11, nx.get_node_attributes(graph11, 'pos'), with_labels = True)
        #plt.savefig("simple_path.png") 
        plt.show()   
        
        
    return tour_data.found_tour, tour_data.optimal_value, time_diff, tour_data[0]
    