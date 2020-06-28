#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:38:53 2020

@author: salihemredevrim
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist 
from scipy.spatial.distance import squareform 
#from node2vec import Node2Vec
import networkx as nx
import warnings
import random 
import itertools
import pulp as plp
from copy import deepcopy
import dgl
warnings.filterwarnings("ignore")

#%% Initialize problem 
def init_prob(tsp_id, data_name, verbose, variable_type):

    tsp_instance = tsp_instance_(tsp_id, data_name)   
    problem1 = tsp_prob(tsp_instance, verbose, variable_type)
    return problem1

def init_prob_tsplib(tsp_id, data_name, verbose, variable_type):

    tsp_instance = tsp_instance_tsplib(tsp_id, data_name)   
    problem1 = tsp_prob(tsp_instance, verbose, variable_type)
    return problem1

#%% Create a part of the graph used in advanced combs
def create_graph_partial(node_list, coordinates_dict, X_soln, plot_flag):
    
    # load nodes
    graph = nx.Graph()
    graph.add_nodes_from(node_list)
    
    if len(X_soln) > 0:
      
        for k in range(len(X_soln)):
            graph.add_edge(X_soln.loc[k][0], X_soln.loc[k][1], capacity=1.0)
    
        if plot_flag == 1:
            X_soln['color'] = X_soln.apply(lambda x: 'b' if x.x_value >= 0.99  else 'r', axis=1)
            colors = X_soln['color'].to_list()

            #set coordinates and plot
            nx.set_node_attributes(graph, coordinates_dict, 'pos')    
            nx.draw(graph, nx.get_node_attributes(graph, 'pos'), with_labels = True, edge_color=colors)
            #plt.savefig("simple_path.png") 
            plt.show() 
    
    return graph

#%% Load/prepare tsp instance from train/test set
class tsp_instance_: 
    def __init__(self, tsp_id, data_name):
        
        self.tsp_id = tsp_id
        
        coordinates = pd.read_csv(data_name).reset_index(drop=False).rename(columns={'index': 'node'})
        coordinates["x"] = coordinates["x"]*1000
        coordinates["y"] = coordinates["y"]*1000
        coordinates['zip'] = list(zip(coordinates.x, coordinates.y))
        self.coordinates_dict = pd.Series(coordinates.zip.values, index=coordinates.node).to_dict()
        
        # coordinates -> euclidean distance 
        # to array 
        arr1 = coordinates[['x', 'y']].values
        dist1 = pdist(arr1, 'euclidean')
        tsp_distance = round(pd.DataFrame(squareform(dist1)), 0) #for precision
        
        tsp_distance = tsp_distance.iloc[:, 1:].stack().reset_index().rename(columns={'level_0': 'origin', 'level_1':'destination', 0:'distance'})
        self.tsp_distance = tsp_distance[tsp_distance['origin'] < tsp_distance['destination']].reset_index(drop=True)
        
        # number of nodes in the network    
        self.n_nodes = tsp_distance.destination.max() + 1
        
        # node list
        self.node_list = list(set(tsp_distance['origin'].append(tsp_distance['destination'])))
        
#%% Load/prepare tsp instance from tsplib excels 
class tsp_instance_tsplib: 
    def __init__(self, tsp_id, data_name):
        
        self.tsp_id = tsp_id
        
        # distance pairs and coordinates
        tsp_distance = pd.read_excel(data_name, sheet_name = 'distance_matrix')
        tsp_distance = tsp_distance.iloc[:, 1:].stack().reset_index().rename(columns={'level_0': 'origin', 'level_1':'destination', 0:'distance'})
        self.tsp_distance = tsp_distance[tsp_distance['origin'] < tsp_distance['destination']].reset_index(drop=True)

        coordinates = pd.read_excel(data_name, sheet_name = 'tsp')
        coordinates = coordinates[coordinates.columns[-3:]]
        coordinates = coordinates.reset_index(drop=False).drop(columns=['node']).rename(columns={'index':'node'})
        
        coordinates['zip'] = list(zip(coordinates.coor1, coordinates.coor2))
        self.coordinates_dict = pd.Series(coordinates.zip.values, index=coordinates.node).to_dict()
        
        # number of nodes in the network    
        self.n_nodes = tsp_distance.destination.max() + 1
        
        # node list
        self.node_list = list(set(tsp_distance['origin'].append(tsp_distance['destination'])))

#%% Define the problem class and lp relaxation
class tsp_prob:
    def __init__(self, tsp_instance, verbose, variable_type):
        
        # init problem info
        self.tsp_id = tsp_instance.tsp_id
        self.n_nodes = tsp_instance.n_nodes
        self.coordinates_dict = tsp_instance.coordinates_dict 
        self.node_list = tsp_instance.node_list 
        self.complete_flag = 0
        self.objective_val = 0 
        self.verbose = verbose
        
        # define problem
        self.prob = plp.LpProblem(name='TSP_'+str(self.tsp_id), sense= plp.LpMinimize)
        
        # define variable X (FLOAT)
        self.or_dest = list(zip(tsp_instance.tsp_distance.origin, tsp_instance.tsp_distance.destination))
        if variable_type == 'continuous':
            self.X = {(i,j): plp.LpVariable(cat=plp.LpContinuous, lowBound=0, upBound=1, name="x_{0}_{1}".format(i,j)) for (i, j) in self.or_dest}
        elif variable_type == 'binary':
            self.X = {(i,j): plp.LpVariable(cat=plp.LpBinary, name="x_{0}_{1}".format(i,j)) for (i, j) in self.or_dest}
        
        # add objective and degree constraints
        self.def_objective(tsp_instance)
        self.degree_con(tsp_instance)
        if self.verbose == 1:
            self.log()
        
        # initial solution
        self.solve_lp_relax()
        self.graph = self.create_graph()
        self.check_if_complete()
                
        # OBJECTIVE FUNCTION
    def def_objective(self, tsp_instance):
        # create distance dict and set objective
        self.distance = {(i,j): tsp_instance.tsp_distance[(tsp_instance.tsp_distance['origin'] == i) & (tsp_instance.tsp_distance['destination'] == j)]['distance'].item() for (i,j) in self.or_dest}
        # self.prob += plp.lpSum(self.X[i,j] * distance[i,j] for (i, j) in self.or_dest)
        objective = plp.lpSum(self.X[i,j] * self.distance[i,j] for (i, j) in self.or_dest)
        self.prob.setObjective(objective)
           
        # DEGREE CONSTRAINTS
        # there must be (exactly) two links per node
    def degree_con(self, tsp_instance):
        for k in self.node_list: 
            
            origin_list = tsp_instance.tsp_distance[tsp_instance.tsp_distance['origin'] == k]['destination'].to_list()
            dest_list = tsp_instance.tsp_distance[tsp_instance.tsp_distance['destination'] == k]['origin'].to_list()
        
            constraint = plp.LpConstraint(
                    e = plp.lpSum(self.X[k, m] for m in origin_list) + plp.lpSum(self.X[n, k] for n in dest_list),
                    sense= plp.LpConstraintEQ, #Equal to
                    rhs = 2.0)
        
            self.prob.addConstraint(constraint)      
    
        # LOG FILE 
    def log(self):  
        self.prob.writeLP('TSP_'+str(self.tsp_id)+'_model.lp')
        
        # ADD CONSTRAINT
    def add_constraint(self, constraint):
        self.prob.addConstraint(constraint)  

    # SOLVE GIVEN LP RELAXATION  
    def solve_lp_relax(self):
        # solve only the degree LP relaxation (variables are float)
        self.prob.solve() 
        
        if self.verbose == 1:
            print("Status:", plp.LpStatus[self.prob.status])
            self.log()
    
        # load solution
        opt_df = pd.DataFrame.from_dict(self.X, orient="index", columns = ["variable_object"])
        opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["origin", "destination"])
        opt_df.reset_index(inplace=True)
        opt_df["x_value"] = opt_df["variable_object"].apply(lambda item: item.varValue)

        # keep solutions in df
        self.X_soln = opt_df[opt_df["x_value"] > 0.0].reset_index(drop=True)
        self.objective_val = round(self.prob.objective.value(), 2)
        
    # UPDATE SOLUTION GRAPH
    def create_graph(self):
    
        # load nodes
        # create an undirected graph
        graph = nx.Graph()
        graph.add_nodes_from(self.node_list)
        
        if len(self.X_soln) > 0:
           
            for k in range(len(self.X_soln)):
                graph.add_edge(self.X_soln.loc[k][0], self.X_soln.loc[k][1], weight = self.X_soln.loc[k]['x_value'])
        
        if self.verbose == 1:
            self.X_soln['color'] = self.X_soln.apply(lambda x: 'b' if x.x_value >= 0.99 else 'r', axis=1)
            colors = self.X_soln['color'].to_list()
        
            # set coordinates and plot
            nx.set_node_attributes(graph, self.coordinates_dict, 'pos')    
            nx.draw(graph, nx.get_node_attributes(graph, 'pos'), with_labels = True, edge_color=colors)
            #plt.savefig("simple_path.png") 
            plt.show()   
        
        return graph 
    
    # CHECK WHETER SOLUTION IS INTEGER AND COMPLETE TOUR
    def check_if_complete(self):      
        # check if all edges == 1
        connections1 = self.X_soln[self.X_soln['x_value'] < 1.0].reset_index(drop=True)
        
        if len(connections1) == 0: # no fractional solution CHECKED! 
            fully_connected = nx.is_connected(self.graph)
        
            if fully_connected == 1: # Fully connected CHECKED!
                self.complete_flag = 1
                
                if self.verbose == 1:
                    print('****************************************************')
                    print('Tour is complete!')
     
        
    # Action: SUBTOUR ELIMINATION CONSTRAINTS 
    def subtour_elimn(self):
     
        counter = 0 
        
        # first check whether nodes are fully connected
        if nx.is_connected(self.graph) == 1: 
            if self.verbose == 1:
                print('NO effective subtour elimination')
                
            return counter
        
        # find isolated cycles
        check1 = nx.cycle_basis(self.graph)
        list1 = []
        
        # #merge cycles to find broader cycles
        # for k in range(len(check1)): 
        #     cycle1 = check1[k]
        #     j = k+1
        #     while j > k and j < len(check1):
        #         cycle2 = check1[j]
        #         intersection_flag = any(item in cycle2 for item in cycle1)
        #         if intersection_flag == True: 
        #            cycle1 = cycle1 + cycle2
        #            cycle1 = list(set(cycle1))
        #         j = j+1
                
        #     check1.append(cycle1)   
            
        # loop over all subtours
        for x in range(len(check1)):
            
            list1 = check1[x]
            len11 = len(list1)
            edge_sum = np.sum(self.X_soln[(self.X_soln['origin'].isin(list1)) & (self.X_soln['destination'].isin(list1))]['x_value'])
                
            #since subtour elimination constraints cannot fix cycles with sum of edges < len - 1
            #no need to add those as constraints
                
            if len11 > 2 and edge_sum >= len11 - 1:
                # add the constraint 
                # sum(x(Delta(S))) >= 2
                    
                # create in-out pairs 
                outer_subtour = [q for q in self.node_list if q not in list1] 
                product0 =  pd.DataFrame(list(itertools.product(list1, outer_subtour)))
                swap0 = np.where(product0[0] < product0[1], [product0[0], product0[1]],[product0[1], product0[0]])
                pairs_for_constraint = list(zip(swap0[0], swap0[1]))
    
                constraint = plp.LpConstraint(
                    e = plp.lpSum(self.X[o, d] for (o,d) in pairs_for_constraint),
                    sense= plp.LpConstraintGE, #Greater than and equal to
                    rhs = 2)
            
                self.add_constraint(constraint)
                counter = counter+1
        
        if self.verbose == 1:    
            print('# of subtour elimination constraints inserted:', counter)    
              
        return counter
    
    
    # Find handles as subcycles 
    def detect_handles(self, cycle_cutoff, reverse_flag):
        
        #an assumption here! 
        #When selecting cycles (handles), consider edges having values less than 1 only as suggested in W.Cook's book     
        #detect edges < 1
        connections_fractional = self.X_soln[self.X_soln.x_value <= cycle_cutoff].reset_index(drop=True)
     
        #find cycles including detected nodes (depth search)
        cycles = create_graph_partial(self.node_list, self.coordinates_dict, connections_fractional, self.verbose)
               
        #find cycles (depth search)
        handles = nx.cycle_basis(cycles)
        
        #merge cycles to find broader cycles
        for k in range(len(handles)): 
            cycle1 = handles[k]
            j = k+1
            while j > k and j < len(handles):
                cycle2 = handles[j]
                intersection_flag = any(item in cycle2 for item in cycle1)
                if intersection_flag == True: 
                   cycle1 = cycle1 + cycle2
                   cycle1 = list(set(cycle1))
                j = j+1
                
            handles.append(cycle1)   
            
        #add connected nodes (if they have edges >= 1) into the handle 
        handles_2 = []
        
        for j in range(len(handles)):
            handle = handles[j]
            inter1 = self.X_soln[(self.X_soln['origin'].isin(handle)) & (~self.X_soln['destination'].isin(handle))].reset_index(drop=True)
            inter2 = self.X_soln[(~self.X_soln['origin'].isin(handle)) & (self.X_soln['destination'].isin(handle))].reset_index(drop=True)
            merge1 = inter1[['destination', 'x_value']].append(inter2[['origin', 'x_value']].rename(columns= {"origin":"destination"}))
            merge1 = merge1.groupby('destination')['x_value'].sum().reset_index(drop=False)
            merge1 = merge1[merge1.x_value > 1]
            add_me = list(merge1.destination)
            
            handle = handle + add_me
            handle = list(set(handle))
            
            handles_2.append(handle)
        
        #drop duplicates 
        handles_2 = [list(x) for x in set(tuple(x) for x in handles_2)]     
        
        #reverse_flag -> 0 small to large
        #reverse_flag -> 1 large to small
        #reverse_flag -> 2 no sort just randomize
        
        if reverse_flag == 0:
        #sort based on type of inequality (ie. for blossom inside-> outside; for complex comb, other way around)
            handles_2.sort(key=len, reverse = 0)
        elif reverse_flag == 1:
            handles_2.sort(key=len, reverse = 1)
        else:
            #some randomization here 
            random.shuffle(handles_2) #shuffle list
        
        return handles_2, cycles


    # Find a blossom inequality
    #THIS IS USED FOR ONLY PATH INEQ (NO REAL ACTION)   
    def find_a_blossom(self, cycle_cutoff, teeth_cutoff):
        
        #teeth cutoff:in order to define inequalities, this cutoff is set since creating teeth with high cut values is best to find a violated inequality as suggested and observed.  
        #detect handles  
        #Reverse flag = RANDOMIZE
        handle, graph = self.detect_handles(cycle_cutoff, 2)   
        comb_handle = []
        pairs = []
        comb_flag = 0     
        merge_flag = 0
        
        if len(handle) == 0: 
            if self.verbose == 1:
                print('no possible blossom here!')
            return comb_handle, pairs, comb_flag 
       
        for k in range(len(handle)): 
            
            len1 = len(handle[k])
                
            if len1 >= 3: #must be greater than or equal to 3 
                comb_handle = handle[k]
                pairs = []
                intersection_list = comb_handle.copy() #keep nodes to compare and avoid intersections in teeth 
  
                for j in comb_handle: 
                    
                    #here we are using connections which have all edges in the current solution
                    destination = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['origin'] == j)]['destination']
                    origin = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['destination'] == j)]['origin']
             
                    #clean up, eliminate nodes in the intersection list
                    destination = [x for x in destination if x not in intersection_list] 
                    origin = [q for q in origin if q not in intersection_list] 
                                
                    if len(destination) > 0: 
                        for p in range(len(destination)):    
                            destination1 = destination[p]
                            pairs.append(tuple((int(j), destination1)))
                            intersection_list.append(destination1)
                            
                    if len(origin) > 0: 
                        for p in range(len(origin)):    
                            origin1 = origin[p]
                            pairs.append(tuple((int(j), origin1)))
                            intersection_list.append(origin1)
                                              
                pairs = list(set(pairs)) #take distinct pairs for teeth 
                if len(pairs) > 0:
                    length_teeth = len(pd.DataFrame(pairs)[0].unique())
                else: 
                    length_teeth = 0
                                
                if length_teeth >= 3:
                        
                    if length_teeth%2==0: #must be odd, otherwise merge last two 
                        merge_flag = 1 
                        length_teeth = length_teeth - 1
                                                   
    #**************************************PREPARE CONSTRAINTS************************************************************************************************************************************************
                         
                    #outbond nodes of handle
                    outer_comb_handle = [t for t in self.node_list if t not in comb_handle] 
                        
                    #pairwise combinations
                    product0 =  pd.DataFrame(list(itertools.product(comb_handle, outer_comb_handle)))
                    swap0 = np.where(product0[0] < product0[1], [product0[0], product0[1]],[product0[1], product0[0]])
                    handle_for_constraint = list(zip(swap0[0], swap0[1]))
                     
                    #populate all possible outbound edge options for each tooth (node in Tj -> node in V\Tj)
                    teeth_for_constraint = []
                    xd1 = pd.DataFrame(pairs)
                    xd1.sort_values(by=0)
                    xd1["h_rank"]=xd1.groupby([0]).ngroup()
                    
                    if merge_flag == 1:
                        max1 = xd1.h_rank.max()
                        xd1["h_rank"] = xd1.apply(lambda x: max1 - 1 if x.h_rank == max1 else x.h_rank, axis = 1)
          
                    for l in list(xd1["h_rank"].unique()):
                        
                        base = list(np.unique(xd1[xd1["h_rank"] == l][[0,1]] ))
                        outer_base = [r for r in self.node_list if r not in base] 
                        #pairwise combinations
                        product1 =  pd.DataFrame(list(itertools.product(base, outer_base)))
                        swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                        list1 = list(zip(swap1[0], swap1[1]))
                        teeth_for_constraint = teeth_for_constraint + list1 
                       
                    #add constraint
                    #derived the constraint using formulation in Delta notation 
                    #x(Delta(H)) + sum(x(Delta(Tj))) >= 3s+1 
                       
                    constraint = plp.LpConstraint(
                        e = plp.lpSum(self.X[o, d] for (o, d) in handle_for_constraint) + 
                            plp.lpSum(self.X[k, l] for (k, l) in teeth_for_constraint),
                        sense= plp.LpConstraintGE, #Greater than and equal to
                        rhs = (3*length_teeth + 1))
            
                    self.add_constraint(constraint)
                        
                    comb_flag = 1
                    if self.verbose == 1:
                        print('added a comb w 1 handle!')
                        print('handle:', comb_handle)
                        print('teeth', pairs)
                        
                    #return right after finding an inequality
                    return comb_handle, pairs, comb_flag  
          
        return comb_handle, pairs, comb_flag  


    #Action: FIND ALL BLOSSOM INEQUALITIES IN THE CURRENT SOLUTION (all |Tj| = 2)
    def find_multi_blossoms(self, cycle_cutoff, teeth_cutoff):
        
        #teeth cutoff:in order to define inequalities, this cutoff is set since creating teeth with high cut values is best to find a violated inequality as suggested and observed.  
        #detect handles  
        #Reverse flag = RANDOMIZE
        handle, graph = self.detect_handles(cycle_cutoff, 2) 
        counter = 0    

        if len(handle) == 0: 
            if self.verbose == 1:
                print('no possible blossom here!')
            return counter
       
        for k in range(len(handle)): 
            len1 = len(handle[k])
            merge_flag = 0
                
            if len1 >= 3: #must be greater than or equal to 3 
                comb_handle = handle[k]
                pairs = []
                intersection_list = comb_handle.copy() #keep nodes to compare and avoid intersections in teeth 
                    
                for j in comb_handle: 
                        
                    #here we are using connections which have all edges in the current solution
                    destination = self.X_soln[(self.X_soln ['x_value'] >= teeth_cutoff) & (self.X_soln['origin'] == j)]['destination']
                    origin = self.X_soln[(self.X_soln ['x_value'] >= teeth_cutoff) & (self.X_soln['destination'] == j)]['origin']
                        
                    #clean up, eliminate nodes in the intersection list
                    destination = [x for x in destination if x not in intersection_list] 
                    origin = [q for q in origin if q not in intersection_list] 
                                
                    if len(destination) > 0: 
                        for p in range(len(destination)):    
                            destination1 = destination[p]
                            pairs.append(tuple((int(j), destination1)))
                            intersection_list.append(destination1)
                            
                    if len(origin) > 0: 
                        for p in range(len(origin)):    
                            origin1 = origin[p]
                            pairs.append(tuple((int(j), origin1)))
                            intersection_list.append(origin1)
                                              
                pairs = list(set(pairs)) #take distinct pairs for teeth 
                if len(pairs) > 0:
                    length_teeth = len(pd.DataFrame(pairs)[0].unique())
                else: 
                    length_teeth = 0
                    
                if length_teeth >= 3:
                        
                    if length_teeth%2==0: #must be odd, otherwise merge last two 
                        merge_flag = 1 
                        length_teeth = length_teeth - 1
                                                   
    #**************************************PREPARE CONSTRAINTS************************************************************************************************************************************************
                                                         
                    #outbond nodes of handle
                    outer_comb_handle = [r for r in self.node_list if r not in comb_handle] 
                        
                    #pairwise combinations
                    product0 =  pd.DataFrame(list(itertools.product(comb_handle, outer_comb_handle)))
                    swap0 = np.where(product0[0] < product0[1], [product0[0], product0[1]],[product0[1], product0[0]])
                    handle_for_constraint = list(zip(swap0[0], swap0[1]))
                     
                    #populate all possible outbound edge options for each tooth (node in Tj -> node in V\Tj)
                    teeth_for_constraint = []
                    xd1 = pd.DataFrame(pairs)
                    xd1.sort_values(by=0)
                    xd1["h_rank"]=xd1.groupby([0]).ngroup()
                    
                    if merge_flag == 1:
                        max1 = xd1.h_rank.max()
                        xd1["h_rank"] = xd1.apply(lambda x: max1 - 1 if x.h_rank == max1 else x.h_rank, axis = 1)
          
                    for l in list(xd1["h_rank"].unique()):
                        
                        base = list(np.unique(xd1[xd1["h_rank"] == l][[0,1]] ))
                        outer_base = [r for r in self.node_list if r not in base] 
                        #pairwise combinations
                        product1 =  pd.DataFrame(list(itertools.product(base, outer_base)))
                        swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                        list1 = list(zip(swap1[0], swap1[1]))
                        teeth_for_constraint = teeth_for_constraint + list1 
                                       
                    #add constraint
                    #derived the constraint using formulation in Delta notation 
                    #x(Delta(H)) + sum(x(Delta(Tj))) >= 3s+1 
                       
                    constraint = plp.LpConstraint(
                        e = plp.lpSum(self.X[o, d] for (o, d) in handle_for_constraint) + 
                            plp.lpSum(self.X[k, l] for (k, l) in teeth_for_constraint),
                        sense= plp.LpConstraintGE, #Greater than and equal to
                        rhs = (3*length_teeth + 1))
            
                    self.add_constraint(constraint)
                        
                    counter = counter + 1 
                    
                    if self.verbose == 1:
                        print('added a comb w 1 handle!')
                        print('handle:', comb_handle)
                        print('teeth', pairs)
                        
                    #clear all
                    comb_handle = []
                    pairs = []
    
        if self.verbose == 1:      
            print('# of blossom inequalities inserted:', counter)   
              
        return counter
    
    
    #Action: COMPLEX COMB INEQUALITIES (try to find inner triangles inside the handle in full graph)
    def find_adv_combs(self, cycle_cutoff, teeth_cutoff):
        
        #teeth cutoff:in order to define inequalities, this cutoff is set since creating teeth with high cut values is best to find a violated inequality as suggested and observed.  
        #detect handles  
        #Reverse flag = RANDOMIZE
        handle, graph = self.detect_handles(cycle_cutoff, 2) 
        
        #number of nodes must be > 3 
        handle_3 = [] 
        
        for m in range(len(handle)): 
            handle_0 = handle[m]
            if len(handle_0) > 3:
                handle_3.append(handle_0)
    
        counter = 0 
        
        if len(handle_3) == 0: 
            if self.verbose == 1:
                print('no possible complex comb here!')
            return counter 
       
        else:
            
            for k in range(len(handle_3)): 
                
                comb_handle = handle_3[k]
                
                #find inner subset 
                X_soln_inner = self.X_soln[(self.X_soln['origin'].isin(comb_handle)) | (self.X_soln ['destination'].isin(comb_handle))].reset_index(drop=True)
                node_list_inner = list(set(X_soln_inner['origin'].append(X_soln_inner['destination'])))
                node_list_inner.sort()
                coordinates_dict_inner = {key: self.coordinates_dict[key] for key in self.coordinates_dict.keys() and node_list_inner}
    
                #find cycles inside the handle             
                inner_graph = create_graph_partial(node_list_inner, coordinates_dict_inner, X_soln_inner, self.verbose)
                inner_cycles = nx.cycle_basis(inner_graph)
                
                #sort them shorter to longer
                inner_cycles.sort(key=len)
           
                if len(inner_cycles) > 1: 
    
                    #so first find possible (not intersecting) teeth then remaining cycle(s) will be handle and will be search for additional, basic teeth if needed
                    teeth_list = []
                    handle_list = []
                    intersection_teeth = [] 
                    
                    for j in range(len(inner_cycles)):
                        tooth_0 = inner_cycles[j]
                        intersection_flag = any(item in tooth_0 for item in intersection_teeth)
                        
                        if intersection_flag == 0:
                            teeth_list.append(tooth_0)
                            intersection_teeth = list(set(intersection_teeth + tooth_0))
                        else: #remainings will be handle 
                            handle_list = list(set(handle_list + tooth_0))
    
                    #each teeth must have at least one node that is not included in handle 
                    range1 = len(teeth_list)
                    tooth_out = []
                    
                    for z in range(range1): 
                        tooth_00 = teeth_list[z]
                        intersection_flag2 = all(item in handle_list for item in tooth_00)
                        
                        if intersection_flag2 == 1:
                            tooth_out.append(tooth_00)
                            
                    #remove teeth out
                    teeth_list = [item for item in teeth_list if item not in tooth_out]                    
                    
                    #check basic tooth/teeth in remaining nodes in handle 
                    #avoid_list = list(np.unique(teeth_list)) #they cannot intersect with complex teeth
                    avoid_list = list(set(x for l in teeth_list for x in l)) 
                    pairs = []
                    
                    for m in handle_list:
                        if m not in avoid_list:
                            
                            #here we are using connections which have all edges in the current solution
                            destination = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['origin'] == m)]['destination']
                            origin = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['destination'] == m)]['origin']
                        
                            #clean up, eliminate nodes in the avoid list and handle too
                            destination = [x for x in destination if x not in avoid_list] 
                            origin = [q for q in origin if q not in avoid_list] 
                            
                            destination = [n for n in destination if n not in handle_list] 
                            origin = [t for t in origin if t not in handle_list]  
                            
                            if len(destination) > 0: 
                            
                                #if still there is more than 1 node, pick one randomly
                                destination = destination[random.randint(0, len(destination)-1)]
                                pairs.append(tuple((int(m), destination)))
                                avoid_list.append(destination)
                                avoid_list.append(m)
                            
                            elif len(origin) > 0: 
                            
                                #if still there is more than 1 node, pick one randomly
                                origin = origin[random.randint(0, len(origin)-1)]
                                pairs.append(tuple((int(m), origin)))
                                avoid_list.append(origin)
                                avoid_list.append(m)
                                                  
                    #check number of basic or complex teeth found
                    teeth_list2 = teeth_list + pairs
                    length_teeth = len(teeth_list2)
                    
                    if length_teeth < 3: # no comb here
                        if self.verbose == 1:
                            print('no possible complex comb here!')
                        #turn back 
                        teeth_list = []
                        handle_list = []
                        intersection_teeth = [] 
                        pairs = []
                   
                    else: 
                        #if number of basic or complex teeth is odd then use all 
                        #else drop one of them            
                        if length_teeth%2==0: 
                            teeth_list2.pop(random.randrange(length_teeth))
                            length_teeth = length_teeth - 1
                            
    #**************************************PREPARE CONSTRAINTS************************************************************************************************************************************************
                        
                        comb_handle = handle_list.copy()
                            
                        #outbond nodes of handle
                        outer_comb_handle = [x for x in self.node_list if x not in comb_handle] 
                        
                        #pairwise combinations
                        product0 =  pd.DataFrame(list(itertools.product(comb_handle, outer_comb_handle)))
                        swap0 = np.where(product0[0] < product0[1], [product0[0], product0[1]],[product0[1], product0[0]])
                        handle_for_constraint = list(zip(swap0[0], swap0[1]))
                     
                        #populate all possible outbound edge options for each tooth (node in Tj -> node in V\Tj)
                        teeth_for_constraint = []
             
                        for e in range(length_teeth):
                            base = list(np.unique(teeth_list2[e]))
                            outer_base = [w for w in self.node_list if w not in base] 
                            #pairwise combinations
                            product1 =  pd.DataFrame(list(itertools.product(base, outer_base)))
                            swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                            list1 = list(zip(swap1[0], swap1[1]))
                            teeth_for_constraint = teeth_for_constraint + list1 
                       
                       #add constraint
                       #derived the constraint using formulation in Delta notation 
                       #x(Delta(H)) + sum(x(Delta(Tj))) >= 3s+1 
                       
                        constraint = plp.LpConstraint(
                        e = plp.lpSum(self.X[o, d] for (o, d) in handle_for_constraint) + 
                            plp.lpSum(self.X[k, l] for (k, l) in teeth_for_constraint),
                        sense= plp.LpConstraintGE, #Greater than and equal to
                        rhs = (3*length_teeth + 1))
            
                        self.add_constraint(constraint)
                        
                        counter = counter + 1
                        if self.verbose == 1:
                            print('added a comb w 1 handle!')
                            print('handle:', comb_handle)
                            print('teeth', teeth_list2)
                        
                        #clear all 
                        comb_handle = [] 
                        teeth_list2 = [] 
                        teeth_list = []
                        handle_list = []
                        intersection_teeth = [] 
                        pairs = []  

        if self.verbose == 1:      
            print('# of complex comb inequalities inserted:', counter)   
                      
        return counter    
 
    
    #Action: CLIQUE TREE INEQUALITIES (2 HANDLES ONLY)
    def find_clique_tree_2(self, cycle_cutoff, teeth_cutoff):
     
        #detect handles  
        #Reverse flag = NO
        selected_handles, graph = self.detect_handles(cycle_cutoff, 0) 
        
        #number of handle candidates
        n_handles = len(selected_handles)
        counter = 0 
        merge_flag1 = 0 
        merge_flag2 = 0
        
        #if less than 2 handles found: stop
        if n_handles <= 1: 
            if self.verbose == 1:
                print('no potential clique tree (2) here!')
            return counter
        
        else:    
            path = []
            path_check = []
            break_flag = 0 
            
            #find a common/intersecting tooth among handle candidates
            for k in range(n_handles):
                
                if break_flag == 1: #break for loop if you find a pair of handles
                    break 
                
                j = k+1
                while k < j and j < n_handles and break_flag == 0: 
                    #pairwise comparison of handles to see if there is a connection (USING FULL GRAPH)
                    path_flag = nx.has_path(self.graph, source=selected_handles[k][0], target=selected_handles[j][0])
                    
                    if path_flag == 1:
                        path = nx.shortest_path(self.graph, source=selected_handles[k][0], target=selected_handles[j][0])
                        #Due to definition there must be at least one node of Tintersection outside the H1 and H2 
                        both_handles = selected_handles[k] + selected_handles[j]
                        path_check = [x for x in path if x not in  both_handles]
                        
                        if len(path_check) >= 1: 
                            break_flag = 1 #if you find one path break the loop
                            handle_1 = selected_handles[k]
                            handle_2 = selected_handles[j]
                            
                            #so path is intersecting tooth between two handles  
                            #eliminate nodes in handle1 and 2 except start and end nodes 
                            end1 = path[-1] #redundant but just to initialize
                            start1 = path[0]
                            
                            for q in path: 
                                if q in handle_1: 
                                    start1 = q
                                elif q in handle_2: 
                                    end1 = q 
                                    break
                        
                            start1 = path.index(start1)
                            start2 = path.index(end1)
                            path = path[start1:start2+1]
                                            
                    j = j+1     
            
            #we should find additional (even number of) teeth for each handle
            if len(path_check) < 1: 
                if self.verbose == 1:
                    print('no potential clique tree (2) here!')
                return counter
                        
            else:
              
                #find some non-intersecting teeth
                intersection_list2 = list(set(path.copy() + handle_1 + handle_2))
                pairs1 = []
                pairs2 = []
                
                for t in handle_1: 
                     #here we are using connections which have all edges in the current solution
                     destination = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['origin'] == t)]['destination']
                     origin = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['destination'] == t)]['origin']
                                     
                     #clean up, eliminate nodes in the intersection list
                     destination = [x for x in destination if x not in intersection_list2] 
                     origin = [x for x in origin if x not in intersection_list2] 
                        
                     if len(destination) > 0: 
                        for p in range(len(destination)):    
                            destination1 = destination[p]
                            pairs1.append(tuple((int(t), destination1)))
                            intersection_list2.append(destination1)
                            
                     if len(origin) > 0:
                         for p in range(len(origin)): 
                             origin1 = origin[p]
                             pairs1.append(tuple((int(t), origin1)))
                             intersection_list2.append(origin1)
                
                     intersection_list2.append(int(t))
             
                #teeth connected to the second handle
                for t in handle_2: 
                    #here we are using connections which have all edges in the current solution
                    destination = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['origin'] == t)]['destination']
                    origin = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['destination'] == t)]['origin']
                                        
                    #clean up, eliminate nodes in the intersection list
                    destination = [x for x in destination if x not in intersection_list2] 
                    origin = [x for x in origin if x not in intersection_list2] 
                        
                    if len(destination) > 0: 
                        for p in range(len(destination)):   
                            destination1 = destination[p]
                            pairs2.append(tuple((int(t), destination1)))
                            intersection_list2.append(destination1)
                    
                    if len(origin) > 0: 
                        for p in range(len(origin)):   
                            origin1 = origin[p]
                            pairs2.append(tuple((int(t), origin1)))
                            intersection_list2.append(origin1)
                
                    intersection_list2.append(int(t))
                                 
                pairs1 = list(set(list(pairs1)))
                pairs2 = list(set(list(pairs2)))
                
                if len(pairs1) > 0:
                    length_teeth1 = len(pd.DataFrame(pairs1)[0].unique())
                else: 
                    length_teeth1 = 0
                    
                if len(pairs2) > 0:
                    length_teeth2 = len(pd.DataFrame(pairs2)[0].unique())
                else: 
                    length_teeth2 = 0
                    
                # # of non-intersecting teeth must be even and at least 2 per handle 
                if length_teeth1 > 1 and length_teeth2 > 1:
                    
                    if length_teeth1%2 == 1:
                        merge_flag1 = 1 
                        length_teeth1 = length_teeth1 - 1
                        
                    if length_teeth2%2 == 1:
                        merge_flag2 = 1 
                        length_teeth2 = length_teeth2 - 1

    #**************************************PREPARE CONSTRAINTS************************************************************************************************************************************************
                      
                    #handle pairs for the constraint
                    outer_handle_1 = [x for x in self.node_list if x not in handle_1]
                    outer_handle_2 = [x for x in self.node_list if x not in handle_2]
                    
                    product1 =  pd.DataFrame(list(itertools.product(handle_1, outer_handle_1)))
                    swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                    list1 = list(zip(swap1[0], swap1[1]))
                    
                    product2 = pd.DataFrame(list(itertools.product(handle_2, outer_handle_2)))
                    swap2 = np.where(product2[0] < product2[1], [product2[0], product2[1]],[product2[1], product2[0]])
                    list2 = list(zip(swap2[0], swap2[1]))
                    
                    handle_for_constraint = list1 + list2
                        
                    #create intersection tooth
                    inter_tooth = path 
                    outer_inter_tooth = [x for x in self.node_list if x not in inter_tooth]
                    product3 = pd.DataFrame(list(itertools.product(inter_tooth, outer_inter_tooth)))
                    swap3 = np.where(product3[0] < product3[1], [product3[0], product3[1]],[product3[1], product3[0]])
                    list3 = list(zip(swap3[0], swap3[1]))
                  
                    #finalize the pairs with non-intersecting teeth 
                    teeth_for_constraint = list3.copy()
                    #1ST HANDLE'S TEETH             
                    xd1 = pd.DataFrame(pairs1)
                    xd1.sort_values(by=0)
                    xd1["h_rank"]=xd1.groupby([0]).ngroup()
                    
                    if merge_flag1 == 1:
                        max1 = xd1.h_rank.max()
                        xd1["h_rank"] = xd1.apply(lambda x: max1 - 1 if x.h_rank == max1 else x.h_rank, axis = 1)
          
                    for l in list(xd1["h_rank"].unique()):
                        base = list(np.unique(xd1[xd1["h_rank"] == l][[0,1]] ))
                        outer_base = [r for r in self.node_list if r not in base] 
                        #pairwise combinations
                        product1 =  pd.DataFrame(list(itertools.product(base, outer_base)))
                        swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                        list1 = list(zip(swap1[0], swap1[1]))
                        teeth_for_constraint = teeth_for_constraint + list1 
                    
                   #2ND HANDLE'S TEETH             
                    xd1 = pd.DataFrame(pairs2)
                    xd1.sort_values(by=0)
                    xd1["h_rank"]=xd1.groupby([0]).ngroup()
                    
                    if merge_flag2 == 1:
                        max1 = xd1.h_rank.max()
                        xd1["h_rank"] = xd1.apply(lambda x: max1 - 1 if x.h_rank == max1 else x.h_rank, axis = 1)
          
                    for l in list(xd1["h_rank"].unique()):
                        base = list(np.unique(xd1[xd1["h_rank"] == l][[0,1]] ))
                        outer_base = [r for r in self.node_list if r not in base] 
                        #pairwise combinations
                        product1 =  pd.DataFrame(list(itertools.product(base, outer_base)))
                        swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                        list1 = list(zip(swap1[0], swap1[1]))
                        teeth_for_constraint = teeth_for_constraint + list1                     

                    #add constraint
                    #derived the constraint using formula with Delta notation
                    #sum(Delta(E(Hi))) + sum(Delta(E(Tj))) >= 2r + 3s - 1
                    n_of_teeth = length_teeth1 + length_teeth2 + 1
                    constraint = plp.LpConstraint(
                        e = plp.lpSum(self.X[o, d] for (o, d) in handle_for_constraint) + 
                            plp.lpSum(self.X[k, l] for (k, l) in teeth_for_constraint),
                            sense= plp.LpConstraintGE, #Greater than and equal to
                            rhs = (2*2 + 3*n_of_teeth - 1))
             
                    self.add_constraint(constraint)
                    
                    #return right after finding an inequality   
                    if self.verbose == 1:
                        print('added a clique tree w 2 handles!')
                    counter = counter + 2 
                    return counter 

        if self.verbose == 1:    
            print('# of clique trees inserted:', counter)   
              
        return counter    


    #Action: BASIC ENVELOPE (2 HANDLES 1 NON-INTERSECTING TOOTH NODE ONLY)
    def find_basic_envelope(self, cycle_cutoff, teeth_cutoff):
     
        #detect handles  
        #Reverse flag = NO
        selected_handles, graph = self.detect_handles(cycle_cutoff, 0) 
        
        #handle sizes -> 3 
        selected_handles = [item for item in selected_handles if len(item) == 3]
        
        #number of handle candidates
        n_handles = len(selected_handles)
        counter = 0 
                
        #if less than 2 handles found: stop
        if n_handles <= 1: 
            if self.verbose == 1:
                print('no potential envelope here!')
            return counter
        
        else:    
            
            break_flag = 0 
            
            #find a common/intersecting tooth among handle candidates
            for k in range(n_handles):
                
                if break_flag == 1:
                    break 
                
                j = k + 1
                while k < j and j < n_handles and break_flag == 0: 
                    pairs_dom = []
                    handle_1 = selected_handles[k]
                    intersection_list = handle_1.copy()
                    handle_2 = selected_handles[j]
                    
                    for t in handle_1: 
                        #here we are using connections which have all edges in the current solution
                        destination = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['origin'] == t)]['destination']
                        origin = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['destination'] == t)]['origin']
                          
                        # clean up handle 1 nodes
                        destination = [x for x in destination if x not in intersection_list] 
                        origin = [x for x in origin if x not in intersection_list] 
                         
                        #get only if connected node is in the second handle
                        destination_dom = [x for x in destination if x in handle_2] 
                        origin_dom = [x for x in origin if x in handle_2] 
                        
                        if len(destination_dom) > 0: 
                            for p in range(len(destination_dom)):    
                                destination1 = destination_dom[p]
                                pairs_dom.append(tuple((int(t), destination1)))
                                intersection_list.append(destination1)
                            
                        if len(origin_dom) > 0:
                            for p in range(len(origin_dom)): 
                                origin1 = origin_dom[p]
                                pairs_dom.append(tuple((int(t), origin1)))
                                intersection_list.append(origin1)
                                        
                    j = j + 1        
                    if len(pairs_dom) > 0:
                        if len(list(pd.DataFrame(pairs_dom)[0].unique())) == 2: 
                            # 2 fully connected teeth and 1 not connected  
                
                            alien1 = [x for x in handle_1 if x not in list(pd.DataFrame(pairs_dom)[0].unique())] 
                            alien2 = [x for x in handle_2 if x not in list(pd.DataFrame(pairs_dom)[1].unique())] 
                            
                            if len(alien1) > 0:
                                destination = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['origin'] == alien1[0])]['destination']
                                origin = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['destination'] == alien1[0])]['origin']
                            else:
                                destination = []
                                origin = []
                                 
                            # clean up handle 1 nodes
                            destination = [x for x in destination if x not in handle_2] 
                            destination = [x for x in destination if x not in handle_1] 
                            origin = [x for x in origin if x not in handle_2] 
                            origin = [x for x in origin if x not in handle_1] 
                        
                            if len(destination) > 0: 
                                alien_node = destination[0]
                            elif len(origin) > 0: 
                                alien_node = origin[0]
                            else: 
                                alien_node = -1
                        
                            if alien_node != -1: 
                                last_tooth = list([alien1[0], alien2[0], alien_node])
                                break_flag = 1 
                                counter = 1
                                    
            if counter == 1: 
    #**************************************PREPARE CONSTRAINTS************************************************************************************************************************************************
                # Handles 
                product1 =  pd.DataFrame(list(itertools.product(handle_1, handle_1)))
                swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                list1 = list(zip(swap1[0], swap1[1]))
                    
                product2 = pd.DataFrame(list(itertools.product(handle_2, handle_2)))
                swap2 = np.where(product2[0] < product2[1], [product2[0], product2[1]],[product2[1], product2[0]])
                list2 = list(zip(swap2[0], swap2[1]))
                    
                handle_for_constraint = list1 + list2
                handle_for_constraint = list(set(handle_for_constraint))
                handle_for_constraint = [x for x in handle_for_constraint if x[0] < x[1]]
                            
                # Teeth 
                product3 = pd.DataFrame(list(itertools.product(last_tooth, last_tooth)))
                swap3 = np.where(product3[0] < product3[1], [product3[0], product3[1]],[product3[1], product3[0]])
                list3 = list(zip(swap3[0], swap3[1]))
                list3 = list(set(list3))
                list3 = [x for x in list3 if x[0] < x[1]]
                
                product4 = pd.DataFrame(list(itertools.product(pairs_dom[0], pairs_dom[0])))
                swap4 = np.where(product4[0] < product4[1], [product4[0], product4[1]],[product4[1], product4[0]])
                list4 = list(zip(swap4[0], swap4[1]))
                
                product5 = pd.DataFrame(list(itertools.product(pairs_dom[1], pairs_dom[1])))
                swap5 = np.where(product5[0] < product5[1], [product5[0], product5[1]],[product5[1], product5[0]])
                list5 = list(zip(swap5[0], swap5[1]))
                
                dom_teeth = list(set(list4 + list5))
                dom_teeth = [x for x in dom_teeth if x[0] < x[1]]
                
                #add constraint
                #derived the constraint using formula with Delta notation
                #x(sum(E(Hi))) + x(sum(E(Tnon-intersecting))) + 2*x(sum(E(Tintersecting))) <= 8
                constraint = plp.LpConstraint(
                        e = plp.lpSum(self.X[o, d] for (o, d) in handle_for_constraint) + 
                            plp.lpSum(self.X[y, u] for (y, u) in list3) + 
                            2*plp.lpSum(self.X[k, l] for (k, l) in dom_teeth),
                            sense= plp.LpConstraintLE, #Less than and equal to
                            rhs = 8)
             
                self.add_constraint(constraint)

        if self.verbose == 1:    
            print('# of envelopes inserted:', counter)   
              
        return counter    


    # Action: PATH INEQUALITY INEQUALITY (2 HANDLES ONLY)
    def find_blossom_n_path(self, cycle_cutoff, teeth_cutoff_comb, teeth_cutoff_path):
    
        # first a comb should be found and added to the inequality and solved (as w. cook suggested)    
        handle_list, teeth_list, comb_flag = self.find_a_blossom(cycle_cutoff, teeth_cutoff_comb)
        counter = 0 
        connections_full = self.X_soln
        
        if comb_flag == 0: 
            if self.verbose == 1:
                print('no path here!')
            return counter, connections_full 
        
        # else increase counter since we added a blossom equality
        counter = counter + 1 
        # solve lp relaxation with added comb 
        # connections_full -> New X_soln
        self.solve_lp_relax()
        connections_full = self.X_soln
        
        handle_1 = np.unique(handle_list).tolist() #H_1 - fixed
        teeth_1 = teeth_list
        teeth_1 = [x for x in teeth_1 if x not in handle_1] #T_1 fixed
        n_of_teeth = len(teeth_1)
        
        if len(handle_list) == 0 or n_of_teeth <= 2 or len(teeth_1)%2==0: 
            if self.verbose == 1:
                print('no path here!')
            return counter, connections_full 
            
        else:
             pairs = [] # keeps first set of teeth to second teeth
             duplicate_list = list(np.unique(teeth_1)) #keeps duplicates
             betas = [] # keeps coefficients in constraints
             
             # detect following nodes after tooth
             for j in list(np.unique(teeth_1)): 
                 if j not in handle_1:
                     count_beta = 0 
                     # Teeth cutoff changed and reversed here!!
                     destination = connections_full[(connections_full['x_value'] <= teeth_cutoff_path) & (connections_full['origin'] == j)]['destination']
                     origin = connections_full[(connections_full['x_value'] <= teeth_cutoff_path) & (connections_full['destination'] == j)]['origin']
                   
                     # clean up, eliminate nodes in the intersection list
                     destination = [x for x in destination if x not in duplicate_list] 
                     origin = [x for x in origin if x not in duplicate_list] 
                                                 
                     if len(destination) > 0:
                         destination = destination[random.randint(0, len(destination)-1)]
                         pairs.append(tuple((j, destination)))
                         duplicate_list.append(destination)
                         count_beta = count_beta + 1
                    
                     elif len(origin) > 0:
                         origin = origin[random.randint(0, len(origin)-1)]
                         pairs.append(tuple((j, origin)))
                         duplicate_list.append(origin)
                         count_beta = count_beta + 1 
                            
                     # if there is no second tooth detected, then assign first tooth itself (no effect)
                     betas.append(count_beta)
                          
             # define handle_2 (H_2) include second nodes in teeth_1 if there are third nodes detected
             handle_2 = handle_1.copy() # handle 2 contains handle 1 
             
             for k in range(len(pairs)):
                 handle_2.append(pairs[k][0])
              
             # create outer lists for delta  
             betas = [2 if x==0 else x for x in betas]
             handle_2 = list(set(handle_2))
             
             if len(handle_1) >= len(handle_2): 
                 if self.verbose == 1:
                     print('no path here!')
                 return counter, connections_full  
             else:  
    
    #**************************************PREPARE CONSTRAINTS************************************************************************************************************************************************
                 
                 # handle pairs for the constraint
                 outer_handle_1 = [x for x in self.node_list if x not in handle_1]
                 outer_handle_2 = [x for x in self.node_list if x not in handle_2]
                    
                 product1 =  pd.DataFrame(list(itertools.product(handle_1, outer_handle_1)))
                 swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                 list1 = list(zip(swap1[0], swap1[1]))
                                        
                 product2 =  pd.DataFrame(list(itertools.product(handle_2, outer_handle_2)))
                 swap2 = np.where(product2[0] < product2[1], [product2[0], product2[1]],[product2[1], product2[0]])
                 list2 = list(zip(swap2[0], swap2[1]))
                    
                 handle_for_constraint = list1 + list2
                 
                 # create tooth for Delta notation
                 teeth_11 = pd.DataFrame(teeth_1)
                 pairs_11 = pd.DataFrame(pairs)
             
                 teeth_111 = pd.merge(teeth_11, pairs_11, how='left', left_on=0, right_on=0)
                 teeth_111 = pd.merge(teeth_111, pairs_11, how='left', left_on='1_x', right_on=0)
             
                 # populate all possible outbound edge options for each tooth (node in Tj -> node in V\Tj)
                 teeth_for_constraint = pd.DataFrame()
             
                 for l in range(n_of_teeth):
                    base = teeth_111.iloc[l].to_list()
                    base = list(np.unique(base))
                    base = [x for x in base if ~np.isnan(x)]
                    outer_base = [x for x in self.node_list if x not in base] 
                    # pairwise combinations
                    product1 =  pd.DataFrame(list(itertools.product(base, outer_base)))
                    swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                    list1 = pd.DataFrame({'origin':swap1[0],'destination':swap1[1]})
                    if len(base) >= 3:
                        list1['betas'] = 1
                    else:
                        list1['betas'] = 2
                    teeth_for_constraint = teeth_for_constraint.append(list1)
            
                    #add constraint
                    #derived the constraint using formula with Delta notation
                    #sum(Alfai * x(Delta(Hi))) + sum(Betaj * x(Delta(Tj))) >= (s+1) sum(Alfai) + 2*sum(Betaj)
                    #Alfai == 1 for both since the number handles is fixed (2)  
                    
                 constraint = plp.LpConstraint(
                        e = plp.lpSum(self.X[o, d] for (o, d) in handle_for_constraint) + 
                            plp.lpSum(self.X[int(teeth_for_constraint['origin'].iloc[t]), int(teeth_for_constraint['destination'].iloc[t])] * teeth_for_constraint['betas'].iloc[t] for t in range(len(teeth_for_constraint))),
                            sense= plp.LpConstraintGE, #Greater than and equal to
                            rhs = 2 * (n_of_teeth + 1) + 2 * np.sum(betas))
                    
                 self.add_constraint(constraint)
                 
                 if self.verbose == 1:
                     print('added a path (2) inequality!')
                     print('handle_1 in the path is:', handle_1)
                     print('handle_2 in the path is:', handle_2)
                     
                 counter = counter + 1 
                 handle_list = [] #empty lists
                 teeth_list = []
                 return counter, connections_full 
             
        if self.verbose == 1:    
            print('# of path constraints inserted:', counter) 
            
        return counter, connections_full     


    # Action: BIPARTITION INEQUALITY (1 intersecting tooth)
    def find_bipartition(self, cycle_cutoff, teeth_cutoff):
        
        # detect handles  
        # Reverse flag = RANDOMIZE
        selected_handles, graph = self.detect_handles(cycle_cutoff, 2)  
        # number of handle candidates
        n_handles = len(selected_handles)
        counter = 0 
        
        # if less than 2 handles found: stop
        if n_handles <= 1: 
            if self.verbose == 1:
                print('no potential bipartition here!')
            return counter
            
        else:    
            
            handle_pool = []
            break_flag = 0 
            intersecting_tooth = []
            
            # find connected edges between cycles 
            for k in range(n_handles):
                j = k+1
     
                if break_flag == 1: # break for loop if you find a handle k and some additional handles
                    break
                
                while k < j and j < n_handles: 
                # pairwise comparison of handles to see if there is a connection
                # first there should be no common nodes in handle candidates but edges
                    
                    itersection_flag = any(item in selected_handles[k] for item in selected_handles[j])
                    itersection_flag_2 = any(item in list(set(x for l in handle_pool for x in l)) for item in selected_handles[j])
                    
                    if itersection_flag == 0 and itersection_flag_2 == 0: 
                        
                        path_flag = nx.has_path(self.graph, source=selected_handles[k][0], target=selected_handles[j][0])
                    
                        if path_flag == 1:
                            path_check_list = []
                            
                            if len(handle_pool) == 0: 
                                path =  nx.shortest_path(self.graph, source=selected_handles[k][0], target=selected_handles[j][0])
                                both_handles = selected_handles[k] + selected_handles[j]
                                path_check = [x for x in path if x not in  both_handles] 
                            
                            else:
                                path_check_list = []
                                for p in range(len(handle_pool)): # compare handle j with all handles in the pool if all path check edges are 0 then add j to the pool 
                                    
                                    path =  nx.shortest_path(self.graph, source=handle_pool[p][0], target=selected_handles[j][0])
                                    both_handles = handle_pool[p] + selected_handles[j]
                                    path_check = [x for x in path if x not in  both_handles] 
                                    path_check_list.append(path_check)
                            
                                path_check = max(path_check_list) # since all must be 0 in order to be eligible to be added into pool 
                           
                            # handles must have one-to-one connections
                            if len(path_check) == 0:
                                #print('found a potential bipartition here!')
                                handle_pool.append(selected_handles[k])
                                handle_pool.append(selected_handles[j]) 
                                break_flag = 1 # but continue inner loop to catch more handles 
                                # eliminate nodes in handle1 and 2 except start and end nodes to find nodes in the intersecting tooth 
                                for t in path: 
                                    if t in selected_handles[k]: 
                                        start1 = t
                                    elif t in selected_handles[j]: 
                                        end1 = t 
                                        break
                        
                                start1 = path.index(start1)
                                start2 = path.index(end1)
                                path = path[start1:start2+1]
                                intersecting_tooth.append(path)
                                                        
                    j = j+1
                    if break_flag == 1: 
                        break 
            
            # take distinct path elements and handles 
                    
            # we should find additional even number of teeth for each handle
            len_handle_pool = len(handle_pool)  
            handle_pool = list(set([tuple(sorted(x)) for x in handle_pool]))  
            intersecting_tooth = list(set(x for l in intersecting_tooth for x in l))   
            n_in_intersecting_tooth = len(intersecting_tooth)
            
            if n_in_intersecting_tooth < 2: 
                if self.verbose == 1:
                    print('no potential bipartition here!')
                return counter
            
            else:
                
                # coefficient in the constraint will be t/(t-1)
                coefficient_for_intersecting_tooth = n_in_intersecting_tooth / (n_in_intersecting_tooth - 1)
    
                # find non-intersecting teeth
                intersection_list2 = list(set(intersecting_tooth + list(set(x for l in handle_pool for x in l))))
                tooth_pairs = []
                
                for k in range(len_handle_pool):
                    handle_picked = handle_pool[k]
                    
                    for n in handle_picked:
                        if n not in intersecting_tooth:
                        
                            destination = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['origin'] == n)]['destination']
                            origin = self.X_soln[(self.X_soln['x_value'] >= teeth_cutoff) & (self.X_soln['destination'] == n)]['origin']
                   
                            # clean up origin and destination (cannot be in intersection list)
                            destination = [x for x in destination if x not in intersection_list2] 
                            origin = [x for x in origin if x not in intersection_list2] 
                        
                            if len(destination) > 0: 
                                for p in range(len(destination)): 
                                    destination1 = destination[p]
                                    tooth_pairs.append(tuple((int(n), destination1)))
                                    intersection_list2.append(destination1)
                            
                            if len(origin) > 0:
                                for p in range(len(origin)):
                                    origin1 = origin[p]
                                    tooth_pairs.append(tuple((int(n), origin1)))
                                    intersection_list2.append(origin1)
                
                            intersection_list2.append(int(n))
    
                tooth_pairs = list(set(list(tooth_pairs)))   
                
                if len(tooth_pairs) > 0:
                    length_tooth_pairs = len(pd.DataFrame(tooth_pairs)[0].unique())  
                else: 
                    length_tooth_pairs = 0
                    
                # # of non-intersecting teeth must be even
                if length_tooth_pairs >= 4 and length_tooth_pairs%2 == 0:
    
    #**************************************PREPARE CONSTRAINTS************************************************************************************************************************************************ 
                    
                    # create pairs for handles for constraint
                    handle_for_constraint = []
                    
                    for l in range(len_handle_pool):
                        base = handle_pool[l]
                        base = list(np.unique(base))
                        base = [x for x in base if ~np.isnan(x)]
                        outer_base = [x for x in self.node_list if x not in base] 
                        # pairwise combinations
                        product1 =  pd.DataFrame(list(itertools.product(base, outer_base)))
                        swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                        list1 = list(zip(swap1[0], swap1[1]))
                        handle_for_constraint = handle_for_constraint + list1
    
                    # create intersecting tooth for handles for constraint
                    outer_intersecting_tooth = [x for x in self.node_list if x not in intersecting_tooth]
                    product2 =  pd.DataFrame(list(itertools.product(intersecting_tooth, outer_intersecting_tooth)))
                    swap2 = np.where(product2[0] < product2[1], [product2[0], product2[1]], [product2[1], product2[0]])
                    
                    intersecting_tooth_for_constraint = list(zip(swap2[0], swap2[1]))
                    
                    # other teeth 
                    teeth_for_constraint = []      
                    xd1 = pd.DataFrame(tooth_pairs)
                    xd1.sort_values(by=0)
                    xd1["h_rank"]=xd1.groupby([0]).ngroup()

                    for l in list(xd1["h_rank"].unique()):
                        base = list(np.unique(xd1[xd1["h_rank"] == l][[0,1]] ))
                        outer_base = [r for r in self.node_list if r not in base] 
                        # pairwise combinations
                        product1 =  pd.DataFrame(list(itertools.product(base, outer_base)))
                        swap1 = np.where(product1[0] < product1[1], [product1[0], product1[1]],[product1[1], product1[0]])
                        list1 = list(zip(swap1[0], swap1[1]))
                        teeth_for_constraint = teeth_for_constraint + list1 
                                        
                 # add constraint
                    #derived the constraint using formula with Delta
                    #sum(Delta(E(Hi))) + sum(Delta(E(Tj others))) + coeff * Delta(E(Tj intersection)) >= 2* (n_other_teeth + coeff) + sum over h (hi + 1): hi is number of teeth that handle intersects 
                    #in this simplified settings, the last component will be n_of_other_teeth + n_in_intersecting_tooth 
                    
                    constraint = plp.LpConstraint(
                        e = plp.lpSum(self.X[o, d] for (o, d) in handle_for_constraint) + 
                            plp.lpSum(self.X[k, l] for (k, l) in teeth_for_constraint) +
                            coefficient_for_intersecting_tooth * plp.lpSum(self.X[m, n] for (m, n) in intersecting_tooth_for_constraint),
                            sense= plp.LpConstraintGE, #Greater than and equal to
                            rhs = 2*(coefficient_for_intersecting_tooth + length_tooth_pairs) +  2 * len(handle_pool) + length_tooth_pairs)
                    
                    self.add_constraint(constraint)                
                    
                    if self.verbose == 1:
                        print('added a bipartition inequality with # of handles is:', n_in_intersecting_tooth)
                        print('intersection tooth is:', intersecting_tooth)
                    counter = counter + n_in_intersecting_tooth
                    return counter
       
        if self.verbose == 1:    
            print('# of bipartition constraints inserted:', counter) 
                           
        return counter   


    # Action: CROWN INEQUALITY (8 vertex sets only)
    def find_crown_8(self, connection_check_flag):
        
        crown_flag = 0 
        
        # Check if it's fully connected first, otherwise iterations needed!
        if connection_check_flag == 1: 
            fully_connected = nx.is_connected(self.graph)
            if fully_connected == 0: 
                if self.verbose == 1: 
                    print('no connection, no possible crown 8 here!')
                return crown_flag
            
        uncovered_nodes = []
        
        if self.n_nodes < 8:
            if self.verbose == 1:
                print('no possible crown 8 here!')
            return crown_flag
        
        # Initial tree
        xd1 = pd.DataFrame(self.coordinates_dict).T.sort_values([0,1]).reset_index(drop=False)
        mid_node = xd1.iloc[int(len(xd1)/2)]['index']
        tree1 = nx.dfs_tree(self.graph, mid_node)
        tree1_leaves = pd.DataFrame(list(tree1.edges)).reset_index(drop=True)
        
        tree1_leaves['split'] = 0
        for k in range(len(tree1_leaves)-1):
            check1 = tree1_leaves.iloc[k][1].astype(int)
            check2 = tree1_leaves.iloc[k+1][0].astype(int)
            if check1 - check2 != 0.0:
                tree1_leaves['split'].iloc[k+1] = 1
        
        tree1_leaves['split'] = np.cumsum(tree1_leaves['split'])
        
        if self.verbose == 1:
            print('number of initial splits:', tree1_leaves['split'].max() + 1)
        
        # if fully connected go with 1 loop only else;
        if len(tree1.nodes) != self.n_nodes: 
        
            uncovered_nodes = [item for item in self.node_list if item not in tree1.nodes]
            
            while len(uncovered_nodes) > 0: 
                # pick one node carelessly
                mid_node = uncovered_nodes[0]
                tree2 = nx.dfs_tree(self.graph, mid_node)
                tree2_leaves = pd.DataFrame(list(tree2.edges)).reset_index(drop=True) 
                tree2_leaves['split'] = 0
                
                for k in range(len(tree2_leaves)-1):
                    check1 = tree2_leaves.iloc[k][1].astype(int)
                    check2 = tree2_leaves.iloc[k+1][0].astype(int)
                    if check1 - check2 != 0.0:
                        tree2_leaves['split'].iloc[k+1] = 1
                
                tree2_leaves['split'] = np.cumsum(tree2_leaves['split'])
                if self.verbose == 1:
                    print('number of island splits:', tree2_leaves['split'].max() + 1)
        
                prev_max_split = tree1_leaves['split'].max()    
                tree2_leaves['split'] = tree2_leaves['split'] + prev_max_split + 1
                
                # append to the mainland and update uncovered list
                tree1_leaves = tree1_leaves.append(tree2_leaves).reset_index(drop=True)        
                uncovered_nodes = [item for item in uncovered_nodes if item not in tree2.nodes]
        
        tree1_leaves = tree1_leaves.reset_index(drop=False)    
            
        # If number of splits is equal to 8 do nothing 
        # If less than 8 split more 
        # Else merge until to reach 8 
        
        if  len(tree1_leaves['split'].unique()) == 8: 
            tree_finalmente = tree1_leaves
        
        elif len(tree1_leaves['split'].unique()) < 8: 
            # split until reach to 8 
            diff = 8 - len(tree1_leaves['split'].unique())
            count_cut = 0 
            break_flag = 0
            keep_index = []
            count_ = 0 
            while break_flag == 0 and count_ <= (self.n_nodes/2): # to avoid redundant computations
                
                range1 = len(tree1_leaves['split'].unique())      
                j = random.randint(0, range1-1)     
                partial = tree1_leaves[tree1_leaves['split'] == j]
                count_ = count_ + 1 
                if len(partial) > 2:
                    random1 = random.randint(1, len(partial)-1)
                    
                    if random1 not in keep_index:
                        keep_index.append(partial['index'].iloc[random1])
                        count_cut = count_cut + 1
                
                        if count_cut == diff:
                            break_flag = 1 
                                            
            # reset splits in tree finalmente
            tree_finalmente = tree1_leaves.copy()
            tree_finalmente['change_flag'] = tree_finalmente.apply(lambda x: 1 if x["index"] in keep_index else 0, axis=1)
            tree_finalmente['change_flag'] = np.cumsum(tree_finalmente['change_flag'])
            tree_finalmente['split'] = tree_finalmente['split'] + tree_finalmente['change_flag']
            
        elif len(tree1_leaves['split'].unique()) > 8 and len(tree1_leaves['split'].unique()) <= 32: # to avoid redundant complexity
            # merge until reach to 8 
            diff = len(tree1_leaves['split'].unique()) - 8
            count_merge = 0 
            break_flag = 0
            keep_merge = []
            count_ = 0
            while break_flag == 0 and count_ <= (self.n_nodes/2): # to avoid redundant computations
                count_ = count_ + 1 
                for j in range(len(tree1_leaves['split'].unique())-1): 
                    # merge consecutive splits if they have common nodes else no 
                    partial1 = tree1_leaves[tree1_leaves['split'] == j][0]
                    partial2 = tree1_leaves[tree1_leaves['split'] == j+1][0]
                    
                    partial2 = partial2.iloc[0]
                    partial1 = partial1.iloc[0]
                    
                    path_flag = nx.has_path(self.graph, source=int(partial1), target=int(partial2))
                    
                    if path_flag == 1:
                        keep_merge.append(tuple((j, j+1)))
                        count_merge = count_merge + 1
                
                    if count_merge == diff:
                        break_flag = 1 
                        break 
        
            # reset splits in tree finalmente
            tree_finalmente = tree1_leaves.copy()
            
            for p in range(len(keep_merge)):
                change_ = keep_merge[p][0]
                with_ = keep_merge[p][1]
                
                tree_finalmente['split'] = tree_finalmente.apply(lambda x: with_ if x["split"] == change_ else x["split"], axis=1)
           
            # rerank splits 
            tree_finalmente.sort_values(['split', 'index'])
        
            tree_finalmente["split"] = tree_finalmente['split'].rank(method="dense", ascending=True)
            tree_finalmente["split"] = tree_finalmente["split"] - 1 # to start from 0 

        else:
            return crown_flag     
            
        # FINAL STEP TO UPDATE BUNCHES AND FIND LOCATIONS
        iter_ = tree_finalmente['split'].unique()
        
        if len(iter_) == 8: # to avoid any error 
            covered_list = [] 
            bunches = [] 
            df_bunches = pd.DataFrame()
            
            for k in iter_: 
                partial = list(set(list(tree_finalmente[tree_finalmente['split'] == k][0]) + list(tree_finalmente[tree_finalmente['split'] == k][1])))       
                partial = [x for x in partial if x not in covered_list]
                bunches.append(partial)
                df_bunches_1 = pd.DataFrame(partial) 
                df_bunches_1['split'] = k 
                df_bunches = df_bunches.append(df_bunches_1)
                covered_list = list(set(covered_list + partial))
                
            if len(bunches) == 8: # to avoid any error     
                dist = pd.DataFrame(self.distance, index=[0]).T.reset_index(drop=False)
                dist1 = dist.copy()
                
                dist = dist.rename(columns={'level_0':'left_side', 'level_1':'right_side'})
                dist1 = dist1.rename(columns={'level_1':'left_side', 'level_0':'right_side'})
                dist1 = dist1[['left_side', 'right_side', 0]]
                dist = dist.append(dist1).reset_index(drop=True)
            
                df_bunches = df_bunches.reset_index(drop=True)
            
                # ASSIGN CIRCULAR LOCATION
                # GREEDY SEARCH
                # Calculate closest points to each other coordinates
                
                dist_split = pd.merge(dist, df_bunches, how='left', left_on='left_side', right_on=0)
                dist_split = pd.merge(dist_split, df_bunches, how='left', left_on='right_side', right_on=0).rename(columns={'0_x':'distance'})
                dist_split = dist_split[['split_x', 'split_y', 'distance']].groupby(['split_x', 'split_y'])['distance'].min().reset_index(drop=False)
                dist_split = dist_split[dist_split['split_x'] != dist_split['split_y']]
                dist_split['dist_rank'] = dist_split.groupby('split_x')['distance'].rank(method='first')
               
                circle_ = [0]
                origin_node = 0
                for k in range(len(iter_)-1): 
                    high_next_point = dist_split[dist_split['split_x'] == origin_node]
                    high_next_point = high_next_point[~high_next_point['split_y'].isin(circle_)]
                    high_next_point = high_next_point[high_next_point.dist_rank == high_next_point.dist_rank.min()]['split_y']
                    circle_.append(high_next_point.iloc[0])
                    origin_node = high_next_point.iloc[0]
         
                circle_ = pd.DataFrame(circle_).reset_index(drop=False)
                circle_['index'] =  circle_['index'] + 1
                keep_circle = pd.DataFrame()
                # assign costs in the circle
                for k in range(len(iter_)): 
                    for j in range(len(iter_)): 
                        index1 = circle_[circle_[0] == k]['index'].item()
                        index2 = circle_[circle_[0] == j]['index'].item()
                        distance = abs(index1-index2) if abs(index1-index2) <= 4 else 8-abs(index1-index2)
                
                        cost = 3 if distance == 1 else (4 if distance ==2 else (5 if distance==3 else 2))
                        obj1 = {'split1':k, 'split2':j, 'cost':cost}
                        keep_circle = keep_circle.append(pd.DataFrame(obj1, index=[0]))
                
                
                keep_circle = keep_circle[keep_circle['split1'] != keep_circle['split2']].reset_index(drop=True)    
                keep_circle = keep_circle[['split1', 'split2', 'cost']]       
                       
                # DEFINE THE CONSTRAINT 
                const1 = pd.merge(keep_circle, df_bunches, how='outer', left_on='split1', right_on='split')
                const1 = pd.merge(const1, df_bunches, how='outer', left_on='split2', right_on='split')
                const1 = const1[['0_x', '0_y', 'cost']]
                const1 = const1[const1['0_x'] < const1['0_y']].reset_index(drop=True)             
        
                constraint = plp.LpConstraint(
                                e = plp.lpSum(self.X[int(const1.iloc[k]['0_x']), int(const1.iloc[k]['0_y'])] * int(const1.iloc[k]['cost']) for k in range(len(const1))),
                                sense= plp.LpConstraintGE, #Greater than and equal to
                                rhs = 22)
                
                self.add_constraint(constraint)
                
                crown_flag = 1
                if self.verbose == 1: 
                    print('added a crown with 8 vertex sets here!')
                
                return crown_flag 
                    
        return crown_flag                    

    # Action: CROWN INEQUALITY (#N, DERIVED THE PATH FROM TREES)
    def find_crown_more(self, connection_check_flag):
        
        crown_flag = 0 
        if self.n_nodes < 8:
            if self.verbose == 1:
                print('no possible crown (more) here!')
            return crown_flag
        
        # Check if it's fully connected first, otherwise iterations needed!
        if connection_check_flag == 1: 
            fully_connected = nx.is_connected(self.graph)
            if fully_connected == 0: 
                if self.verbose == 1: 
                    print('no connection, no possible crown (more) here!')
                return crown_flag           
            
        K_ = int(np.floor(self.n_nodes/4))
        closest_4k = K_*4
        diff = self.n_nodes - closest_4k
        
        dist = pd.DataFrame(self.distance, index=[0]).T.reset_index(drop=False)
        dist1 = dist.copy()
        dist = dist.rename(columns={'level_0':'left_side', 'level_1':'right_side'})
        dist1 = dist1.rename(columns={'level_1':'left_side', 'level_0':'right_side'})
        dist1 = dist1[['left_side', 'right_side', 0]]
        dist = dist.append(dist1).reset_index(drop=True)
            
        dist['dist_rank'] = dist.groupby('left_side')[0].rank(method='first')
        
         # GREEDY SEARCH FOR CIRCLE
        circle_ = [0]
        origin_node = 0
        for k in range(self.n_nodes-1):
            high_next_point = dist[dist['left_side'] == origin_node]
            high_next_point = high_next_point[~high_next_point['right_side'].isin(circle_)]
            high_next_point = high_next_point[high_next_point.dist_rank == high_next_point.dist_rank.min()]['right_side']
            circle_.append(high_next_point.iloc[0])
            origin_node = high_next_point.iloc[0]
                        
        circle_ = pd.DataFrame(circle_).reset_index(drop=False).rename(columns={0:'vertex'})
        circle_['index'] =  circle_['index'] + 1
        if diff > 0: 
            circle_['index'].iloc[-diff:] = closest_4k
                          
        keep_circle = pd.DataFrame()
        # assign costs in the circle
        for k in range(self.n_nodes): 
            for j in range(self.n_nodes): 
                index1 = circle_[circle_['vertex'] == k]['index'].item()
                index2 = circle_[circle_['vertex'] == j]['index'].item()
                distance = abs(index1-index2) if abs(index1-index2) <= 2*K_ else closest_4k-abs(index1-index2)
                
                cost = closest_4k - 6 + distance if (distance <= 2*K_ - 1) else (2*K_ - 2 if distance == 2*K_ else 0)
                obj1 = {'split1':index1, 'split2':index2, 'vertex1':k, 'vertex2':j, 'cost':cost}
                keep_circle = keep_circle.append(pd.DataFrame(obj1, index=[0]))
                
                        
        keep_circle = keep_circle[keep_circle['vertex1'] != keep_circle['vertex2']].reset_index(drop=True)    
        keep_circle['cost'] = keep_circle.apply(lambda x: 0 if x.split1 == x.split2 else x.cost, axis=1)
        
                                   
        # DEFINE THE CONSTRAINT 
        const1 = keep_circle[['vertex1', 'vertex2', 'cost']]
        const1 = const1[const1['vertex1'] < const1['vertex2']].reset_index(drop=True)
        
        a0 = 12*K_*(K_- 1) - 2        
        
        constraint = plp.LpConstraint(
                                e = plp.lpSum(self.X[int(const1.iloc[k]['vertex1']), int(const1.iloc[k]['vertex2'])] * int(const1.iloc[k]['cost'])  for k in range(len(const1))),
                                sense= plp.LpConstraintGE, #Greater than and equal to
                                rhs = a0)
                
        self.add_constraint(constraint)
        
        crown_flag = K_
        if self.verbose == 1: 
            print('added a crown with vertex sets:', closest_4k)
                
        return crown_flag 

#%% Branching action 
def add_branch(problem, node, connection, regret, action): 
    
    if action == 0:
        node1 = node
        node2 = connection
        
    elif action == 1:
        node1 = node
        node2 = connection
        
    elif action == 2:
        node1 = node
        node2 = regret
        
    elif action == 3:
        node1 = node
        node2 = regret
        
    else:
        node1 = connection
        node2 = regret
    
    if node1 < node2: 
        node11 = node1
        node22 = node2
    else:
        node11 = node2
        node22 = node1
        
    if action%2==0:
        rhs1 = 0.0
    else:
        rhs1 = 1.0
            
    constraint = plp.LpConstraint(
                    e = problem.X[int(node11), int(node22)],
                        sense= plp.LpConstraintEQ, #Equal to
                        rhs = rhs1)
                
    problem.add_constraint(constraint)  
    
    return problem
                     
#%% Temporal actions
#Action 6: Set one of the variables with fractional solution to 1 (Top to middle)
def branch_1T(problem, branch_cutoff):
    
    branch_flag = 0 
    branch1 = problem.X_soln[(problem.X_soln['x_value'] < 0.99) & (problem.X_soln['x_value'] >= branch_cutoff)].reset_index(drop=True)
      
    if len(branch1) > 0: 
        #start from the top 
        branch1 = branch1[branch1['x_value'] == branch1['x_value'].max()]
        
        #RANDOMIZED
        branch1 = branch1.sample()
    
        #add constraint
        #variable_x_y == 1 
        constraint = plp.LpConstraint(
                    e = problem.X[int(branch1.origin), int(branch1.destination)],
                        sense= plp.LpConstraintEQ, #Equal to
                        rhs = 1)
                
        problem.add_constraint(constraint)                
        
        if problem.verbose == 1:
            print('set varible to 1T:', str(branch1.variable_object.item()))
        branch_flag = 1
    
    return branch_flag 

#%% Action 7: Set one of the variables with fractional solution to 0 (Bottom to middle)
def branch_0B(problem, branch_cutoff):
    
    branch_flag = 0 
    branch0 = problem.X_soln[(problem.X_soln['x_value'] > 0.01) & (problem.X_soln['x_value'] <= branch_cutoff)].reset_index(drop=True)   
    
    if len(branch0) > 0: 
        #start from the bottom 
        branch0 = branch0[branch0['x_value'] == branch0['x_value'].min()]
        
        #RANDOMIZED
        branch0 = branch0.sample()
    
        #add constraint
        #variable_x_y == 0 
        constraint = plp.LpConstraint(
                    e = problem.X[int(branch0.origin), int(branch0.destination)],
                        sense= plp.LpConstraintEQ, #Equal to
                        rhs = 0)
                
        problem.add_constraint(constraint)                
         
        if problem.verbose == 1:
            print('set varible to 0B:', str(branch0.variable_object.item()))
        branch_flag = 1
    
    return branch_flag     

#%% Action 8: Set one of the variables with fractional solution to 0 (Middle to bottom)
def branch_0M(problem, branch_cutoff):
    
    branch_flag = 0 
    branch0 = problem.X_soln[(problem.X_soln['x_value'] > 0.01) & (problem.X_soln['x_value'] <= branch_cutoff)].reset_index(drop=True)   
    
    if len(branch0) > 0: 
        #start from the middle 
        branch0 = branch0[branch0['x_value'] == branch0['x_value'].max()]
        
        #RANDOMIZED
        branch0 = branch0.sample()
    
        #add constraint
        #variable_x_y == 0 
        constraint = plp.LpConstraint(
                    e = problem.X[int(branch0.origin), int(branch0.destination)],
                        sense= plp.LpConstraintEQ, #Equal to
                        rhs = 0)
                
        problem.add_constraint(constraint)                
        if problem.verbose == 1:     
            print('set varible to 0M:', str(branch0.variable_object.item()))
        branch_flag = 1

    return branch_flag    

#%% Action 9: Set one of the variables with fractional solution to 1 (Middle to top)
def branch_1M(problem, branch_cutoff):
    
    branch_flag = 0 
    branch1 = problem.X_soln[(problem.X_soln['x_value'] < 0.99) & (problem.X_soln['x_value'] >= branch_cutoff)].reset_index(drop=True)
       
    if len(branch1) > 0: 
        #start from the top 
        branch1 = branch1[branch1['x_value'] == branch1['x_value'].min()]
        
        #RANDOMIZED
        branch1 = branch1.sample()
    
        #add constraint
        #variable_x_y == 1 
        constraint = plp.LpConstraint(
                    e = problem.X[int(branch1.origin), int(branch1.destination)],
                        sense= plp.LpConstraintEQ, #Equal to
                        rhs = 1)
                
        problem.add_constraint(constraint)                
        
        if problem.verbose == 1:
            print('set varible to 1M:', str(branch1.variable_object.item()))
        branch_flag = 1
 
    return branch_flag

#%%State handcrafted
# def state_handcraft(problem):
    
#     #input1: is connected or not 
#     node1 = int(nx.is_connected(problem.graph))
    
#     #input2: ratio of fractional edges 
#     #in perfect world num of edges == num of nodes
#     lenX = problem.n_nodes 
#     node2 = len(problem.X_soln[problem.X_soln.x_value < 1.0]) / lenX if lenX > 0 else 0 
    
#     #input3: ratio of 0.5 in fractional solutions 
#     node3 = len(problem.X_soln[problem.X_soln.x_value == 0.5])/(node2*lenX) if node2 > 0 else 0
    
#     #input4: ratio of <0.5 in fractional solutions 
#     node4 = len(problem.X_soln[problem.X_soln.x_value < 0.5])/(node2*lenX) if node2 > 0 else 0
    
#     #input5: ratio of > 0.5 in solutions 
#     node5 = len(problem.X_soln[(problem.X_soln.x_value > 0.5) & (problem.X_soln.x_value < 1.0)])/(node2*lenX) if node2 > 0 else 0
    
#     #input6: ratio of nodes having more than 2 connections 
#     node6 = problem.X_soln.groupby(['origin'])["destination"].count().reset_index(drop=False).rename(columns= {"origin":"node", "destination":"count"})
#     node61 = problem.X_soln.groupby(['destination'])["origin"].count().reset_index(drop=False).rename(columns= {"destination":"node", "origin":"count"})
#     node611 = node6.append(node61).groupby(['node'])["count"].sum().reset_index(drop=False)
#     node611["count"] = node611["count"].astype(float) 
    
#     node6 = len(node611[node611["count"] > 2.0]) / lenX if lenX > 0 else 0 
    
#     #node7: input max of number of connections 
#     node7 = max(node611["count"])
    
#     #node8: min value of edges 
#     node8 = min(problem.X_soln["x_value"])
    
#     #node9: max value of (fractional) edges 
#     node9 = max(problem.X_soln[problem.X_soln.x_value < 1.0]['x_value']) if node2 > 0 else 1
    
#     #update problem state
#     problem.state = pd.DataFrame([node1, node2, node3, node4, node5, node6, node7, node8, node9]).T
    
#     return problem

#%%
# def state_node2vec(problem, emb_dim): 
#     #Create node embeddings 
#     #DFS
#     node2vec = Node2Vec(problem.graph, dimensions=emb_dim, walk_length=20, num_walks=int(problem.n_nodes * 2), p=1, q=0.5, workers=2, quiet=True)
#     model = node2vec.fit()
    
#     vec = [] 
#     for node in problem.node_list:
#         vec_ = model.wv.get_vector(str(node))
#         vec.append(vec_)
    
#     vec1 = pd.DataFrame(vec).reset_index(drop=False)
    
#     if problem.verbose == 1:
#         plt.scatter(x=vec1[0], y=vec1[1])
#         plt.show()
    
#     #utilizing stats for node -> graph level
#     #update problem's state
#     problem.state = pd.DataFrame(list(vec1.sum())[1:] +  list(vec1.max())[1:] + list(vec1.min())[1:] + list(vec1.min())[1:] + list(vec1.std())[1:]).T
    
#     return problem

#%% Solve lp relaxation, take action, add constraint, update problem 
def solve_action_update(problem, action, initial_objective, old_state, cycle_cutoff):
    
    old_solution = problem.X_soln[['origin', 'destination', 'x_value']]
    old_objective = problem.objective_val
    problem_2 = deepcopy(problem) # to avoid adding redundant constraints keep previous set of constraints
    delta_obj = 0 
    connection_full = problem.X_soln
    
    if action==0: 
        # action 0
        # subtour elimination 
        count_const = problem.subtour_elimn()
    
    elif action==1:
        # action 1
        # multiple blossom inequalities
        count_const = problem.find_multi_blossoms(cycle_cutoff, 0.1)
 
    elif action==2:
        # action 2 
        # complex comb inequalities (|H| > 3)        
        count_const = problem.find_adv_combs(cycle_cutoff, 0.1)
 
    elif action==3:
        # action 3
        # clique tree inequality (with 2 handles)
        count_const = problem.find_clique_tree_2(cycle_cutoff, 0.1)
    
    elif action==4:
        # action 4 
        # blossom + consecutive path inequalities
        count_const, connection_full = problem.find_blossom_n_path(cycle_cutoff, 0.1, 1)
    
    elif action==5:
        # action 5
        # bipartition inequality (with 1 intersecting tooth)
        count_const = problem.find_bipartition(cycle_cutoff, 0.1)
    
    elif action==6:
        # action 6
        count_const = problem.find_basic_envelope(cycle_cutoff, 0.1)
    
    elif action==7: 
        # action 7
        # Crown 8
        count_const = problem.find_crown_8(0)
        
    elif action==8: 
        # action 8
        # Crown multi
        count_const = problem.find_crown_more(0)
    
    elif action==9:
        count_const = branch_1T(problem, 0.5)
    
    elif action==10:
        count_const = branch_0B(problem, 0.5)
        
    elif action==11:
        count_const = branch_0M(problem, 0.5)
        
    elif action==12:
        count_const = branch_1M(problem, 0.5)
    
    # solve lp relaxation 
    problem.solve_lp_relax()
    
    # objective delta
    new_solution = problem.X_soln[['origin', 'destination', 'x_value']]
    new_objective = problem.objective_val
    
    change_flag = 1 - int(old_solution.equals(new_solution))

    # update the graph, complete_flag and state if there is a change 
    if change_flag == 1: 
        problem.graph = problem.create_graph()
        graph = problem.graph.copy()
        problem.check_if_complete()
        delta_obj = float(-1 if new_objective - old_objective is None else abs(new_objective - old_objective))
        
        graph.add_edges_from(zip(graph.nodes(), graph.nodes()))
        graph = dgl.DGLGraph(graph)
        
        new_state = graph # nx graph -> dgl graph
    else:
        new_state = old_state # already a dgl graph 
  
    if action == 4: #fix 
        change_flag_2 = 1 - int(connection_full.equals(new_solution))
        if change_flag_2 == 0: 
            if count_const == 2:
                count_const = 1
                    
    return problem, delta_obj, change_flag, new_state, count_const 

#%% Solve lp relaxation, take action, add constraint, update problem 
def solve_action_check(problem, action, cycle_cutoff):
    
    old_solution = problem.X_soln[['origin', 'destination', 'x_value']]
    old_objective = problem.objective_val
    problem_2 = deepcopy(problem) #to avoid adding redundant constraints keep previous set of constraints
    connection_full = problem.X_soln
    
    if action==0: 
        # action 0
        # subtour elimination 
        count_const = problem.subtour_elimn()
    
    elif action==1:
        # action 1
        # multiple blossom inequalities
        count_const = problem.find_multi_blossoms(cycle_cutoff, 0.1)
 
    elif action==2:
        # action 2 
        # complex comb inequalities (|H| > 3)        
        count_const = problem.find_adv_combs(cycle_cutoff, 0.1)
 
    elif action==3:
        # action 3
        # clique tree inequality (with 2 handles)
        count_const = problem.find_clique_tree_2(cycle_cutoff, 0.1)
    
    elif action==4:
        # action 4 
        # blossom + consecutive path inequalities
        count_const, connection_full = problem.find_blossom_n_path(cycle_cutoff, 0.1, 1)
    
    elif action==5:
        # action 5
        # bipartition inequality (with 1 intersecting tooth)
        count_const = problem.find_bipartition(cycle_cutoff, 0.1)
    
    elif action==6:
        # action 6
        count_const = problem.find_basic_envelope(cycle_cutoff, 0.1)
    
    elif action==7: 
        # action 7
        # Crown 8
        count_const = problem.find_crown_8(0)
    
    elif action==8: 
        # action 8
        # Crown multi
        count_const = problem.find_crown_more(0)
  
    #solve lp relaxation 
    problem.solve_lp_relax()
    
    #objective delta
    new_solution = problem.X_soln[['origin', 'destination', 'x_value']]
    change_flag = 1 - int(old_solution.equals(new_solution))
    new_objective = problem.objective_val
    
    delta_obj = float(-1 if new_objective - old_objective is None else abs(new_objective - old_objective))
    
    if action == 4: #fix 
      change_flag_2 = 1 - int(connection_full.equals(new_solution))
      if change_flag_2 == 0: 
          if count_const == 2:
              count_const = 1
    
    return problem_2, change_flag, count_const, delta_obj 

#%%

print('Valid inequalities version: 9')
print('No hard restrictions for teeth assignments')