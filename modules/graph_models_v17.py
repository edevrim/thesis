#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:18:08 2020

@author: salihemredevrim
"""

# Graph Neural Networks and Node Embeddings i
import torch as th
import torch.nn as nn
import torch.nn.functional as F
#from nodevectors import Node2Vec #uses CSR matrices
import numpy as np
import pandas as pd

import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.softmax import edge_softmax
import dgl.function as fn
import seaborn as sns
import matplotlib.pyplot as plt
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
#from google.colab import files
#%%# https://docs.dgl.ai/tutorials/basics/4_batch.html
def collect_graphs(graphs_and_labels):
    # graphs_and_labels: pairs of graph and label
    graphs, labels = map(list, zip(*graphs_and_labels))
    batched_graph = dgl.batch(graphs)
    return batched_graph, th.tensor(labels)

#%% Graph Convolution Networks (varied on number of convolutions, mean and max)
# 4 VARIABLES - 2 CONV LAYERS - MEAN POOLING - SIGMOID
class GCN2_MN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN2_MN, self).__init__()
        # two convolution layers for node representation
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        # two dense layers for q value calculation with dropout
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))     
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_emb = F.relu(self.layer1(graph_emb))
        h_emb = self.drop_layer1(h_emb)
        pred = F.sigmoid(self.layer2(h_emb))
        
        return pred

#%%# 4 VARIABLES - 2 CONV LAYERS - MAX POOLING - SIGMOID
class GCN2_MX(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN2_MX, self).__init__()
        # two convolution layers for node representation
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        # two dense layers for q value calculation with dropout
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))     
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.max_nodes(graph, 'h')
        h_emb = F.relu(self.layer1(graph_emb))
        h_emb = self.drop_layer1(h_emb)
        pred = F.sigmoid(self.layer2(h_emb))
        
        return pred

#%%# 4 VARIABLES - 2 CONV LAYERS - SUM POOLING - SIGMOID
class GCN2_SU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN2_SU, self).__init__()
        # two convolution layers for node representation
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        # two dense layers for q value calculation with dropout
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))     
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.sum_nodes(graph, 'h')
        h_emb = F.relu(self.layer1(graph_emb))
        h_emb = self.drop_layer1(h_emb)
        pred = F.sigmoid(self.layer2(h_emb))
        
        return pred
    
#%%# 4 VARIABLES - 4 CONV LAYERS - MEAN POOLING - SIGMOID
class GCN3_MN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN3_MN, self).__init__()
        # two convolution layers for node representation
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)

        # two dense layers for q value calculation with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer1 = nn.Linear(hidden_dim, output_dim)
        
                
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))   
        h_co = F.relu(self.conv3(graph, h_co))   
        h_co = F.relu(self.conv4(graph, h_co))  
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_emb = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.layer1(h_emb))
        
        return pred

#%%# 4 VARIABLES - 5 CONV LAYERS - MEAN POOLING - 2 DENSE LAYERS AND SIGMOID
class GCN5_MN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN5_MN, self).__init__()
        # two convolution layers for node representation
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)
        self.hidden_dim_2 = int(hidden_dim/2)

        # two dense layers for q value calculation with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer1 = nn.Linear(hidden_dim, self.hidden_dim_2)
        self.drop_layer2 = nn.Dropout(p=drop_out)
        self.layer2 = nn.Linear(self.hidden_dim_2, output_dim)
        
                
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))   
        h_co = F.relu(self.conv3(graph, h_co))   
        h_co = F.relu(self.conv4(graph, h_co))  
        h_co = F.relu(self.conv5(graph, h_co))  
        
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_emb = self.drop_layer1(graph_emb)
        h_emb = F.relu(self.layer1(h_emb))
        h_emb2 = self.drop_layer1(h_emb)
        pred = F.sigmoid(self.layer2(h_emb2))
        
        return pred, graph_emb, graph.ndata['h'] 

#%%# 4 VARIABLES - 5 CONV LAYERS - MEAN POOLING - 2 DENSE LAYERS AND SIGMOID WITH NEGATIVES
class GCN5_MN_NEG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN5_MN_NEG, self).__init__()
        # two convolution layers for node representation
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)
        self.hidden_dim_2 = int(hidden_dim/2)

        # two dense layers for q value calculation with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer1 = nn.Linear(hidden_dim, self.hidden_dim_2)
        self.drop_layer2 = nn.Dropout(p=drop_out)
        self.layer2 = nn.Linear(self.hidden_dim_2, output_dim)
        
                
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()
        
        h11 = -1*h1
        h22 = -1*h2
        h33 = -1*h3
        h44 = -1*h4

        h_ = th.cat((h1, h2, h3, h4, h11, h22, h33, h44), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))   
        h_co = F.relu(self.conv3(graph, h_co))   
        h_co = F.relu(self.conv4(graph, h_co))  
        h_co = F.relu(self.conv5(graph, h_co))  
        
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_emb = self.drop_layer1(graph_emb)
        h_emb = F.relu(self.layer1(h_emb))
        h_emb2 = self.drop_layer1(h_emb)
        pred = F.sigmoid(self.layer2(h_emb2))
        
        return pred, graph_emb, graph.ndata['h'] 
    
#%%# 4 VARIABLES - 5 CONV LAYERS - MEAN POOLING - 2 DENSE LAYERS AND SIGMOID WITH COORDINATES
class GCN5_MN_COORD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN5_MN_COORD, self).__init__()
        # two convolution layers for node representation
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)
        self.hidden_dim_2 = int(hidden_dim/2)

        # two dense layers for q value calculation with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer1 = nn.Linear(hidden_dim, self.hidden_dim_2)
        self.drop_layer2 = nn.Dropout(p=drop_out)
        self.layer2 = nn.Linear(self.hidden_dim_2, output_dim)
        
                
    def forward(self, graph, coord):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = coord["x"].view(-1, 1).float()
        h3 = coord["y"].view(-1, 1).float()

        h_ = th.cat((h1, h2, h3), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))   
        h_co = F.relu(self.conv3(graph, h_co))   
        h_co = F.relu(self.conv4(graph, h_co))  
        h_co = F.relu(self.conv5(graph, h_co))  
        
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_emb = self.drop_layer1(graph_emb)
        h_emb = F.relu(self.layer1(h_emb))
        h_emb2 = self.drop_layer1(h_emb)
        pred = F.sigmoid(self.layer2(h_emb2))
        
        return pred, graph_emb, graph.ndata['h'] 
#%%# 4 VARIABLES - 5 CONV LAYERS - MEAN POOLING - 2 DENSE LAYERS AND SIGMOID
class GCN5_MN_TANH(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN5_MN_TANH, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)
        self.hidden_dim_2 = int(hidden_dim/2)

        # two dense layers for q value calculation with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer1 = nn.Linear(hidden_dim, self.hidden_dim_2)
        self.layer2 = nn.Linear(self.hidden_dim_2, output_dim)
                        
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (tanh)
        h_co = F.tanh(self.conv1(graph, h_))
        h_co = F.tanh(self.conv2(graph, h_co))   
        h_co = F.tanh(self.conv3(graph, h_co))   
        h_co = F.tanh(self.conv4(graph, h_co)) 
        h_co = F.tanh(self.conv5(graph, h_co)) 
        
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_emb = F.leaky_relu(self.layer1(graph_emb), negative_slope=0.01, inplace=False)
        h_emb2 = self.drop_layer1(h_emb)
        pred = F.sigmoid(self.layer2(h_emb2))
        
        return pred, graph_emb, graph.ndata['h']     
    
#%%# 4 VARIABLES - 5 CONV LAYERS - MEAN POOLING - 2 DENSE LAYERS AND SIGMOID
class GCN5_MN_SIG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN5_MN_SIG, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)

        # two dense layers for q value calculation with dropout
        self.layer1 = nn.Linear(hidden_dim, output_dim)
        self.drop_layer1 = nn.Dropout(p=drop_out)
                        
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (sigmoid)
        h_co = F.sigmoid(self.conv1(graph, h_))
        h_co = F.sigmoid(self.conv2(graph, h_co))   
        h_co = F.sigmoid(self.conv3(graph, h_co))   
        h_co = F.sigmoid(self.conv4(graph, h_co)) 
        h_co = F.sigmoid(self.conv5(graph, h_co)) 
        
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_emb2 = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.layer1(h_emb2))
        
        return pred, graph_emb, graph.ndata['h']   
    
#%%# 4 VARIABLES - 4 CONV LAYERS - MEAN POOLING - (SOFTMAX COULD BE APPLIED LATER)
class GCN_GC_MN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN_GC_MN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)

        # two dense layers for q value calculation with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer1 = nn.Linear(hidden_dim, output_dim)
        
                
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))   
        h_co = F.relu(self.conv3(graph, h_co))   
        h_co = F.relu(self.conv4(graph, h_co))  
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_emb = self.drop_layer1(graph_emb)
        pred = self.layer1(h_emb)
        
        return pred

#%%# 4 VARIABLES - 5 CONV LAYERS - MAX POOLING - 2 DENSE LAYERS AND SIGMOID
class GCN5_MX(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN5_MX, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)
        self.hidden_dim_2 = int(hidden_dim/2)

        # two dense layers for q value calculation with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer1 = nn.Linear(hidden_dim, self.hidden_dim_2)
        self.drop_layer2 = nn.Dropout(p=drop_out)
        self.layer2 = nn.Linear(self.hidden_dim_2, output_dim)
        
                
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))   
        h_co = F.relu(self.conv3(graph, h_co))   
        h_co = F.relu(self.conv4(graph, h_co))  
        h_co = F.relu(self.conv5(graph, h_co))  
        
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.max_nodes(graph, 'h')
        h_emb = self.drop_layer1(graph_emb)
        h_emb = F.relu(self.layer1(h_emb))
        h_emb2 = self.drop_layer1(h_emb)
        pred = F.sigmoid(self.layer2(h_emb2))
        
        return pred, graph_emb, graph.ndata['h'] 
    
#%%# 4 VARIABLES - 4 CONV LAYERS - MAX POOLING - SIGMOID
class GCN3_MX(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN3_MX, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)

        # two dense layers for q value calculation with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer1 = nn.Linear(hidden_dim, output_dim)
                
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))   
        h_co = F.relu(self.conv3(graph, h_co)) 
        h_co = F.relu(self.conv4(graph, h_co)) 
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.max_nodes(graph, 'h')
        h_emb = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.layer1(h_emb))
        
        return pred

#%%# 4 VARIABLES - 2 CONV LAYERS - SUM POOLING - SIGMOID
class GCN3_SU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN3_SU, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)

        # two dense layers for q value calculation with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer1 = nn.Linear(hidden_dim, output_dim)
                
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))   
        h_co = F.relu(self.conv3(graph, h_co))   
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.sum_nodes(graph, 'h')
        h_emb = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.layer1(h_emb))
        
        return pred

#%%# 4 VARIABLES - 5 CONV LAYERS - SUM POOLING - 2 DENSE LAYERS AND SIGMOID
class GCN5_SU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out):
        super(GCN5_SU, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.conv4 = GraphConv(hidden_dim, hidden_dim)
        self.conv5 = GraphConv(hidden_dim, hidden_dim)
        self.hidden_dim_2 = int(hidden_dim/2)

        # two dense layers for q value calculation with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer1 = nn.Linear(hidden_dim, self.hidden_dim_2)
        self.drop_layer2 = nn.Dropout(p=drop_out)
        self.layer2 = nn.Linear(self.hidden_dim_2, output_dim)
        
                
    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        # and binary variable if node has fractional value 
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        # perform graph convolution and activation function (relu)
        h_co = F.relu(self.conv1(graph, h_))
        h_co = F.relu(self.conv2(graph, h_co))   
        h_co = F.relu(self.conv3(graph, h_co))   
        h_co = F.relu(self.conv4(graph, h_co))  
        h_co = F.relu(self.conv5(graph, h_co))  
        
        graph.ndata['h'] = h_co
        # calculate graph representation
        graph_emb = dgl.sum_nodes(graph, 'h')
        h_emb = self.drop_layer1(graph_emb)
        h_emb = F.relu(self.layer1(h_emb))
        h_emb2 = self.drop_layer1(h_emb)
        pred = F.sigmoid(self.layer2(h_emb2))
        
        return pred, graph_emb, graph.ndata['h']     
    
#%% Graph Attention Network 
#https://discuss.dgl.ai/t/gat-for-graph-classification/366/3
class GATLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 alpha=0.2,
                 agg_activation=F.relu):
        super(GATLayer, self).__init__()

        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.attn_l = nn.Parameter(th.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_r = nn.Parameter(th.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        self.agg_activation=agg_activation
        
        #th.nn.init.xavier_normal_(self.fc.weight)

    def clean_data(self):
        ndata_names = ['ft', 'a1', 'a2']
        edata_names = ['a_drop']
        for name in ndata_names:
            self.g.ndata.pop(name)
        for name in edata_names:
            self.g.edata.pop(name)

    def forward(self, feat, bg):
        # prepare, inputs are of shape V x F, V the number of nodes, F the dim of input features
        self.g = bg
        h = self.feat_drop(feat)
        # V x K x F', K number of heads, F' dim of transformed features
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))
        head_ft = ft.transpose(0, 1)                              # K x V x F'
        a1 = th.bmm(head_ft, self.attn_l).transpose(0, 1)      # V x K x 1
        a2 = th.bmm(head_ft, self.attn_r).transpose(0, 1)      # V x K x 1
        self.g.ndata.update({'ft' : ft, 'a1' : a1, 'a2' : a2})
        # 1. compute edge attention
        self.g.apply_edges(self.edge_attention)
        # 2. compute softmax in two parts: exp(x - max(x)) and sum(exp(x - max(x)))
        self.edge_softmax()
        # 2. compute the aggregated node features scaled by the dropped,
        # unnormalized attention values.
        self.g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
        # 3. apply normalizer
        ret = self.g.ndata['ft']                                  # V x K x F'
        ret = ret.flatten(1)

        if self.agg_activation is not None:
            ret = self.agg_activation(ret)

        # Clean ndata and edata
        self.clean_data()

        return ret

    def edge_attention(self, edges):
        # an edge UDF to compute un-normalized attention values from src and dst
        a = self.activation(edges.src['a1'] + edges.dst['a2'])
        return {'a' : a}

    def edge_softmax(self):
        attention = self.softmax(self.g, self.g.edata.pop('a'))
        # Dropout attention scores and save them
        self.g.edata['a_drop'] = self.attn_drop(attention)
        
#%%# 4 VARIABLES - 2 GAT LAYERS - MEAN POOLING - SIGMOID
class GAT_MN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, drop_out):
        super(GAT_MN, self).__init__()

        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
            
        # 1 dense layer with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.classify = nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        h_at = self.gat1(h_, graph)
        h_at = self.gat2(h_at, graph)
        
        graph.ndata['h'] = h_at

        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_at2 = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.classify(h_at2))

        return pred       

#%%# 4 VARIABLES - 3 GAT LAYERS - MEAN POOLING - SIGMOID
class GAT3_MN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, drop_out):
        super(GAT3_MN, self).__init__()

        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
        self.gat3 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
            
        # 1 dense layer with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.classify = nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        h_at = self.gat1(h_, graph)
        h_at = self.gat2(h_at, graph)
        h_at = self.gat3(h_at, graph)
        
        graph.ndata['h'] = h_at

        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_at2 = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.classify(h_at2))

        return pred, graph_emb, graph.ndata['h']       

#%%# 4 VARIABLES - 3 GAT LAYERS - MEAN POOLING - SIGMOID WITH NEGATIVES
class GAT3_MN_NEG(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, drop_out):
        super(GAT3_MN_NEG, self).__init__()

        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
        self.gat3 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
            
        # 1 dense layer with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.classify = nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h11 = -1*h1
        h22 = -1*h2
        h33 = -1*h3
        h44 = -1*h4

        h_ = th.cat((h1, h2, h3, h4, h11, h22, h33, h44), 1)
        
        h_at = self.gat1(h_, graph)
        h_at = self.gat2(h_at, graph)
        h_at = self.gat3(h_at, graph)
        
        graph.ndata['h'] = h_at

        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_at2 = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.classify(h_at2))

        return pred, graph_emb, graph.ndata['h']    
    
#%%# 4 VARIABLES - 3 GAT LAYERS - MEAN POOLING - SIGMOID WITH COORDINATES
class GAT3_MN_COORD(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, drop_out):
        super(GAT3_MN_COORD, self).__init__()

        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
        self.gat3 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
            
        # 1 dense layer with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.classify = nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, graph, coord):
        # graph: a dgl graph
        # use node degree as the initial node feature
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = coord["x"].view(-1, 1).float()
        h3 = coord["y"].view(-1, 1).float()

        h_ = th.cat((h1, h2, h3), 1)
        
        h_at = self.gat1(h_, graph)
        h_at = self.gat2(h_at, graph)
        h_at = self.gat3(h_at, graph)
        
        graph.ndata['h'] = h_at

        # calculate graph representation
        graph_emb = dgl.mean_nodes(graph, 'h')
        h_at2 = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.classify(h_at2))

        return pred, graph_emb, graph.ndata['h']    
    
#%%# 4 VARIABLES - 2 GAT LAYERS - MAX POOLING - SIGMOID
class GAT_MX(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, drop_out):
        super(GAT_MX, self).__init__()

        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
            
        # 1 dense layer with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.classify = nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        h_at = self.gat1(h_, graph)
        h_at = self.gat2(h_at, graph)
        
        graph.ndata['h'] = h_at

        # calculate graph representation
        graph_emb = dgl.max_nodes(graph, 'h')
        h_at2 = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.classify(h_at2))

        return pred       

#%%# 4 VARIABLES - 3 GAT LAYERS - MAX POOLING - SIGMOID
class GAT3_MX(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, drop_out):
        super(GAT3_MX, self).__init__()

        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
        self.gat3 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
            
        # 1 dense layer with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.classify = nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        h_at = self.gat1(h_, graph)
        h_at = self.gat2(h_at, graph)
        h_at = self.gat3(h_at, graph)
        
        graph.ndata['h'] = h_at

        # calculate graph representation
        graph_emb = dgl.max_nodes(graph, 'h')
        h_at2 = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.classify(h_at2))

        return pred, graph_emb, graph.ndata['h'] 
      
#%%# 4 VARIABLES - 2 GAT LAYERS - SUM POOLING - SIGMOID
class GAT_SU(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, drop_out):
        super(GAT_SU, self).__init__()

        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
            
        # 1 dense layer with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.classify = nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        h_at = self.gat1(h_, graph)
        h_at = self.gat2(h_at, graph)
        
        graph.ndata['h'] = h_at

        # calculate graph representation
        graph_emb = dgl.sum_nodes(graph, 'h')
        h_at2 = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.classify(h_at2))

        return pred       

#%%# 4 VARIABLES - 3 GAT LAYERS - SUM POOLING - SIGMOID
class GAT3_SU(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes, drop_out):
        super(GAT3_SU, self).__init__()

        self.gat1 = GATLayer(in_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
        self.gat3 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads)
            
        # 1 dense layer with dropout
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.classify = nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, graph):
        # graph: a dgl graph
        # use node degree as the initial node feature
        h = graph.in_degrees()     
        h1 = h.view(-1, 1).float()
        h2 = (h - 3) > 0
        h2 = h2.view(-1, 1).float()
        h3 = 3/h1
        h4 = (h - 4) > 0
        h4 = h4.view(-1, 1).float()

        h_ = th.cat((h1, h2, h3, h4), 1)
        
        h_at = self.gat1(h_, graph)
        h_at = self.gat2(h_at, graph)
        h_at = self.gat3(h_at, graph)
        
        graph.ndata['h'] = h_at

        # calculate graph representation
        graph_emb = dgl.sum_nodes(graph, 'h')
        h_at2 = self.drop_layer1(graph_emb)
        pred = F.sigmoid(self.classify(h_at2))

        return pred, graph_emb, graph.ndata['h'] 
       
#%% Save model
def save_model(model, MODEL_TAG):
    th.save(model.state_dict(), MODEL_TAG+".pth.tar")
    #files.download(MODEL_TAG+".pth.tar") 

#%% Node2vec with DFS and BFS for node-level embeddings
def node2vec_dfs(graph1, emb_dim, verbose1): 
    #Create node embeddings 
    #DFS
    #https://github.com/VHRanger/nodevectors/blob/master/nodevectors/embedders.py  
    n2v = Node2Vec(
            n_components = emb_dim,
            walklen=int(graph1.number_of_nodes()/4),
            epochs=int(graph1.number_of_nodes() * 2.5),
            return_weight=1,
            neighbor_weight=2)

    n2v.fit(graph1.to_networkx(), verbose=verbose1)

    # Make embedding result matrix
    nodes = graph1.nodes().tolist()
    embeds = np.empty((len(nodes), emb_dim))
    for i in nodes:
        embeds[i] = n2v.predict(i)
    
    embeds = pd.DataFrame(embeds)
    g_sum = embeds.sum().reset_index(drop=False) 
    g_mean = embeds.mean().reset_index(drop=False) 
    g_max = embeds.max().reset_index(drop=False) 
    g_min = embeds.min().reset_index(drop=False) 
    g_std = embeds.std().reset_index(drop=False)
            
    return embeds, g_sum, g_mean, g_max, g_min, g_std

#%% BFS
def node2vec_bfs(graph1, emb_dim, verbose1): 
    #Create node embeddings 
    #DFS
    #https://github.com/VHRanger/nodevectors/blob/master/nodevectors/embedders.py  
    n2v = Node2Vec(
            n_components = emb_dim,
            walklen=int(graph1.number_of_nodes()/4),
            epochs=int(graph1.number_of_nodes() * 2.5),
            return_weight=1,
            neighbor_weight=0.5)

    n2v.fit(graph1.to_networkx(), verbose=verbose1)

    # Make embedding result matrix
    nodes = graph1.nodes().tolist()
    embeds = np.empty((len(nodes), emb_dim))
    for i in nodes:
        embeds[i] = n2v.predict(i)

    embeds = pd.DataFrame(embeds)  
    g_sum = embeds.sum().reset_index(drop=False) 
    g_mean = embeds.mean().reset_index(drop=False) 
    g_max = embeds.max().reset_index(drop=False) 
    g_min = embeds.min().reset_index(drop=False) 
    g_std = embeds.std().reset_index(drop=False)
            
    return embeds, g_sum, g_mean, g_max, g_min, g_std
#%% NN for embeddings 
class NNN_(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out):
        super(NNN_, self).__init__()
        
        
        self.hidden_dim_1 = int(input_dim/1)
        self.hidden_dim_2 = int(input_dim/2)
        self.hidden_dim_3 = int(input_dim/4)

        # three dense layers for q value calculation with dropout
        self.layer1 = nn.Linear(input_dim, self.hidden_dim_1) # input is the size of embeddings
        self.drop_layer1 = nn.Dropout(p=drop_out)
        self.layer2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.drop_layer2 = nn.Dropout(p=drop_out)
        self.layer3 = nn.Linear(self.hidden_dim_2, self.hidden_dim_3)
        self.drop_layer3 = nn.Dropout(p=drop_out)
        self.layer4 = nn.Linear(self.hidden_dim_3, output_dim)     
 
        
    def forward(self, embeds):
        h = F.leaky_relu(self.layer1(embeds), negative_slope=0.01, inplace=False)
        h = self.drop_layer1(h)
        h_2 = F.leaky_relu(self.layer2(h), negative_slope=0.01, inplace=False)
        h_2 = self.drop_layer2(h_2)
        h_3 = F.leaky_relu(self.layer3(h_2), negative_slope=0.01, inplace=False)
        h_4 = self.drop_layer3(h_3)
        h_5 = self.layer4(h_4)
        
        return h_5

#%%
# https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
def draw_roc_curve(testy, yhat):
  # calculate roc curves
  fpr, tpr, thresholds = roc_curve(testy, yhat)
  # plot the roc curve for the model
  plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
  plt.plot(fpr, tpr, marker='.', label='Logistic')
  # axis labels
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.legend()
  # show the plot
  plt.show()

  # threshold 
  # get the best threshold
  J = tpr - fpr
  ix = np.argmax(J)
  best_thresh = thresholds[ix]
  print('According to ROC Curve - Best Threshold=%f' % (best_thresh))
  
  # roc score
  roc_score = roc_auc_score(testy, yhat)

  return best_thresh, roc_score

#%%
# https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
def draw_precision_recall_curve(testy, yhat):
  
  precision, recall, thresholds = precision_recall_curve(testy, yhat)
  # convert to f score
  fscore = (2 * precision * recall) / (precision + recall)
  # locate the index of the largest f score
  ix = np.argmax(fscore)
  print('According to precision recall curve: Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
  # plot the roc curve for the model
  no_skill = len(testy[testy==1]) / len(testy)
  plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
  plt.plot(recall, precision, marker='.', label='Logistic')
  plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
  # axis labels
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.legend()
  # show the plot
  plt.show()

  return fscore[ix], thresholds[ix]

#%%
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')
 
def find_threshold(target, pred):
  # define thresholds
  thresholds = np.arange(0, 1, 0.01)
  # evaluate each threshold
  scores = [f1_score(target, to_labels(pred, t)) for t in thresholds]
  # get best threshold
  ix = np.argmax(scores)
  print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
  return thresholds[ix]

