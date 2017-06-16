#!/usr/bin/env python
import MDAnalysis as mda
import networkx as nx
from MDAnalysis.analysis.distances import distance_array

import mdsynthesis as mds
import pandas as pd
import dask
import dask.array as da
from dask.delayed import delayed
from dask import multiprocessing
from dask.multiprocessing import get
from dask.distributed import Client, progress
import numpy as np
import math
import time, glob, os, sys
from MDAnalysis import Writer 
import scipy
from scipy import spatial
from scipy.spatial.distance import cdist

Scheduler_IP = sys.argv[1]
#SLURM_JOBID = sys.argv[2]
print (Scheduler_IP)
print (type (Scheduler_IP))
#print (Client(Scheduler_IP))
c = Client(Scheduler_IP)
print (c.get_versions(check=True))
print(sys.path)

def Leaflet_finder(traj, i, j,len_atom, cutoff):
    atom = np.load(traj)
    g1 = atom[i:i+len_chunks]
    g2 = atom[j:j+len_chunks]
    block = np.zeros((len(g1),len(g2)),dtype=float)
    block[:,:] = cdist(g1, g2) <= cutoff
    S = scipy.sparse.dok_matrix((len_atom, len_atom))
    S[i:i+len_chunks, j:j+len_chunks] = block
    leaflet = sorted(nx.connected_components(nx.Graph(S>0)), key=len, reverse=True)
    l_connected = [] # Keep only connected components
    for lng in range(len(leaflet)):
        if(len(leaflet[lng])>1):l_connected.append(leaflet[lng])

    return list(l_connected)

input_data = ['atom_pos_132K.npy','atom_pos_262K.npy','atom_pos_524K.npy']

with open('data.txt', mode='w') as file:
    N_procs = 1024
    
    for traj in input_data:
    	c.restart()

    	start1 = time.time()
        atom = np.load(traj)
        len_chunks = int(len(atom)/np.sqrt(N_procs))

        start2 = time.time()
    	l = []
    	for ii, i in enumerate(range(0, len(atom), len_chunks)):
    	    jj = ii 
    	    len_atom = len(atom)
    	    for j in range(0, len(atom), len_chunks):
                l.append(delayed(Leaflet_finder)(traj, i, j, len_atom, 15))       

        l = delayed(l)         

        start3 = time.time()

        l_f = l.compute(get=c.get)

        start4 = time.time()

        connected = []
        [connected.append(i) for i in l_f]         

#	results = []
#	map(results.extend, connected)
        results = connected

        chk = True
    	j = 0
     	while chk == True and len(results)>1:
    	      chk = False
    	      item1 = results[j]
              ind = []
   	          for i, item2 in enumerate(results):
       	          if set(item1).intersection(set(item2)) and i!=j:
                     results[j] = list(set(list(item1) + list(item2)))
                     ind.append(i) 
                     item1 = results[j]
                     chk = True 
   
              results_new = [i for jj, i in enumerate(results) if jj not in ind]
    	      results = results_new
              j = j+1
 
        start5 = time.time()

        t_comp1 = start3-start2
    	t_comp2 = start4-start3
        t_init = start2-start1 
        t_conn = start5-start4

        file.write("{} {} {} {}\n".format(t_init, t_comp1, t_comp2, t_conn))
        file.flush()
