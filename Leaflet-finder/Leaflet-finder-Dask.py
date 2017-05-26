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
import MDAnalysis as mda
import numpy as np
import math
import time, glob, os, sys
from MDAnalysis import Writer 
import scipy
import networkx as nx

Scheduler_IP = sys.argv[1]
#SLURM_JOBID = sys.argv[2]
print (Scheduler_IP)
print (type (Scheduler_IP))
#print (Client(Scheduler_IP))
c = Client(Scheduler_IP)
print (c.get_versions(check=True))
print(sys.path)

def Leaflet_finder(block, atoms, cutoff, len_chunks, block_id=None):
    id_0 = block_id[0]
    id_1 = block_id[1]

#     print(len_chunks, len(atoms))
    block[:,:] = distance_array(atoms[id_0*len_chunks:(id_0+1)*len_chunks], atoms[id_1*len_chunks:(id_1+1)*len_chunks])
    adj = np.array(block < cutoff)
    S = scipy.sparse.dok_matrix((len(atoms), len(atoms)))
    S[id_0*len_chunks:(id_0+1)*len_chunks, id_1*len_chunks:(id_1+1)*len_chunks] = adj
    l = np.array({i: item for i, item 
                  in enumerate(sorted(nx.connected_components(nx.Graph(S>0))))},
                 dtype=np.object).reshape(1,1)
    
    return l

input_data = ['atom_pos_132K.npy','atom_pos_262K.npy','atom_pos_524K.npy']

with open('data.txt', mode='w') as file:
    N_procs = 64

    for traj in input_data:
        c.restart()

        start1 = time.time()
        atom = np.load(traj)
        len_chunks = int(len(atom)/np.sqrt(N_procs))
        dist_matrix = da.zeros((len(atom),len(atom)), dtype=float, chunks=(len_chunks,len_chunks))        

        start2 = time.time()

	a = dist_matrix.map_blocks(Leaflet_finder, atom, 15.0, len_chunks, dtype=np.object).compute()

        start3 = time.time()
         
	connected = []
	for i in a.flatten():
   	   l = []
   	   [l.append(i[key]) for key in i.keys()]
   	   connected.append(l)
        
        results = [] # Keep only connected components
	for leaflet in connected:
    	   for lng in range(len(leaflet)):
               if(len(leaflet[lng])>1):results.append(leaflet[lng])

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

        start4 = time.time()

        file.write("{} {} {} \n".format(start2-start1, start3-start2, start4-start3))
#        file.write(results)
        file.flush()

