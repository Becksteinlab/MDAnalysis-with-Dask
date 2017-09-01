#!/usr/bin/env python
import MDAnalysis as mda
import networkx as nx
from MDAnalysis.analysis.distances import distance_array
import dask
import dask.array as da
from dask.delayed import delayed
from dask.multiprocessing import get
from dask.distributed import Client, progress
import numpy as np
import math
import time, glob, os, sys
import scipy
from scipy import spatial
from scipy.spatial.distance import cdist
from distributed.diagnostics.plugin import SchedulerPlugin
from MDAnalysis import Writer

def submitCustomProfiler(profname,dask_scheduler):
    prof = MyProfiler(profname)
    dask_scheduler.add_plugin(prof)

def removeCustomProfiler(dask_scheduler):
    dask_scheduler.remove_plugin(dask_scheduler.plugins[-1])

class MyProfiler(SchedulerPlugin):
    def __init__(self,profname):
        self.profile = profname

    def transition(self,key,start,finish,*args,**kargs):
        if start == 'processing' and finish == 'memory':
            with open(self.profile,'a') as ProFile:
                ProFile.write('{}\n'.format([key,start,finish,kargs['worker'],kargs['thread'],kargs['startstops']]))


Scheduler_IP = sys.argv[1]
#SLURM_JOBID = sys.argv[2]
print (Scheduler_IP)
print (type (Scheduler_IP))
c = Client(Scheduler_IP)

def Leaflet_finder(traj, i, j, ii, jj, len_chunks, cutoff):
    g1 = np.load(os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),traj))), mmap_mode='r')[i:i+len_chunks]
    g2 = np.load(os.path.abspath(os.path.normpath(os.path.join(os.getcwd(),traj))), mmap_mode='r')[j:j+len_chunks]

    block = np.zeros((len(g1),len(g2)),dtype=float)
    block[:,:] = cdist(g1, g2) <= cutoff
    adj_list = np.where(block[:,:] == True)        
    adj_list = np.vstack(adj_list)

    adj_list[0] = adj_list[0]+ii*len_chunks
    adj_list[1] = adj_list[1]+jj*len_chunks

    if adj_list.shape[1] == 0:
        adj_list=np.zeros((2,1))

    graph = nx.Graph()
    edges = [(adj_list[0,k],adj_list[1,k]) for k in range(0,adj_list.shape[1])]
    graph.add_edges_from(edges)
    leaflet = sorted(nx.connected_components(graph), key=len, reverse=True)
    l_connected = [] # Keep only connected components
    for lng in range(len(leaflet)):
        l_connected.append(leaflet[lng])

    return list(l_connected)

input_data = ['atom_pos_132K.npy','atom_pos_262K.npy','atom_pos_524K.npy']

with open('data.txt', mode='w') as file:
    N_procs = 1024
    
    for traj in input_data:
        for k in range(3): 
            c.restart()
            c.run_on_scheduler(submitCustomProfiler,'leaflet_{}_{}.txt'.format(traj,jj))
    	    
            start1 = time.time()
            atom = np.load(traj)
            len_chunks = int(len(atom)/np.sqrt(N_procs))

            start2 = time.time()
    	    l = []
            for ii, i in enumerate(range(0, len(atom), len_chunks)):
                len_atom = len(atom)
                for jj, j in enumerate(range(0, len(atom), len_chunks)):
                    l.append(delayed(Leaflet_finder)(traj, i, j, ii, jj, len_chunks, 15))

	        l = delayed(l)         

            start3 = time.time()

            l_f = l.compute(get=c.get)

            start4 = time.time()

            results = []
            [[results.append(j) for j in i] for i in l_f]

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

            file.write("{} {} {} {} {} {}\n".format(traj, k, t_init, t_comp1, t_comp2, t_conn))
            file.flush()
            c.run_on_scheduler(removeCustomProfiler)
