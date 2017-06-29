#!/usr/bin/env python
import MDAnalysis as mda
import networkx as nx
from MDAnalysis.analysis.distances import distance_array
import dask
import dask.array as da
from dask.delayed import delayed
from dask.multiprocessing import get
from dask.distributed import Client, progress
import MDAnalysis as mda
import numpy as np
import time, glob, os, sys
import scipy
import networkx as nx
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
#print (Client(Scheduler_IP))
c = Client(Scheduler_IP)

def Leaflet_finder(block, traj, cutoff, len_atom, len_chunks, block_id=None):
    id_0 = block_id[0]
    id_1 = block_id[1]

    block[:,:] = cdist(np.load(traj, mmap_mode='r')[id_0*len_chunks:(id_0+1)*len_chunks], np.load(traj, mmap_mode='r')[id_1*len_chunks:(id_1+1)*len_chunks]) <= cutoff 
    adj_list = np.where(block[:,:] == True)        
    adj_list = np.vstack(adj_list)

    adj_list[0] = adj_list[0]+id_0*len_chunks
    adj_list[1] = adj_list[1]+id_1*len_chunks

    if adj_list.shape[1] == 0:
        adj_list=np.zeros((2,1))
        
    graph = nx.Graph()
    edges = [(adj_list[0,k],adj_list[1,k]) for k in range(0,adj_list.shape[1])]
    graph.add_edges_from(edges)
    l = np.array({i: item for i, item in enumerate(sorted(nx.connected_components(graph)))}, dtype=np.object).reshape(1,1)
    
    return l

input_data = ['atom_pos_132K.npy','atom_pos_262K.npy','atom_pos_524K.npy']

with open('data.txt', mode='w') as file:
    N_procs = 1024
    for traj in input_data:
        for k in range(3):
            c.restart()
            c.run_on_scheduler(submitCustomProfiler,'leaflet_{}_{}.txt'.format(traj,jj))

            start1 = time.time()
            atom = np.load(traj)
            len_atom = len(atom)      
            len_chunks = int(len(atom)/np.sqrt(N_procs))
            dist_matrix = da.zeros((len(atom),len(atom)), dtype=float, chunks=(len_chunks,len_chunks))        

            start2 = time.time()

            a = dist_matrix.map_blocks(Leaflet_finder, traj, 15.0, len_atom, len_chunks, dtype=np.object).compute()

            start3 = time.time()
         
            connected = []
            for i in a.flatten():
                l = []
                [l.append(i[key]) for key in i.keys()]
                connected.append(l)

            results = [] # Keep only connected components
            for leaflet in connected:
                for lng in range(len(leaflet)):
                    results.append(leaflet[lng])

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


            file.write("{} {} {} {} {} \n".format(traj, k, start2-start1, start3-start2, start4-start3))
            file.flush()
            c.run_on_scheduler(removeCustomProfiler)
