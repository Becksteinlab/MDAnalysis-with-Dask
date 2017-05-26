#!/usr/bin/env python
import warnings; warnings.simplefilter('ignore')
import pandas as pd
import dask
from dask.delayed import delayed
from dask import multiprocessing
from dask.multiprocessing import get
from dask.distributed import Client, progress
import MDAnalysis as mda
import numpy as np
import math
import time, glob, os, sys
from MDAnalysis import Writer 
from MDAnalysis.analysis import psa
import datreant.core as dtr

Scheduler_IP = sys.argv[1]
#SLURM_JOBID = sys.argv[2]
print (Scheduler_IP)
print (type (Scheduler_IP))
#print (Client(Scheduler_IP))
c = Client(Scheduler_IP)
print (c.get_versions(check=True))
print(sys.path)

def sort_filenames(traj):
    traj_sorted = []
    import re
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    for infile in sorted(traj, key=numericalSort):
        traj_sorted.append(infile)
        
    return traj_sorted


# In[5]:

def PSA_hausdorff(block, Bl_size0, Bl_size1, i_bl, j_bl):
    subD = np.zeros([Bl_size0, Bl_size1])
#     D = pd.DataFrame(index=range(Len), columns=range(Len))
#     D = D.fillna(0)
#     print (i_bl, j_bl)
    
    for i in range(Bl_size0):
        for j in range(Bl_size1):   
            traj1 = block[i]
            traj2 = block[j+Bl_size0]
            #--------------------------------------
            P = traj1
            Q = traj2
            #--------------------------------------
            subD[i,j] = psa.hausdorff(P, Q)
#           psa.discrete_frechet(P, Q)
#             df = pd.DataFrame(subD, index=None, columns=None) 
#             D = insert_df(i_bl, j_bl, Bl_size0, Bl_size1, df, D)
      
    return subD


# In[6]:

def insert_df(il, jl, w0, w1, df, D):
    df.index = range(il,il+w0)
    df.columns = range(jl,jl+w1)
    D.iloc[il:il+w0, jl:jl+w1] = df
#     if il != jl:
#         D.iloc[jl:jl+w1,il:il+w0] = df.transpose()
        
    return D


# In[7]:

def PSA_Parallel(trj_list, traj, N_blocks, N_procs):
    Block_size = int(len(traj)/N_blocks)
    print (len(traj), Block_size)
    re = len(traj) % Block_size
    
    if re == 0: 
        Bl_size_0 = N_procs*[Block_size]
        Bl_size_1 = N_procs*[Block_size]
    else:
        Bl_size_0 = (N_procs-1)*[Block_size] + [re] 
        Bl_size_1 = (N_procs-1)*[Block_size] + [re]
        
    N = 0
    D_list = []
    for ii, i in enumerate(range(0, len(traj), Block_size)):
        for jj, j in enumerate(range(0, len(traj), Block_size)):
            print ('i {} and j {}'.format(i,j))
            #-------------------------------------------------------------  
            block = trj_list[i:i+Bl_size_0[ii]] + trj_list[j:j+Bl_size_1[jj]]
            D_list.append(delayed(PSA_hausdorff)(block, Bl_size_0[ii], Bl_size_1[jj], i, j))
            
    return D_list


# In[8]:

def Reduction_Process(res_stacked):
    df = res_stacked[0]
    for i in range(len(res_stacked)-1):
        df = df.add(res_stacked[i+1])
    return df


# In[11]:

with open('data.txt', mode='w') as file:
    N_procs = 64
    N_blocks = int(np.sqrt(N_procs))
    s = dtr.Treant('PSA-data')
    # s.trees.names
    # traj1 = s.glob('trj_aa_*.npz.npy').abspaths
    traj2 = s.glob('trj_aa_*.npz.npy').abspaths
    # traj1 = sort_filenames(traj1)
    # traj2 = sort_filenames(traj2)
    traj = traj2[:128]
    #-------------------------------------
    size = ['small','medium','large']  #'small','medium'
    for k in size:
        c.restart()
        trj_list = []
        start1 = time.time()
        if k=='small':
            trj_list = [np.load(tr) for tr in traj]
        elif k=='medium':
            trj_list = [np.hstack((np.load(tr),np.load(tr))) for tr in traj]
        else:
            trj_list = [np.hstack((np.load(tr),np.load(tr),np.load(tr),np.load(tr))) for tr in traj]


        output = PSA_Parallel(trj_list, traj, N_blocks, N_procs)
        
        output = delayed(output)
#        start1 = time.time()
        res_stacked = output.compute(get=c.get)
        tot_time = time.time()-start1

#         start2 = time.time()
#         D = Reduction_Process(res_stacked)
#         reduc_time = time.time()-start2
        reduc_time = 0
        
        file.write("{} {} {}\n".format(k, tot_time, reduc_time))  
        file.flush()

