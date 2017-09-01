#!/usr/bin/env python
import warnings; warnings.simplefilter('ignore')
import mdsynthesis as mds
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
from mdsynthesis import Sim
from MDAnalysis.analysis import psa
import datreant.core as dtr
import matplotlib
import matplotlib.pyplot as plt


Scheduler_IP = sys.argv[1]
print (Scheduler_IP)
print (type (Scheduler_IP))
c = Client(Scheduler_IP)
print (mda.__version__)

# In[2]:

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


# In[8]:

def PSA_hausdorff(Len, Bl_size0, Bl_size1, i_bl, j_bl, k, traj):
    subD = np.zeros([Bl_size0, Bl_size1])
    block = traj[i_bl:i_bl+Bl_size0] + traj[j_bl:j_bl+Bl_size1]

    trj_list = []
    #-----------------------------------------------
    if k=='small':
       trj_list = [np.load(tr) for tr in block]
    elif k=='medium':
       trj_list = [np.hstack((np.load(tr),np.load(tr))) for tr in block]
    else:
       trj_list = [np.hstack((np.load(tr),np.load(tr),np.load(tr),np.load(tr))) for tr in block]
    #------------------------------------------------
    for i in range(Bl_size0):
        for j in range(Bl_size1):   
            #--------------------------------------
            traj1 = trj_list[i]
            traj2 = trj_list[j+Bl_size0]
            #--------------------------------------
            P = traj1
            Q = traj2
            #--------------------------------------
            subD[i,j] = psa.hausdorff(P, Q)

    return subD


# In[10]:

def PSA_Parallel(traj, N_blocks, N_procs, k):
    Block_size = int(len(traj)/N_blocks)
    print (Block_size)
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
            D_list.append(delayed(PSA_hausdorff)(len(traj), Bl_size_0[ii], Bl_size_1[jj], i, j, k, traj))
            jj += 1
      
    return D_list


# In[12]:

with open('data.txt', mode='w') as file:
    N_procs = 16
    N_blocks = int(np.sqrt(N_procs))
    s = dtr.Treant('../PSA-data')
    traj2 = s.glob('trj_aa_*.npz.npy').abspaths
    traj = traj2[:128]
    print(len(traj), N_blocks, N_procs)
    #-------------------------------------
    size = ['small','medium','large']
    for k in size:
        start1 = time.time()
        output = PSA_Parallel(traj, N_blocks, N_procs, k)
        output = delayed(output)
        res_stacked = output.compute(get=c.get)
        tot_time = time.time()-start1
        
        print(tot_time, reduc_time)
        file.write("{} {} {}\n".format(k, N_procs, tot_time))
        file.flush()
