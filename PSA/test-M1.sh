#!/usr/bin/env bash
# #!/bin/bash

#SBATCH -J parallel_RMSD_XTC             # Job name
#SBATCH -o slurm.%j.out                  # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err                  # STDERR (%j = JobId)
#SBATCH --partition=compute
# #SBATCH --constraint="large_scratch"
#SBATCH --nodes=43                        # Total number of nodes requested (16 cores/node). You may delete this line if wanted
#SBATCH --ntasks-per-node=24             # Total number of mpi tasks requested
#SBATCH --export=ALL
#SBATCH -t 48:00:00                      # wall time (D-HH:MM)
#SBATCH --mail-user=mkhoshle@asu.edu     # email address
#SBATCH --mail-type=all                  # type of mail to send

#The next line is required if the user has more than one project
# #SBATCH -A A-yourproject # <-- Allocation name to charge job against

export WKDIR=`pwd`

source /home/mkhoshle/miniconda2/envs/py35/bin/activate py35

cd $WKDIR
echo $WKDIR

my_file=PSA_Parallel
#my_file=test

export SCHEDULER=`hostname`
echo SCHEDULER: $SCHEDULER
dask-scheduler --port=8786 &
sleep 5

hostnodes=`scontrol show hostnames $SLURM_NODELIST`
echo $hostnodes

for host in $hostnodes; do
    echo "Working on $host ...."
    ssh $host $WKDIR/workers.sh $SCHEDULER &
    sleep 1
done

echo "====-get to work-===="

echo "Launching $my_file.py"
/home/mkhoshle/miniconda2/envs/py35/bin/python $my_file.py $SCHEDULER:8786

