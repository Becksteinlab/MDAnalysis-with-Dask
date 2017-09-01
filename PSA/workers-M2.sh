#!/usr/bin/env bash

source /home/mkhoshle/miniconda2/envs/py35/bin/activate py35

echo "CHECKING dask-worker: $(which dask-worker)"
echo "CHECKING python: $(python --version 2>&1)"

echo "SCHEDULER: $1"
dask-worker --nprocs 24 --nthreads 1 ${1}:8786
