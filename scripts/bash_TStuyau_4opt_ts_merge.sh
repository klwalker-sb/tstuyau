#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o TStuyau_tsmerge.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_tsmerge.%N.%a.%j.err # STDERR
#SBATCH --job-name="TStyuau_tsm"
#SBATCH --array=950
###################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 4000))

## Set permissions on output files
umask 002

###################################################################
### Settables:

SIS=("gcvi" "kndvi" "ndmi" "nbr")
START_PAD="2018-01-01"
START="2018-03-01"
END_PAD="2024-09-01"
END="2024-07-01"

METHOD='STAC'
IMGTYPE='LS2'
RES=10.0
PROCSEQ='mu.br.cga'
NCHUNKS=512
SKIP_INTERVAL=7
SKIP_YEARS=1
ROVERWRITE="False"
SM_CHUNKS=512
PREFILL_GAPS="False"
DTS_MAX_WIN=61
DTS_MIN_WIN=15
PREFILL_YEARS=2
DTS_t=5
PREFILL_MAX_DAYS=80
PREFILL_WMAX=75
PREFILL_WMIN=21
RMOUT="True"
SMOOTH_METH="wh"

###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel"
BACKUP_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"

###################################################################
### activate the virtual environment
conda activate venv.tstuyau_pipe

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

STEP="reconstruct"
MERGE="True"
## COMP variables control naming of ts and brdf folders -- simpler naming if false
COMPSEN=False
COMPRES=False
COMPPROC=False

#############################################
# Turn off NumPy parallelism and rely on dask
#############################################
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# This should be sufficient for OpenBlas and MKL
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
################################################

# Do for each index:
for SI in "${SIS[@]}"
do

CONFIG_UPDATES="grids:[${GRIDS}] 
res:${RES}
image_type:${IMGTYPE}
procseq:${PROCSEQ} 
main_path:"${MAIN_DIR}/${PROJECT}/stac/grid"
backup_path:"${BACKUP_DIR}/${PROJECT}/stac/grid"
dlMehod:${METHOD}
num_workers:${SLURM_CPUS_ON_NODE} 
io:n_chunks:${NCHUNKS}
compare_imgtype:${COMPSEN}
compare_res:${COMPRES}
compare_procseq:${COMPPROC}
reconstruct:merge_ts:${MERGE}
reconstruct:start_pad:${START_PAD} 
reconstruct:end_pad:${END_PAD} reconstruct:start:${START} reconstruct:end:${END} 
reconstruct:skip_interval:${SKIP_INTERVAL} reconstruct:skip_years:${SKIP_YEARS}
reconstruct:si:${SI} reconstruct:overwrite:${ROVERWRITE} reconstruct:chunks:${SM_CHUNKS}
reconstruct:smooth_kwargs:max_window:${DTS_MAX_WIN}
reconstruct:smooth_kwargs:min_window:${DTS_MIN_WIN}
reconstruct:smooth_kwargs:prefill_max_years:${PREFILL_YEARS}
reconstruct:smooth_kwargs:prefill_gaps:${PREFILL_GAPS} reconstruct:smooth_kwargs:t:${DTS_t}
reconstruct:smooth_kwargs:prefill_max_days:${PREFILL_MAX_DAYS}
reconstruct:smooth_kwargs:prefill_wmax:${PREFILL_WMAX}
reconstruct:smooth_kwargs:prefill_wmax:${PREFILL_WMIN}
reconstruct:smooth_kwargs:remove_outliers:${RMOUT}
reconstruct:smooth_kwargs:smooth_method:${SMOOTH_METH}
clean:remove_items:${REMOVE_ITEMS}"

tuyau $STEP --config-updates $CONFIG_UPDATES

done
conda deactivate
