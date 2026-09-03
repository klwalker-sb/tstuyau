#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-02:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_mosaic2.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_mosaic2.%N.%a.%j.err # STDERR
#SBATCH --job-name="mosaic2"

TILES='1to10'
YR=2024
MOD="base4Poly6_bal200mix4_${YR}_filtsmh-sp-tsfilt"
TESTING=True   ## This is for previous mosaic. If True, will find inputs in the scratch dir
SAVEME=True
###################################################################
### Project settings
PREFIX='CELPy'
MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
SCRATCH_DIR="/home/scratch-cel"
PROJECT="paraguay_lc"
###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

### activate the virtual environment
conda activate tstuyau_pipe

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="
grids:''
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT}
buffer:$BUF
scratch_dir:${SCRATCH_DIR}/${PROJECT}
classify:chunked_map:True
classify:chunk_tile:$TILES
classify:test:$TESTING
classify:prefix:$PREFIX
classify:name:$MOD
classify:save_mosaic:$SAVEME"

tuyau mosaic --config-updates $CONFIG_UPDATES

conda deactivate
