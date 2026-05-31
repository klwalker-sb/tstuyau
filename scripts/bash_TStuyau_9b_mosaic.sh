#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_mosaic.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_mosaic.%N.%a.%j.err # STDERR
#SBATCH --job-name="mosaic"
###################################################################

###################################################################
### Settables
### note CELLS can be a list of gridcells to mosaic or the path to a .csv file with a list of cells.
###     if a .csv file, the file basename will be start the basename of the output mosaic.

#CELLS="/home/downspout-cel/paraguay_lc/mosaics/lists/CELpy_DistrictSamp.csv"
CELLS="/home/downspout-cel/paraguay_lc/mosaics/lists/CELPy_Tile2.csv"
MOD="base4Poly6_bal200mix6_21_LC32_RF_2021"
TESTING=False   ## if TESTING=True, will send output to scratch directory
SAVEME=True
COMP_DIR="input_dir"  ## 'input_dir' | 'backup' | 'tmp'  (which drive to look for the comp folder with the input images)
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
SCRATCH_DIR="/home/scratch-cel"
PROJECT="paraguay_lc"

###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

### activate the virtual environment
conda activate venv.tstuyau_pipe

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:$CELLS
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT}/stac/grid
scratch_dir:${SCRATCH_DIR}/${PROJECT}
classify:comp_dir:$COMP_DIR
classify:test:$TESTING
classify:name:$MOD
classify:save_mosaic:$SAVEME"

tuyau mosaic --config-updates $CONFIG_UPDATES

conda deactivate
