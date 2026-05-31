#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_status.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_status.%N.%a.%j.err # STDERR
#SBATCH --job-name="status"
###################################################################

###################################################################
### Settables
### note CELLS can be a list of gridcells to mosaic or the path to a .csv file with a list of cells.
###     if a .csv file, the file basename will be start the basename of the output mosaic.

STARTSTOP='[2024200,2025100]'
SIS=("gcvi","kndvi","nbr","ndmi")
OUT='/home/images'
GRID_FILE="/home/sandbox-cel/LUCinLA_grid_8858.gpkg"
#CONTEXT="/home/sandbox-cel/paraguay_lc/vector/EParaguayOutline.shp"
#CONTEXT="/home/sandbox-cel/paraguay_lc/vector/Paraguay_OutlineAll.shp"
CONTEXT=("/home/sandbox-cel/paraguay_lc/vector/EParaguayOutline.shp","/home/sandbox-cel/paraguay_lc/vector/Paraguay_OutlineAll.shp")
## Zomme options: none, 'completed', or a grid cell # 
ZOOM='3750'
FILTER='PARAGUAY'
FILTERCOL='CEL_projec'
#ZOOM='completed'
OFFSET=200000
KEEPEXIST=True
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
SCRATCH_DIR="/home/scratch-cel"
PROJECT="paraguay_lc"
GRID=
###################################################################
### activate the virtual environment
conda activate venv.tuyau

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

CONFIG_UPDATES="grid_file=$GRID_FILE
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT}/stac/grids
status:zoom:$ZOOM
status:period:$STARTSTOP
status:out_path:$OUT
status:offset:$OFFSET
status:filter:$FILTER
status:filter_column:$FILTERCOL
status:other_layers:$CONTEXT
status:use_existing:$KEEPEXIST
feature_model:spec_indices:$SIS
log_level:INFO"

tuyau status --config-updates $CONFIG_UPDATES

conda deactivate
