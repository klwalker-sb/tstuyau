#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 8 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_ptvals.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_ptvals.%N.%a.%j.err # STDERR
#SBATCH --job-name="ptvals"

###################################################################
########## Settables #####################################################

CELLS="${BK_DIR}/${PROJECT}/cell_lists/Training_cells.csv"
IMGTYPE='LS2'
SI='ndvi-raw'
YRS=[2022,2024]
LOADSAMP=True
PTFILE="/home/downspout-cel/paraguay_lc/vector/pts_training/SamplePts_Dec2024.csv"
FILTCOL='Class'
FILTCLASS='35'

### The following only matters if making new points from polygons (POLYS != None & LOADSMP ==False)
#LOADSAMP=False
POLYS=None
NEWEST=2022
OLDEST=2010
YRCOL='yrObs'
NPTS=5
SEED=88

###################################################################
### Project settings

######  project calendar 
STARTMO=6
STARTWET=306
ENDWET=61
STARTDRY=183
ENDDRY=259

EPSG=8858
MAIN_DIR="/home/sandbox-cel/"
BK_DIR="/home/downspout-cel/"
PROJECT="paraguay_lc"
PROJECT_DIR ="${MAIN_DIR}/${PROJECT}"
GRID_FILE="${PROJECT_DIR}/project_grid_${EPSG}.gpkg"
OUTDIR="${PROJECT_DIR}/vector/sampleData"

###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

### activate the virtual environment
conda activate venv.tstuyau_pipe

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:$CELLS
main_path:"${MAIN_DIR}/${PROJECT}/stac/grid"
backup_path:"${BK_DIR}/${PROJECT}/stac/grid" 
scratch_dir:$SCRATCH 
grid_file:$GRID_FILE
image_type:$IMGTYPE
feature_model:spec_indices:$SI
sample_model:train_yrs:$YRS
calendar:first_mo:$STARTMO
calendar:start_wet:$STARTWET
calendar:end_wet:$ENDWET
calendar:start_dry:$STARTDRY
calendar:end_dry:$ENDDRY
sample_model:load_samp:$LOADSAMP
sample_model:point_file:$PTFILE
sample_model:filter_col:$FILTCOL
sample_model:filter_class:$FILTCLASS
sample_model:poly_file:$POLYS
sample_model:npts:$NPTS
sample_model:obs_col:$YRCOL
sample_model:newest:$NEWEST
sample_model:oldest:$OLDEST
sample_model:poly_samp_seed:$SEED
sample_model:point_samp_dir:$OUTDIR
masking:maxval:10000"

tuyau sample_timeseries --config-updates $CONFIG_UPDATES

conda deactivate
