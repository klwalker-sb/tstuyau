#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_tsplot.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_tsplot.%N.%a.%j.err # STDERR
#SBATCH --job-name="tsplot"

###################################################################
########## Settables #####################################################

#CELLS="${BK_DIR}/${PROJECT}/cell_lists/Training_cells.csv"
CELLS='[176]'
IMTYPE='LC2'
SI='nbr'
START='2022-07-01'
END='2023-12-01'
FILTCOL='Grass_ID'
FILTCLASS='ARIJUN'
DPI=300
COORDS='[-30.917517,28.4026,-30.871873,28.391676]'
###################################################################
### Project settings

######  project calendar 
STARTMO=11
STARTWET=1
#ENDWET=120
ENDWET=61
STARTDRY=181
ENDDRY=300

MAIN_DIR="/home/sandbox-cel/"
BK_DIR="/home/downspout-cel/"
PROJECT="biltong"
PROJECT_DIR="${MAIN_DIR}/${PROJECT}"
GRID_FILE="${PROJECT_DIR}/vector/biltong_grid_utm35S.gpkg"
OUTDIR="${PROJECT_DIR}/vector/sampleData"
###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

### activate the virtual environment
conda activate venv.tuyau

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:$CELLS
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT}/stac/grid 
scratch_dir:$SCRATCH 
grid_file:$GRID_FILE
image_type:$IMTYPE
dlmethod:stac
reconstruct:si:$SI
reconstruct:chunks:256
reconstruct:res:10
nodata:0
sample_model:point_samp_dir:$OUTDIR
masking:maxval:10000
plot:coords:$COORDS
plot:start:$START
plot:end:$END
plot:dpi:$DPI
io:gdal_cachemax:512
plot:out_path:$OUTDIR
numworkers:4"

tuyau plot_timeseries --config-updates $CONFIG_UPDATES

conda deactivate
