#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o TStuyau_zonal.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_zonal.%N.%a.%j.err # STDERR
#SBATCH --job-name="zonal"

###################################################################

###################################################################
### Project settings
PREFIX='CELPy'
MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
SCRATCH_DIR="/home/scratch-cel"
PROJECT="paraguay_lc"
PROJECTV="Py_0"

###################################################################
### Settables

OUTDIR="${BK_DIR}/$PROJECT/outputs/summary_data"
MAPNAME='CELPyTileAll_base4Poly6_bal200mix4_LC36_1723_RF_2021'

## MAPDIR For CEL products:
MAPDIR="${BK_DIR}/$PROJECT/mosaics"
## MAPDIR For other prods:
#MAPDIR="${BK_DIR}/$PROJECT/lc_prods"

## For E & W Paraguay:
POLYS="/home/downspout-cel/paraguay_lc/vector/zonal_polys/Py/PyAll2_8858.shp"
## Note: CLASSMOD needs to be in SCHEMATIC_MOD dict (in lookup.py) or...
CLASSMOD='veg_with_crop'
OUTCATS='All'

MAPDICT='/home/downspout-cel/paraguay_lc/lc_prods/prod_dict.json'
#MAPDICT=
###################################################################
### activate the virtual environment
conda activate venv.tuyau

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT} 
project_name:$PROJECT
project_ver:$PROJECTV
scratch_dir:$SCRATCH_DIR
explore:zones:$POLYS
explore:out_dir:$OUTDIR
explore:product_dir:$MAPDIR
explore:product_name:$MAPNAME
explore:map_type:cat
schematic_model:summary_mod:$CLASSMOD
explore:printcats:$OUTCATS
explore:product_dict:$MAPDICT
"

tuyau summarize_raster --config-updates $CONFIG_UPDATES

conda deactivate
