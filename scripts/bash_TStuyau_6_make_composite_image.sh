#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
##SBATCH -p basic
#SBATCH -o TStuyau_comp.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_comp.%N.%a.%j.err # STDERR
#SBATCH --job-name="TStuyau_comp"
#SBATCH --array=620-650%8
###################################################################
### Modify --array above with grid cell numbers to run as array job.

GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
### GRID_ID=$(($SLURM_ARRAY_TASK_ID + 3000))

###################################################################
### Settables

#Note SPEC_INDEX is a folder in the IMGDIR containing the time series images
SPEC_INDEX='evi2'
MODYR=2021
IMGTYPE='LS2'
RES=10.0
PROCSEQ='mu.br.cga'
FEATMOD='base6svh23'

BANDS="[minv_yr,maxv_yr,amp_yr]"
#BANDS="[Jan_20,Jun_20,Nov_20]"
#BANDS="[Jan_20,Feb_20,Mar_20,Apr_20,May_20,Jun_20,Jul_20,Aug_20,Sep_20,Oct_20,Nov_20,Dec_20]"

### If creating multiyr mosaic, set MULTIYR as list (e.g. ("[2022,2023,2024,2025]")
###    with this option, can only use one spec_index and band currently
MULTIYR=None
### For multi-year composite (works with single variable band)
#MULTIYR=[2022,2023,2025]
#BANDS="[cv_wet]"

## If OUT == 'archive', final composites will be sent to the cell directory in the bk dir
##   If OUT == 'tmp', final composites will be sent to a single 'comp' folder in the temp drive (to be mosaicked into a final product)
OUT='archive'

###  Pheno allows for extra padding around season and more complex statistics.
###    the following parameters only matter if PHENO=True
PHENO=False
PHENOSIS="[kndvi]"
PHENOVARS="[posv_wet]"
PHENOPAD="[30,0]"
SIGDIF=500
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
SCRATCH_DIR="/home/scratch-cel"
PROJECT="paraguay_lc"
PROJVER='Py_0'
## COMP variables control naming of ts and brdf folders -- simpler naming if false
COMPSEN=False
COMPRES=False
COMPPROC=False
######  project calendar 
STARTMO=6
STARTWET=306
ENDWET=61
STARTDRY=183
ENDDRY=259
###################################################################

export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"

### activate the virtual environment
conda activate venv.tstuyau_pipe

###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################

CONFIG_UPDATES="grids:$GRID_ID
res:${RES}
image_type:${IMGTYPE}
procseq:${PROCSEQ}
project_ver:$PROJVER
main_path:${MAIN_DIR}/${PROJECT}/stac/grid
backup_path:${BK_DIR}/${PROJECT}/stac/grid
scratch_dir:${SCRATCH_DIR}/${PROJECT}/composites
classify:out_yrs:$MULTIYR
feature_model:start_yr:$MODYR
feature_model:spec_indices:$SPEC_INDEX
feature_model:si_vars:$BANDS
feature_model:treat_out:$OUT
calendar:first_mo:$STARTMO
calendar:start_wet:$STARTWET
calendar:end_wet:$ENDWET
calendar:start_dry:$STARTDRY
calendar:end_dry:$ENDDRY
feature_model:use_pheno:$PHENO
feature_model:spec_indices_pheno:$PHENOSIS 
feature_model:pheno_vars:$PHENOVARS 
feature_model:pheno_pad_days:$PHENOPAD
feature_model:pheno_sigdif:$SIGDIF
compare_imgtype:${COMPSEN}
compare_res:${COMPRES}
compare_procseq:${COMPPROC}
"

tuyau make_ts_composite --config-updates $CONFIG_UPDATES

conda deactivate
