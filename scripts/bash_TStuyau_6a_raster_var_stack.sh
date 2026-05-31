#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-24:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o rfstack.%N.%a.%j.out # STDOUT
#SBATCH -e rfstack.%N.%a.%j.err # STDERR
#SBATCH --job-name="rfstack"
#SBATCH --array=715
###################################################################
### If a running smallish number of cells, 
###    can enter in array above and use these values as the cell input:

#CELLS=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
CELLS="$(($SLURM_ARRAY_TASK_ID + 3000))"

### If running a lot of cells, can use a list: 
#CELLS="/home/downspout-cel/paraguay_lc/cell_lists/CellsP2.csv"
### To parallelize, can split list into multiple lists in a directory:
#CELLS="/home/downspout-cel/paraguay_lc/cell_lists/TrainingCells/${SLURM_ARRAY_TASK_ID}.csv"
###################################################################
###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
PROJECT="paraguay_lc"
PROJECT_VER="Py_0"
SINGDICT="${BK_DIR}/${PROJECT}/ancillary_var_dict.json"
MODDICT="${BK_DIR}/${PROJECT}/Feature_Models.json"
SEG_PATH="${BK_DIR}/${PROJECT}/Segmentations"
POLY_MOD='cnet4VIk'

### Scratch dir (to save intermedite products). 
###    If blank, saves to 'comp' folder of TSDIR for each cell
#SCRATCH=""
SCRATCH="/home/scratch-cel/stackprods"

######  project calendar 
STARTMO=6
STARTWET=306
ENDWET=61
STARTDRY=183
ENDDRY=259

IMGTYPE='LS2'
PROSEQ='m0.br.cga'
RES=10.0
#################################################################
### Settables

MODNAME='base4Poly6'
OVERWRITE=True
STARTYR=2024

OUT='archive'
####### New model specs -- only matter if model is not already in the MODDICT
SIs="[kndvi,gcvi,ndmi,nbr]"
SIVARS="[maxv-yr,minv-yr,amp-yr,avg-yr,sd-yr,Jan-20,Feb-20,Mar-20,Apr-20,May-20,Jun-20,Jul-20,Aug-20,Sep-20,Oct-20,Nov-20,Dec-20,maxv-wet,minv-wet,med-wet,cv-wet,maxv-dry,minv-dry,med-dry,cv-dry]"
SING="[BH,SH,Chaco,Cer]"
#POLYVARS="[]"
POLYPATH="${SEG_PATH}/${POLY_MOD}/feats_EO_8pt5_$((STARTYR+1))"
POLYVARS="[poly_ext,poly_dst,poly_cropbnds,poly_area,poly_APrEf,poly_NovDecStd]"
PHENOSIS="[kndvi]"
PHENOVARS="[posv.500-wet]"
PHENOPAD="[30,0]"
SIGDIF=500
## BASETHRES is set based on data for peak calculations, so should be None here
BASETHRESH=None
IMGBUF=None
COMBOBANDS="[]"

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
backup_path:${BK_DIR}/${PROJECT}/stac/grids 
scratch_dir:$SCRATCH
project_ver:$PROJECT_VER
image_type:$IMGTYPE
procseq:$PROSEQ
res:$RES
feature_model:name:$MODNAME 
feature_model:start_yr:$STARTYR 
feature_model:start_mo:$STARTMO 
feature_model:spec_indices:$SIs 
feature_model:si_vars:$SIVARS 
feature_model:spec_indices_pheno:$PHENOSIS 
feature_model:pheno_vars:$PHENOVARS 
feature_model:pheno_pad_days:$PHENOPAD
feature_model:pheno_sigdif:$SIGDIF
feature_model:pheno_basethresh:$BASETHRESH
feature_model:pheno_imgbuf:$IMGBUF
feature_model:feature_mod_dict:$MODDICT 
feature_model:ancillary_vars:$SING 
feature_model:singelton_var_dict:$SINGDICT 
feature_model:poly_vars:$POLYVARS 
feature_model:poly_var_path:$POLYPATH
feature_model:combo_bands:$COMBOBANDS 
feature_model:treat_out:$OUT 
feature_model:overwrite:$OVERWRITE
calendar:first_mo:$STARTMO
calendar:start_wet:$STARTWET
calendar:end_wet:$ENDWET
calendar:start_dry:$STARTDRY
calendar:end_dry:$ENDDRY"

tuyau make_var_stack --config-updates $CONFIG_UPDATES

conda deactivate
