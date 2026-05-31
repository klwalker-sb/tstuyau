#!/bin/bash -l
#
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-08:00 # time (D-HH:MM)
##SBATCH -p basic
#SBATCH -o TStuyau_glcm.%N.%a.%j.out # STDOUT
#SBATCH -e TStuyau_glcm.%N.%a.%j.err # STDERR
#SBATCH --job-name="TStuyau_glcm"
#SBATCH --array=824

###################################################################
### Modify --array above with grid cell numbers to run as array job.

#GRID_ID=$SLURM_ARRAY_TASK_ID
### note: if grid cell > 999, enter last three digits in array and use
GRID_ID=$(($SLURM_ARRAY_TASK_ID + 3000))

###################################################################
### Settables

#Note SPEC_INDEX is a folder in the IMGDIR containing the time series images
SPEC_INDEX='kndvi'
MODYR=2024
## TH is angle value. can be 0-3 (quadrant) or 'mix4' to get average of all 4 quadrants
##  0 = horizontal, 1 = diag up-right, 2= vertical, 3= diag up-left
TH='mix4'
VAR='med'
#BANDS="[med.glcm.corr.w5.c100.th0-yr]"
### If creating multiyr mosaic, set MULTIYR as list (e.g. ("[2022,2023,2024,2025]")
###    with this option, can only use one spec_index and band currently
MULTIYR=None
### For multi-year composite (works with single variable band)
#MULTIYR=[2022,2023,2025]
#BANDS="[cv_wet]"

## If OUT == 'archive', final composites will be sent to the cell directory in the bk dir
##   If OUT == 'tmp', final composites will be sent to a single 'comp' folder in the temp drive (to be mosaicked into a final product)
OUT='archive'
PROJVER='Py_0'

###  Pheno allows for extra padding around season and more complex statistics.
###    the following parameters only matter if PHENO=True
PHENO=False

###################################################################
### Project settings

MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
SCRATCH_DIR="/home/scratch-cel"
PROJECT="paraguay_lc"

######  project calendar 
STARTMO=7
STARTWET=306
ENDWET=61
STARTDRY=183
ENDDRY=259


## will cycle through window, c_vals and textures to generate glcm with each combination of values
##  if TH='mix4, will calculate all four primary angle paths (vertical, diag right, horizontal, diag left) and return average

## Wins are the moving windows. Must be odd. e.g. (5 11 25) or just (11)
#WINS=(5 11 25)

## C_VALS are the total number of possible values (maxval after rescaling data)
#C_VALS=(32 64 100 255)

## TEXTURE options: ('homogeneity' 'dissimilarity' 'contrast' 'correlation' 'entropy''variiance' 'energy' 'ASM")
##  just need to enter forst 3 letters
#TEXTURES=('homo' 'diss' 'cont' 'ent')

WINS=(5 11 25)
C_VALS=(64 100)
TEXTURES=('ent')


###################################################################
### activate the virtual environment
conda activate venv.tuyau
###################################################################
###################################################################
### SHOULD NOT NEED TO MODIFY BELOW
###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"


# Nested loops to iterate through every combination
for win in "${WINS[@]}"; do
    for c in "${C_VALS[@]}"; do
        for tex in "${TEXTURES[@]}"; do
                
       	    # Construct and update the BANDS variable
            BANDS="[${VAR}.glcm.${tex}.w${win}.c${c}.th${TH}-yr]"
                
 	    CONFIG_UPDATES="grids:$GRID_ID
	    main_path:${MAIN_DIR}/${PROJECT}/stac/grid
 	    backup_path:${BK_DIR}/${PROJECT}/stac/grids
 	    scratch_dir:${SCRATCH_DIR}/${PROJECT}/composites
 	    classify:out_yrs:$MULTIYR
 	    project_ver:$PROJVER
 	    feature_model:start_yr:$MODYR
 	    feature_model:spec_indices:$SPEC_INDEX
 	    feature_model:si_vars:$BANDS
	    feature_model:treat_out:$OUT
	    calendar:first_mo:$STARTMO
	    calendar:start_wet:$STARTWET
	    calendar:end_wet:$ENDWET
	    calendar:start_dry:$STARTDRY
	    calendar:end_dry:$ENDDRY
	    "
		tuyau make_ts_composite --config-updates $CONFIG_UPDATES
                
        done
    done
done

conda deactivate
