#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-1:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o stac_rawcomp.%N.%a.%j.out # STDOUT
#SBATCH -e stac_rawcomp.%N.%a.%j.err # STDERR
#SBATCH --job-name="rawcomp"
#SBATCH --array=379

GRID_ID="$SLURM_ARRAY_TASK_ID"
RES=10.0
PROCSEQ='mu.br.cga'
IMGTYPE='LS2'
FEATMOD='base6svh23'
SIS=("wi-raw" "nbr-raw")
BANDS="[cv-yr,cv-wet,cv-dry]"
MODYR=2023
## If OUT == 'archive', final composites will be sent to the cell directory in the bk dir
##   If OUT == 'tmp', final composites will be sent to a single 'comp' folder in the temp drive (to be mosaicked into a final product)
OUT='archive'

###################################################################
### Project settings
MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
SCRATCH_DIR="/home/scratch-cel"
PROJECT="biltong"
######  project calendar
STARTMO=11
STARTWET=1
ENDWET=120
STARTDRY=181
ENDDRY=300

# Set permissions on output files
umask 002
#############################################
# Turn off NumPy parallelism and rely on dask
#############################################
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# This should be sufficient for OpenBlas and MKL
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
################################################

METHOD='STAC'
NCHUNKS=512
SM_CHUNKS=512

###############################
# DO NOT MODIFY BELOW THIS LINE
###############################

# activate the virtual environment
conda activate venv.tuyau

for SI in "${SIS[@]}"

do

	CONFIG_UPDATES="grids:[${GRID_ID}]
	res:${RES}
    image_type:${IMGTYPE}
    procseq:${PROCSEQ}
	masking:sat_sensors:${SAT_SENSORS}
	main_path:${MAIN_DIR}/${PROJECT}/stac/grid
	backup_path:${BK_DIR}/${PROJECT}/stac/grid
	scratch_dir:${SCRATCH_DIR}/${PROJECT}/stackprods
	dlMehod:${METHOD}
	num_workers:${SLURM_CPUS_ON_NODE}
	io.n_chunks:${NCHUNKS} 
	reconstruct:chunks:${SM_CHUNKS}
	reconstruct:rewrite_win:False
	reconstruct:overwrite:True
	feature_model:spec_indces:$SI
	classify:out_yrs:None
	reconstruct:si:$SI
	feature_model:start_yr:$MODYR
	feature_model:si_vars:$BANDS
	feature_model:treat_out:$OUT
	calendar:first_mo:$STARTMO
	calendar:start_wet:$STARTWET
	calendar:end_wet:$ENDWET
	calendar:start_dry:$STARTDRY
	calendar:end_dry:$ENDDRY"

tuyau make_ts_composite --config-updates $CONFIG_UPDATES

done
conda deactivate
