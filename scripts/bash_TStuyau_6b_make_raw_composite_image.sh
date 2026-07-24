#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 4 # number of cores
#SBATCH -t 0-1:00 # time (D-HH:MM)
#SBATCH -p basic 
#SBATCH -o tstuyau_bl_rawcomp.%N.%a.%j.out # STDOUT
#SBATCH -e tstuyau_bl_rawcomp.%N.%a.%j.err # STDERR
#SBATCH --job-name="rawcomp"
#SBATCH --array=203

GRID_ID="$SLURM_ARRAY_TASK_ID"

SIs=("kndvi-raw")
BANDS="[amp-wet,cv-wet,minv-wet]"
MODYR=2023
MULTIYR=None
### For multi-year composite (works with single variable band)
#MULTIYR=[2022,2023,2025]
#BANDS="[cv_wet]"

## If OUT == 'archive', final composites will be sent to the cell directory in the bk dir
##   If OUT == 'tmp', final composites will be sent to a single 'comp' folder in the temp drive (to be mosaicked into a final product)
OUT='tmp'

###  Pheno allows for extra padding around season and more complex statistics.
###    the following parameters only matter if PHENO=True
PHENO=False
PHENOSIS="[mirbi-raw]"
#PHENOVARS="[posv_wet]"
PHENOVARS="[burn.p1200.doy-Monthly]"
PHENOPAD="[15,15]"
##  these only matter for delta calculations (sigdif, burn, etc.)
SIGDIF=1200
RANGE_PREEVENT="[1400,3200]"
RANGE_POSTEVENT="[3200,10000]"
IMGBUF=3
MAKETS=True
###################################################################
### Project settings
MAIN_DIR="/home/sandbox-cel"
BK_DIR="/home/downspout-cel"
SCRATCH_DIR="/home/scratch-cel"
PROJECT="biltong"
RES=10.0
IMGTYPE='LS2'
PROCSEQ='mu.br.cga'
NCHUNKS=512
SM_CHUNKS=51
METHOD='STAC'

######  project calendar
STARTMO=11
STARTWET=1
ENDWET=120
STARTDRY=181
ENDDRY=300

####################################################
##activate the virtual environment
conda activate venv.tuyau
# Set permissions on output files
umask 002
#####################################################

#####################################################
# SHOULD NOT NEED TO MODIFY BELOW
#####################################################

for SI in "${SIs[@]}"

do
	CONFIG_UPDATES="grids:{${GRID_ID}}
	res:${RES}
	image_type:${IMGTYPE}
	procseq:${PROCSEQ}
    dlMehod:${METHOD}
	main_path:${MAIN_DIR}/${PROJECT}/stac/grid
	backup_path:${BK_DIR}/${PROJECT}/stac/grid
	scratch_dir:${SCRATCH_DIR}/${PROJECT}/stackprods  
	feature_model:ts_type:raw
    reconstruct:nodata:0
    reconstruct:exclude:'X'
    reconstruct:chunks:${SM_CHUNKS}
    num_workers:${SLURM_CPUS_ON_NODE}
    io.n_chunks:${NCHUNKS}
    reconstruct:rewrite_win:False
    reconstruct:overwrite:$MAKETS
	feature_model:spec_indces:$SI
	classify:out_yrs:$MULTIYR
	feature_model:start_yr:$MODYR
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
    feature_model:pheno_basethresh_pre:$RANGE_PREEVENT
    feature_model:pheno_basethresh_post:$RANGE_POSTEVENT
    feature_model:pheno_imgbuf:$IMGBUF

    "

tuyau make_ts_composite --config-updates $CONFIG_UPDATES

done
conda deactivate
