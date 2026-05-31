#!/bin/bash -l

#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -p basic
#SBATCH -o tstuyau_checkDls.%N.%a.%j.out # STDOUT
#SBATCH -e tstuyau_checkDls.%N.%a.%j.err # STDERR
#SBATCH --job-name="checkDls"

#Settables:

DBPATH='/home/downspout-cel/paraguay_lc/cell_processing_dl.csv'
ARCHIVE='~/archive/eostac_logs'
LOGS='.'
START="['2000-01-01','2025-10-01']"
#IGNORE=('2025-10-01--2025-12-31')
PREFIX='stacdl1_py'

#Activate the virtual environment (which relys on anaconda)
###################################################################
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_ON_NODE}"
### activate the virtual environment
conda activate venv.tuyau


CONFIG_UPDATES="
status:period:$STARTSTOP
status:archive_path:$ARCHIVE
status:download_db_path:$DBPATH
status:log_path:$LOGS
status:ignore_dates:$IGNORE
status:log_prefix:$PREFIX
"
tuyau dl_check --config-updates $CONFIG_UPDATES

conda deactivate
