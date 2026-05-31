import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import os
from ..handler import logger
from .project import ProjectPaths
from .constants import FILENAME_DATE_INDEX, FILENAME_DATE_INDEX_GEE

def update_processing_rec(fbase, processing_db, ppaths, stage, reprocess=True, prep_rerun=False, treat_brdf='flag_X'):
    
    if stage == 'download':
        rec = processing_db.index[processing_db.index == fbase].tolist()
        logger.info(f'rec = {rec}')
    elif stage == 'brdf':
        rec = processing_db.index[processing_db['brdf_id'].str.startswith(fbase, na = False)].tolist()
    else:
        logger.warning('current stages are download and brdf')

    if len(rec) == 0:
        logger.warning(f'{fbase} is no longer found in the database')
    else:
        if reprocess == False:
            if prep_rerun == False:
                ## Flag records rather than deleting them so they will not be reprocessed in future
                processing_db.loc[processing_db.index == rec[0], 'skip'] = True
                processing_db.loc[processing_db.index == rec[0], 'skip_reason'] = 'user QC'
        else: 
            logger.info(f'removing all occurences of {fbase} -- brdf and record in processing database')
        
        if stage == 'download':
            ## flag row for deletion from database after all files are processed 
            ##    (full record will be deleted, so no need to further modify record) 
            processing_db.loc[processing_db.index == rec[0], 'remove_entry'] = 1
        
        ## find any existing brdf file and delete/flag it 
        if 'brdf_id' in processing_db.columns:
            brdf = processing_db.loc[processing_db.index == rec[0], 'brdf_id'].item()
            logger.debug(f'brdf={brdf}')
            if isinstance(brdf,str):
                brdf0 = brdf.split('.')[0]
                logger.info(f'brdf_id for {rec} is: {brdf}')
                if type(brdf) == str and '.nc' in brdf:
                    ## file might not be exact match because info is sometimes added at end of brdf file names (e.g. during coreg)
                    brdf_match = ppaths.ms.parent.joinpath('brdf').glob(f'{brdf0}*')
                    for b in brdf_match:
                        #brdf_file = ppaths.ms.parent.joinpath(f'brdf/{b}')
                        if prep_rerun or reprocess or (stage == 'download') or (treat_brdf == 'delete'):
                        ## if existing brdf is not deleted, brdf script will prevent rewriting
                            logger.info(f'deleting corresponding brdf file {b}')
                            b.unlink()
                        elif treat_brdf.startswith('flag'):
                            alt= treat_brdf.split('_')[1]
                            if not b.stem.endswith(alt):
                                b.rename(Path(b.parent, f"{b.stem}_{alt}{b.suffix}")) 
                        #TODO: can add move method here
                else: logger.info(f'brdf file was previously recorded as {brdf} in database')    
            else: logger.info('brdf is not in database')
        else: logger.info('brdfs have not been processed for this grid cell')         
        
        if (stage == 'brdf') and (reprocess or prep_rerun):
            ## flag for redownload in case original file was corrupted
            processing_db.loc[rec[0], 'redownload'] = True
            ## remove brdf record from database so it can be recreated
            processing_db.loc[rec[0], 'brdf_id'] = np.nan
            newrec = processing_db.loc[processing_db.index == rec[0]][['redownload','brdf_id']]
            logger.info(f'updated database entry to: {newrec}')

        if prep_rerun:
            if stage == 'brdf':
                dl = processing_db.loc[rec[0], 'dl']
                logger.info(f'dl to copy: {dl}')
                if rec[0].startswith('L'):
                    copy_to = ppaths.ms.parent.joinpath('brdfs_to_rerun/landsat')
                    copy_files = ppaths.ms.parent.joinpath('landsat').glob(f'{rec[0]}*')
                    logger.info(f'copying files {copy_files}')
                else:
                    copy_to = ppaths.ms.parent.joinpath('brdfs_to_rerun/sentinel2')
                    copy_files = ppaths.ms.parent.joinpath('sentinel2').glob(f'{rec[0]}*')
                for cf in copy_files:
                    cf_base = cf.name
                    logger.info(f'copying files {copy_files}')
                    shutil.copy(cf, copy_to.joinpath(cf_base))
    
    return processing_db
    
def clean(params):

    """
    Removes files that are not needed for further processing. 
    These can be the original downloads and other copies of pre-processed images or corrupted files that need to be reprocessed.
    In the case of the former, the record remains in the database so that these files are not reprocessed in any updating procedures.
    In the case of the later (corrupted files), the record needs to be removed from the database so that the processing step can be rerun.
    Note that the 'reprocess' parameter is badly named (TODO: consider renaming). 'reprocess' means that the record is removed from the database
       such that it will reprocess if the process is run again. If the intention is to flag the file to be reprocessed specifically, 'prep-rerun' is used.
    """

    for grid in params['grids']:

        logger.info(f'  Cleaning directory for grid {grid} ...')

        ppaths = ProjectPaths(params, grid=grid)
        
        if ppaths.ms.parent.joinpath('processing.info').is_file():
            processing_db = pd.read_pickle(ppaths.ms.parent.joinpath('processing.info'))
            logger.info(f'processing db has {processing_db.shape[0]} enteries')
        else:
            processing_db = None

        if params['clean']['prep_rerun']:
            ## make new folder to store temp copy of downloads to reprocess
            rerun_dir = ppaths.ms.parent.joinpath('brdfs_to_rerun')
            if rerun_dir.is_dir():
                shutil.rmtree(str(rerun_dir))
            logger.info(f'making folder at {rerun_dir}')
            rerun_landsat = ppaths.ms.parent.joinpath('brdfs_to_rerun/landsat')
            rerun_sentinel = ppaths.ms.parent.joinpath('brdfs_to_rerun/sentinel2')
            rerun_landsat.mkdir(parents=True, exist_ok=True)
            rerun_sentinel.mkdir(parents=True, exist_ok=True)
        
        ## with params['clean']['xlist'], can supply a list of files to remove
        ##    format = path to .csv file containing list of basenames
        ##    by default, this list is not supplied and files are cleaned by date/sensor/filetype
        
        if params['clean']['xlist']:
            if isinstance(params['clean']['xlist'], list):
                xlist = params['clean']['xlist']
            else:
                listfile = ppaths.ms.parent.joinpath(params['clean']['xlist'])
                logger.info(f'getting files to clean from list: {listfile}')
                with open(listfile, "r") as inlist: \
                    xlist = list(map(str, inlist.read().replace('\n',',').split(',')))
                logger.info(f'there are {len(xlist)} images to remove')
                
            for fn in xlist:
                logger.info(f'cleaning {fn}')
                if fn.startswith('L3'):
                    fbase = str(fn)[0:35]
                    if params['clean']['prep_rerun']:
                        logger.info(f'prep_rerun')
                        update_processing_rec(fbase, processing_db, ppaths, stage='brdf', reprocess=False, prep_rerun=True)
                    elif params['clean']['treat_brdf'].startswith('flag'):
                        alt= params['clean']['treat_brdf'].split('_')[1]
                        try:
                            (Path(ppaths.ms)/f'{fn}.nc').rename(ppaths.ms / f"{fn}_{alt}.nc") 
                            update_processing_rec(fbase, processing_db, ppaths, stage='brdf', 
                                              reprocess=False, prep_rerun=False, treat_brdf=params['clean']['treat_brdf'])
                        except Exception as e: 
                            logger.warning(f'ERROR {e}') 
                else:
                    fbase = str(fn).split('.')[0]
                    update_processing_rec(fbase, processing_db, ppaths, stage='download', reprocess=False, prep_rerun=False)

            pd.to_pickle(processing_db, ppaths.ms.parent.joinpath('processing.info'))
        
        else:
            xlist = None
            
            ## Note images are recorded in 'processing.info' and will not be reprocessed (download or brdf) 
            ##   if the file is removed unless the record is also deleted from 'processing.info'. 
            ##   To delete the record, set params['clean']['delete_record'] to True.
            ##   If params['clean']['remove_items'] is 'downloads', this will also remove corresponding 
            ##   brdf file so that full reprocessing can occur.
        
            delete_record = params['clean']['delete_record']
        
            ## Can clean files for any or all of the following sensors: S2A,S2B,LT05,LE07,LC08,LC09
            ##    with params['clean']['sat_sensors']
            ##    cleans all sensors if params['clean']['sat_sensors'] == 'All' (default)
        
            sensors = params['clean']['sat_sensors']
            if isinstance(sensors,str):
                sensors = [sensors]

            ## Can set the date range for images to remove with params['clean']['date_range']
            ## which is [YYYYMMDD, YYYYMMDD]. If a single image is to be cleaned, can use [YYYYMMDD]
            ## if all files are to be cleaned (all dates), use [0] (default)
       
            date_range = params['clean']['date_range']
            if  date_range[0] == 0:
                date_start = 19000101
                date_end = 20990101
            else:
                date_start = int(date_range[0])
                if len(date_range) > 1:
                    date_end = int(date_range[1])
                else:
                    date_end = date_start

            logger.info(f'cleaning files from sensors: {sensors} for dates: {date_start} to {date_end}')
        
            ## directories to be cleaned set by params['clean']['remove_items']
            ##    'gee' (if gee) or 'downloads' (if stac) removes all raw downloads for selected sensors and dates.
            ##        if params['clean']['delete_record'] == True, will also delete brdf file and database record
            ##    'brdf' removes brdf files and clears brdf entry from database record so that it can be reprocessed
            ##    'nocoreg' removes all files in the backup directory (pre-coreged brdfs) created during check_coreg 
        
            for path_to_remove in params['clean']['remove_items']:
                logger.info(f'cleaning files from {path_to_remove}...')

                if path_to_remove == 'gee':
                
                    # TODO: complete this as with 'downloads' if using gee methods 

                    for fn in ppaths.ms.parent.joinpath('gee').glob('*.nc'):
                        fn.unlink()

                    for fn in ppaths.ms.parent.joinpath('gee').glob('*.xml'):
                        fn.unlink()

                    for fn in ppaths.ms.parent.joinpath('gee').glob('*.txt'):
                        fn.unlink()

                    # Delete temporary directories
                    for pobj in ppaths.ms.parent.joinpath('gee').glob('**/*'):

                        if pobj.is_dir():
                            for fn in pobj.glob('**/*'):
                                fn.unlink()
                            shutil.rmtree(str(pobj))
        
                elif path_to_remove == 'downloads':
                    brdf_path = ppaths.ms.parent.joinpath('brdf')
                
                    if processing_db is not None:
                        processing_db['remove_entry'] = 0
                    else:
                        logger.info('processing database does not exist')
                        delete_record = False

                    if any(s in  ['All','LS2'] for s in sensors) or any(s.upper().startswith('L') for s in sensors):
                        landsat_path = ppaths.ms.parent.joinpath('landsat')
                        logger.info(f'removing files in {landsat_path}...')
                    
                        senstart = ['L'] if sensors == 'All' else [s.upper() for s in sensors if s.upper().startswith('L')]

                        for fn in [f for f in ppaths.ms.parent.joinpath('landsat').glob('*.tif') \
                               if f.name.startswith(tuple(senstart)) \
                               and int(f.name.split('_')[3][:8]) >= date_start \
                               and int(f.name.split('_')[3][:8]) <= date_end]:
                            
                            fbase = fn.name.split('.')[0]
                            if delete_record:
                                update_processing_rec(fbase, processing_db, ppaths, stage='download', reprocess=True)
                        
                            # Remove downloaded file itself
                            if fn.exists():   ## maybe already deleted in update_db
                                logger.info(f'deleting {fn}')
                                fn.unlink()
                        
                            # Remove corresponding .xml and .txt files
                            for fx in ppaths.ms.parent.joinpath('landsat').glob(f'{fbase}*'):
                                logger.info(f'also deleting {fx}')
                            fx.unlink()

                    if any(s in ['All','LS2'] for s in sensors) or any(s.upper().startswith('S2') for s in sensors):
                        # Remove all downloaded files in sentinel2 folder
                        sen2_path = ppaths.ms.parent.joinpath('sentinel2')
                        logger.info(f'removing files in {sen2_path}...')
                    
                        senstart = ['S2'] if any(s in ['All','LS2'] for s in sensors) else [s for s in sensors if s.upper().startswith('S2')]

                        for fn in [f for f in ppaths.ms.parent.joinpath('sentinel2').glob('*.tif') \
                               if f.name.startswith(tuple(senstart)) \
                               and int(f.name.split('_')[2][:8]) >= date_start \
                               and int(f.name.split('_')[2][:8]) <= date_end]:
                            
                            fbase = fn.name.split('.')[0]
                            if delete_record:  
                                update_processing_rec(fbase, processing_db, ppaths, stage='download', reprocess=True)
                        
                            # Remove downloaded file itself
                            if fn.exists():   ## maybe already deleted in update_db
                                logger.info(f'deleting {fn}')
                                fn.unlink()
                        
                            # Remove corresponding .xml and .txt files
                            for fx in ppaths.ms.parent.joinpath('sentinel2').glob(f'{fbase}*'):
                                logger.info(f'also deleting {fx}')
                                fx.unlink()
                            
                    # Remove temp files
                    for fn in ppaths.ms.glob('*temp*'):
                        fn.unlink()

                    # Remove 'no data' files
                    for fn in ppaths.ms.glob('*.nodata'):
                        fn.unlink()

                    # Remove window tracker
                    for fn in ppaths.ms.parent.glob('*.window'):
                        fn.unlink()
                
                    if delete_record:
                        processing_db = processing_db[processing_db['remove_entry'] == 0]
                        logger.info(f'processing db now has {processing_db.shape[0]} images')
                    
                    pd.to_pickle(processing_db, ppaths.ms.parent.joinpath('processing.info'))
                
                elif path_to_remove == 'brdf':
                    brdf_path = ppaths.ms
                    
                    if processing_db is not None:
                        if 'brdf_id' in processing_db.columns:
                            update_db = True
                        else:
                            logger.info('brdfs have not been processed yet')
                            update_db = False
                    else:
                        logger.info('processing database does not exist')
                        update_db = False
               
                    if params['dlMethod'] == 'GEE':
                        date_pos=FILENAME_DATE_INDEX_GEE
                        prepend_str='netcdf:'
                    else:
                        date_pos=FILENAME_DATE_INDEX
                        prepend_str=''
        
                    senstart = ['L','S'] if sensors in ['All','LS2'] else [s.upper() for s in sensors]
                
                    for fn in [f for f in brdf_path.glob('*.nc') \
                               if f.name.split('_')[1].startswith(tuple(senstart)) \
                               and int(f.name.split('_')[date_pos][:8]) >= date_start \
                               and int(f.name.split('_')[date_pos][:8]) <= date_end]:
                        
                        fbase = fn.name[0:35]
                        if update_db:
                            update_processing_rec(fbase, processing_db, ppaths, stage='brdf', reprocess=True)
                               
                        if fn.exists():   ## maybe already deleted in update_db
                            logger.info(f'deleting {fn}') 
                            fn.unlink()
                        
                elif path_to_remove == 'processing_db':
    
                    if ppaths.ms.parent.joinpath('processing.info').is_file():
                        ppaths.ms.parent.joinpath('processing.info').unlink()
                    
                    '''
                    # Remove angles directories
                    for fdir in ppaths.ms.parent.joinpath('tmp').glob('angle*'):
                        if fdir.is_dir():
                            for fn in fdir.glob('**/*'):
                                fn.unlink()
                            shutil.rmtree(str(fdir))

                    # Remove downloaded GTiff files
                    for fn in ppaths.ms.parent.joinpath('tmp').glob('*.TIF'):
                        fn.unlink()

                    for fn in ppaths.ms.parent.joinpath('tmp').glob('*.tif'):
                        fn.unlink()

                    # Remove downloaded j2 files
                    for fn in ppaths.ms.parent.joinpath('tmp').glob('*.jp2'):
                        fn.unlink()

                    # Remove downloaded XML files
                    for fn in ppaths.ms.parent.joinpath('tmp').glob('*.xml'):
                        fn.unlink()

                    # Remove downloaded temporary files
                    for fn in ppaths.ms.parent.joinpath('tmp').glob('*.gstmp'):
                        fn.unlink()
                    '''

                elif path_to_remove == 'old_seg':

                    if ppaths.seg.is_dir():

                        if (ppaths.seg / 'edge_probas.tif').is_file():
                            (ppaths.seg / 'edge_probas.tif').unlink()

                        if (ppaths.seg / 'seg.tif').is_file():
                            (ppaths.seg / 'seg.tif').unlink()

                elif path_to_remove == 'masks':

                    if ppaths.masks.is_dir():
                        for fn in ppaths.masks.glob('*.tif'):
                            fn.unlink()

                elif path_to_remove == 'nodata':

                    if ppaths.nodata.is_dir():
                        for fn in ppaths.nodata.glob('*.tif'):
                            fn.unlink()

                elif path_to_remove == 'nocoreg':

                    logger.info(f'removing precoreg files from {ppaths.s2_nocoreg}...')
                    if ppaths.s2_nocoreg.is_dir():
                        for fn in ppaths.s2_nocoreg.glob('*.tif'):
                            fn.unlink()
                        for fn in ppaths.s2_nocoreg.glob('*.nc'):
                            fn.unlink()

                elif path_to_remove == 'fusion':
    
                    if ppaths.fusion.is_dir():
                        for fn in ppaths.fusion.glob('*.tif'):
                            fn.unlink()

                elif path_to_remove == 'sis':

                    for root, dirs, files in os.walk(str(ppaths.ts)):

                        if files:

                            for fn in Path(root).glob('*.window'):
                                fn.unlink()

                            for fn in Path(root).glob('*.reindex'):
                                fn.unlink()

                            for fn in Path(root).glob('*.tif'):
                                fn.unlink()

                elif path_to_remove == 'vrts':

                    # Remove VRT files
                    for root, dirs, files in os.walk(str(ppaths.ts)):
                        if files:
                            for fn in Path(root).glob('*.vrt'):
                                fn.unlink()

        logger.info(f'  Finished cleaning grid {grid}.')
