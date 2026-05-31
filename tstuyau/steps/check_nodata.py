import sys
import shutil
import math
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import pyproj
#import pickle
import xarray as xr
import rasterio as rio
from shapely.geometry import box
import geowombat as gw

from ..handler import logger
from .project import ProjectPaths
from .check_status import read_db


def reconstruct_db(processing_info_path,landsat_path,sentinel2_path,brdf_path):
   
    """ 
    This checks for an existing processing.info database and creates one if needed from download and brdf folders.
    This is only for cases of corruption or accidental deletion. -- 
    processing.info is normally created as files are downloaded -- 
    Note: It is best to use original database whenever possible, as this will not recreate error notes,
    nor populate the numpix or coreg shift_x and shift_y columns that are in the original db
    """
    
    modified = False
    if Path(brdf_path).is_dir():
        brdf_files = list(Path(brdf_path).glob('*.nc'))
    else:
        brdf_files = []
    if Path(landsat_path).is_dir():
        landsat_files = list(Path(landsat_files).glob('*.tif'))
    else:
        landsat_files = list(Path(landsat_files).glob('*.tif'))
    if Path(sentinel2_path).is_dir():
        sentinel2_files = list(Path(sentinel2_files).glob('*.tif'))
    else:
        sentinel2_files = []
        
    if len(landsat_files) + len(sentinel2_files) + len(brdf_files) == 0:
        logger.warning('no images have been downloaded')
    else:
        ## Make new processing db if it does not already exist:
        if not processing_info_path.is_file():
            processing_dict = {}
            # First check for for files in the brdf folder (these are at the most processed stage)
            if len(brdf_files) > 0:
                for b in brdf_files:
                    # get corresponding dl id:
                    if b.stem.split("_")[1].startswith('L'):
                        dlid = '{}_{}_{}_{}_{}_{}'.format(b.stem.split("_")[1],
                                                         'L2SP',
                                                          b.stem.split("_")[2][4:10],
                                                          b.stem.split("_")[3],
                                                          b.stem.split("_")[2][10:12],
                                                          b.stem.split("_")[2][12:14])
                    else:
                        dlid = '{}_{}_{}_{}_{}'.format(b.stem.split("_")[1],
                                                       b.stem.split("_")[2][4:9],
                                                       b.stem.split("_")[3],
                                                       b.stem.split("_")[2][9:10],
                                                       b.stem.split("_")[2][10:13])
                    bp = 'True' if b.stem.split('_')[0] == 'L3B' else ('False' if b.stem.split('_')[0] == 'L3A' else np.nan)
                    processing_dict[dlid] = {'dl':'{landsat_path}',
                                           'beforeDB':True,
                                           'redownload':False,
                                           'brdf_id':f'{b.stem}',
                                           'brdf':'True',
                                           'brdf_error':np.nan,
                                           'bandpass':bp}
            # If no files in brdf folder, reconstruct db from download folders
            else:
                for f in landsat_files:
                    processing_dict[f.stem]={'dl': f'{landsat_path}/{f.stem}',
                                                             'beforeDB':True, 'redownload':False}
                for s in sentinel2_files:
                    processing_dict[s.stem]={'dl':f'{sentinel2_path}/{s.stem}',
                                                             'beforeDB':True, 'redownload':False}
            new_processing_info = pd.DataFrame.from_dict(processing_dict,orient='index')
            new_processing_info.rename_axis('id', axis=1, inplace=True)
            pd.to_pickle(new_processing_info, processing_info_path)
            logger.info(f'{len(new_processing_info)} images downloaded and added to database.')
            
        # read in existing db (can be the one that was just created or pre-existing):
        processing_db = pd.read_pickle(processing_info_path)
        
        ## to fix issues from older version of db already created for some cells:
        if 'id' not in processing_db:
            processing_db.rename_axis('id', axis=1, inplace=True)
        #if processing_db.index != 'id':
        #    logger.info('removing original index column and setting it to id column')
        #    processing_db.set_index('id', drop=True, inplace=True)
        
        logger.info(f'{len(processing_db)} records in db. {len(landsat_files)} landsat and {len(sentinel2_files)} sentinel images in downloads.')

        if len(processing_db) >= len(landsat_files) + len(sentinel2_files):
            logger.info('all downloaded images have probably been added to db already')
        else:
            logger.info('adding images to db...')
            new_dls = {}
            for f in landsat_files:
                if f.stem in processing_db.values:
                    continue
                else:
                    new_dls[f.stem]={'dl':f'{landsat_path}/{f}','beforeDB':True}
            for s in sentinel2_files:
                if s.stem in processing_db.values:
                    continue
                else:
                    new_dls[s.stem]={'dl':f'{sentinel2_path}/{s}',',beforeDB':True}
        
            if len(new_dls)>0:
                new_dl_db = pd.DataFrame.from_dict(new_dls,orient='index')
                new_dl_db.rename_axis('id', axis=1, inplace=True)
                processing_db.append(new_dl_db)
                modified = True
            
        if Path(brdf_path).is_dir(): 
            if 'brdf' in processing_db:
                logger.info('brdf data already in database')
            
            else: 
                logger.info('adding brdf info to db...')
                processing_db['brdf_id'] = np.nan
                processing_db['brdf_error'] = np.nan
                processing_db['brdf'] = np.nan
                processing_db['bandpass'] = np.nan
                for idx, row in processing_db.iterrows():
                    match=None
                    logger.debug(f'idx: {idx}')
                    for fi in list(Path(brdf_path).glob('*.nc')):
                        if idx.startswith('S'):  
                            if (idx.split('_')[1] in fi.stem.split('_')[2]) and (idx.split('_')[2] == fi.stem.split('_')[3]):
                                match = fi
                        elif idx.startswith('L'): 
                            if (idx.split('_')[0] == fi.stem.split('_')[1]) and (idx.split('_')[2] in fi.stem.split('_')[2]) and (
                                idx.split('_')[3] == fi.stem.split('_')[3]):
                                match = fi
                    logger.debug(f'match:{match}')
                    processing_db.at[idx,'brdf_id']=match
                    if match:
                        if match.split('_')[0] == 'L3B':
                            processing_db.at[idx,'bandpass']=True
                        elif match.split('_')[0] == 'L3A':
                            processing_db.at[idx,'bandpass']=False
                
                modified = True
            
            num_coreged_files = len(list(Path(brdf_path).glob('*coreg.nc')))
            logger.info(f'{num_coreged_files} images have been coreged')
            if num_coreged_files == 0:
                logger.info('coregistration has not yet occured. Processing database is up to date')
            else:
                if 'shift_x' in processing_db:
                    logger.info('coreg data has already been added to database')
                else:
                    logger.info('adding coreg info to db...')
                    processing_db['coreg'] = np.nan
                    processing_db['shift_x'] = np.nan
                    processing_db['shift_y'] = np.nan
                    processing_db['coreg_error'] = np.nan
                    for idx, row in processing_db.iterrows():
                        match=None
                        logger.debug(f'idx:{idx} \n')
                        for fi in list(Path(brdf_path).glob('*.nc')):
                            if idx.startswith('S'):
                                if (idx.split('_')[1] in fi.stem.split('_')[2]) and (idx.split('_')[2] == fi.stem.split('_')[3]):
                                     match = fi 
                            elif idx.startswith('L'): 
                                if (idx.split('_')[0] == fi.stem.split('_')[1]) and (idx.split('_')[2] in fi.stem.split('_')[2]) and (
                                    idx.split('_')[3] == fi.stem.split('_')[3]):
                                    match = fi
                        logger.debug(f'match:{match}')
                        if match:
                            if 'coreg' in match:
                                processing_db.at[idx,'coreg']=True
                            elif match.endswith('X.nc'):
                                processing_db.at[idx,'coreg']=False
                                processing_db.at[idx,'coreg_error']='unknown'
                            else:
                                processing_db.at[idx,'coreg']='NaN'                           
                    modified = True                        
        else:
            logger.info('brdfs have not yet been created. Processing database is up to date')

        if modified:
            pd.to_pickle(processing_db, processing_info_path)
            logger.info('saving new database')
        
        return processing_db


def get_img_list_from_db(in_dir, grid_cell, img_type, yrs=None, data_source='stac'):
    """
        returns list of images in database for year range (yrs) for selected directory (raw or brdf)
        and sensor ('L' for Landsat, 'S2' for Sentinel, 'LS2' for both).
        yrs is a list in format [YYYY, YYYY] -- (if [YYYYdoy, YYYYdoy] or [YYYY-MM-dd, YYYY-MMM-dd], etc., will only use YYYY parts)
    """
    
    if data_source == 'stac':
        scene_info_combo = (Path(in_dir)/f"{int(grid_cell):06d}/processing.info")
        if scene_info_combo.is_dir(): 
            df = read_db(scene_info_combo,'current')
            if yrs:
                df_out0 = df[(df['date']>=int('{yrs[0][:4]}0101')) & (df['date']<int('{yrs[1][:4]}0101'))]
        else:
            ## The following is all for backwards compatibility for files processed with previous versions of eostac
            if 'brdf' not in str(in_dir):
            #original downloads for stac data are in separate landsat and sentinel2 folders. Each has its own scene.info file
                if img_type.lower().startswith('l'):
                    scene_info_l = (Path(in_dir)/f"{int(grid_cell):06d}/landsat/scene.info")
                    if not scene_info_l.is_dir():
                        logger.warning(f'There is no scene.info file in the Landsat directory for cell {grid_cell}')
                        landsat_df = None
                    else: landsat_df = read_db(scene_info_l,'old')
                    if img_type == 'LS2':
                        scene_info_s = (Path(in_dir)/f"{int(grid_cell):06d}/sentinel2/scene.info")
                        if not scene_info_s.is_dir():
                            logger.warning(f"There is no scene.info file in the Sentinel directory for cell {grid_cell}")
                            sentinel_df = None
                        else:
                            sentinel_df = read_db(scene_info_s,'old')
                            if not landsat_df:
                                df = sentinel_df
                            else:
                                df = pd.concat([landsat_df,sentinel_df],axis=0)
                    else:
                        df = landsat_df        
                elif img_type == 'S2':
                    scene_info_s = (Path(in_dir)/f"{int(grid_cell):06d}/sentinel2/scene.info")
                    if not scene_info_s.is_dir():
                        logger.warning(f"There is no scene.info file in the Sentinel directory for cell {grid_cell}")
                    else: df = read_db(scene_info_s,'old')
                else:
                    logger.warning(f"current image types are LS2, S2, L, lt05, le07, lc08, lc09. You put: {img_type}")           
            else:
                scene_info = Path(in_dir)/'scene.info'
                df = read_db(scene_info,'old')
            if yrs:
                df_out0 = df[f'{yrs[0]}-01-01:{yrs[1]}-12-31']
        
        if img_type !='LS2':
            logger.info(f'filtering returned dataset to {img_type}...')
            if len(img_type) == 1:
                df_out = df_out0.loc[df_out0.sensor.str.startswith(img_type.lower())]
            else:
                df_out = df_out0.loc[df.sensor==img_type]
        else:
            df_out = df_out0

    else: 
        logger.warning(f"need to add method to script for data source {data_source}")

    return df_out

def separate_missing_db_files(grid_cell, params):

    ppaths = ProjectPaths(params, grid=grid_cell)
    img_type = params['img_type']
    data_source = params['dlMethod'] 
    yrs = params['status']['period']
    ts_type = params['status']['ts_type']

    if ts_type == 'raw':
        in_dir = ppaths.ms.parent
        df = get_img_list_from_db(in_dir, grid_cell, img_type, yrs, data_source)          
        df['file_path'] = df.out_id.apply(lambda x: in_dir.joinpath(f'{x}'))
    elif ts_type == 'unprocessed':
        if img_type.lower().startswith('l'):
            in_dir = ppaths.ms.parent/'landsat'
        elif img_type == 'S2':
            in_dir = ppaths.ms.parent/'sentinel2'    
        df['file_path'] = df.id.apply(lambda x:in_dir.joinpath(f'{x}.tif'))

    df['file_path_exists'] = df.file_path.apply(lambda x: Path(x).is_file())
    df_existing = df.loc[df.file_path_exists]
    df_missing = df.loc[~df.file_path_exists]
    
    return df_existing, df_missing


def get_valid_pix_per(img_path):
    """
    returns the percent (as integer 0-100) of pixels with data for the input image
    for original .tif files and original or processed .nc files
    returns -99 if no file is found that matches the request
        (this is so that all brdfs are not marked as bad if this is run when 
        original file is no longer available (e.g. after cleaning downloads)
    """
    
    logger.info(f'getting pix count for {img_path}')

    if isinstance(img_path, str):
        img_path = Path(img_path)
        
    if img_path.suffix == '.tif':
        ## .tif file names should always be the same as the original download
        if img_path.exists():
            with rio.open(img_path) as src:
                no_data = src.nodata
                img = src.read(4)
            allpix = img.shape[0]*img.shape[1]
            nanpix = np.count_nonzero(np.isnan(img))
            #nanpix = np.count_nonzero(no_data)
            validpix = allpix-nanpix
        else:
            logger.warning(f'{img_path} no longer exists')
        
    elif img_path.suffix == '.nc':
        ## file might not be exact (e.g. if coreg has been run, coreg info is added at end of brdf file names)
        img_base = img_path.stem
        brdf_match = list(img_path.parent.glob(f'{img_base}*'))
        if len(brdf_match) > 0:
            with xr.open_dataset(brdf_match[0]) as xrimg:
                xr_idx = xrimg['nir']
            xr_idx_valid = xr_idx.where(xr_idx < 10000)
            allpix = xr_idx.count() 
            validpix = xr_idx_valid.count()
        else:
            logger.warning(f' no brdf file found for {img_path}')
        
    try:
        if allpix == 0:
            validper = 0
        else:
            #validper = int(100*validpix/float(allpix))
            ## above checks if > min percent of scene area is not NaN, but scene area can be a sliver of cell area 
            ## below will ensure that > min percent of entire cell is coevered 
            ##   (note there is actually a bit more than 4,000,000 pixels per scene depending on buffer size)  
            validper = int(100*validpix/float(4000000))  
        return validper
    
    except:
        return -99

def check_valid_pixels(ppaths, grid, params, check_missing_files=False):
    """
    Adds fields 'ValidPix_orig' and 'ValidPix_brdf' to processing database
    """
    
    if check_missing_files:
        df_brdf = separate_missing_db_files(ppaths, grid, params)[0]
    else:
        df_brdf = pd.read_pickle(ppaths.ms.parent.joinpath('processing.info'))
    
    df_brdf['ValidPix_orig'] = df_brdf.apply(lambda x: get_valid_pix_per(x['dl']), axis=1)
    df_brdf['ValidPix_brdf'] = df_brdf.apply(lambda x: -99 if x['brdf'] !=True else 
                                             get_valid_pix_per(ppaths.ms.joinpath(x['brdf_id'])),axis=1)
    out_df = ppaths.ms.parent.joinpath(f'processed_imgs_{grid}.info')
    df_brdf.to_pickle(out_df)
        
    return df_brdf

def move_nodata(params):

    """
    Moves or flags images with little to no data.
    If 'move_method' is set to 'nodata_folder' in parameters, will send to designated folder.
         otherwise, if set to 'flag_?', will flag brdf file by adding ? to the end of the file basename.
         (? can be anything, but should probably end with an X to be consistent with later processing methods)
    Can specify what qualifies as little data with filesize (if using 'filesize' method) 
         or percent valid pixels (if using 'pixel_check' method) 
    This targets the brdf directory, as brdfs are already processed before bringing in tstuyau
    Will flag brdfs that do not match with original pixel count for optional reprocessing
    Marks flagged images skipped in processing database
    """
    with gw.config.update(sensor='s2l7'):
        
        for grid in params['grids']:
            
            ppaths = ProjectPaths(params, grid=grid)
            
            if not ppaths.ms.is_dir():
                logger.warning(f"  The BRDF directory for grid {grid} does not exist.")
                continue
                
            if not ppaths.ms.parent.joinpath('processing.info').is_file:
                processing_db = pd.read_pickle(ppaths.ms.parent.joinpath('processing.info'))
                logger.info(f'found processing db with {processing_db.shape[0]} enteries')
            else:
                processing_db = None
                #TODO: make new one with method above 

            image_list = list(ppaths.ms.glob('L*.nc'))
            if not image_list:
                logger.warning(f"  No brdf images found for grid {grid}.")
                continue
                
            logger.info(f"  Checking grid {grid} ...")
            
            if params['move_no_data']['id_method']=='filesize':
                logger.info(f"moving all files with filesize < {params['move_no_data']['filesize']}")
                to_move = [fn for fn in image_list if fn.stat().st_size < params['move_no_data']['filesize']]
                ## TODO: mark in db['skip'] and [skip_reason] in processing database

            elif params ['move_no_data']['id_method']=='pixel_check':
                ## check pixel count for both download and brdf to check whether processing was bad for valid download data
                db = check_valid_pixels(ppaths, grid, params)
                
                ## check where brdf pixels != download pixels
                ## TODO: fix 'ValidPix_orig' for Sentinel images (always finding 100%) and remove 'numpix' from these calcs
                ##   or can just use 'numpix', but is calculated at download stage and won't be avaiable if db missing/corrupted.
                db['rerun_brdf'] = db.apply(lambda x: True if ((x['numpix'] > 200000) & 
                                          ((x['ValidPix_orig'] + 1) / (x['ValidPix_brdf'] + 1) >= 2)) else False, axis=1)
                to_rerun = db[db['rerun_brdf']==True]
                to_rerun[['brdf_id']].to_csv(ppaths.ms.parent.joinpath('rerun_brdfs.csv'), index=False, header=False) 
                logger.info(f'flagging {to_rerun.shape[0]} failed brdfs in database to rerun')

                weak_images = db[((db['ValidPix_orig'] < params['move_no_data']['exclude_below']) & 
                                  (db['ValidPix_orig'] > -1)) | (db['numpix'] < 200000)]
                
                ## mark as excluded in database: TODO: remove hardcoded 200000 and remove numpix if Sentinel check is fixed
                db['skip']= np.where(((db['ValidPix_orig'] < params['move_no_data']['exclude_below']) 
                                      & (db['ValidPix_orig'] > -1)) | (db['numpix'] < 200000), True, db['skip'])
                db['skip_reason']= np.where(((db['ValidPix_orig'] < params['move_no_data']['exclude_below']) 
                                            & (db['ValidPix_orig'] > -1))| (db['numpix'] < 200000),'not enough valid data', db['skip_reason'])
                
                ## resave db (TODO: maybe make this safer)
                pd.to_pickle(db, ppaths.ms.parent.joinpath('processing.info'))
                
                ## match brdf records to actual files for images to be removed
                to_move = []
                for img in weak_images['brdf_id']:
                    if img:
                        logger.debug(img)
                        img_base = img.split('.')[0]
                        match = list(ppaths.ms.glob(f'{img_base}*'))
                        if len(match) > 0:
                            to_move.append(match[0])
                    
            elif params['move_no_data']['id_method'] == 'quarters':
                offset = int(params['move_no_data']['offset'])
                to_move = []
                for fn in image_list:
                    with gw.open(str(fn),
                        chunks=512) as src:
                        i = int(src.gw.nrows / 2)
                        j = int(src.gw.ncols / 2)
                    flagged = False
                    
                    # Check center block
                    mean_value = src.sel(band='red')[i-offset:i+offset, j-offset:j+offset].data\
                                        .mean().compute(num_workers=params['num_workers'])
                    if 1 <= mean_value < 20:
                        to_move.append(fn)  
                        flagged = True
                        
                    # Check the top quarter
                    if not flagged:
                        i = int(src.gw.nrows * 0.25)
                        mean_value = src.sel(band='red')[:i, :].data.mean().compute(num_workers=params['num_workers'])
                        if 1 <= mean_value < 20:
                            to_move.append(fn)
                            flagged = True

                    # Check the bottom quarter
                    if not flagged:
                        i = int(src.gw.nrows * 0.25)
                        mean_value = src.sel(band='red')[-i:, :].data.mean().compute(num_workers=params['num_workers'])
                        if 1 <= mean_value < 20:
                            to_move.append(fn)
                            flagged = True

                    # Check the left quarter
                    if not flagged:
                        j = int(src.gw.ncols * 0.25)
                        mean_value = src.sel(band='red')[:, :j].data.mean().compute(num_workers=params['num_workers'])
                        if 1 <= mean_value < 20:
                            to_move.append(fn)
                            flagged = True

                    # Check the right quarter
                    if not flagged:
                        j = int(src.gw.ncols * 0.25)
                        mean_value = src.sel(band='red')[:, -j:].data.mean().compute(num_workers=params['num_workers'])
                        if 1 <= mean_value < 20:
                            to_move.append(fn)
                            flagged = True
                          
            if params['move_no_data']['move_method'] == 'nodata_folder':
                nodata_dir = ppaths.ms.parent.joinpath('brdf_nodata')
                nodata_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f'moving {len(to_move)} files with little to no data into {nodata_dir}')
                for f in to_move:
                    f.replace(ppaths.nodata.joinpath(f.name))
                    #logger.info(f'moving {f.stem}')
                    #shutil.move(f,os.path.join(nodata_dir,fbase))  ## the above is cleaner. if it works remove this  
                    ## if in downloads, can remove angle files as well, but this is targeting brdf folder
                    #ppaths.ms.joinpath(f.name.replace('MTD.tif', 'MTD_angles.tif'))\
                    #.replace(ppaths.nodata.joinpath(f.name.replace('MTD.tif', 'MTD_angles.tif')))
                          
            elif params['move_no_data']['move_method'].startswith('flag'):
                for f in to_move:
                    alt= params['move_no_data']['move_method'].split('_')[1]
                    p = Path(f)
                    p.rename(Path(p.parent, f"{p.stem}_{alt}{p.suffix}"))
                
           
