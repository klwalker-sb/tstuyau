from pathlib import Path
from datetime import datetime
import concurrent.futures
import numpy as np
import pandas as pd
import rasterio as rio
from affine import Affine
#from rasterio.enums import Resampling as RioResampling
#from rasterio.warp import reproject
from rasterio.transform import from_bounds
import geowombat as gw
from geowombat.core import sort_images_by_date
import xarray as xr
import rastercrf as rcrf
from tqdm import tqdm

from .io import extract_ref_profile
from ..handler import logger
from ..db import TuyauDataBase
from . import utils
from .project import ProjectPaths
from .mask_utils import compute_cloud_crf, compute_cast_shadow, compute_self_shadow, buffer_mask
from .gee_ingest import IngestFromGoogle
from .constants import FILENAME_DATE_INDEX, FILENAME_DATE_INDEX_GEE, FILENAME_DATE_START_INDEX, FILENAME_DATE_END_INDEX
from .lookup import SENSORS, MASKS
from .processing_tracker import match_brdf_files_to_record

def update_db_mask_tracker(processing_db,image_id,mask_type):
    
    existing_masks = processing_db.loc[image_id, 'masking']
    if pd.isna(existing_masks) or existing_masks == 'native masks only':
        processing_db.loc[image_id, 'masking'] = mask_type
    elif mask_type in existing_masks:
        pass
    else:
        processing_db.loc[image_id, 'masking'] = processing_db.loc[image_id, 'masking'] + f';{mask_type}'

    return processing_db


def write_mask(arr, out_path, template=None, img_path=None, chunks=None,
               open_kwargs=None, nodata_val=255, dtype='uint8'):
    '''
    Writes a single-band mask array to out_path, matched to
    template's grid (crs, transform, width/height, block sizes).
 
    Can pass a pre-built <template> -- a gw.open()'d DataArray (band already dropped) whose
    crs, transform, and gw.ncols/nrows/col_chunks/row_chunks define the output grid/profile.
    (so the scene is only opened once), 
    or Can pass <img_path> (plus optional parameters to feed into gw
    '''
    if template is None:
        if img_path is None:
            logger.warning('write_mask needs either template or img_path')
            return
        open_kwargs = open_kwargs or {}
        if chunks is not None:
            open_kwargs = {'chunks': chunks, **open_kwargs}
        with gw.open(img_path, **open_kwargs) as src:
            template = src.isel(band=0, drop=True) if 'band' in src.dims else src
            template = template.squeeze(drop=True)

    out_profile = extract_ref_profile(template, count=1, nodata=nodata_val, dtype=dtype)
    arr = arr.astype(dtype)
 
    with rio.open(out_path, 'w', **out_profile) as dst:
        dst.write(arr, 1)


def mask_clouds_CRF(params, ppaths, processing_db):
    '''
    Method using conditional Random Fields trained on clouds, shadows, water, and clear land.
    (original code from jgrss)  TODO: integrate this into other code. 
    Currently saves masks with date names, so cannot be used along with individual images.  
    '''
    mask_args = MASKS['cloud_CRF']
    maskname = mask_args['maskname']
    
    # Setup the CRF object
    crf_clf_clouds = rcrf.CRFClassifier()
    crf_clf_shadows = rcrf.CRFClassifier()
    # lgb_clf = rcrf.LGBMClassifier()

    if params['masking']['deep_crf']:
        lcrf_clf = rcrf.LSTMCRFClassifier(params['masking']['predict_labels'], params['masking']['batch_size'])
        lstm_model_name = str(rcrf.model_path(params['masking']['lstm_model_name']))
        lcrf_clf.from_file(lstm_model_name)

    else:
        lcrf_clf = None

    # Get the full path to the model
    crf_cloud_model_name = str(rcrf.model_path(params['masking']['crf_cloud_model_name']))
    crf_shadow_model_name = str(rcrf.model_path(params['masking']['crf_shadow_model_name']))
    # lgb_model_name = str(rcrf.model_path(params['masking']['lgb_model_name']))

    # Load the model
    crf_clf_clouds.from_file(crf_cloud_model_name)
    crf_clf_shadows.from_file(crf_shadow_model_name)
    # lgb_clf.from_file(lgb_model_name)
    lgb_clf = None

    pred_kwargs = dict(count=1,
                        dtype='uint8',
                        nodata=255,
                        driver='GTiff',
                        tiled=True,
                        compress='lzw')

    if params['dlMethod'] == 'GEE':
        date_pos=FILENAME_DATE_INDEX_GEE
        prepend_str='netcdf:'
    else:
        date_pos=FILENAME_DATE_INDEX
        prepend_str=''

    # Get a list of the input images
    sensors = params['masking']['sat_sensors']
    if isinstance(sensors,list):
        if (any(s.startswith('S2') for s in sensors)) and (any(s.startswith('L') for s in sensors)):
            sensor='LS2'
        else:
            sensor = sensors[0]
    else:
        sensor = sensors
    
    skip_flag = params['masking']['skip_flag']  
    
    if (sensor == 'LS2') or (sensor == 'All'):
        search_str = f'*.nc' if skip_flag is None else f"*[!{skip_flag}].nc"
    else:
        senstr = SENSORS[sensor]['matchstr']
        search_str = f'L3?_{senstr}*.nc' if skip_flag is None else f"L3?_{senstr}*[!{skip_flag}].nc"
        
    image_dict = sort_images_by_date(ppaths.ms,
                                    search_str,
                                    date_pos=date_pos,
                                    date_start=0,
                                    date_end=8,
                                    prepend_str=prepend_str)

    proc_names = list(image_dict.keys())
    #logger.info(f'image list = {proc_names}')

    l8_image = [fn for fn in proc_names if Path(fn).name.startswith('LC08')][-1]

    # Set the output kwargs
    with gw.open(f'{l8_image}:blue',
                    chunks=params['masking']['chunks']) as src:

        ref_bounds = src.gw.bounds

    pred_kwargs['crs'] = src.crs
    pred_kwargs['transform'] = src.transform
    pred_kwargs['blockxsize'] = src.gw.col_chunks
    pred_kwargs['blockysize'] = src.gw.row_chunks
    pred_kwargs['width'] = src.gw.ncols
    pred_kwargs['height'] = src.gw.nrows

    def time_generator(rpath_mask, full_time_list, batch_size):
        """This uses a sliding window of batch_size (with checks to ensure that same date images are together in a batch) to provide CRF model
        temporal context when deciding whether pixels are cloud or not. TODO: fix bug of same-date images getting same name (causing several problems)
        by using same naming convention as other methods.

        Note, processing dictionary is passed in here, for consistency, but it is not used. TODO: update at end as with other masking methods
        """
        for fidx in range(0, len(full_time_list)-batch_size):
            file_time_list = []
            image_dates = []
            yidx = 0

            # Fill the list until the number of unique items equals the batch size
            while len(list(set(image_dates))) < batch_size:
                if fidx+yidx+1 >= len(full_time_list):
                    break

                fn = full_time_list[fidx+yidx]
                yidx += 1
                try:
                    with gw.open(f'{fn}:swir2', chunks=params['masking']['chunks']) as src:
                        pass
                except:
                    continue
                # The image date
                fn_dt = datetime.strptime(Path(fn).name.split('_')[3][:8], '%Y%m%d')
                file_time_list.append(fn)
                image_dates.append(fn_dt)

            # Continue to check the end for additional duplicates
            while True:
                if fidx+yidx+1 >= len(full_time_list):
                    break
                fn = full_time_list[fidx+yidx]
                yidx += 1
                try:
                    with gw.open(f'{fn}:swir2', chunks=params['masking']['chunks']) as src:
                        pass
                except:
                    continue
                fn_dt = datetime.strptime(Path(fn).name.split('_')[3][:8], '%Y%m%d')
                if len(list(set(image_dates + [fn_dt]))) > batch_size:
                    break
                file_time_list.append(fn)
                image_dates.append(fn_dt)
            # Check if the files in the batch have been processed
            existing_files = []
            future_files = []

            for fn, fn_dt in zip(file_time_list, image_dates):
                outfile = rpath_mask / f'{fn_dt.year}{fn_dt.month:02d}{fn_dt.day:02d}.tif'
                outfile = rpath_mask / f'{Path(fn).stem}_{maskname}.tif'
                existing_files.append(outfile.is_file())
                future_files.append(outfile)
            
            if not all(existing_files):
                yield file_time_list, image_dates, future_files
            else:
                yield None, None, None

        bfidx = 0
        # Backfill the end
        while True:
            file_time_list_ = full_time_list[len(full_time_list)-batch_size-bfidx:len(full_time_list)]
            file_time_list = []

            for fn in file_time_list_:
                try:
                    with gw.open(f'{fn}:swir2', chunks=params['masking']['chunks']) as src:
                        pass
                    file_time_list.append(fn)
                except:
                    pass
            image_dates = [datetime.strptime(Path(fn).name.split('_')[3][:8], '%Y%m%d') for fn in file_time_list]
            if len(list(set(file_time_list))) == batch_size:
                break
            bfidx -= 1

        existing_files = []
        future_files = []

        for fn, fn_dt in zip(file_time_list, image_dates):

            outfile = rpath_mask / f'{fn_dt.year}{fn_dt.month:02d}{fn_dt.day:02d}.tif'

            if params['masking']['overwrite']:
                if outfile.is_file():
                    outfile.unlink()

            existing_files.append(outfile.is_file())
            future_files.append(outfile)

        if not all(existing_files):
            yield file_time_list, image_dates, future_files
        else:
            yield None, None, None

        with rio.Env(GDAL_CACHEMAX=params['io']['gdal_cachemax']):

            with gw.config.update(sensor=params['masking']['sensor'],
                                  ref_bounds=ref_bounds,
                                  ref_res=params['masking']['ref_res'],
                                  ignore_warnings=True):

                futures = []

                with concurrent.futures.ProcessPoolExecutor(max_workers=params['num_workers']) as executor:

                    for image_batch_list, dates_batch_list, future_files in time_generator(ppaths.masks,
                                                                                           proc_names,
                                                                                           params['masking']['batch_size']):

                        if image_batch_list:

                            f = executor.submit(mask_data,
                                                image_batch_list,
                                                dates_batch_list,
                                                params['masking']['chunks'],
                                                params['nodata'],
                                                params['masking']['resampling'],
                                                future_files,
                                                crf_clf_clouds,
                                                crf_clf_shadows,
                                                lgb_clf,
                                                lcrf_clf,
                                                deep_crf=params['masking']['deep_crf'],
                                                band_names=params['masking']['band_names'],
                                                sensor=params['masking']['sensor'],
                                                cloud_labels=params['masking']['cloud_labels'],
                                                shadow_labels=params['masking']['shadow_labels'],
                                                num_workers=1,
                                                cloud_proba_thresh=params['masking']['cloud_proba_thresh'],
                                                shadow_proba_thresh=params['masking']['shadow_proba_thresh'],
                                                pred_kwargs=pred_kwargs)

                            futures.append(f)

                    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                        res = f.result()

def mask_clouds_s2cloudless(params, ppaths, grid, input_dir, processing_db, open_kwargs):
    '''
    Downloads s2cloudless masks from GEE and applies them to Sentinel-2 images.
 
    processing_db/mask_info accepted for consistency with the other
    masking functions; not yet used below since the thresholding
    logic itself is still a TODO (unchanged from the original).
    '''
    logger.info(f'  first downloading s2cloudless masks from GEE ...')
    mask_args = MASKS['s2cloudless']  ##TODO: use this
    params['image_type'] = ['S2cp']
    ig = IngestFromGoogle(verbose=1)
    gee = ig.ingest_from_gee(params, grid, ppaths)
    logger.info(f'  now applying masks to Sentinel images ...')

    skip_flag = params['masking']['skip_flag']

    db_col = mask_args['db_col']
    maskname = mask_args['maskname']
    out_dir = Path(input_dir).parent / mask_args['mask_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    file_to_scene = match_brdf_files_to_record(input_dir, processing_db, exclude=skip_flag, method=params['dlMethod'])
    logger.info(f'preparing masks for {len(file_to_scene)} images')
 
    n_ok, n_skip = 0, 0
    '''
    for img_path, scene_id in file_to_scene.items():
        try:
            with gw.open(img_path, chunks=chunks, **open_kwargs) as src:
                template = src.isel(band=0, drop=True) if 'band' in src.dims else src
                template = template.squeeze(drop=True)
            arr=
            # TODO: finish this here!
            ## mask pixels >95? 60? (100 is max prob of cloud, 255 is nodata)
            ## add s2cloudless_thresh in params
            ## also, consider cloud shadows as per
            ## https://towardsdatascience.com/creating-sentinel-2-truly-cloudless-mosaics-with-microsoft-planetary-computer-7392a2c0d96c/
            write_mask(arr, out_path, template=template, nodata_val=255)
   
            n_ok += 1
            processing_db.loc[scene_id, db_col] = str(out_path)
            update_db_mask_tracker(processing_db,scene_id,s2coudless')

            except Exception as e:
                logger.warning(f'Failed to make shade mask for {img_path.name}: {e}')
                n_skip += 1
                # Rename the file that failed masking with the skip_flag at the end
                p = Path(img_path)
                p.rename(p.parent / f'{p.stem}_MASKFAIL{skip_flag}{p.suffix}')
                processing_db.loc[scene_id, db_col] = 'ERROR'
 
    logger.info(f'Shade masks: {n_ok} created, {n_skip} skipped -> {out_dir}')
    '''

def mask_shade(params, input_dir, processing_db, open_kwargs):
    '''
    Computes a combined cast-shadow + self-shadow mask for each BRDF scene in this grid cell, using the cell's DEM and the per-scene
    sun zenith/azimuth already stored in processing_db (added by eostac).
    '''
    
    mask_args = MASKS['terrain_shade']
    db_col = mask_args['db_col']
    maskname = mask_args['maskname']
    out_dir = Path(input_dir).parent / mask_args['mask_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
 
    res = params['res']
    skip_flag = params['masking']['skip_flag']
    
    dem_path = Path(input_dir).parent / 'terrain' / 'dem.tif'
    
    file_to_scene = match_brdf_files_to_record(input_dir, processing_db, exclude=skip_flag, method=params['dlMethod'])
    logger.info(f'preparing masks for {len(file_to_scene)} images')
 
    n_ok, n_skip = 0, 0
    for img_path, scene_id in file_to_scene.items():
 
        row = processing_db.loc[scene_id]
        zenith = row.get('sun_zenith')
        azimuth = row.get('sun_azimuth')
        if pd.isna(zenith) or pd.isna(azimuth):
            logger.warning(f'No sun angles for {scene_id}; skipping {img_path.name}')
            n_skip += 1
            continue
         
        if 'coreg' in str(img_path):
            out_path = out_dir / f"{img_path.stem.split('_coreg')[0]}_{maskname}.tif"
        elif '_.nc' in str(img_path):
             out_path = out_dir / f"{img_path.name.split('_.nc')[0]}_{maskname}.tif"
        else: 
            out_path = out_dir / f"{img_path.stem}_{maskname}.tif"
            
        try:
            with gw.open(img_path, **open_kwargs) as src:
                template = src.isel(band=0, drop=True) if 'band' in src.dims else src
                template = template.squeeze(drop=True)
                cellsize = float(res)

            with gw.config.update(
                ref_bounds=template.gw.bounds,
                ref_crs=template.crs,
                ref_res=res,
                nodata=255,
                ignore_warnings=True,
             ):
                with gw.open(dem_path, resampling='bilinear') as dem_src:
                    dem_da = dem_src.squeeze(drop=True).load()
                dem = dem_da.values.astype(np.float32)
 
                cast_shadow = compute_cast_shadow(float(zenith), float(azimuth), dem=dem, cellsize=cellsize)
                self_shadow = compute_self_shadow(float(zenith), float(azimuth), dem_da=dem_da)
                shadow_mask = cast_shadow | self_shadow
                shadow_mask = buffer_mask(shadow_mask, buffer_px=params['masking']['buffer_px'])
                shadow_mask = shadow_mask.astype('uint8')

                write_mask(shadow_mask, out_path, template=template, nodata_val=255)
            
            n_ok += 1
            processing_db.loc[scene_id, db_col] = str(out_path)
            update_db_mask_tracker(processing_db,scene_id,'shade')

            
        except Exception as e:
            logger.warning(f'Failed to make shade mask for {img_path.name}: {e}')
            n_skip += 1
            # Rename the file that failed masking with the skip_flag at the end
            p = Path(img_path)
            p.rename(p.parent / f'{p.stem}_MASKFAIL{skip_flag}{p.suffix}')
            processing_db.loc[scene_id, db_col] = 'ERROR'
 
    logger.info(f'Shade masks: {n_ok} created, {n_skip} skipped -> {out_dir}')

                        
def make_masks(params):
    """
    Creates masks for clouds (and cloud shadows) or shaded terrain

    if <params.masking.method> is 'CRF", uses Conditional Random Field method above to mask clouds and cloud shadows. 
    if <params.masking.method> is 's2cloudless", uses s2cloudless masks from GEE to mask clouds and cloud shadows.
    if <params.masking.method> is 'terrain_shade' Computes a combined cast-shadow + self-shadow mask based on a DEM file and the 
            per-scene sun zenith/azimuth already stored in processing.info (from processes in eostac)
    otherwise, uses native masks (does nothing here) and adds 'native masks only' to 'masking' field in Processing info.
    """   

    mask_method = params['masking']['method']
    
    input_img_dir = 'brdf'
    if params['masking']['img_dir']:
        if (params['masking']['img_dir'] !='brdf') & (params['masking']['img_dir'] !='None'):
            input_img_dir = params['masking']['img_dir']    
    else:
        params['masking']['img_dir'] = 'brdf'

    n_chunks = 512
    if params['reconstruct']['chunks']:
        params['reconstruct']['chunks']
    
    open_kwargs = (
        {'engine': 'h5netcdf', 'chunks': {'band': -1, 'y': n_chunks, 'x': n_chunks}} 
        if input_img_dir == 'brdf' 
        else {'chunks': {'band': 1, 'y': n_chunks, 'x': n_chunks}}
    )
                
    cells = utils.get_cell_list_from_grid_param(params['grids'])
    for grid in cells:

        ppaths = ProjectPaths(params, grid=grid)
        if input_img_dir == 'brdf':
            img_dir = ppaths.ms
        else:
            img_dir = ppaths.ms.parent/input_img_dir

        processing_db = pd.read_pickle(ppaths.ms.parent/'processing.info')
        db_col = MASKS[mask_method]['db_col']
        if db_col not in processing_db:  ## This is the master db that tracks the progress of each individual image
                processing_db[db_col] = np.nan
        if 'masking' not in processing_db:
            processing_db['masking'] = np.nan

        ## This is a general process db that tracks what processes have been run for each cell
        db = TuyauDataBase(str(ppaths.ms.parent / f'{int(grid):06d}.db'))

        check_download_db = params['status']['check_downloads']
        
        if check_download_db:
            msensors = params['masking']['sat_sensors']
            if isinstance(msensors, str):
                msensors = [msensors] 
            if any (s in msensors for s in ['All', 'AllRaw', 'LS2']):
                msensors = ['S2','S2cp','LT05','LE07','LC08','LC09']
            elif 'L' in msensors:
                msensors = ['LT05','LE07','LC08','LC09']
            elif 'S' in msensors:
                msensors = ['S2']
            
            ## Check downloads TODO: update for stac method
            for sen in msensors:
                senlab = SENSORS[sen]['sensor']
                senpath = SENSORS[sen]['GEE']
                if not db.eosvault_is_complete(senlab, senpath):
                    logger.warning(f'  The {senlab} {senpath} downloads for grid {grid} are incomplete.')
                    continue
            # Check post-processing
            if ppaths.gee.is_dir():
                ## note: we are excluding files that end with 'angles' and 'cloudless' from the glob by excluding words that 
                ##    end in s. For more precise method, might need to use list version:
                ## e.g. [f for f in ppaths.gee if 'LT05' in os.path.basename(f) and "angles" not in os.path.basename(f)]
                if any (s in msensors for s in ['L','All','LS2','LT05']):
                    if list(ppaths.gee.glob('LT05*[!s].nc')):
                        logger.warning(f'  The LT05 post-processing for grid {grid} is incomplete.')
                        continue

                if any (s in msensors for s in ['L','All','LS2','LE07']):
                    if list(ppaths.gee.glob('LE07*[!_s].nc')):
                        logger.warning(f'  The LE07 post-processing for grid {grid} is incomplete.')
                        continue

                if any (s in msensors for s in ['L','All','LS2','LC08']):   
                    if list(ppaths.gee.glob('LC08*[!_s].nc')):
                        logger.warning(f'  The LC08 post-processing for grid {grid} is incomplete.')
                        continue

                if any (s in msensors for s in ['L','All','LS2','LC09']):  
                    if list(ppaths.gee.glob('LC09*[!_s].nc')):
                        logger.warning(f'  The LC09 post-processing for grid {grid} is incomplete.')
                        continue

                if any (s in msensors for s in ['S', 'All','LS2','S2']):  
                    if list(ppaths.gee.glob('L1C*[!_s].nc')):
                        logger.warning(f'  The S-2 L1C post-processing for grid {grid} is incomplete.')
                        continue

        if not db.table_exists:
            db.remove()
            db.create(exists_ok=True)
            db.insert(grid)

        if params['status']['reset_db'] or params['masking']['overwrite']:
            db.reset(grid, 'mask')
            
        # Check if the step is complete
        if db.is_complete(grid, 'mask'):
            logger.warning(f'  The masking step is complete.')
            continue

        logger.info(f'  Masks being created for grid {grid} ...')
        
        if not img_dir.is_dir():
            logger.warning(f' Directory: {str(img_dir)} for grid {grid} does not exist.')
            continue
        
        if params['masking']['retry_masks'] == True:
            logger.info(f'resetting file names in {str(img_dir)} to retry masking')
            failed_masks = list(img_dir.glob('*MASKFAILX*.nc'))
            for f in failed_masks:
                clean_name = f.name.replace("_MASKFAILX", "")
                f.rename(f.with_name(clean_name))
            
        if mask_method == 'cloud_CRF':    
            mask_clouds_CRF(params, ppaths, processing_db)

            mask_clouds_s2cloudless(params, ppaths, grid, img_dir, processing_db, open_kwargs)
        
        elif mask_method == 'terrain_shade':
            mask_shade(params, img_dir, processing_db, open_kwargs)

        else:
            logger.warning(f'ERROR: unrecognized mask method: {mask_method}') 
            
        processing_db['masking'] = processing_db['masking'].fillna('native masks only')
        pd.to_pickle(processing_db, ppaths.ms.parent / 'processing.info')
        db.update(grid, 'mask')
