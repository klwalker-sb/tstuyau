from pathlib import Path
from datetime import datetime
import concurrent.futures

from ..handler import logger
from ..db import TuyauDataBase
from .project import ProjectPaths
from .mask_utils import mask_data
from .gee_ingest import IngestFromGoogle
from .constants import FILENAME_DATE_INDEX, FILENAME_DATE_INDEX_GEE, FILENAME_DATE_START_INDEX, FILENAME_DATE_END_INDEX
from .lookup import SENSORS
import numpy as np
import pandas as pd
import rasterio as rio
import geowombat as gw
from geowombat.core import sort_images_by_date
import rastercrf as rcrf
from tqdm import tqdm
            
def mask_clouds_CRF(params, ppaths):
    '''
    Method using conditional Random Fields trained on clouds, shadows, water, and clear land.
    (original code from jgrss)
    '''
    
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

    # Get a list of the co-registered images
    sensors = params['masking']['sat_sensors']
    if isinstance(sensors,list):
        if (any(s.startswith('S2') for s in sensors)) and (any(s.startswith('L') for s in sensors)):
            sensor='LS2'
        else:
            sensor = sensors[0]
    else:
        sensor = sensors
    
    skip_flag = params['reconstruct']['exclude']  
    
    if (sensor == 'LS2') or (sensor == 'All'):
        search_str = f"*[!{skip_flag}].nc"
    else:
        senstr = SENSORS[sensor]['matchstr']
        search_str = f"L3?_{senstr}*[!{skip_flag}].nc"
        
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

                existing_files.append(outfile.is_file())
                future_files.append(outfile)

            if not all(existing_files):
                yield file_time_list, image_dates, sorted(list(set(future_files)))
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
            yield file_time_list, image_dates, sorted(list(set(future_files)))
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

                        
def mask_clouds(params):

    """
    Masks clouds and cloud shadows

    if params['masking']['method'] is 'CRF", uses Conditional Random Field method above. 
    if params['masking']['method'] is 's2cloudless", uses s2cloudless masks from GEE.
    otherwise, uses native masks (does nothing here) and adds 'native masks only' to 'masking' field in Processing info.
    """   
    for grid in params['grids']:

        ppaths = ProjectPaths(params, grid=grid)

        processing_db = pd.read_pickle(ppaths.ms.parent/'processing.info')
        if 'masking' not in processing_db:
            ## This is the master db that tracks the progress of each individual image
             processing_db['masking'] = np.nan

        ## This is a general process db that tracks what processes have been run for each cell
        db = TuyauDataBase(str(ppaths.ms.parent / f'{int(grid):06d}.db'))
        
        msensors = params['masking']['sat_sensors']
        if isinstance(msensors, str):
            msensors = [msensors] 
        if any (s in msensors for s in ['All', 'AllRaw', 'LS2']):
            msensors = ['S2','S2cp','LT05','LE07','LC08','LC09']
        elif 'L' in msensors:
            msensors = ['LT05','LE07','LC08','LC09']
        elif 'S' in msensors:
            msensors = ['S2']

        if not params['status']['check_downloads']:
            check_download_db = False
        else:
            check_download_db = True

        if check_download_db:
            ## Check downloads
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
            logger.warning(f'  The cloud mask step is complete.')
            continue

        logger.info(f'  Masks being created for grid {grid} ...')

        if params['dlMethod'] == 'GEE':
            date_pos=FILENAME_DATE_INDEX_GEE
            prepend_str='netcdf:'
        else:
            date_pos=FILENAME_DATE_INDEX
            prepend_str=''

        if params['masking']['method'] == 'CRF':
            
            if not ppaths.ms.is_dir():
                logger.warning(f' The BRDF directory for grid {grid} does not exist.')
                continue
                        
            mask_clouds_CRF(params)

        elif params['masking']['method'] == 's2cloudless':
            logger.info(f'  first downloading s2cloudless masks from GEE ...')
            params['image_type'] = ['S2cp']
            ig = IngestFromGoogle(verbose=1)
            gee = ig.ingest_from_gee(params, grid, ppaths)
            logger.info(f'  now applyting masks to Sentinel images ...')

            #TODO: finish this here!
            ## mask pixels >95? 60? (100 is max prob of cloud, 255 is nodata)
            ## add s2cloudless_thresh in params
            ## add shadow masks eg: 
            ## https://towardsdatascience.com/creating-sentinel-2-truly-cloudless-mosaics-with-microsoft-planetary-computer-7392a2c0d96c/
        db.update(grid, 'mask')
