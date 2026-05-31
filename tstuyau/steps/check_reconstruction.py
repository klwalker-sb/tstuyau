from pathlib import Path
from datetime import datetime, timedelta
import concurrent.futures

from ..handler import logger
from ..db import TuyauDataBase
from .project import ProjectPaths, get_tsdir_name
from .time_series_utils import smooth
from .io import TimeSeriesLoader
from .constants import FILENAME_DATE_INDEX, FILENAME_DATE_START_INDEX, FILENAME_DATE_END_INDEX, FILENAME_DATE_INDEX_GEE
from .lookup import SENSORS
from .spec_indices import SI_DICT
from .date_utils import get_date_range
from . import prechecks

import geowombat as gw
from geowombat.core import sort_images_by_date
from geowombat.core.windows import get_window_offsets

import numpy as np
import pandas as pd
import rasterio as rio
import yaml
from tqdm import tqdm
import sys

LANDSAT_LIKE_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

def _write_layer(layer, w, date, ts_dir, pad, profile, tags):
    layer_name = f'{int(date.year):d}{int(date.timetuple().tm_yday):03d}.tif'
    output_layer = ts_dir / layer_name

    if not output_layer.is_file():
        # Create the file
        with rio.open(output_layer, mode='w', **profile) as dst:
            if tags:
                dst.update_tags(**tags)

    # Difference between padded and regular windows
    w_row_offset = abs(w[0].row_off - w[1].row_off)
    w_col_offset = abs(w[0].col_off - w[1].col_off)

    with rio.open(output_layer, mode='r+') as dst:
        dst.write(
            np.uint16(
                layer[
                    w_row_offset:w_row_offset+w[0].height,
                    w_col_offset:w_col_offset+w[0].width
                ]
            ) if isinstance(pad, int) else np.uint16(layer),
            indexes=1,
            window=w[0]
        )

    return layer
    #return None


def _update_progress(ts_dir, grid, params, chunk_id):
    ''''
    Sets the window tracker at the last chunk run for an image
    ''' 
    with open(ts_dir / f'{grid:06d}.window', mode='r') as pf:
        window_tracker = yaml.load(pf, Loader=yaml.FullLoader)

    window_tracker[params['reconstruct']['chunks']]['latest'] = chunk_id

    with open(ts_dir / f'{grid:06d}.window', mode='w') as pf:
        yaml.dump(window_tracker, pf, default_flow_style=False)

def reconstruct(params):
    """
    Calculates indices and reconstructs a time series into weekly intervals using various data smoothing and gap filling options.
       output rasters are integers (val*10000) named YYYYdoy.tif stored in the time series directory for the cell (ppaths.ts) 
    If spec_index has 'raw': just calculates indices directly (no smoothing) and outputs rasters 
       for all input images in a defined time period into a scratch folder (ppaths.scratch)
           the time period is defined based on the first variable set in params['feature_model']['si_vars'] (_yr, _wet, or _dry), 
           the calendar parameters to define the sesaons, and the model year (params['feature_model']['start_yr']).
        
    vegetation indices are defined in SpecIndices in spec_indices.py (called through TimeSeriesLoader) and io.py. 
        any new index need to be added to the SI_DICT as well as SpecIndices methods.
    """
    
    si = params['reconstruct']['si']
    #if '.' in si:
    #    si_extraparam = si.split('.')[1].split('-')[0]
    #else:
    #    si_extraparam = None
    if si.endswith('raw'):
        params['reconstruct']['merge_ts'] = False
        season = params['feature_model']['si_vars'][0].split('-')[1] 
        params['reconstruct']['start'], params['reconstruct']['end'] = get_date_range(params['feature_model']['start_yr'],season,params)
        if params['feature_model']['pheno_pad_days']:
            params['reconstruct']['start_pad'] = (datetime.strptime(params['reconstruct']['start'], '%Y-%m-%d') 
                                                  - timedelta(days=params['feature_model']['pheno_pad_days'][0])).strftime("%Y-%m-%d")
            params['reconstruct']['end_pad'] = (datetime.strptime(params['reconstruct']['end'], '%Y-%m-%d') 
                                                  + timedelta(days=params['feature_model']['pheno_pad_days'][1])).strftime("%Y-%m-%d")
        else:
            params['reconstruct']['start_pad'] = params['reconstruct']['start']
            params['reconstruct']['end_pad'] = params['reconstruct']['end']
        logger.info(f"calculating index for images from {params['reconstruct']['start_pad']} to {params['reconstruct']['end_pad']}")

    for grid in params['grids']:
        ppaths = ProjectPaths(params, grid=grid)
        db = TuyauDataBase(str(ppaths.ms.parent / f'{int(grid):06d}_tuyau.db'))

        if params['reconstruct']['use_masks']:
            # Check if the step is complete
            if not db.is_complete(grid, 'mask'):
                logger.warning(f'  The cloud mask step is not complete.')
                continue

        # Check if other pre-processing steps are complete
        if not db.is_complete(grid, 'preprocess'):
            logger.warning(f'  The pre-processing step is incomplete.')
            continue

        if not getattr(ppaths, 'proc').is_dir():
            logger.warning(f'  The input directory for grid {grid} does not exist.')
            continue

        if params['reconstruct']['use_masks']:
            prechecks.precheck_reconstruct(grid, ppaths, params)

        ## Set directory to store output ts data 
        ## root folder name is <img_type>-<res>-<procseq> if comparative models are being run, but parts are dropped for simplicity if not
        ts_root = get_tsdir_name(params)
        ## base ts folder is named <si>-<ts_type>.  
        if si.endswith('raw'):
            ## indices calculated from raw images are currently sent to the scratch drive 
            ##      because they cost nothing to recreate (UNLESS BRDFs are deleted!)
            ts_dir = ppaths.scratch / 'raw' / ts_root / si / str(params['feature_model']['start_yr'])
        elif params['reconstruct']['merge_ts']:
            ts_dir = ppaths.ts / si / 'sub_ts'
            ## clean out any reminant window tracker so that new data can be run
            if ts_dir.is_dir() and len(list(ts_dir.glob('*.tif'))) < 5:
                winfile = Path(ts_dir) /f'{int(grid):06d}.window'
                if winfile.is_file():
                    winfile.unlink()
        else:
            ts_dir = ppaths.ts / si

        ts_dir.mkdir(parents=True, exist_ok=True)
        
        ppaths.clean_temp(ts_dir)

        if params['reconstruct']['overwrite']:
            if (ts_dir / f'{grid:06d}.window').is_file():
                (ts_dir / f'{grid:06d}.window').unlink()

            if (ts_dir / f'{grid:06d}.reindex').is_file():
                (ts_dir / f'{grid:06d}.reindex').unlink()

            for file_path in ts_dir.glob('*.tif'):
                file_path.unlink()

        ## Check if the file is complete (only if rewrite_win parameter is set to False)
        if not params['reconstruct']['rewrite_win']:
            if (ts_dir / f'{grid:06d}.window').is_file():
                with open(ts_dir / f'{grid:06d}.window', mode='r') as pf:
                    window_tracker = yaml.load(pf, Loader=yaml.FullLoader)

                if int(window_tracker[params['reconstruct']['chunks']]['latest']) == 1e9:
                    logger.warning('  The reconstruct step is complete.')
                    continue

        ## Get images from final processing directory to process
        if params['dlMethod'] == 'GEE':
            date_pos=FILENAME_DATE_INDEX_GEE
            prepend_str='netcdf:'
        else:
            date_pos=FILENAME_DATE_INDEX
            prepend_str=''

        ## exclude should include 'X' but can add other letters. Will exclude anything ending with ANY letter in this string
        skip_flag = params['reconstruct']['exclude']  
        sensors = params['image_type']
        if isinstance(sensors,list):
            if (any(s.startswith('S2') for s in sensors)) and (any(s.startswith('L') for s in sensors)):
                sensor='LS2'
            else:
                sensor = sensors[0]
        else:
            sensor = sensors
        
        if (sensor == 'LS2') or (sensor == 'All'):
            search_str = f"*[!{skip_flag}].nc"
        else:
            senstr = SENSORS[sensor]['matchstr']
            search_str = f"L3?_{senstr}*[!{skip_flag}].nc"

        image_dict = sort_images_by_date(
            getattr(ppaths, 'proc'),
            search_str,
            date_start=FILENAME_DATE_START_INDEX,
            date_end=FILENAME_DATE_END_INDEX,
            date_pos=date_pos,
            prepend_str=prepend_str
        )

        img_names = list(image_dict.keys())
        logger.debug(f' ALL valid images in directory (not yet filtered to date): {img_names}')
        img_times = list(image_dict.values())

        l8_image = [str(fn) for fn in img_names if 'LC08' in Path(fn).name][-1]

        # Window padding for moving window smoothing
        if si.endswith('raw'):
            pad = 0
            pidx = 0
        elif params['reconstruct']['smooth_kwargs']['prefill_gaps']:
            pad = params['reconstruct']['smooth_kwargs']['prefill_wmax']
            pidx = 1
        else:
            pad = params['reconstruct']['smooth_kwargs']['k']*2 if params['reconstruct']['smooth_kwargs']['spt_smoothing'] else None
            pidx = 1 if params['reconstruct']['smooth_kwargs']['spt_smoothing'] else 0
            if isinstance(pad, int):
                if pad % 2 == 0:
                    pad += 1
        
        sidx = si
        ## sis may be passed in with parameters attached (e.g. savi.100 and/or with ts info (e.g. savi-raw or savi.100-raw). '_' is legacy only.
        if '.' in si:
            si = si.split('.')[0]
        else:
            if any(m in si for m in ('-', '_')):
                sidx = si.split(m)[0]
        if sidx in SI_DICT.keys():
            band_names = SI_DICT[sidx]['band_names']
        else:
            logger.exception(f"  Model does not recognize {sidx}. Supported spectral indices are {SI_DICT.keys()}")
            raise NameError

        with rio.Env(GDAL_CACHEMAX=params['io']['gdal_cachemax']):
            with gw.open(
                        l8_image,
                        band_names=LANDSAT_LIKE_BANDS,
                        chunks={'band' : -1,
                            'y' : params['reconstruct']['chunks'],
                            'x' : params['reconstruct']['chunks']},
                        engine='h5netcdf'
                    ) as src:
                ref_bounds = src.gw.bounds

                if isinstance(src.crs, str) and (src.crs.lower().startswith('epsg') or src.crs.lower().startswith(':epsg')):
                    ref_crs = int(src.crs.split(':')[-1])
                else:
                    ref_crs = src.crs

                profile = dict(
                    blockxsize=src.gw.col_chunks,
                    blockysize=src.gw.row_chunks,
                    crs=src.gw.crs_to_pyproj.to_wkt(),
                    transform=src.gw.affine,
                    driver='GTiff',
                    count=1,
                    height=src.gw.nrows,
                    width=src.gw.ncols,
                    nodata=0,
                    dtype='uint16',
                    compress='lzw',
                    tiled=True
                )

                windows = get_window_offsets(
                    src.gw.nrows,
                    src.gw.ncols,
                    params['reconstruct']['chunks'],
                    params['reconstruct']['chunks'],
                    return_as='list',
                    padding=(pad, pad, pad, pad)
                )

            ## Open the images into an array
            with gw.config.update(
                sensor='l7',
                ref_res=float(params['reconstruct']['res']),
                ref_bounds=ref_bounds,
                ref_crs=ref_crs,
                ignore_warnings=True
            ):
                # Iterate by window for memory management
                for widx, w in enumerate(windows):
                    rw = params['reconstruct']['rewrite_win']
                    logger.info(f'widx ={widx},index={si}')
                    logger.info(f'rw = {rw}')

                    if params['reconstruct']['rewrite_win']:
                        if widx < int(params['reconstruct']['start_win']):
                            continue

                        window_tracker = {params['reconstruct']['chunks']: {'latest': -999}}
                        with open(ts_dir / f'{grid:06d}.window', mode='w') as pf:
                            yaml.dump(window_tracker, pf, default_flow_style=False)

                    elif (ts_dir / f'{grid:06d}.window').is_file():
                        with open(ts_dir / f'{grid:06d}.window', mode='r') as pf:
                            window_tracker = yaml.load(pf, Loader=yaml.FullLoader)

                        if widx < int(window_tracker[params['reconstruct']['chunks']]['latest']):
                            continue

                    else:
                        window_tracker = {params['reconstruct']['chunks']: {'latest': -999}}
                        with open(ts_dir / f'{grid:06d}.window', mode='w') as pf:
                            yaml.dump(window_tracker, pf, default_flow_style=False)

                    logger.info(f'  Loading window data for window {widx+1} out of {len(windows)} ...')

                    # Get the chunk slice
                    slicer = (
                        slice(0, None),  # time
                        slice(0, None),  # bands
                        slice(w[pidx].row_off, w[pidx].row_off+w[pidx].height),
                        slice(w[pidx].col_off, w[pidx].col_off+w[pidx].width)
                    )

                    # Setup a DataFrame of all the image names and dates
                    time_band_df = pd.DataFrame(data=img_names,
                                                index=img_times,
                                                columns=['image_path'])
                    logger.debug(f'time_band_df for valid all images in directory (not yet filtered to date):\n {time_band_df}')

                    if params['reconstruct']['use_masks']:

                        # Get the potential mask images from the image dates
                        mask_images = sorted(list(set([str(ppaths.masks.joinpath(f'{fn_dt.year}{fn_dt.month:02d}{fn_dt.day:02d}.tif'))
                                                       for fn_dt in img_times])))

                        # Get existing mask files
                        mask_images = [fn for fn in mask_images if Path(fn).is_file()]

                        # Get the mask date from the filename
                        mask_times = [datetime.strptime(Path(fn).stem, '%Y%m%d') for fn in mask_images]

                        time_mask_df = pd.DataFrame(
                            data=mask_images,
                            index=mask_times,
                            columns=['image_path']
                        )

                    # Get the padding datetime
                    start_pad_dt = datetime.strptime(params['reconstruct']['start_pad'], '%Y-%m-%d')
                    end_pad_dt = datetime.strptime(params['reconstruct']['end_pad'], '%Y-%m-%d')

                    start_dt = datetime.strptime(params['reconstruct']['start'], '%Y-%m-%d')
                    end_dt = datetime.strptime(params['reconstruct']['end'], '%Y-%m-%d')
                    if not params['reconstruct']['skip_years']:
                        params['reconstruct']['skip_years'] == 1
                    if 'raw' not in si:
                        # Iterate over each annual slice
                        unique_yrs = sorted(list(set([dt.year for dt in time_band_df.index.to_pydatetime()])))
                    else:
                        unique_yrs = list(set([int(params['reconstruct']['start'][:4]),int(params['reconstruct']['end'][:4])])) 
                        logger.debug(f"unique yrs: are {unique_yrs} for {params['reconstruct']['start']} to {params['reconstruct']['end']}.")
                    
                    for yidx in range(0, len(unique_yrs), params['reconstruct']['skip_years']):
                        year = unique_yrs[yidx]
                        if year < start_pad_dt.year:
                            continue
                        if year > end_pad_dt.year:
                            continue
                        
                        logger.info(f'working on year {year}...') 
                            
                        # Padded datetimes
                        if start_dt.month - start_pad_dt.month >= 0:
                            start_pad_dt_slice = datetime.strptime(f'{year}-{start_pad_dt.month}-{start_pad_dt.day}', '%Y-%m-%d')
                        else:
                            start_pad_dt_slice = datetime.strptime(f'{year-1}-{start_pad_dt.month}-{start_pad_dt.day}', '%Y-%m-%d')
                        
                        if end_pad_dt.month - end_dt.month >= 0:
                            end_pad_dt_slice = datetime.strptime(f"{year+params['reconstruct']['skip_years']}-{end_pad_dt.month}-{end_pad_dt.day}", '%Y-%m-%d')
                        else:
                            end_pad_dt_slice = datetime.strptime(f"{year+params['reconstruct']['skip_years']+1}-{end_pad_dt.month}-{end_pad_dt.day}", '%Y-%m-%d')
                        
                        if end_pad_dt_slice > end_pad_dt:
                            end_pad_dt_slice = end_pad_dt
                    
                        # Un-padded datetimes
                        start_dt_slice = datetime.strptime(f'{year}-{start_dt.month}-{start_dt.day}', '%Y-%m-%d')
                        end_dt_slice = datetime.strptime(f"{year+params['reconstruct']['skip_years']}-{end_dt.month}-{end_dt.day}", '%Y-%m-%d')
                        
                        time_band_df_slice = time_band_df.loc[start_pad_dt_slice:end_pad_dt_slice]
                        #imgs_used = time_band_df_slice['image_path'].apply(lambda x: Path(x).stem)
                        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                            logger.debug(f"images for this time period: {time_band_df_slice['image_path'].apply(lambda x: Path(x).stem)} \n")
                        ## note: time_band_df_slice is all input images for the period. 
                        ##     duplicate dates are handled with time series loader and final set is in real_img_times below.

                        # if params['reconstruct']['use_masks']:
                        #
                        #     time_mask_df_slice = time_mask_df.loc[start_pad_dt_slice:end_pad_dt_slice]
                        #     mask_image_list = time_mask_df_slice.image_path.values.tolist()
                        #     mask_time_list = time_mask_df_slice.index.to_pydatetime().tolist()
                        #     mask_band_names = ['mask']
                        #
                        # else:
                        #
                        #     mask_image_list = time_band_df_slice.image_path.values.tolist()[:2]
                        #     mask_time_list = [1, 2]
                        #     mask_band_names = [band_names[0]]

                        #logger.info('loading tsl...')
                        tsl = TimeSeriesLoader(
                            time_band_df_slice,
                            start_pad_dt_slice,
                            end_pad_dt_slice,
                            band_names,
                            slicer,
                            params
                        )

                        if params['reconstruct']['load_on_cluster']:
                            real_img_times, y = tsl.load_on_cluster()
                        else:
                            # real_img_times, y = tsl.load()
                            real_img_times, y = tsl.load_netcdf()
                            
                        tags = {}
                        
                        if 'raw' not in si:
                            logger.info(f'  Smoothing data for grid {grid} ...')
                            # Smooth the data and re-grid to weekly intervals
                            xinfo, stack = smooth(
                            real_img_times,
                                y,
                                datetime.strftime(start_dt_slice, '%Y-%m-%d'),
                                datetime.strftime(end_dt_slice, '%Y-%m-%d'),
                                n_jobs=params['num_workers'],
                                **params['reconstruct']['smooth_kwargs']
                            )
                            out_data = zip(xinfo.skip_slice[xinfo.write_skip_idx],stack[xinfo.write_skip_idx])
                            for bidx, dt in enumerate(xinfo.skip_slice[xinfo.write_skip_idx]):
                                tags[f'BAND_{bidx+1:03d}'] = dt.strftime('%Y%m%d')   
                        
                        else:
                            ## using all of the inputs (no smoothing or reindexing) for raw calculations
                            s = 10000
                            if params['masking']['maxval']:
                                s = params['masking']['maxval']
                            out_data = zip(real_img_times,y * s)
                            for bidx, dt in enumerate(real_img_times):
                                tags[f'BAND_{bidx+1:03d}'] = dt.strftime('%Y%m%d')   
                                                    
                        logger.info(f'  Writing reconstructed results for grid {grid} ...')

                        ####################
                        # Write data to file
                        ####################
                        

                        with concurrent.futures.ProcessPoolExecutor(max_workers=params['num_workers']) as executor:
                            futures = [
                                executor.submit(
                                    _write_layer,
                                    layer,
                                    w,
                                    date,
                                    ts_dir,
                                    pad,
                                    profile,
                                    tags
                                ) for date, layer in out_data
                            ]
                            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                                logger.debug(f'f={f}')
                                res = f.result()
                                logger.debug(f'res = {res}')
                            #except Exception as ex:

                    _update_progress(ts_dir, grid, params, widx)

                    if params['reconstruct']['rewrite_win']:
                        if (params['reconstruct']['start_win'] + params['reconstruct']['win_batchsize']) == int(widx)+1:
                            if int(widx) < 16:
                                sys.exit(0)

                if 'raw' not in si:
                    _update_progress(ts_dir, grid, params, 1e9)
                    db.update(grid, 'reconstruct')

                logger.info(f'Grid cell {grid} is complete.')

                
