import concurrent.futures
import copy
import sys
from datetime import datetime, timedelta
from pathlib import Path

import geowombat as gw
import numpy as np
import pandas as pd
import rasterio as rio
import yaml
from geowombat.core.windows import get_window_offsets
from tqdm import tqdm

from ..db import TuyauDataBase
from ..handler import logger
from . import utils
from .date_utils import get_date_range, date_str_to_utc
from .io import TimeSeriesLoader
from .lookup import SENSORS
from .mask_utils import apply_masks_to_images, get_mask_kwargs
from .project import ProjectPaths, get_tsdir_name, cleanup_tracker, get_raw_ts_dir
from .spec_indices import SI_DICT
from .time_series_utils import smooth
from .utils import (get_cell_list_from_grid_param,
                    get_sorted_brdf_images, 
                    get_brdf_sort_kwargs, 
                    get_family_from_name, 
                    get_sensor_family,
                    get_sensors,
                    filter_images_by_date)
                    

LANDSAT_LIKE_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

 
def _yaml_load(path):
    with open(path, mode='r') as pf:
        return yaml.load(pf, Loader=yaml.FullLoader)
 
 
def _yaml_dump(obj, path):
    with open(path, mode='w') as pf:
        yaml.dump(obj, pf, default_flow_style=False)

 
def _get_band_names(si):
    '''
    sis may be passed in with parameters attached (e.g. savi.100 and/or with ts info (e.g. savi-raw or savi.100-raw).
    '_' is legacy only.
    '''
    sidx = si.split('.')[0]
    if m := next((char for char in ('-', '_') if char in sidx), None):
        sidx = sidx.split(m)[0]
    if sidx not in SI_DICT:
        logger.warning(f"  Model does not recognize {sidx}. Supported spectral indices are {SI_DICT.keys()}")
        raise NameError
    return SI_DICT[sidx]['band_names']
 
 
def _set_raw_period(params):
    '''
    Raw indices: period is from the first si_var (-yr, -wet, -dry), the calendar params and the model year,
    padded with <feature_model:pheno_pad_days> if set. Written to params['reconstruct'] start/end/start_pad/end_pad.
    '''
    rc = params['reconstruct']
    season = params['feature_model']['si_vars'][0].split('-')[1]
    rc['start'], rc['end'] = get_date_range(params['feature_model']['start_yr'], season, params)
    pad_days = params['feature_model']['pheno_pad_days']
    if pad_days:
        rc['start_pad'] = (date_str_to_utc(rc['start']) - timedelta(days=pad_days[0])).strftime("%Y-%m-%d")
        rc['end_pad'] = (date_str_to_utc(rc['end']) + timedelta(days=pad_days[1])).strftime("%Y-%m-%d")
    else:
        rc['start_pad'] = rc['start']
        rc['end_pad'] = rc['end']
    logger.info(f"calculating index for images from {rc['start_pad']} to {rc['end_pad']}")
 
 
def _get_out_dir(ppaths, params, si, grid):
    '''
    Directory where reconstruct writes its outputs for a cell. 
    '''
    if si.endswith('raw'):
        ## indices calculated from raw images are currently sent to the scratch drive
        ##      because they cost nothing to recreate (UNLESS BRDFs are deleted!)
        return get_raw_ts_dir(ppaths, params, si, params['feature_model']['start_yr'])
    if params['reconstruct']['merge_ts']:
        ts_dir = ppaths.ts / si / 'sub_ts'
        cleanup_tracker(ts_dir, '.tif', 5, grid)
        return ts_dir
    return ppaths.ts / si

 
def _skip_window(winfile, params, widx):
    '''
    Window tracker logic: True if this window was already run and should be skipped.
    Also (re)initializes the tracker where needed.
    '''
    rc = params['reconstruct']
    chunks = rc['chunks']
    if rc['rewrite_win']:
        if widx < int(rc['start_win']):
            return True
        _yaml_dump({chunks: {'latest': -999}}, winfile)
    elif winfile.is_file():
        if widx < int(_yaml_load(winfile)[chunks]['latest']):
            return True
    else:
        _yaml_dump({chunks: {'latest': -999}}, winfile)
    return False
 
 
def _year_slice_dates(year, start_dt, end_dt, start_pad_dt, end_pad_dt, skip_years):
    '''Padded and un-padded datetimes for one annual slice -> (start_pad, end_pad, start, end).'''
    if start_dt.month - start_pad_dt.month >= 0:
        start_pad = date_str_to_utc(f'{year}-{start_pad_dt.month}-{start_pad_dt.day}')
    else:
        start_pad = date_str_to_utc(f'{year-1}-{start_pad_dt.month}-{start_pad_dt.day}')
 
    add_yrs = skip_years if end_dt.year - start_dt.year > 0 else 0
 
    if end_pad_dt.month - end_dt.month >= 0:
        end_pad = date_str_to_utc(f"{year+add_yrs}-{end_pad_dt.month}-{end_pad_dt.day}")
    else:
        end_pad = date_str_to_utc(f"{year+add_yrs+1}-{end_pad_dt.month}-{end_pad_dt.day}")
    end_pad = min(end_pad, end_pad_dt)
 
    start = date_str_to_utc(f'{year}-{start_dt.month}-{start_dt.day}')
    end = date_str_to_utc(f"{year+add_yrs}-{end_dt.month}-{end_dt.day}")
    return start_pad, end_pad, start, end
 
 
def _get_year_slices(time_band_df, si, start_dt, end_dt, start_pad_dt, end_pad_dt, skip_years):
    '''All annual slices to run: list of (year, start_pad, end_pad, start, end). Independent of the window, so made once.'''
    if 'raw' not in si:
        unique_yrs = sorted({dt.year for dt in time_band_df.index.to_pydatetime()})
    else:
        unique_yrs = sorted({start_dt.year, end_dt.year})
        logger.debug(f"unique yrs: are {unique_yrs} for {start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}.")
 
    slices = []
    for yidx in range(0, len(unique_yrs), skip_years):
        year = unique_yrs[yidx]
        if year < start_pad_dt.year or year > end_pad_dt.year:
            continue
        slices.append((year, *_year_slice_dates(year, start_dt, end_dt, start_pad_dt, end_pad_dt, skip_years)))
    return slices


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

def reconstruct(params, cells=None, files=None, out_dir=None):
    """
    Calculates indices and reconstructs a time series into weekly intervals using various data smoothing and gap filling options.
       output rasters are integers (val*10000) named YYYYdoy.tif stored in the time series directory for the cell (ppaths.ts) 
    If spec_index has 'raw': just calculates indices directly (no smoothing) and outputs rasters 
       for all input images in a defined time period into a scratch folder (ppaths.scratch)
           the time period is defined based on the first variable set in params['feature_model']['si_vars'] (-yr, -wet, or -dry), 
           the calendar parameters to define the sesaons, and the model year (params['feature_model']['start_yr']).
        
    vegetation indices are defined in SpecIndices in spec_indices.py (called through TimeSeriesLoader) and io.py. 
        any new index need to be added to the SI_DICT as well as SpecIndices methods.

    Option to apply masks at this stage with params<reconstruction><use_masks>. A mask must already exist for each image with the 
        same base name (anything before _coreg) as the image to be masked. If masks do not exist, run make_masks() first.

    Optional arguments (all default to the legacy behavior):
        cells:   list of cells to run instead of the cells in params['grids']
        files:   list of brdf file names (just the names) to run through the raw calculation instead of a season. The period is set to the
                 dates of those files, and nothing is deleted or smoothed. Raw indices only. params is not modified.
        out_dir: required with files; where the YYYYdoy.tif outputs are written -- to not overwrite files in true ts_dir
    """

    if files is not None:
        if out_dir is None:
            raise ValueError("out_dir is required when files is given")
        if not params['reconstruct']['si'].endswith('raw'):
            raise ValueError("files can only be used with raw indices (si must end with 'raw')")
        params = copy.deepcopy(params)  ## so none of the overrides below leak back to the caller
        params['reconstruct']['rewrite_win'] = False
        params['reconstruct']['overwrite'] = False
        file_names = {Path(f).name for f in files}
    else:
        file_names = None
 
    rc = params['reconstruct']
    si = rc['si']
 
    if si.endswith('raw'):
        rc['merge_ts'] = False
        if files is None:
            _set_raw_period(params)
 
    if cells is None:
        cells = get_cell_list_from_grid_param(params['grids'])

    sort_kwargs = get_brdf_sort_kwargs(params)
    skip_flag = rc['skip_flag'] or None
    ## with files=, the file names do the selecting, so allow any sensor
    sensor_tags = 'All' if files is not None else params['image_type']

    for grid in cells:
        ppaths = ProjectPaths(params, grid=grid)
        db = TuyauDataBase(str(ppaths.ms.parent / f'{int(grid):06d}_tuyau.db'))

        #if params['reconstruct']['use_masks']:
            # Check if the step is complete
            #if not db.is_complete(grid, 'mask'):
            #    logger.warning(f'  The mask step is not complete.')
            #    continue

        # Check if other pre-processing steps are complete
        if not db.is_complete(grid, 'preprocess'):
            logger.warning('  The pre-processing step is incomplete.')
            continue

        if not ppaths.proc.is_dir():
            logger.warning(f'  The input directory for grid {grid} does not exist.')
            continue
        in_dir = ppaths.proc

        #if params['reconstruct']['use_masks']:
        #    prechecks.precheck_reconstruct(grid, ppaths, params)

        ## Set directory to store output ts data 
        ## root folder name is <img_type>-<res>-<procseq> if comparative models are being run, but parts are dropped for simplicity if not
        ts_dir = Path(out_dir) if files is not None else _get_out_dir(ppaths, params, si, grid)
        winfile = Path(ts_dir) /f'{int(grid):06d}.window'
        ts_dir.mkdir(parents=True, exist_ok=True)
        
        ppaths.clean_temp(ts_dir)

        if files is not None:
            ## fresh window tracker for a one-off run, but leave any other images in out_dir alone
            winfile.unlink(missing_ok=True)
            (ts_dir / f'{int(grid):06d}.reindex').unlink(missing_ok=True)
 
        if rc['overwrite']:
            winfile.unlink(missing_ok=True)
            (ts_dir / f'{int(grid):06d}.reindex').unlink(missing_ok=True)
            for file_path in ts_dir.glob('*.tif'):
                file_path.unlink()
 
        ## Check if the file is complete (only if rewrite_win parameter is set to False)
        if not rc['rewrite_win'] and winfile.is_file():
            if int(_yaml_load(winfile)[rc['chunks']]['latest']) == 1e9:
                logger.warning('  The reconstruct step is complete.')
                continue

        ## Get images from final processing directory to process
        image_dict = get_sorted_brdf_images(in_dir, sensor_tags, skip_flag, sort_kwargs, names=file_names)
 
        if files is not None:
            if not image_dict:
                logger.warning(f'  None of the requested files {sorted(file_names)} were found in {in_dir} for grid {grid}.')
                continue
            ## period = the dates of the requested files, no padding
            img_dates = sorted(image_dict.values())
            rc['start'] = rc['start_pad'] = img_dates[0].strftime('%Y-%m-%d')
            rc['end'] = rc['end_pad'] = img_dates[-1].strftime('%Y-%m-%d')
            rc['skip_years'] = max(rc['skip_years'] or 1, img_dates[-1].year - img_dates[0].year + 1)
            logger.info(f"calculating index for {sorted(file_names)} ({rc['start']} to {rc['end']})")
 
        img_names = list(image_dict.keys())
        logger.debug(f' ALL valid images in directory (not yet filtered to date): {img_names}')
        img_times = list(image_dict.values())
 
        ## this is just a template for gw_open (a Landsat 8 image if there is one, otherwise the last image)
        l8_images = [str(fn) for fn in img_names if 'LC08' in Path(fn).name]
        l8_image = l8_images[-1] if l8_images else str(img_names[-1])

        # Window padding for moving window smoothing
        if si.endswith('raw'):
            pad = 0
            pidx = 0
        elif rc['smooth_kwargs']['prefill_gaps']:
            pad = rc['smooth_kwargs']['prefill_wmax']
            pidx = 1
        else:
            pad = rc['smooth_kwargs']['k']*2 if rc['smooth_kwargs']['spt_smoothing'] else None
            pidx = 1 if rc['smooth_kwargs']['spt_smoothing'] else 0
            if isinstance(pad, int) and (pad % 2 == 0):
                pad += 1
 
        band_names = _get_band_names(si)
 
        start_dt, end_dt = date_str_to_utc(rc['start']), date_str_to_utc(rc['end'])
        start_pad_dt, end_pad_dt = date_str_to_utc(rc['start_pad']), date_str_to_utc(rc['end_pad'])
 
        if rc['use_masks']:
            '''Apply masks to all images and save with same names in temp folder, then move to chunked reading
            must be done at this stage to preserve 1:1 mask:image matching prior to merging images from same dates.
            Can be done simultaneously during the chunked read in the case of cloud masks (TOOD: reinstate this option),
            but terrain masks need to see a larger landsacpe and are better run on the full image.
            '''
            logger.info('applying masks to raw images...')
 
            mask_args = get_mask_kwargs(ppaths.ms, params)
            mask_dir = ppaths.ms.parent / mask_args['mask_dir']

            if not mask_dir.is_dir():
                logger.warning(f'  There is no directory with masks called {mask_dir} for cell {grid}.'
                               f'Check masking params and/or run make_masks()')
                continue
 
            masked_img_dir = ppaths.scratch / 'masked'
            masked_img_dir.mkdir(parents=True, exist_ok=True)
            filtered_images = filter_images_by_date(image_dict, rc['start_pad'], rc['end_pad'])
            logger.info(f'applying masks to {len(filtered_images)} images')
            mask_imgs_in_dir(img_dir, mask_args, out_dir=None, images=filtered_images, params=None)
 
            image_dict = get_sorted_brdf_images(masked_img_dir, sensor_tags, skip_flag, sort_kwargs, names=file_names)
 
            img_names = list(image_dict.keys())
            logger.debug(f' ALL valid images in directory (not yet filtered to date): {img_names}')
            img_times = list(image_dict.values())
 
        # Setup a DataFrame of all the image names and dates
        time_band_df = pd.DataFrame(data=img_names, index=img_times, columns=['image_path'])
        logger.debug(f'time_band_df for valid all images in directory (not yet filtered to date):\n {time_band_df}')
 
        skip_years = rc['skip_years'] or 1
        year_slices = _get_year_slices(time_band_df, si, start_dt, end_dt, start_pad_dt, end_pad_dt, skip_years)
 
        with rio.Env(GDAL_CACHEMAX=params['io']['gdal_cachemax']):
            with gw.open(
                        l8_image,
                        band_names=LANDSAT_LIKE_BANDS,
                        chunks={'band' : -1,
                            'y' : rc['chunks'],
                            'x' : rc['chunks']},
                        engine='h5netcdf'
                    ) as src:
                ref_bounds = src.gw.bounds
 
                if isinstance(src.crs, str) and (src.crs.lower().startswith('epsg') or src.crs.lower().startswith(':epsg')):
                    ref_crs = int(src.crs.split(':')[-1])
                else:
                    ref_crs = src.crs
 
                profile = {
                    'blockxsize': src.gw.col_chunks,
                    'blockysize': src.gw.row_chunks,
                    'crs': src.gw.crs_to_pyproj.to_wkt(),
                    'transform': src.gw.affine,
                    'driver': 'GTiff',
                    'count': 1,
                    'height': src.gw.nrows,
                    'width': src.gw.ncols,
                    'nodata': 0,
                    'dtype': 'uint16',
                    'compress': 'lzw',
                    'tiled': True
                }
 
                windows = get_window_offsets(
                    src.gw.nrows,
                    src.gw.ncols,
                    rc['chunks'],
                    rc['chunks'],
                    return_as='list',
                    padding=(pad, pad, pad, pad)
                )

                        ## Open the images into an array
            with gw.config.update(
                sensor='l7',
                ref_res=float(rc['res']),
                ref_bounds=ref_bounds,
                ref_crs=ref_crs,
                ignore_warnings=True
            ):
                # Iterate by window for memory management
                for widx, w in enumerate(windows):
                    logger.info(f"widx ={widx},index={si}, rewrite_win = {rc['rewrite_win']}")
 
                    if _skip_window(winfile, params, widx):
                        continue
 
                    logger.info(f'  Loading window data for window {widx+1} out of {len(windows)} ...')
 
                    # Get the chunk slice
                    slicer = (
                        slice(0, None),  # time
                        slice(0, None),  # bands
                        slice(w[pidx].row_off, w[pidx].row_off+w[pidx].height),
                        slice(w[pidx].col_off, w[pidx].col_off+w[pidx].width)
                    )


                    for year, start_pad_dt_slice, end_pad_dt_slice, start_dt_slice, end_dt_slice in year_slices:
                        logger.info(f'working on year {year}...')
 
                        time_band_df_slice = time_band_df.loc[start_pad_dt_slice:end_pad_dt_slice]
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

                        tsl = TimeSeriesLoader(
                            time_band_df_slice,
                            start_pad_dt_slice,
                            end_pad_dt_slice,
                            band_names,
                            slicer,
                            params
                        )
 
                        if rc['load_on_cluster']:
                            real_img_times, y = tsl.load_on_cluster()
                        else:
                            real_img_times, y = tsl.load_netcdf()
 
                        tags = {}
 
                        if 'raw' not in si:
                            logger.info(f'  Smoothing data for grid {grid} ...')
                            # Smooth the data and re-grid to weekly intervals
                            xinfo, stack = smooth(
                                real_img_times,
                                y,
                                start_dt_slice.strftime('%Y-%m-%d'),
                                end_dt_slice.strftime('%Y-%m-%d'),
                                n_jobs=params['num_workers'],
                                **rc['smooth_kwargs']
                            )
                            out_data = zip(xinfo.skip_slice[xinfo.write_skip_idx], stack[xinfo.write_skip_idx])
                            for bidx, dt in enumerate(xinfo.skip_slice[xinfo.write_skip_idx]):
                                tags[f'BAND_{bidx+1:03d}'] = dt.strftime('%Y%m%d')
 
                        else:
                            ## using all of the inputs (no smoothing or reindexing) for raw calculations
                            s = 10000
                            if params['masking']['maxval']:
                                s = params['masking']['maxval']
                            out_data = zip(real_img_times, y * s)
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
 
                    _update_progress(ts_dir, grid, params, widx)
 
                    if rc['rewrite_win'] and (
                        (rc['start_win'] + rc['win_batchsize']) == int(widx)+1) and (
                        int(widx) < 16):
 
                        sys.exit(0)
 
                if 'raw' not in si:
                    _update_progress(ts_dir, grid, params, 1e9)
                    db.update(grid, 'reconstruct')
 
                logger.info(f'Grid cell {grid} is complete.')


                
