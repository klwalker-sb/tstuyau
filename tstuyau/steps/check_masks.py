import concurrent.futures
from collections import OrderedDict
from datetime import datetime, timezone
import math
import inspect
from pathlib import Path

from affine import Affine
import dask
import geowombat as gw
import numpy as np
import pandas as pd
import rastercrf as rcrf
import rasterio as rio
from rasterio.enums import Resampling
#from rasterio.warp import reproject
from rasterio.vrt import WarpedVRT
import scipy.ndimage as ndi
from tqdm import tqdm

from ..db import TuyauDataBase
from .date_utils import date_str_to_utc
from ..handler import logger
from .constants import FILENAME_DATE_INDEX, FILENAME_DATE_INDEX_GEE
from .gee_ingest import IngestFromGoogle
from .io import extract_ref_profile
from .lookup import MASKS, SENSORS
from .mask_utils import buffer_mask, compute_cast_shadow, compute_self_shadow, extract_sun_angles, get_mask_kwargs
from .processing_tracker import match_brdf_files_to_record, match_dl_files_to_record
from .project import ProjectPaths
from .utils import (get_cell_list_from_grid_param,
                    get_sorted_images, 
                    get_sensor_param_for_mask,
                    filter_images_by_date,
                    filter_images_by_sensor,
                    get_sensor_code_from_filename, 
                    expand_sensors, 
                    get_sensor_family, 
                    get_sensors)

LANDSAT_LIKE_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']


SHADOWCAST_DEFAULTS = {
    'common': dict(
        # --- decoding the native mask ---
        min_cloud_conf=2,              # Landsat cloud confidence 1 low / 2 medium / 3 high
        min_shadow_conf=1,             # Landsat shadow confidence needed to use the native shadow flag
        use_cirrus=False,
        use_dilated=False,             # Landsat: also count the native dilated-cloud bit as cloud
        s2_cloud_classes=(8, 9),       # SCL classes counted as cloud
        native_shadow_mode='candidate',  # 'off' | 'candidate' (extra evidence only) | 'all' (mask every flagged pixel)
        use_terrain=True,              # keep terrain-shade pixels out of the shadow candidates if that mask exists
        cloud_band='blue',             # growing clouds: image band ...
        cloud_band_min=None,           # ... and minimum value (reflectance x 10000); None = no growing
        # --- cast_cloud_shadows ---
        dark_nir=700., dark_swir2=800.,
        shadow_match_nir=350., shadow_match_swir2=600.,   # stricter thresholds used only to choose WHERE the shadow is
        min_similarity=0.11, min_inside=0.3, h_min=200., h_max=12000., max_samples=20000,
        project_core=True, rel_dark=None, rel_win=151,
        buffer_min_px=1, min_cloud_px=1,
        shadow_grow_px=0, shadow_nir_max=1000., shadow_swir2_max=900.,
        # --- bigger-than-cell native mask ---
        pad_height=0,                  # max cloud height (m) looked for OUTSIDE the cell; 0 = mask is the cell's grid
        # --- run control ---
        verbose=True,                  # per-image summary and the five largest cloud objects in the log
        raise_errors=False,            # True: stop at the first failing image instead of marking it 'ERROR'
    ),
    'landsat': dict(
        cloud_dilate=2, shadow_dilate=1, step_px=2, min_obj_px=10, edge_grow_px=0, cloud_grow_px=0,
        native_shadow_mode='all', min_shadow_conf=1, shadow_grow_px=0, pad_height=0,
    ),
    'aentinel-2': dict(
        cloud_dilate=5, shadow_dilate=2, step_px=4, min_obj_px=30, buffer_min_px=10, cloud_grow_px=50, edge_grow_px=15,
        native_shadow_mode='candidate', min_shadow_conf=2, shadow_grow_px=50, pad_height=6000,
    ),
}

def get_shadowcast_settings(params, sensor):
    """SHADOWCAST_DEFAULTS['common'] + ['<sensor>'] ('landsat' or 'sentinel-2'), overridden by params['masking'] (see top)."""
    cfg = {**SHADOWCAST_DEFAULTS['common'], **SHADOWCAST_DEFAULTS[sensor]}
    for key, default in cfg.items():
        val = get_sensor_param_for_mask(params, key, sensor)
        if val is None:
            continue
        if isinstance(val, str):                                  # CLI / config values arrive as text
            s = val.strip()
            if isinstance(default, bool):
                val = s.lower() in ('1', 'true', 'yes')
            elif isinstance(default, tuple):
                val = tuple(int(x) for x in s.strip('()[] ').split(',') if x.strip())
            elif not isinstance(default, str):
                val = float(s)
        if isinstance(val, (list, tuple)):
            val = tuple(val)
        elif not isinstance(val, (bool, str)):                    # numbers: 0 = not set, negative = switch off
            if val == 0:
                continue
            val = max(val, 0)
            if isinstance(default, bool):
                val = bool(val)
            elif isinstance(default, int):
                val = int(val)
            elif isinstance(default, float):
                val = float(val)
        cfg[key] = val
    if cfg['native_shadow_mode'] not in ('off', 'candidate', 'all'):
        raise ValueError(f"native_shadow_mode must be off/candidate/all, got {cfg['native_shadow_mode']!r}")
    if cfg['cloud_band'] not in LANDSAT_LIKE_BANDS:
        raise ValueError(f"cloud_band must be one of {LANDSAT_LIKE_BANDS}, got {cfg['cloud_band']!r}")
    return cfg


def cast_cloud_shadows(cloud, nir, swir2, zenith, azimuth, res, water=None, fill=None,
                       terrain=None, dark_nir=700., dark_swir2=800., cloud_dilate=2,
                       shadow_dilate=1, h_min=200., h_max=12000., step_px=2, min_obj_px=10,
                       min_similarity=0.11, max_samples=20000, seed=0, native_shadow=None,
                       native_mode='candidate', buffer_min_px=1, min_cloud_px=1, stats=None, cloud_grow=None,
                       cloud_grow_px=0, min_inside=0.5, shadow_match_nir=None, shadow_match_swir2=None, project_core=False,
                       rel_dark=None, rel_win=151, view_zenith=0., view_azimuth=0., shadow_grow_px=0, 
                       shadow_nir_max=None, shadow_swir2_max=None, edge_grow_px=0):
    """
    Writes Cloud mask files. Returns (uint8 array, n_cloud_objects, n_matched).
    Output codes: 0 clear, 1 cloud, 2 cloud shadow, 255 nodata.
    
    Cloud masking method that uses native mask to locate clouds but aggressively filters flagged shaded area to
    reduce masking of dark areas that are not cloud shadow (e.g. burn scars). Cloud shadows are determined based on projection
    of clouds onto the landscape using metadata regarding azimuth and viewing geometry. (this is best done using a native mask
       that is larger than the grid cell to capture clouds just off the edge)
    All located clouds and shadows are dilated to surrounding pixels that meet spectral thresholds suggesting cloud/shadow 
    
    <cloud>,<water>,<fill> are boolean arrays (from the native mask); 
    band vals (blue,nir,swir2) are reflectance x 10000 (NaN = nodata);
    <terrain> is an optional terrain-shade array (nonzero = shade) to be excluded from cloud shadow candidates;
    <native_shadow> is an boolean array of the sensor's own shadow flag
       if <native_mode> is 'candidate', <native_shadow> is used to find potential shadows, but only kept if a 
       plausible source cloud is also found. if <native_mode> is 'all', all shadows from original mask are kept
    Each cloud object is projected along the anti-sun direction for heights <h_min> to <h_max> (both in m),
        and flagged as casting a shadow only if, at its best-matching height, at least <min_similarity> of the projected 
        footprint is dark in BOTH NIR and SWIR2 (pixle values less than <dark_nir> and <dark_swir2>). 
    Note that Sentinel-2 views up to ~10 degrees off nadir, which at 2 km cloud height is a ~400 m offset, thus
        corrections are applied using the view angle from the metadata (<view_azimuth> and <view_zenith>) in addition to 
        the direct <azimuth> and <zenith> information. 
        For nadir-looking satellites, like Landsat, <view_azimuth> and <view_zenith> are 0
    If the cloud mask covers a larger area than the bands (so that clouds outside the cell can cast shadows into it), 
        all bands are placed on a larger canvas with fill=True where there is no band data: 
        those pixels are treated as unknown when a height is scored (not as 'not dark') and never become shadow. 
        Since clouds on fill pixels have no band data to inform their growth, <edge_grow_px> is used to dilated them by this
          many pixels; <min_inside> is then lowered (e.g. 0.3) so a shadow that only partly falls on the cell can still match.
    """
    H, W = cloud.shape
    box = np.ones((3, 3), bool)
    lab, _ = ndi.label(cloud, structure=box)
    size = np.bincount(lab.ravel())
    size[0] = 0                                           # background is not a cluster
    cloud = cloud & (size[lab] >= min_cloud_px)           # drop speckle entirely
    fill = np.zeros_like(cloud) if fill is None else fill
    if cloud_grow is not None and cloud_grow_px > 0:                  # grow cores into connected cloud-like pixels
        seeds = cloud & (size[lab] >= buffer_min_px)
        zone = ndi.binary_dilation(seeds, structure=box, iterations=cloud_grow_px)
        cloud = cloud | ndi.binary_propagation(seeds, structure=box, mask=seeds | (cloud_grow & zone & ~fill))
    if edge_grow_px > 0:                                  # no band to grow into outside the data: plain dilation
        cloud = cloud | (ndi.binary_dilation(cloud & fill, structure=box, iterations=edge_grow_px) & fill)
    core = cloud.copy()                                   # cloud before the masking buffer
    if cloud_dilate > 0:                                  # buffer only clusters of enough pixels
        cloud |= ndi.binary_dilation(cloud & (size[lab] >= buffer_min_px), structure=box,
                                     iterations=cloud_dilate)

    # shadow candidates: dark in NIR and SWIR2, and not cloud / nodata / water / terrain shade
    with np.errstate(invalid='ignore'):
        dark = (nir < dark_nir) & (swir2 < dark_swir2)
    rel = None
    if rel_dark is not None:                # much darker than the neighbourhood in NIR and SWIR2
        usable = np.isfinite(nir) & np.isfinite(swir2) & ~cloud & ~fill
        if water is not None:
            usable &= ~water
        wgt = ndi.uniform_filter(usable.astype(np.float32), size=rel_win, mode='constant')
        rel = wgt > 0.2                      # enough clear neighbours to define a background
        for band in (nir, swir2):
            bg = ndi.uniform_filter(np.where(usable, band, 0).astype(np.float32), size=rel_win,
                                    mode='constant') / np.maximum(wgt, 1e-3)
            with np.errstate(invalid='ignore'):
                rel &= band < rel_dark * bg
        dark |= rel
    if native_shadow is not None and native_mode == 'candidate':
        dark |= native_shadow
    excl = cloud | fill                       # pixels that can never be shadow
    if water is not None:
        excl = excl | water
    if terrain is not None:
        excl = excl | terrain.astype(bool)
    dark &= ~excl

    score = dark                            # what a candidate height is judged on
    if shadow_match_nir is not None or shadow_match_swir2 is not None:
        with np.errstate(invalid='ignore'):
            score = dark & (nir < (dark_nir if shadow_match_nir is None else shadow_match_nir)) \
                    & (swir2 < (dark_swir2 if shadow_match_swir2 is None else shadow_match_swir2))
        
    if stats is not None:                   # optional diagnostics, filled in for the caller
        stats['dark_pct'] = 100.0 * dark.sum() / max((~fill).sum(), 1)
        stats['match_pct'] = 100.0 * score.sum() / max((~fill).sum(), 1)
        stats['best_sims'], stats['objects'] = [], []
        stats['offset_per_m'] = None         # filled below: ground offset (m) per metre of cloud height

    ## (row, col) shifts of a cloud footprint for each candidate height; azimuth points TO the
    ## sun, so the shadow falls the opposite way (rows grow southward, cols eastward)
    ## per metre of cloud height the shadow moves away from the sun azimuth and the cloud's apparent
    ## position moves away from the satellite (parallax): offset = height * (sun term - view term)

    az, tan_el = math.radians(azimuth), math.tan(math.radians(max(90.0 - zenith, 1.0)))
    vaz, tan_v = math.radians(view_azimuth), math.tan(math.radians(view_zenith))
    east = -math.sin(az) / tan_el + tan_v * math.sin(vaz)       # metres east per metre of height
    north = -math.cos(az) / tan_el + tan_v * math.cos(vaz)      # metres north per metre of height
    per_m = math.hypot(east, north)                              # ground offset per metre of height
    heights = np.arange(h_min, h_max + step_px * res / per_m, step_px * res / per_m)
    shifts = np.unique(np.stack([np.rint(-north * heights / res),
                                 np.rint(east * heights / res)], axis=1).astype(int), axis=0)
    shifts = shifts[(np.abs(shifts) <= max(H, W)).all(axis=1)]

    if stats is not None:
        stats['offset_per_m'] = per_m

    rng = np.random.default_rng(seed)
    vr, vc = np.nonzero(~fill)                           # extent of the pixels that have band data
    reach = (vr.min(), vr.max(), vc.min(), vc.max()) if vr.size else None
    labels, _ = ndi.label(cloud, structure=box)
    shadow = np.zeros_like(cloud)
    n_obj = n_hit = 0
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        rr, cc = np.nonzero((labels[sl] == i) & (core[sl] if project_core else True))
        if rr.size < min_obj_px:
            continue
        rr, cc = rr + sl[0].start, cc + sl[1].start
        if reach is None or (rr.max() + shifts[:, 0].max() < reach[0] or rr.min() + shifts[:, 0].min() > reach[1]
                             or cc.max() + shifts[:, 1].max() < reach[2] or cc.min() + shifts[:, 1].min() > reach[3]):
            continue                                     # none of its possible shadows can fall on data
        n_obj += 1
        sub = rng.choice(rr.size, max_samples, replace=False) if rr.size > max_samples else slice(None)
        sr, sc = rr[sub], cc[sub]

        best_sim, best = 0.0, None
        for dr, dc in shifts:
            r2, c2 = sr + dr, sc + dc
            ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
            ok[ok] = ~fill[r2[ok], c2[ok]]                 # pixels without band data are unknown, not 'not dark'
            if ok.sum() < min_inside * sr.size:
                continue
            sim = score[r2[ok], c2[ok]].mean()
            if sim > best_sim:
                best_sim, best = sim, (dr, dc)

        if stats is not None:
            fp = (None, None)                # mean NIR / SWIR2 under the best footprint (diagnostic)
            if best is not None:
                r2, c2 = rr + best[0], cc + best[1]
                ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
                fp = (float(np.nanmean(nir[r2[ok], c2[ok]])), float(np.nanmean(swir2[r2[ok], c2[ok]])))
            stats['best_sims'].append(best_sim)
            stats['objects'].append((rr.size, int(rr.mean()), int(cc.mean()), best_sim, best, *fp))
        if best is not None and best_sim >= min_similarity:
            n_hit += 1
            r2, c2 = rr + best[0], cc + best[1]
            ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
            shadow[r2[ok], c2[ok]] = True

    shadow &= dark                         # keep only genuinely dark pixels in matched footprints
    if shadow_grow_px > 0 and shadow.any():   # grow into connected dark pixels near the matched shadows
        loose = dark
        if shadow_nir_max is not None or shadow_swir2_max is not None:
            with np.errstate(invalid='ignore'):
                loose = dark | (((nir < (np.inf if shadow_nir_max is None else shadow_nir_max))
                                 & (swir2 < (np.inf if shadow_swir2_max is None else shadow_swir2_max))) & ~excl)
        zone = ndi.binary_dilation(shadow, structure=box, iterations=shadow_grow_px)
        shadow = shadow | ndi.binary_propagation(shadow, structure=box, mask=shadow | (loose & zone))

    if shadow_dilate > 0:
        shadow = ndi.binary_dilation(shadow, structure=box, iterations=shadow_dilate)
    if native_shadow is not None and native_mode == 'all':
        shadow |= native_shadow
    shadow &= ~cloud

    out = np.zeros((H, W), np.uint8)
    out[shadow], out[cloud], out[fill] = 2, 1, 255
    return out, n_obj, n_hit

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
    or can pass <img_path> (plus optional parameters to feed into gw
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

    pred_kwargs = {'count': 1,
                        'dtype': 'uint8',
                        'nodata': 255,
                        'driver': 'GTiff',
                        'tiled': True,
                        'compress': 'lzw'}

    if params['dlMethod'] == 'GEE':
        sort_kwargs = dict(date_start=0, date_end=8, date_pos=FILENAME_DATE_INDEX_GEE, prepend_str='netcdf:')
    else:
        sort_kwargs = dict(date_start=0, date_end=8, date_pos=FILENAME_DATE_INDEX, prepend_str='')

    image_dict = get_sorted_brdf_images(ppaths.ms, params['masking']['sat_sensors'],
                                        params['masking']['skip_flag'], sort_kwargs)
    if not image_dict:
        logger.warning(f'No brdf images found in {ppaths.ms} for sensors {params["masking"]["sat_sensors"]}')
        return

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
        for fidx in range(len(full_time_list)-batch_size):
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
                except Exception:
                    continue
                # The image date
                fn_dt = datetime.strptime(Path(fn).name.split('_')[3][:8], '%Y%m%d').replace(tzinfo=timezone.utc)
                file_time_list.append(fn)
                image_dates.append(fn_dt)

            # Continue to check the end for additional duplicates
            while True:
                if fidx+yidx+1 >= len(full_time_list):
                    break
                fn = full_time_list[fidx+yidx]
                yidx += 1
                try:
                    with gw.open(f'{fn}:swir2', chunks=params['masking']['chunks']):
                        pass
                except Exception:
                    continue
                fn_dt = datetime.strptime(Path(fn).name.split('_')[3][:8], '%Y%m%d').replace(tzinfo=timezone.utc)
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
                    with gw.open(f'{fn}:swir2', chunks=params['masking']['chunks']):
                        pass
                    file_time_list.append(fn)
                except Exception as e:
                    logger.warning(f'ERROR opening file {fn}: {e}')
            image_dates = [datetime.strptime(Path(fn).name.split('_')[3][:8], 
                                             '%Y%m%d').replace(tzinfo=timezone.utc) for fn in file_time_list]
            if len(list(set(file_time_list))) == batch_size:
                break
            bfidx -= 1

        existing_files = []
        future_files = []

        for fn, fn_dt in zip(file_time_list, image_dates):

            outfile = rpath_mask / f'{fn_dt.year}{fn_dt.month:02d}{fn_dt.day:02d}.tif'

            if params['masking']['overwrite'] and (outfile.is_file()):
                    outfile.unlink()

            existing_files.append(outfile.is_file())
            future_files.append(outfile)

        if not all(existing_files):
            yield file_time_list, image_dates, future_files
        else:
            yield None, None, None

        with rio.Env(GDAL_CACHEMAX=params['io']['gdal_cachemax']), gw.config.update(
            sensor=params['masking']['sensor'],
            ref_bounds=ref_bounds,
            ref_res=params['masking']['ref_res'],
            ignore_warnings=True,
        ):
            futures = []

            with concurrent.futures.ProcessPoolExecutor(max_workers=params['num_workers']) as executor:

                for image_batch_list, dates_batch_list, future_files in time_generator(ppaths.masks,
                                                                                           proc_names,
                                                                                           params['masking']['batch_size']):

                    if image_batch_list:

                        f = executor.submit(compute_cloud_crf,
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
                    _res = f.result()

def mask_clouds_s2cloudless(params, ppaths, grid, input_dir, processing_db, open_kwargs):
    '''
    Downloads s2cloudless masks from GEE and applies them to Sentinel-2 images.
 
    processing_db/mask_info accepted for consistency with the other
    masking functions; not yet used below since the thresholding
    logic itself is still a TODO (unchanged from the original).
    '''
    logger.info('  first downloading s2cloudless masks from GEE ...')
    mask_args = MASKS['s2cloudless']  ##TODO: use this
    params['image_type'] = ['S2cp']
    ig = IngestFromGoogle(verbose=1)
    ig.ingest_from_gee(params, grid, ppaths)
    logger.info('  now applying masks to Sentinel images ...')

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

def mask_cloudshadows(params, input_dir, processing_db, sensor): 
    """Cloud + cast-shadow mask for every image of one sensor ('landsat' or 'sentinel-2') in input_dir.
    Settings: SHADOWCAST_DEFAULTS and MASKS (sensor sections), overridden by params['masking']. The native QA / SCL
    mask is read from MASKS[...][sensor]['in_dir'] / ['in_suffix'] (or next to the image when in_dir is None). If
    pad_height > 0 that mask may cover more than the cell: the cell's bands are placed on a canvas padded on the
    sides the clouds lie on, and the result is cropped back to the cell.
    """

    cfg = get_shadowcast_settings(params, sensor)
    ma = get_mask_kwargs(input_dir, params, sensor=sensor)
    out_dir, maskname, db_col = Path(ma['mask_dir']), ma['maskname'], ma['db_col']
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_dir = Path(input_dir).parent
    qa_dir = None if ma['in_dir'] in (None, '', 'None') else cell_dir / ma['in_dir']
    terrain_args = MASKS['terrain_shade']
    terrain_dir = cell_dir / terrain_args['mask_dir']
 
    res = float(params['res'])
    skip_flag = params['masking']['skip_flag']
    nir_band, swir_band = LANDSAT_LIKE_BANDS.index('nir') + 1, LANDSAT_LIKE_BANDS.index('swir2') + 1
    cloud_band = LANDSAT_LIKE_BANDS.index(cfg['cloud_band']) + 1
    cast_kw = {k: v for k, v in cfg.items() if k in inspect.signature(cast_cloud_shadows).parameters}
    mode, pad = cfg['native_shadow_mode'], cfg['pad_height']
 
    if params['masking']['img_dir'] == 'brdf':
        file_to_scene = match_brdf_files_to_record(input_dir, processing_db, exclude=skip_flag, method=params['dlMethod'])
    else:
        file_to_scene = match_dl_files_to_record(input_dir, processing_db)
    # (a folder can hold both sensors, e.g. brdf output)
    file_to_scene = {p: s for p, s in file_to_scene.items() if str(s).get_family_from_name(s) == sensor}
    logger.info(f'{sensor}: preparing cloud shadow masks for {len(file_to_scene)} images')
 
    # --- sun / view angles: check once for the scenes about to be masked, extract if any are missing ---
    for c in ('sun_zenith', 'sun_azimuth', 'view_zenith', 'view_azimuth'):
        if c not in processing_db:
            processing_db[c] = np.nan
    angle_cols = ['sun_zenith', 'sun_azimuth'] + (['view_zenith', 'view_azimuth'] if sensor == 'sentinel-2' else [])
    scene_ids = list(dict.fromkeys(file_to_scene.values()))
    if scene_ids and processing_db.loc[scene_ids, angle_cols].isna().any(axis=1).any():
        fam = get_sensor_family(sensor)
        code, folder, metafile = fam['unq'], fam['dir'], fam['metafile']
        logger.info(f'{sensor}: sun/view angles missing for some scenes; extracting from metadata')
        db_path = cell_dir / 'processing.info'
        extract_sun_angles(db_path, cell_dir / folder / 'scene.info', code, metafile)
        fresh = pd.read_pickle(db_path)
        for c in angle_cols:
            if c in fresh:
                processing_db[c] = fresh[c]
 
    n_ok, n_skip = 0, 0
    for img_path, scene_id in file_to_scene.items():
        row = processing_db.loc[scene_id]
        zenith, azimuth = row.get('sun_zenith'), row.get('sun_azimuth')
        if pd.isna(zenith) or pd.isna(azimuth):
            logger.warning(f'No sun angles for {scene_id}; skipping {img_path.name}')
            n_skip += 1
            continue
        zenith, azimuth = float(zenith), float(azimuth)
        view_zenith, view_azimuth = row.get('view_zenith'), row.get('view_azimuth')
        if pd.isna(view_zenith) or pd.isna(view_azimuth):
            view_zenith, view_azimuth = 0.0, 0.0                 # none stored: nadir (no parallax)
        view_zenith, view_azimuth = float(view_zenith), float(view_azimuth)
        base = img_path.stem
        out_path = out_dir / f'{base}_{maskname}.tif'
        qa_path = (qa_dir or img_path.parent) / f"{base}{ma['in_suffix']}{'.tif' if qa_dir else img_path.suffix}"
        
        try:
            with rio.open(img_path) as src:                       # bands are reflectance x 10000
                nir = src.read(nir_band, masked=True).astype(np.float32).filled(np.nan)
                swir = src.read(swir_band, masked=True).astype(np.float32).filled(np.nan)
                cloud_grow = None
                if cfg['cloud_band_min'] is not None:             # cloud-like pixels to grow clouds into
                    cloud_grow = src.read(cloud_band, masked=True).astype(np.float32).filled(np.nan) >= cfg['cloud_band_min']
                profile, cell_tf, cell_crs = src.profile.copy(), src.transform, src.crs
            H0, W0 = nir.shape
            top = left = 0
            if pad > 0:
                # clouds outside the cell cast shadows into it: pad the canvas on the sides they lie on. Per metre of
                # cloud height the shadow moves (east, north) metres, so the cloud lies at -(east, north) from it.
                tan_el = math.tan(math.radians(max(90.0 - zenith, 1.0)))
                tan_v = math.tan(math.radians(view_zenith))
                east = -math.sin(math.radians(azimuth)) / tan_el + tan_v * math.sin(math.radians(view_azimuth))
                north = -math.cos(math.radians(azimuth)) / tan_el + tan_v * math.cos(math.radians(view_azimuth))
                top, bottom = (math.ceil(m * pad / res) for m in (max(0.0, -north), max(0.0, north)))
                left, right = (math.ceil(m * pad / res) for m in (max(0.0, east), max(0.0, -east)))
                # the mask file can be any extent that contains the cell; it is resampled (nearest) onto the cell's
                # grid extended by the padding. Outside the file it is 0 (nodata).
                with rio.open(qa_path) as msrc, WarpedVRT(
                        msrc, crs=cell_crs, transform=cell_tf * Affine.translation(-left, -top),
                        width=W0 + left + right, height=H0 + top + bottom,
                        resampling=Resampling.nearest, nodata=0) as vrt:
                    qa = vrt.read(1)
                cell = (slice(top, top + H0), slice(left, left + W0))
 
                def to_canvas(a, empty):                          # place a cell-sized array on the canvas
                    big = np.full(qa.shape, empty, a.dtype)
                    big[cell] = a
                    return big
                nir, swir = to_canvas(nir, np.nan), to_canvas(swir, np.nan)    # NaN outside the cell -> fill
                cloud_grow = None if cloud_grow is None else to_canvas(cloud_grow, False)
            else:
                with rio.open(qa_path) as src:
                    qa, profile = src.read(1), src.profile.copy()
                if qa.shape != nir.shape:
                    raise ValueError(f'mask layer {qa.shape} and image {nir.shape} are not on the same grid '
                                     f'(a wider mask needs pad_height > 0)')

            if sensor == 'landsat':
                # QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 7 water,
                # 8-9 cloud conf, 10-11 shadow conf, 14-15 cirrus conf
                cloud = (((qa >> 3) & 1) == 1) & (((qa >> 8) & 3) >= cfg['min_cloud_conf'])
                if cfg['use_cirrus']:
                    cloud |= (((qa >> 2) & 1) == 1) & (((qa >> 14) & 3) >= 2)
                if cfg['use_dilated']:
                    cloud |= ((qa >> 1) & 1) == 1
                native = (((qa >> 4) & 1) == 1) & (((qa >> 10) & 3) >= cfg['min_shadow_conf'])
                water, fill = ((qa >> 7) & 1) == 1, (qa & 1) == 1
            else:
                # SCL classes: 0 nodata, 3 cloud shadow, 6 water, 8/9 cloud medium/high, 10 thin cirrus
                cloud = np.isin(qa, cfg['s2_cloud_classes'] + ((10,) if cfg['use_cirrus'] else ()))
                native, water, fill = qa == 3, qa == 6, qa == 0
            fill |= ~np.isfinite(nir) | ~np.isfinite(swir)
 
            terrain = None
            terrain_path = terrain_dir / f"{base}_{terrain_args['maskname']}.tif"
            if cfg['use_terrain'] and terrain_path.exists():
                with rio.open(terrain_path) as src:
                    terrain = src.read(1)
                if terrain.shape != (H0, W0):
                    logger.warning(f'terrain shade mask for {base} is on a different grid; ignoring it')
                    terrain = None
                elif pad > 0:
                    terrain = to_canvas(terrain, 0)
 
            stats = {}
            out, n_obj, n_hit = cast_cloud_shadows(
                cloud, nir, swir, zenith, azimuth, res, water, fill, terrain,
                native_shadow=None if mode == 'off' else native, native_mode=mode, stats=stats,
                cloud_grow=cloud_grow, view_zenith=view_zenith, view_azimuth=view_azimuth, **cast_kw)
            if pad > 0:                                           # back to the cell
                out, native = out[cell], native[cell]
 
            profile.update(driver='GTiff', dtype='uint8', count=1, nodata=255, compress='lzw')
            with rio.open(out_path, 'w', **profile) as dst:
                dst.write(out, 1)
 
            if cfg['verbose']:
                sims = stats['best_sims']
                logger.info(f"{img_path.name}: {n_hit} of {n_obj} cloud objects matched to a shadow; dark candidates "
                            f"{stats['dark_pct']:.1f}% (strict {stats['match_pct']:.2f}%) of valid px; best similarity "
                            f"max {max(sims, default=0):.2f}, median {float(np.median(sims)) if sims else 0:.2f} "
                            f"(min_similarity {cfg['min_similarity']}); sun az/zen {azimuth:.1f}/{zenith:.1f}, "
                            f"view az/zen {view_azimuth:.1f}/{view_zenith:.1f}")
                for n_px, r0, c0, sim, shift, nir_m, swir_m in sorted(stats['objects'], key=lambda o: o[0], reverse=True)[:5]:
                    if shift is not None:       # largest cloud objects: where did the best match fall?
                        h = math.hypot(*shift) * res / stats['offset_per_m']
                        logger.info(f'   cloud object {n_px} px at row {r0}, col {c0}: best similarity {sim:.2f} at shift '
                                    f'(row {shift[0]}, col {shift[1]}) px = cloud height ~{h:.0f} m; '
                                    f'mean NIR / SWIR2 under that footprint {nir_m:.0f} / {swir_m:.0f}')
                if mode != 'off':               # where did the natively flagged shadow pixels end up?
                    tally = {c: int((native & (out == c)).sum()) for c in (2, 1, 255, 0)}
                    logger.info(f'   {int(native.sum())} native shadow px -> {tally[2]} shadow, {tally[1]} cloud (inside '
                                f'cloud buffer), {tally[255]} nodata, {tally[0]} unmasked')
                    if mode == 'all' and tally[0]:
                        logger.warning(f'{img_path.name}: {tally[0]} native shadow px left unmasked in mode "all"')
 
            n_ok += 1
            processing_db.loc[scene_id, db_col] = str(out_path)
            update_db_mask_tracker(processing_db, scene_id, 'shadowcast')


        except Exception as e:
            if cfg['raise_errors']:
                raise
            # no MASKFAIL rename here: the native mask's name is built from the image stem, so renaming the image
            # would orphan it
            logger.warning(f'Failed to make cloud shadow mask for {img_path.name}: {e}')
            n_skip += 1
            processing_db.loc[scene_id, db_col] = 'ERROR'
 
    logger.info(f'{sensor}: cloud shadow masks: {n_ok} created, {n_skip} skipped -> {out_dir}')
        

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

    
def make_aggregate_mask(ppaths, mask_method, mask_dir=None,
                       start_date=None, end_date=None,image_type=None):
    '''
    Aggregates individual masks into three single-band summary products:
      <maskname>_count.tif       number of times each pixel was masked
      <maskname>_per_masked.tif  percent of valid observations that were masked, * 100
      <maskname>_num_clear.tif   number of valid, unmasked observations
    Returns (count_path, per_masked_path, num_clear_path).
    '''
    mask_args = MASKS[mask_method]
    mask_name = mask_args['maskname']
    mask_val = mask_args['mask_val'] 
    mask_val = list(mask_val) if isinstance(mask_val, (list, tuple)) else [mask_val]
    in_dir = ppaths.ms.parent / (mask_dir or mask_args['mask_dir'])
    out_dir = in_dir.parent/'comp'
    out_dir.mkdir(exist_ok=True, parents=True)
    
    sort_kwargs = dict(date_start=0, date_end=8, date_pos=FILENAME_DATE_INDEX, prepend_str='')
    image_dict = get_sorted_images(in_dir, f'*{mask_name}.tif', sort_kwargs, names=None)
    
    if start_date:
        image_dict = filter_images_by_date(image_dict, start_date, end_date)
    if image_type:
        image_dict = filter_images_by_sensor(image_dict, image_type)
    if not image_dict:
        logger.warning(f'No {mask_name} masks found in {in_dir} for the requested dates/sensors.')
        return None, None, None

    files = list(image_dict.keys())
    dates = list(image_dict.values())
    
    with rio.open(files[0]) as src:
        profile = src.profile.copy()
                 
    with gw.open(files, time_names=dates, stack_dim='time') as masksrc:
        masksrc = masksrc.isel(band=0)                  # masks are single band -> (time, y, x)

        valid = masksrc.notnull() & (masksrc != 255)                       # nodata/NaN = no observation
        if mask_val:
            is_masked = masksrc.isin(mask_val)
        else:
            is_masked = masksrc > 0                     # empty mask_val (CRF): any nonzero = masked

        nummasked = is_masked.sum(dim='time')
        numobs = valid.sum(dim='time')
        nummasked, numobs = dask.compute(nummasked.values, numobs.values)   # one pass over the data
                 
    numclear = numobs - nummasked
    permasked = np.where(numobs > 0, 100 * (nummasked / np.maximum(numobs, 1)), 255)
    
    prefix = f'{mask_name}_{image_type}' if image_type else f'{mask_name}'
    if start_date:
        start_text = f"{start_date.split('-')[0]}{start_date.split('-')[1]}"
        end_text = f"{end_date.split('-')[0]}{end_date.split('-')[1]}"
        prefix = f'{prefix}_{start_text}-{end_text}'
    agg_count_out = out_dir / f'{prefix}_count.tif'
    agg_per_out = out_dir / f'{prefix}_per_masked.tif'
    agg_clear_out = out_dir / f'{prefix}_num_clear.tif'
    
    profile.update(count=1, compress='lzw')
    with rio.open(agg_per_out, 'w', **profile) as dst:
        dst.write(permasked, 1)
    profile.update(dtype='int16', nodata=-9999)
    with rio.open(agg_count_out, 'w', **profile) as dst:
        dst.write(nummasked, 1)
    with rio.open(agg_clear_out, 'w', **profile) as dst:
        dst.write(numclear, 1)
    
    return agg_count_out, agg_per_out, agg_clear_out
                        
def make_masks(params):
    """
    Creates masks for clouds (and cloud shadows) or shaded terrain

    if params.masking.method is 'cloud_shadowcast', uses native masks without shadows and casts shadows separately.
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
                
    cells = get_cell_list_from_grid_param(params['grids'])
    for grid in cells:
        ppaths = ProjectPaths(params, grid=int(grid))
        
        if (params['masking']['aggregate']) and (params['masking']['aggregate'] not in ['False','']):
            ## aggregates existing masks -- assumes this process was already run to create the individual masks.
            aggmask, _, _ = make_aggregate_mask(ppaths, mask_method,mask_dir=None,image_type=params['masking']['sat_sensors'],
                    start_date=params['reconstruct']['start'], end_date=params['reconstruct']['end'])
            logger.info(f'saved aggregated mask to: {aggmask}')

        else:
            sensors = get_sensors(params['masking']['sat_sensors'])
            dl_dirs = get_sensor_dirs(ppaths.ms.parent, params['masking']['sat_sensors'])
            if input_img_dir == 'brdf':
                img_dir = ppaths.ms
            elif input_img_dir == 'dl':
                img_dir = dl_dirs[sensors[0]] if sensors else ppaths.ms.parent    # (only used for the checks below)
            else:
                img_dir = ppaths.ms.parent / input_img_dir

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
                msensors = [s for s in expand_sensors(params['masking']['sat_sensors']) if 'GEE' in SENSORS[s]]
                ## Check downloads TODO: update for stac method
                undone = [s for s in msensors if not db.eosvault_is_complete(SENSORS[s]['sensor'], SENSORS[s]['GEE'])]
                if undone:
                    logger.warning(f'  Downloads for {", ".join(undone)} in grid {grid} are incomplete.')
                    continue

                # Check post-processing
                if ppaths.gee.is_dir():
                    ## note: we are excluding files that end with 'angles' and 'cloudless' from the glob by excluding words that 
                    ##    end in s. For more precise method, might need to use list version:
                    ## e.g. [f for f in ppaths.gee if 'LT05' in os.path.basename(f) and "angles" not in os.path.basename(f)]
                    pending = get_pending_gee_files(ppaths.gee, params['masking']['sat_sensors'])
                    if pending:
                        logger.warning(f'  Post-processing for {", ".join(pending)} in grid {grid} is incomplete.')
                        continue

            if not db.table_exists:
                db.remove()
                db.create(exists_ok=True)
                db.insert(grid)

            if params['status']['reset_db'] or params['masking']['overwrite']:
                db.reset(grid, 'mask')
            
            # Check if the step is complete
            if db.is_complete(grid, 'mask'):
                logger.warning('  The masking step is complete.')
                continue

            logger.info(f'  Masks being created for grid {grid} ...')
        
            if not img_dir.is_dir():
                logger.warning(f' Directory: {img_dir!s} for grid {grid} does not exist.')
                continue
        
            if params['masking']['retry_masks'] == True:
                logger.info(f'resetting file names in {img_dir!s} to retry masking')
                failed_masks = list(img_dir.glob('*MASKFAILX*.nc'))
                for f in failed_masks:
                    clean_name = f.name.replace("_MASKFAILX", "")
                    f.rename(f.with_name(clean_name))
        
            if mask_method == 'cloud_shadowcast':
                for sensor in sensors:
                    sensor_dir = dl_dirs[sensor] if input_img_dir == 'dl' else img_dir
                    if not sensor_dir.is_dir():
                        logger.warning(f' Directory: {sensor_dir!s} for grid {grid} does not exist.')
                        continue
                    mask_cloudshadows(params, sensor_dir, processing_db, sensor)
    
            elif mask_method == 'cloud_CRF':    
                mask_clouds_CRF(params, ppaths, processing_db)
            
            elif mask_method == 's2cloudless': 
                mask_clouds_s2cloudless(params, ppaths, grid, img_dir, processing_db, open_kwargs)
        
            elif mask_method == 'terrain_shade':
                mask_shade(params, img_dir, processing_db, open_kwargs)

            else:
                logger.warning(f'ERROR: unrecognized mask method: {mask_method}') 
            
            processing_db['masking'] = processing_db['masking'].fillna('native masks only')
            pd.to_pickle(processing_db, ppaths.ms.parent / 'processing.info')
            db.update(grid, 'mask')
