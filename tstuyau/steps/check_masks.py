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
from rasterio.enums import Resampling as RioResampling
from rasterio.warp import reproject
from scipy.ndimage import rotate as ndi_rotate
import geowombat as gw
import xarray as xr
import xrspatial
from geowombat.core import sort_images_by_date
import rastercrf as rcrf
from tqdm import tqdm

"""
Computes a combined cast-shadow + self-shadow mask for each
BRDF-corrected raster in a grid cell, using that cell's Copernicus
DEM and the per-scene sun zenith/azimuth already stored in
processing.info.

Run as a standalone step, separate from the download/metadata
pipeline. Does NOT yet apply the shiftX/shiftY co-registration
offsets -- that's a follow-on step once masks are validated.
"""

def match_brdf_files(brdf_dir, processing_info, exclude=None):
    """
    Maps each BRDF .tif file to its processing.info scene id, via 'brdf_id' column -- 
          a prefix of the actual filename, which may have extra text appended before '.tif'.
    option to skip files with text that matches <exclude> (usually X)  at end of stem name 
    """
    brdf_dir = Path(brdf_dir)
    candidates = [f for f in brdf_dir.glob('*.tif') if not f.name.endswith(f'{exclude}.tif')]

    id_lookup = processing_info['brdf_id'].dropna()
    id_lookup = id_lookup.loc[id_lookup.str.len().sort_values(ascending=False).index]

    file_to_scene = {}
    unmatched = []
    for f in candidates:
        match = next(
            (scene_id for scene_id, bid in id_lookup.items() if f.stem.startswith(bid)),
            None,
        )
        if match is None:
            unmatched.append(f.name)
            continue
        file_to_scene[f] = match

    if unmatched:
        preview = unmatched[:5]
        logger.warning(
            f"{len(unmatched)} brdf files had no matching brdf_id: "
            f"{preview}{'...' if len(unmatched) > 5 else ''}"
        )

    return file_to_scene


def compute_cast_shadow(
    dem: np.ndarray,
    cellsize: float,
    sun_zenith_deg: float,
    sun_azimuth_deg: float,
) -> np.ndarray:
    """Boolean cast-shadow mask (True = in shadow), via a
    rotate-and-scan horizon algorithm (Corripio 2003 style).

    Rotates the DEM so the sun's azimuth direction aligns with the
    row axis, then for each column marches from the sun-facing edge
    outward, tracking the highest 'projected sun-ray height'
    encountered so far. Any pixel lower than that projected height
    is occluded.

    CAVEAT: the rotation direction/sign convention and the
    center-crop alignment after rotating back have NOT been verified
    against a real scene. Validate visually (e.g. compare against a
    simple hillshade for one known sun position) before trusting this
    at scale -- getting azimuth sign/quadrant wrong is a common,
    easy-to-miss bug in this kind of algorithm.
    """
    sun_elevation_deg = 90.0 - sun_zenith_deg
    if sun_elevation_deg <= 0:
        # Sun below the horizon -- everything is in shadow.
        return np.ones(dem.shape, dtype=bool)

    tan_e = np.tan(np.radians(sun_elevation_deg))

    # Rotate so marching along increasing row index = marching away
    # from the sun. scipy.ndimage.rotate's angle is counterclockwise;
    # verify this against your actual azimuth convention (0=N,
    # clockwise, as used by sun_azimuth in the metadata) before
    # trusting the sign here.
    rot_angle = sun_azimuth_deg
    dem_rot = ndi_rotate(
        dem, angle=rot_angle, reshape=True, order=1, mode='constant', cval=np.nan
    )

    nrows, ncols = dem_rot.shape
    shadow_rot = np.zeros_like(dem_rot, dtype=bool)
    horizon = np.full(ncols, -np.inf)

    for r in range(nrows):
        row = dem_rot[r, :]
        valid = ~np.isnan(row)
        shadow_rot[r, valid] = row[valid] < horizon[valid]
        horizon[valid] = np.maximum(horizon[valid], row[valid])
        horizon -= cellsize * tan_e

    shadow_back = ndi_rotate(
        shadow_rot.astype(np.uint8),
        angle=-rot_angle,
        reshape=True,
        order=0,
        mode='constant',
        cval=0,
    )
    # Center-crop back to the original DEM shape (rotate w/
    # reshape=True pads the array). Verify this crop is centered
    # correctly for your actual array dimensions.
    dh = (shadow_back.shape[0] - dem.shape[0]) // 2
    dw = (shadow_back.shape[1] - dem.shape[1]) // 2
    shadow = shadow_back[dh:dh + dem.shape[0], dw:dw + dem.shape[1]].astype(bool)

    return shadow


def compute_self_shadow(
    dem: np.ndarray,
    sun_zenith_deg: float,
    sun_azimuth_deg: float,
) -> np.ndarray:
    """Boolean self-shadow mask (True = facet faces away from sun),
    via the standard illumination-angle cosine test on slope/aspect.
    """
    dem_da = xr.DataArray(dem, dims=('y', 'x'))
    slope = xrspatial.slope(dem_da).values       # degrees
    aspect = xrspatial.aspect(dem_da).values     # degrees, 0=N clockwise

    slope_r = np.radians(slope)
    aspect_r = np.radians(aspect)
    zenith_r = np.radians(sun_zenith_deg)
    azimuth_r = np.radians(sun_azimuth_deg)

    cos_i = (
        np.cos(zenith_r) * np.cos(slope_r)
        + np.sin(zenith_r) * np.sin(slope_r) * np.cos(azimuth_r - aspect_r)
    )
    return cos_i <= 0


def align_dem_to_raster(dem_path: Path, ref_transform, ref_crs, ref_shape) -> np.ndarray:
    """Returns the DEM as an array matched to the reference raster's
    transform/CRS/shape, reprojecting only if they differ.
    """
    with rasterio.open(dem_path) as dsrc:
        same_grid = (
            dsrc.transform == ref_transform
            and dsrc.crs == ref_crs
            and (dsrc.height, dsrc.width) == ref_shape
        )
        if same_grid:
            return dsrc.read(1).astype(np.float32)

        dem = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(dsrc, 1),
            destination=dem,
            src_transform=dsrc.transform,
            src_crs=dsrc.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=RioResampling.bilinear,
        )
        return dem


def make_shade_mask(
    scene_tif: Path,
    dem_path: Path,
    sun_zenith_deg: float,
    sun_azimuth_deg: float,
    out_path: Path,
) -> Path:
    """Computes and saves a combined cast+self shadow mask for one
    scene, matched to that scene's exact grid.
    """
    with rasterio.open(scene_tif) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
        cellsize = abs(transform.a)  # assumes square pixels

    dem = align_dem_to_raster(dem_path, transform, crs, shape)

    cast_shadow = compute_cast_shadow(dem, cellsize, sun_zenith_deg, sun_azimuth_deg)
    self_shadow = compute_self_shadow(dem, sun_zenith_deg, sun_azimuth_deg)
    shadow_mask = (cast_shadow | self_shadow).astype('uint8')

    out_profile = profile.copy()
    out_profile.update(dtype='uint8', count=1, nodata=255, compress='lzw')

    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(shadow_mask, 1)

    return out_path


"""
terrain_shadow.py

Computes a combined cast-shadow + self-shadow mask for each
BRDF-corrected raster in a grid cell, using that cell's Copernicus
DEM and the per-scene sun zenith/azimuth already stored in
processing.info.

Run as a standalone step, separate from the download/metadata
pipeline. Does NOT yet apply the shiftX/shiftY co-registration
offsets -- that's a follow-on step once masks are validated.
"""

import logging
import typing as T
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import xrspatial
from rasterio.enums import Resampling as RioResampling
from rasterio.warp import reproject
from scipy.ndimage import rotate as ndi_rotate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def match_brdf_files(brdf_dir: Path, processing_info: pd.DataFrame) -> T.Dict[Path, str]:
    """Maps each BRDF .tif file to its processing.info scene id, via
    the 'brdf_id' column -- a prefix of the actual filename, which
    may have extra text appended before '.tif'.

    Files ending in 'X.tif' are excluded (a different product type,
    not an image to mask).
    """
    brdf_dir = Path(brdf_dir)
    candidates = [f for f in brdf_dir.glob('*.tif') if not f.name.endswith('X.tif')]

    id_lookup = processing_info['brdf_id'].dropna()
    # Check longer ids first so a more specific id can't be shadowed
    # by a shorter one that also happens to be a valid prefix.
    id_lookup = id_lookup.loc[id_lookup.str.len().sort_values(ascending=False).index]

    file_to_scene: T.Dict[Path, str] = {}
    unmatched = []
    for f in candidates:
        stem = f.stem
        match = next(
            (scene_id for scene_id, bid in id_lookup.items() if stem.startswith(bid)),
            None,
        )
        if match is None:
            unmatched.append(f.name)
            continue
        file_to_scene[f] = match

    if unmatched:
        preview = unmatched[:5]
        logger.warning(
            f"{len(unmatched)} brdf files had no matching brdf_id: "
            f"{preview}{'...' if len(unmatched) > 5 else ''}"
        )

    return file_to_scene


def compute_cast_shadow(
    dem: np.ndarray,
    cellsize: float,
    sun_zenith_deg: float,
    sun_azimuth_deg: float,
) -> np.ndarray:
    """Boolean cast-shadow mask (True = in shadow), via a
    rotate-and-scan horizon algorithm (Corripio 2003 style).

    Rotates the DEM so the sun's azimuth direction aligns with the
    row axis, then for each column marches from the sun-facing edge
    outward, tracking the highest 'projected sun-ray height'
    encountered so far. Any pixel lower than that projected height
    is occluded.

    CAVEAT: the rotation direction/sign convention and the
    center-crop alignment after rotating back have NOT been verified
    against a real scene. Validate visually (e.g. compare against a
    simple hillshade for one known sun position) before trusting this
    at scale -- getting azimuth sign/quadrant wrong is a common,
    easy-to-miss bug in this kind of algorithm.
    """
    sun_elevation_deg = 90.0 - sun_zenith_deg
    if sun_elevation_deg <= 0:
        # Sun below the horizon -- everything is in shadow.
        return np.ones(dem.shape, dtype=bool)

    tan_e = np.tan(np.radians(sun_elevation_deg))

    # Rotate so marching along increasing row index = marching away
    # from the sun. scipy.ndimage.rotate's angle is counterclockwise;
    # verify this against your actual azimuth convention (0=N,
    # clockwise, as used by sun_azimuth in the metadata) before
    # trusting the sign here.
    rot_angle = sun_azimuth_deg
    dem_rot = ndi_rotate(
        dem, angle=rot_angle, reshape=True, order=1, mode='constant', cval=np.nan
    )

    nrows, ncols = dem_rot.shape
    shadow_rot = np.zeros_like(dem_rot, dtype=bool)
    horizon = np.full(ncols, -np.inf)

    for r in range(nrows):
        row = dem_rot[r, :]
        valid = ~np.isnan(row)
        shadow_rot[r, valid] = row[valid] < horizon[valid]
        horizon[valid] = np.maximum(horizon[valid], row[valid])
        horizon -= cellsize * tan_e

    shadow_back = ndi_rotate(
        shadow_rot.astype(np.uint8),
        angle=-rot_angle,
        reshape=True,
        order=0,
        mode='constant',
        cval=0,
    )
    # Center-crop back to the original DEM shape (rotate w/
    # reshape=True pads the array). Verify this crop is centered
    # correctly for your actual array dimensions.
    dh = (shadow_back.shape[0] - dem.shape[0]) // 2
    dw = (shadow_back.shape[1] - dem.shape[1]) // 2
    shadow = shadow_back[dh:dh + dem.shape[0], dw:dw + dem.shape[1]].astype(bool)

    return shadow


def compute_self_shadow(
    dem: np.ndarray,
    sun_zenith_deg: float,
    sun_azimuth_deg: float,
) -> np.ndarray:
    """Boolean self-shadow mask (True = facet faces away from sun),
    via the standard illumination-angle cosine test on slope/aspect.
    """
    dem_da = xr.DataArray(dem, dims=('y', 'x'))
    slope = xrspatial.slope(dem_da).values       # degrees
    aspect = xrspatial.aspect(dem_da).values     # degrees, 0=N clockwise

    slope_r = np.radians(slope)
    aspect_r = np.radians(aspect)
    zenith_r = np.radians(sun_zenith_deg)
    azimuth_r = np.radians(sun_azimuth_deg)

    cos_i = (
        np.cos(zenith_r) * np.cos(slope_r)
        + np.sin(zenith_r) * np.sin(slope_r) * np.cos(azimuth_r - aspect_r)
    )
    return cos_i <= 0


def align_dem_to_raster(dem_path: Path, ref_transform, ref_crs, ref_shape) -> np.ndarray:
    """Returns the DEM as an array matched to the reference raster's
    transform/CRS/shape, reprojecting only if they differ.
    """
    with rasterio.open(dem_path) as dsrc:
        same_grid = (
            dsrc.transform == ref_transform
            and dsrc.crs == ref_crs
            and (dsrc.height, dsrc.width) == ref_shape
        )
        if same_grid:
            return dsrc.read(1).astype(np.float32)

        dem = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(dsrc, 1),
            destination=dem,
            src_transform=dsrc.transform,
            src_crs=dsrc.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=RioResampling.bilinear,
        )
        return dem


def make_shade_mask(
    scene_tif: Path,
    dem_path: Path,
    sun_zenith_deg: float,
    sun_azimuth_deg: float,
    out_path: Path,
) -> Path:
    """Computes and saves a combined cast+self shadow mask for one
    scene, matched to that scene's exact grid.
    """
    with rasterio.open(scene_tif) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
        cellsize = abs(transform.a)  # assumes square pixels

    dem = align_dem_to_raster(dem_path, transform, crs, shape)

    cast_shadow = compute_cast_shadow(dem, cellsize, sun_zenith_deg, sun_azimuth_deg)
    self_shadow = compute_self_shadow(dem, sun_zenith_deg, sun_azimuth_deg)
    shadow_mask = (cast_shadow | self_shadow).astype('uint8')

    out_profile = profile.copy()
    out_profile.update(dtype='uint8', count=1, nodata=255, compress='lzw')

    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(shadow_mask, 1)

    return out_path

    
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

                        
def mask_images(params):

    """
    Masks clouds (and cloud shadows) or shaded terrain

    if <params.masking.method> is 'CRF", uses Conditional Random Field method above to mask clouds and cloud shadows. 
    if <params.masking.method> is 's2cloudless", uses s2cloudless masks from GEE to mask clouds and cloud shadows.
    if <params.masking.method> is 'terrain_shade' uses...
    otherwise, uses native masks (does nothing here) and adds 'native masks only' to 'masking' field in Processing info.
    """   
    for grid in params['grids']:

        ppaths = ProjectPaths(params, grid=grid)

        processing_db = pd.read_pickle(ppaths.ms.parent/'processing.info')
        if params['masking']['method'] == 'terrain_shade':
            if 'shade_mask' not in processing_db:  ## This is the master db that tracks the progress of each individual image
                processing_db['masking'] = np.nan
        elif 'masking' not in processing_db: ## This is the master db that tracks the progress of each individual image
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

        check_download_db = False
        if not params['status']['check_downloads']:
            check_download_db = True

        if check_download_db:
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
        
        elif params['masking']['method'] == 'terrain_shade':
            logger.info(f'  Creating cast+self shadow masks ...')
            dem_path = ppaths.ms.parent / 'terrain' / 'dem.tif'
            out_dir = ppaths.ms.parent / 'terrain' / 'shade_masks'
                out_dir.mkdir(parents=True, exist_ok=True)
        
            file_to_scene = match_brdf_files(ppaths.ms, processing_db, exclude=params['masking']['exclude'])

            n_ok, n_skip = 0, 0
            for tif_path, scene_id in file_to_scene.items():
                row = processing_db.loc[scene_id]
                zenith = row.get('sun_zenith')
                azimuth = row.get('sun_azimuth')
                if pd.isna(zenith) or pd.isna(azimuth):
                    logger.warning(f"No sun angles for {scene_id}; skipping {tif_path.name}")
                    n_skip += 1
                    continue

                out_path = out_dir / f"{tif_path.stem}_shademask.tif"
                try:
                    make_shade_mask(tif_path, dem_path, float(zenith), float(azimuth), out_path)
                    n_ok += 1
                    row['shade_mask']=out_path
                except Exception as e:
                    logger.warning(f"Failed to make shade mask for {tif_path.name}: {e}")
                    n_skip += 1

            logger.info(f"Shade masks: {n_ok} created, {n_skip} skipped -> {out_dir}")
    
        db.update(grid, 'mask')
