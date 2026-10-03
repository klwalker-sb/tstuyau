import shutil
from pathlib import Path
import string
import random
import geowombat as gw
from geowombat.data import srtm30m_bounding_boxes
from geowombat.core import dask_to_xarray, ndarray_to_xarray
from geowombat.radiometry.topo import calc_slope_delayed, calc_aspect_delayed
from geowombat.moving import moving_window
from rastercrf.util import scale_min_max, transform_data, nd_to_columns, columns_to_nd
import numpy as np
import cv2
import rasterio as rio
from rasterio.windows import Window
import geopandas as gpd
import dask.array as da
import xarray as xr
import xrspatial
from scipy.ndimage import rotate as ndi_rotate, binary_dilation
import dask.array as da
from tqdm import tqdm
from .project import ProjectPaths
from .lookup import MASKS
from .io import extract_profile_geotif, default_profile_netcdf
from ..handler import logger

LANDSAT_LIKE_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

KERNEL_CROSS = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype='uint8')

KERNEL_SQUARE = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype='uint8')

KERNEL_DISC = np.array([[0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0]], dtype='uint8')


def get_mask_kwargs(img_dir, params, mask_args={}):
    
    mask_type = params['masking']['method']
    mask_defaults = MASKS.get(mask_type, {})
    logger.info(f'mask_defaults = {mask_defaults}')
    mask_dir =  params['masking']['mask_dir']
    if (mask_dir is None) or (mask_dir == 'None'):
        mask_dir = Path(img_dir).parent / f"{mask_defaults.get('mask_dir')}"
    mask_name = params['masking']['maskname']
    if (mask_name is None) or (mask_name == 'None'):
        mask_name = mask_defaults.get('maskname')
    mask_val = params['masking']['mask_val']
    if (mask_val is None) or (mask_val == 'None'):
        mask_val = mask_defaults.get('mask_val')
    logger.debug(f'mask_val = {mask_val}')

    if isinstance(mask_val, int):
        mask_start = mask_val
        mask_stop = mask_val
    elif isinstance(mask_val, list):
        mask_start = mask_val[0]
        mask_stop = mask_val[-1]
    else:
        logger.warning(f'need to enter mask vals in params:masking:mask_val or MASKS dict for {mask_type}')
        
    mask_args['mask_type'] = mask_type
    mask_args['mask_dir'] = mask_dir
    mask_args['maskname'] = mask_name
    mask_args['mask_val'] = mask_val
    mask_args['mask_start'] = mask_start
    mask_args['mask_stop'] = mask_stop

    logger.info(f'mask_args: {mask_args}')
    return mask_args
    

def apply_masks_to_images(images, img_dir, masked_img_dir, mask_args={}, params=None):
    """
    Applies existing masks to images and saves to temp dir for use in index construction and smoothing operations
    <images> can be a list of images or a dictionary with image names as keys. 
    <masked_img_dir> is a directory, usually temporary, where the output masked images will be stored.
    <mask_args> is a dictionary built from input masking parameters of MASKS dictionary. If None, needs params passed to build.

    Outputs a masked image with same name and format as the original in the  <masked_img_dir>
    Returns a list of images to process, which is usually the same list as the input <images> unless some images are missing masks and
        params:masking:treat_missing> is set to 'skip'
    """

    if not mask_args:
        mask_args = get_mask_kwargs(img_dir, params)
        
    def get_image_basename(im_path):
        if 'coreg' in im_path.stem:
            im_base = im_path.stem.split('_coreg')[0]
        elif '_.nc' in im_path.name:
            im_base = im_path.name.split('_.nc')[0]
        else:
            im_base = im_path.name.split('.nc')[0]
        return im_base 
    
    mask_dir = Path(img_dir).parent / mask_args['mask_dir']
    mask_start = mask_args['mask_start']
    mask_stop = mask_args['mask_stop']

    set_chunks = None
    if params['reconstruct']['chunks']:
        set_chunks = params['reconstruct']['chunks']
    if (set_chunks is None) or (set_chunks == 'None'):
        set_shunks = 512
        
    if isinstance(images, dict):
        image_names = list(images)
    elif isinstance(images[-1],pathlib.Path):
        image_names = [i.name for i in images]
    elif '/' in images[-1]:
        image_names = [Path(i).name for i in images]
    else: image_names = images

    logger.info(f" checking masks in {str(mask_dir)}...")
    missing = []
    for im in image_names:
        im_base = get_image_basename(Path(im))
        mask_path = mask_dir / f"{im_base}_{mask_args['maskname']}.tif"
        if not mask_path.is_file():
            missing.append(im)
            
    if missing:
        logger.warning(f"{len(missing)} of {len(image_names)} scenes have no {mask_args['maskname']} file in {mask_args['mask_dir']}:")
        logger.warning(f"Missing items: {missing[:5]}{'...' if len(missing) > 5 else ''}. ")
        if params['masking']['treat_missing'].lower() == 'fail':
            logger.warning(f"try running the masks before running this step")
            return
        elif params['masking']['treat_missing'].lower() == 'skip':
            image_names = [item for item in image_names if item not in missing]
            logger.warning(f"skipping these images -- image stack has been reduced to {len(image_names)} images")
        else:
            logger.warning("need to set parameter masking:treat_missing to 'skip' or 'fail'")
            return

    ref_image = image_names[-1]
    input_img_type = Path(ref_image).suffix.lower().replace('.', '')
    if input_img_type == 'tif':
        profile_kwargs = extract_profile_geotif(ref_image, band_names=LANDSAT_LIKE_BANDS, n_chunks=set_chunks)
    elif input_img_type == 'nc':
        profile_kwargs = default_profile_netcdf(ref_image)

    img_open_kwargs = (
        {'engine': 'h5netcdf', 'chunks': {'band': -1, 'y': set_chunks, 'x': set_chunks}} 
        if input_img_type == 'nc' 
        else {'chunks': {'band': 1, 'y': set_chunks, 'x': set_chunks}}
    )
    
    for im in image_names:
        im_path = Path(im)
        logger.debug(f'applying mask to {im}')
        out_path = Path(masked_img_dir) / im_path.name
        im_base = get_image_basename(im_path)
        mask_path = mask_dir / f"{im_base}_{mask_args['maskname']}.tif"
        with gw.open(im_path, band_names=LANDSAT_LIKE_BANDS, **img_open_kwargs) as img_src:
            with gw.open(mask_path) as mask_src:
                mask_da = mask_src.isel(band=0, drop=True) if 'band' in mask_src.dims else mask_src
                ## converting both dask arrays to numpy forces them to use grid alignment rather than coordinate
                ##   because coordinate alignment becomes a problem with images that have been coregistered.
                mask_values = mask_da.values
                img_values = img_src.values
                if mask_values.shape != img_values.shape[-2:]:
                    logger.warning:(f"ERROR -- mask shape {mask_values.shape} != image shape {img_values.shape[-2:]} for {im_path.name}")
                is_kept = (mask_values < mask_start) | (mask_values > mask_stop)
                nodata_fill = profile_kwargs.get('_FillValue', profile_kwargs.get('nodata', 0))
                masked_arr = np.where(is_kept[np.newaxis, :, :], img_values, nodata_fill)
                masked_arr = masked_arr.astype(img_src.dtype)

                ## best to keep output same as input image at this point so that the masksed images can be slotted right back into the 
                ##    reconstruction processing. Final outputs will be dictated by the parameters for the larger process being run.
                if input_img_type == 'tif':
                    profile_kwargs['dtype'] = str(img_src.dtype)
                    out_profile = dict(profile_kwargs)
                    out_profile['count'] = masked_arr.shape[0]
                    with rio.open(out_path, 'w', **out_profile) as dst:
                        dst.write(masked_arr)
                elif input_img_type == 'nc':
                    ## need to convert from numpy back to dask array for gw.to_netcdf to work
                    masked_arr_da = da.from_array(masked_arr, chunks=img_src.data.chunksize)
                    masked_img = img_src.copy(data=masked_arr_da)
                    ## gw.to_netcdf requires a name for the array
                    masked_img.name = 'masked_data'
                    masked_img.gw.to_netcdf(out_path, overwrite=True, **profile_kwargs)

    return image_names
        
def open_masks_with_time_series(ppaths, ts_stack, ds_stack, **mask_args):
    """Opens each scene's mask, matched 1:1 to ts_stack by filename as a single time-stacked boolean DataArray aligned to ds_stack.
    Returns a (time, y, x) DataArray, True where a pixel is in should be excluded, aligned to the same time coordinate as ts_stack itself.
    Raises FileNotFoundError if any scene in ts_stack has no corresponding mask 
    Note: not using currently because running masking through reconstruction is with apply_masks_to_images() 
          is cleaner in the full processign pipeline even though it requires saving the masked images temporarily. 
    """
    mask_paths = []
    missing = []
    
    for p in ts_stack:
        p_base = Path(p).stem.split('_coreg')[0]
        mask_path = Path(mask_dir) / f"{p_base}_{maskname}.tif"
        if not mask_path.is_file():
            missing.append(Path(p).name)
        mask_paths.append(mask_path)
 
    if missing:
        logger.warning(
            f"{len(missing)} of {len(ts_stack)} scenes have no shade mask in "
            f"{mask_dir}: {missing[:5]}{'...' if len(missing) > 5 else ''}. "
        )
 
    with gw.open(mask_paths, time_names=ds_stack) as mask_src:
        mask_da = mask_src.load()
    if 'band' in mask_da.dims:
        mask_da = mask_da.squeeze('band', drop=True)
        
    return (mask_da >= mask_start) & (mask_da <= mask_stop)


def _random_id(string_length):
    """
    Generates a random string of letters and digits
    """

    letters_digits = string.ascii_letters + string.digits

    return ''.join(random.choice(letters_digits) for i in range(string_length))


def apply_binary_mask(in_ras, mask, printmap=False, out_path=None, **profile):
    """
    applies existing binary mask (0 = exclude, 1 = keep) to <in_ras> 
    if <printmap> == True, prints out file to <out_path> and returns <out_path>
       otherwise, returns result as numpy array.
    This is a much simpler structure than apply_masks_to_images() to be used for pre- or post-processing steps
    """
    if isinstance(in_ras, np.ndarray):
        data = in_ras
    elif in_ras.endswith('.tif'):
        with rio.open(in_ras) as src:
            data = src.read()
            profile = src.profile

    with rio.open(mask) as mask_src:
        mask_data = mask_src.read()

    out_data = data * mask_data

    if printmap:
        with rio.open(out_path, 'w', **profile) as dst:
            dst.write(out_data)
        return out_path
    else:
        return out_data
        
def combine_binary_masks(in_masks, printmask=False, out_path=None):
    '''
    combines multiple binary masks into single mask (0 = exclude, 1 = keep)
    A pixel is excluded if it is masked in any of the input masks.
    if <printmask> == True, prints final mask to <out_path> and returns <out_path>
       otherwise, returns final mask as numpy array
    '''
    ## TODO: find the smallest and clip the other(s) to it first
    with rio.open(in_masks[0]) as src:
        data = src.read()
        out_meta = src.meta.copy()
    mask_out = data
    
    for mask in in_masks[1:]:
        with rio.open(mask) as ras:
            data = ras.read()
        mask_out = mask_out * data

    if printmask:
        with rio.open(out_path, 'w', **out_meta) as dst:
            dst.write(mask_out)
        return out_path
    else:
        return mask_out

def buffer_mask(mask, buffer_px=1):
    """
    Expands a boolean mask by <buffer_px> pixels in every direction.
    buffer_px=0 returns mask unchanged
    """
    if buffer_px <= 0:
        return mask
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure, iterations=buffer_px)


def open_dem_aligned(dem_path, ref_bounds, ref_crs, res):
    """
    Opens dem_path aligned (bounds/crs/resolution) to the given
    reference grid, returning a loaded, squeezed DataArray. Used by
    compute_cast_shadow and compute_self_shadow when a pre-aligned
    DEM isn't passed in.
    """
    with gw.config.update(
        ref_bounds=ref_bounds, ref_crs=ref_crs, ref_res=res, nodata=255, ignore_warnings=True
    ):
        with gw.open(dem_path, resampling='bilinear') as dem_src:
            dem_da = dem_src.squeeze(drop=True).load()
    return dem_da


def compute_cast_shadow(sun_zenith_deg, sun_azimuth_deg, dem=None, cellsize=None,
                        dem_path=None, ref_bounds=None, ref_crs=None, res=None):
    """
    Boolean cast-shadow mask (True = in shadow), via a rotate-and-scan horizon algorithm (Corripio 2003 style).

    Rotates the DEM so the sun's azimuth direction aligns with the row axis, then for each column marches 
    from the sun-facing edge outward, tracking the highest 'projected sun-ray height' encountered so far. 
    Any pixel lower than that projected height is shaded.
    """
    if dem is None:
        if dem_path is None:
            logger.warning('ERROR - compute_cast_shadow needs either dem (with cellsize) or dem_path (with ref_bounds, ref_crs, res)')
        dem_da = open_dem_aligned(dem_path, ref_bounds, ref_crs, res)
        dem = dem_da.values.astype(np.float32)
        cellsize = float(res)
    elif cellsize is None:
        logger.warning('ERROR - compute_cast_shadow needs cellsize when dem is passed in directly')
    
    sun_elevation_deg = 90.0 - sun_zenith_deg
    if sun_elevation_deg <= 0:
        # Sun below the horizon -- everything is in shadow.
        return np.ones(dem.shape, dtype=bool)

    tan_e = np.tan(np.radians(sun_elevation_deg))

    # Rotate so marching along increasing row index = marching away
    # from the sun. scipy.ndimage.rotate's angle is counterclockwise;
    # TODO verify this against actual azimuth convention (0=N,
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
    # correctly for the actual array dimensions.
    dh = (shadow_back.shape[0] - dem.shape[0]) // 2
    dw = (shadow_back.shape[1] - dem.shape[1]) // 2
    shadow = shadow_back[dh:dh + dem.shape[0], dw:dw + dem.shape[1]].astype(bool)

    return shadow

def compute_self_shadow(sun_zenith_deg, sun_azimuth_deg, dem_da=None,
                        dem_path=None, ref_bounds=None, ref_crs=None, res=None):
    """
    Boolean self-shadow mask (True = facet faces away from sun),
    via the standard illumination-angle cosine test on slope/aspect (Lambertian cosine-of-incidence formula).
    This is similar to calc_il below, but is boolean (only concerned with whether cos(i) <= 0) 
    and uses a single value for the Zenith and Azumith rather than the angles rasters
       (note: if the Zenith and Azumith are the global scene-based value, actual cell values can differ by up to 0.5deg
       this could alter calculations by up to 1 pixel for 10 m images, but is likely to be a smaller source of error than
       the DEM itself. Buffering masks by 1-3 pixels is recommended given this uncertainity). 
    """
    if dem_da is None:
        if dem_path is None:
            logger.warning('ERROR - compute_self_shadow needs either dem_da or dem_path (with ref_bounds, ref_crs, res)')
        dem_da = open_dem_aligned(dem_path, ref_bounds, ref_crs, res)

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
    
def get_srtm_grids(data, dem_path):

    # Read the SRTM data
    srtm_grid_path_temp = dem_path / f'srtm30m_bounding_boxes_{_random_id(9)}.gpkg'

    # Make a copy of the SRTM grids, read, and delete
    shutil.copy(str(srtm30m_bounding_boxes), str(srtm_grid_path_temp))
    srtm_df = gpd.read_file(srtm_grid_path_temp)
    srtm_grid_path_temp.unlink()

    # Get the grids that intersect the image
    srtm_df_int = srtm_df[srtm_df.geometry.intersects(data.gw.geodataframe.to_crs(epsg=4326).geometry.values[0])]

    zip_paths = []

    for dfn in srtm_df_int.dataFile.values.tolist():

        zip_file = dem_path / f"NASADEM_HGT_{dfn.split('.')[0].lower()}.zip"

        src_zip = f"zip+file://{zip_file}!/{Path(zip_file).stem.split('_')[-1]}.hgt"

        zip_paths.append(src_zip)

    if len(zip_paths) == 1:
        zip_paths = zip_paths[0]
        mosaic = False
    else:
        mosaic = True

    return zip_paths, mosaic


def calc_il(data, dem_path=None, angles=None, num_workers=None, params=None):
    """
    Calculates illumination from DEM (srtm here -- TODO: update for others) from the Lambertian cosine-of-incidence formula
    Zenith and Azumith are pulled from angles DataArray, scaled by 0.01
    Returns the raw il value (a continuous illumination score, roughly -1 to 1)
    """
    if not dem_path:
        ppaths=ProjectPaths(params)
        dem_path = ppaths.dem

    if params is None or not params['topo']['demsource']:
        demsource = 'srtm'
    else:
        demsource = params['topo']['demsource']

    if demsource == 'srtm':
        zip_paths, mosaic = get_srtm_grids(data, srtm_path)

    slope_kwargs = dict(format='MEM',
                        computeEdges=True,
                        alg='ZevenbergenThorne',
                        slopeFormat='degree')

    aspect_kwargs = dict(format='MEM',
                         computeEdges=True,
                         alg='ZevenbergenThorne',
                         trigonometric=False,
                         zeroForFlat=True)

    slope_kwargs['format'] = 'MEM'
    slope_kwargs['slopeFormat'] = 'degree'
    aspect_kwargs['format'] = 'MEM'

    # Force to SRTM resolution
    # proc_dims = (int((data.gw.ncols*data.gw.cellx) / 30.0),
    #              int((data.gw.nrows*data.gw.celly) / 30.0))

    w = int((5 * 30.0) / data.gw.celly)

    if w % 2 == 0:
        w += 1

    with gw.open(zip_paths, mosaic=mosaic, resampling='nearest') as elev:

        elev = elev.gw.transform_crs(dst_res=30.0, resampling='nearest')

        # Slope
        slope_deg = calc_slope_delayed(elev.squeeze().data, proc_dims=(elev.gw.nrows, elev.gw.ncols), w=w, **slope_kwargs)
        slope_deg_fd = da.from_delayed(slope_deg, (elev.gw.nrows, elev.gw.ncols), dtype='float64')

        # Aspect
        aspect_deg = calc_aspect_delayed(elev.squeeze().data, proc_dims=(elev.gw.nrows, elev.gw.ncols), w=w, **aspect_kwargs)
        aspect_deg_fd = da.from_delayed(aspect_deg, (elev.gw.nrows, elev.gw.ncols), dtype='float64')

        # Degrees -> radians
        slope_rad = da.deg2rad(slope_deg_fd)
        aspect_rad = da.deg2rad(aspect_deg_fd)

        slope_rad = dask_to_xarray(elev, slope_rad, ['slope']).assign_attrs(res=(elev.res, elev.res))
        aspect_rad = dask_to_xarray(elev, aspect_rad, ['aspect']).assign_attrs(res=(elev.res, elev.res))

        slope_rad = slope_rad.gw.transform_crs(dst_bounds=data.gw.bounds, dst_res=data.res, resampling='cubic')
        aspect_rad = aspect_rad.gw.transform_crs(dst_bounds=data.gw.bounds, dst_res=data.res, resampling='cubic')

        solar_za = xr.ufuncs.deg2rad(angles.sel(band='sza') * 0.01).expand_dims(dim='band')
        solar_az = xr.ufuncs.deg2rad(angles.sel(band='saa') * 0.01).expand_dims(dim='band')

        cos_z = xr.ufuncs.cos(solar_za)

        il = (xr.ufuncs.cos(slope_rad.sel(band='slope')) *
              cos_z.sel(band='sza') +
              xr.ufuncs.sin(slope_rad.sel(band='slope')) *
              xr.ufuncs.sin(solar_za.sel(band='sza')) *
              xr.ufuncs.cos(solar_az.sel(band='saa') - aspect_rad.sel(band='aspect')))\
            .expand_dims(dim='band')\
            .assign_coords(coords={'band': ['il']})\
            .assign_attrs(**data.attrs)

        if il.gw.has_time_coord:
            il = il.transpose('time', 'band', 'y', 'x')
        else:
            il = il.transpose('band', 'y', 'x')

    # il = cv2.resize(il.compute(num_workers=num_workers),
    #                 (0, 0),
    #                 fy=data.gw.nrows / il.shape[0],
    #                 fx=data.gw.ncols / il.shape[1],
    #                 interpolation=cv2.INTER_CUBIC)

    return il
    
def masks_to_file(sat_bands,
                  cloud_probas,
                  shadow_probas,
                  outfile,
                  cloud_labels,
                  shadow_labels,
                  pred_kwargs,
                  cloud_proba_thresh=0.25,
                  shadow_proba_thresh=0.75,
                  w=3,
                  num_workers=1):

    """
    Args:
        sat_bands (ndarray): The model mask band.
        cloud_probas (ndarray): The cloud model probabilities.
        shadow_probas (ndarray): The shadow model probabilities.
        outfile (str): The output file path and name.
        cloud_labels (list): The cloud class labels.
        shadow_labels (list): The shadow class labels.
        pred_kwargs (dict): Keyword arguments for file creation.
        cloud_proba_thresh (float): The cloud probability minimum threshold.
        shadow_proba_thresh (float): The shadow probability minimum threshold.
        w (int): The moving window size for smoothing probabilities. Currently not used.
        num_workers (int): The number of parallel workers.

    Values:
        clear: 0
        water: 1
        shadow: 2
        snow or ice: 3
        cloud: 4
        cirrus: 5
        adjacent cloud: 6
        adjacent shadow: 7
        fill: 7
    """

    pad = 9

    if cloud_probas.dtype.name in ['float32', 'float64']:

        # for i in range(0, cloud_probas.shape[0]):
        #
        #     if class_labels[i] != 'c':
        #
        #         proba_layer = cloud_probas[i]
        #
        #         proba_layer[(proba_layer > 1) | (proba_layer < 0) | np.isnan(proba_layer) | np.isinf(proba_layer)] = 0
        #
        #         cloud_probas[i] = moving_window(np.ascontiguousarray(np.pad(proba_layer, ((pad, pad), (pad, pad)),
        #                                                                     mode='reflect'), dtype='float64'),
        #                                         stat='mean',
        #                                         w=w,
        #                                         weights=True,
        #                                         n_jobs=num_workers)[pad:-pad, pad:-pad]

        def resample_probas(probas):

            # Resample the probabilities to full extent
            proba_layers = np.zeros((probas.shape[0],
                                     pred_kwargs['height'],
                                     pred_kwargs['width']), dtype='float64')

            for i in range(0, probas.shape[0]):

                proba_layers[i] = np.float64(cv2.resize(np.float32(probas[i]),
                                                        (int(pred_kwargs['width']),
                                                         int(pred_kwargs['height'])),
                                                        interpolation=cv2.INTER_CUBIC))

            return proba_layers

        cloud_proba_layers = resample_probas(cloud_probas)
        shadow_proba_layers = resample_probas(shadow_probas)

        # Threshold the probabilities
        cloud_proba_layers[cloud_labels.index('c')] = \
            np.where(cloud_proba_layers[cloud_labels.index('c')] < cloud_proba_thresh,
                     0,
                     cloud_proba_layers[cloud_labels.index('c')])

        shadow_proba_layers[shadow_labels.index('s')] = \
            np.where(shadow_proba_layers[shadow_labels.index('s')] < shadow_proba_thresh,
                     0,
                     shadow_proba_layers[shadow_labels.index('s')])

    nodata_mask = np.uint8(cv2.resize(sat_bands[-1],
                                      (int(pred_kwargs['width']),
                                       int(pred_kwargs['height'])),
                                      interpolation=cv2.INTER_NEAREST))

    def create_mask_dict(class_labels):

        mask_dict = {255: 255}

        if 'v' in class_labels:
            mask_dict[class_labels.index('v')] = 0
        if 'd' in class_labels:
            mask_dict[class_labels.index('d')] = 0
        if 'b' in class_labels:
            mask_dict[class_labels.index('b')] = 0
        if 'wv' in class_labels:
            mask_dict[class_labels.index('wv')] = 0
        if 'w' in class_labels:
            mask_dict[class_labels.index('w')] = 1
        if 's' in class_labels:
            mask_dict[class_labels.index('s')] = 2
        if 'h' in class_labels:
            mask_dict[class_labels.index('h')] = 0
        if 'hd' in class_labels:
            mask_dict[class_labels.index('hd')] = 0
        if 'hv' in class_labels:
            mask_dict[class_labels.index('hv')] = 0
        if 'c' in class_labels:
            mask_dict[class_labels.index('c')] = 4
        if 'n' in class_labels:
            mask_dict[class_labels.index('n')] = 255

        return mask_dict

    masking_dict = create_mask_dict(cloud_labels)
    shadow_mask_dict = create_mask_dict(shadow_labels)

    if cloud_probas.dtype.name in ['float32', 'float64']:

        def combine_probas(class_labels, proba_layers):

            def merge_probas(proba_array, v1, v2):
                proba_array[class_labels.index(v1)] = proba_array[class_labels.index(v1)] + proba_array[class_labels.index(v2)]
                proba_array[class_labels.index(v2)] = 0
                return proba_array

            if ('v' in class_labels) and ('d' in class_labels):
                proba_layers = merge_probas(proba_layers, 'v', 'd')

            if ('v' in class_labels) and ('b' in class_labels):
                proba_layers = merge_probas(proba_layers, 'v', 'b')

            if ('v' in class_labels) and ('wv' in class_labels):
                proba_layers = merge_probas(proba_layers, 'v', 'wv')

            if ('v' in class_labels) and ('w' in class_labels):
                proba_layers = merge_probas(proba_layers, 'v', 'w')

            if ('hv' in class_labels) and ('hd' in class_labels):
                proba_layers = merge_probas(proba_layers, 'hv', 'hd')

            if ('c' in class_labels) and ('hv' in class_labels):
                proba_layers = merge_probas(proba_layers, 'c', 'hv')

            # Get the class labels for the last time step
            pred_int = proba_layers.argmax(axis=0)

            # 'No data' predictions should not have data
            pred_int[(nodata_mask == 1) | (proba_layers.max(axis=0) == 0)] = class_labels.index('n')

            return pred_int

        cloud_pred_int = combine_probas(cloud_labels, cloud_proba_layers)
        shadow_pred_int = combine_probas(shadow_labels, shadow_proba_layers)

    else:

        pred_int = np.uint8(cv2.resize(cloud_probas,
                                       (int(pred_kwargs['width']),
                                        int(pred_kwargs['height'])),
                                       interpolation=cv2.INTER_NEAREST))

        # 'No data' predictions should not have data
        pred_int[nodata_mask == 1] = cloud_labels.index('n')

    def cleaner(pred_array, pred_str, class_labels, erode_iters=0, open_iters=0, dilate_iters1=0, dilate_iters2=0, close_iters=0):

        """erode --> open to remove small clumps --> dilate to include cloud edges"""

        if erode_iters > 0:
            pred_array_buffer = cv2.morphologyEx(np.uint8(np.where(pred_array == class_labels.index(pred_str), 1, 0)),
                                                 cv2.MORPH_ERODE, KERNEL_CROSS, iterations=erode_iters)
        else:
            pred_array_buffer = np.uint8(np.where(pred_array == class_labels.index(pred_str), 1, 0))

        if open_iters > 0:
            pred_array_buffer = cv2.morphologyEx(np.uint8(pred_array_buffer), cv2.MORPH_OPEN, KERNEL_CROSS, iterations=open_iters)

        if dilate_iters1 > 0:
            pred_array_buffer = cv2.morphologyEx(np.uint8(pred_array_buffer), cv2.MORPH_DILATE, KERNEL_SQUARE, iterations=dilate_iters1)

        if close_iters > 0:
            pred_array_buffer = cv2.morphologyEx(np.uint8(pred_array_buffer), cv2.MORPH_CLOSE, KERNEL_CROSS, iterations=close_iters)

        if dilate_iters2 > 0:

            for iter_ in range(0, dilate_iters2):

                pred_array_buffer = moving_window(np.ascontiguousarray(np.pad(pred_array_buffer, ((pad, pad), (pad, pad)),
                                                                              mode='reflect'), dtype='float64'),
                                                  stat='expand',
                                                  w=5,
                                                  weights=True,
                                                  n_jobs=num_workers)[pad:-pad, pad:-pad]

            return np.uint8(pred_array_buffer)

        else:
            return np.uint8(pred_array_buffer)

    # Clean
    cloud_pred_int_fill_buffer = cleaner(cloud_pred_int, 'n', cloud_labels, erode_iters=0, open_iters=1, dilate_iters1=1, dilate_iters2=0)
    shadow_pred_int_fill_buffer = cleaner(shadow_pred_int, 'n', shadow_labels, erode_iters=0, open_iters=1, dilate_iters1=1, dilate_iters2=0)

    cloud_pred_int_cloud_buffer = cleaner(cloud_pred_int, 'c', cloud_labels, erode_iters=0, open_iters=0, dilate_iters1=0, close_iters=0, dilate_iters2=10)
    cloud_pred_int[cloud_pred_int_cloud_buffer == 1] = cloud_labels.index('c')

    shadow_pred_int_shadow_buffer = cleaner(shadow_pred_int, 's', shadow_labels, erode_iters=0, open_iters=0, dilate_iters1=0, close_iters=0, dilate_iters2=10)
    shadow_pred_int[shadow_pred_int_shadow_buffer == 1] = shadow_labels.index('s')

    cloud_pred_int[cloud_pred_int_fill_buffer == 1] = cloud_labels.index('n')
    shadow_pred_int[shadow_pred_int_fill_buffer == 1] = shadow_labels.index('n')

    def recode_class_labels(pred_int, mask_dict):

        # Recode the values
        pred_value_recoded = np.zeros((pred_int.shape[0], pred_int.shape[1]), dtype='uint8')

        for pred_value in np.unique(pred_int).tolist():

            if pred_value in mask_dict:
                pred_value_recoded[pred_int == pred_value] = mask_dict[pred_value] + 10000
            else:
                pred_value_recoded[pred_int == pred_value] = 10000

        pred_value_recoded -= 10000

        return pred_value_recoded

    # Recode the class labels
    clouds_pred_value_recoded = recode_class_labels(cloud_pred_int, masking_dict)
    shadows_pred_value_recoded = recode_class_labels(shadow_pred_int, shadow_mask_dict)

    # Combine cloud and shadow
    clouds_pred_value_recoded[(shadows_pred_value_recoded == shadow_mask_dict[shadow_labels.index('s')]) &
                              (clouds_pred_value_recoded != masking_dict[cloud_labels.index('c')])] = shadow_mask_dict[shadow_labels.index('s')]

    w = Window(row_off=0, col_off=0, width=cloud_pred_int.shape[1], height=cloud_pred_int.shape[0])

    with rio.Env(GDAL_CACHEMAX=512):

        if not outfile.is_file():

            with rio.open(str(outfile), mode='w', **pred_kwargs) as dst:
                pass

        # with rio.open(str(outfile), mode='r+') as dst:
        #     dst.write(proba_layer, indexes=1, window=w)

        with rio.open(str(outfile), mode='r+') as dst:
            dst.write(clouds_pred_value_recoded, indexes=1, window=w)


# Log-log
def log_func(b1, b2, out_name):

    return (scale_min_max(xr.ufuncs.log(b1.sel(band=b1.band.values.tolist()[0])), 0, 1, np.log(0.01), np.log(1)).clip(0, 1) *
            scale_min_max(xr.ufuncs.log(b2.sel(band=b2.band.values.tolist()[0])), 0, 1, np.log(0.01), np.log(1)).clip(0, 1))\
                    .clip(0, 1)\
                    .expand_dims(dim='band')\
                    .assign_coords({'band': [out_name]})\
                    .transpose('time', 'band', 'y', 'x')


def calc_features(data, scale_factor=0.0001, nodata=65535, il=None):

    """
    Calculates CRF image features

    Args:
        data (DataArray)
        scale_factor (Optional[float])
        nodata (Optional[int | float])
        with_time (Optional[bool])

    Returns:
        ``tuple``
    """

    attrs = data.attrs.copy()

    # Create a mask of 'no data' values
    #
    # All bands have the same 'no data' mask, so we only need to check one.
    # mask = xr.where(data.sel(band='blue') == nodata, 1, 0)\
    #             .astype('uint8')\
    #             .expand_dims(dim='band')\
    #             .assign_coords(band=['mask'])

    mask = data.gw.band_mask(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
                             src_nodata=nodata,
                             dst_clear_val=0,
                             dst_mask_val=1)\
                    .transpose('time', 'band', 'y', 'x')

    mask = xr.where((mask.sel(band='mask') == 1) |
                    (data.max(dim='band') == 0) |
                    (data.min(dim='band') == nodata), 1, 0)\
                .expand_dims(dim='band')\
                .transpose('time', 'band', 'y', 'x')

    # Scale and clip the bands, changing 'no data' values to 0
    # dsrc = data.gw.set_nodata(nodata, 0, (0, 1), 'float64', scale_factor=scale_factor)
    dsrc = xr.where((data == 0) | (data == nodata), 1, data*0.0001).astype('float64')

    # HOT
    hot = dsrc.sel(band='blue') - 0.5 * dsrc.sel(band='red') - 0.08

    hot = (1.0 - (1.0 / (1.0 + xr.ufuncs.exp((1.0 / 0.05) * (hot + 0.075)))))\
                .assign_coords(band='hot')\
                .expand_dims(dim='band')\
                .transpose('time', 'band', 'y', 'x')

    # Bare soil index
    bsi = ((dsrc.sel(band='swir1') + dsrc.sel(band='red')) - (dsrc.sel(band='nir') + dsrc.sel(band='blue'))) / \
          ((dsrc.sel(band='swir1') + dsrc.sel(band='red')) + (dsrc.sel(band='nir') + dsrc.sel(band='blue')))

    bsi = scale_min_max(bsi, 0, 1, -1, 1)\
                .assign_coords(band='bsi')\
                .expand_dims(dim='band')\
                .transpose('time', 'band', 'y', 'x')

    def scale_and_assign(in_data, band1, band2, assign_coord):

        return scale_min_max(gw.norm_diff(in_data, band1, band2), 0, 1, -1, 1)\
                    .assign_coords({'band': [assign_coord]})\
                    .transpose('time', 'band', 'y', 'x')

    # Normalized burn ratio
    nbr = scale_and_assign(dsrc, 'swir2', 'nir', 'nbr')

    # Normalized difference vegetation index
    ndvi = scale_and_assign(dsrc, 'green', 'nir', 'ndvi')

    # Normalized difference water index
    ndwi = scale_and_assign(dsrc, 'nir', 'green', 'ndwi')

    # Normalized difference moisture index
    ndmi = scale_and_assign(dsrc, 'swir1', 'nir', 'ndmi')

    # Shadow index
    shi = ((1.0 - dsrc.sel(band='blue')) *
           (1.0 - dsrc.sel(band='green')) *
           (1.0 - dsrc.sel(band='red'))).clip(-2, 1)\
                .assign_coords(band='shi')\
                .expand_dims(dim='band')

    shi = scale_min_max(shi.sel(band='shi') - ndsi.sel(band='ndvi') - bsi, 0, 1, -2, 1)\
                .assign_coords(band=['shi'])\
                .transpose('time', 'band', 'y', 'x')

    # EVI2
    evi2 = gw.evi2(dsrc) ** 0.33

    # kernel NDVI
    kndvi = gw.kndvi(dsrc)

    # Normalized Difference Snow Index
    ndsi = scale_min_max(gw.norm_diff(dsrc, 'swir1', 'green'), 0, 1, -1, 1)\
                .assign_coords(band=['ndsi'])\
                .transpose('time', 'band', 'y', 'x')

    shi2 = scale_min_max((dsrc.sel(band='blue') + dsrc.sel(band='green') -
                          (dsrc.sel(band='nir') + dsrc.sel(band='swir1'))), 0, 1, -2, 2)\
                .assign_coords(band='shi2')\
                .expand_dims(dim='band')\
                .transpose('time', 'band', 'y', 'x')

    blue = dsrc.sel(band='blue').expand_dims(dim='band').transpose('time', 'band', 'y', 'x')
    green = dsrc.sel(band='green').expand_dims(dim='band').transpose('time', 'band', 'y', 'x')
    red = dsrc.sel(band='red').expand_dims(dim='band').transpose('time', 'band', 'y', 'x')
    nir = dsrc.sel(band='nir').expand_dims(dim='band').transpose('time', 'band', 'y', 'x')
    swir1 = dsrc.sel(band='swir1').expand_dims(dim='band').transpose('time', 'band', 'y', 'x')
    swir2 = dsrc.sel(band='swir2').expand_dims(dim='band').transpose('time', 'band', 'y', 'x')

    log_bg = log_func(blue, green, 'log_bg')
    log_br = log_func(blue, red, 'log_br')
    log_bn = log_func(blue, nir, 'log_bn')
    log_bs1 = log_func(blue, swir1, 'log_bs1')
    log_bs2 = log_func(blue, swir2, 'log_bs2')

    if isinstance(il, xr.DataArray):

        concat_list = (blue, bsi, evi2, green, hot, il, kndvi,
                       log_bg, log_bn, log_br, log_bs1, log_bs2,
                       nbr, ndmi, ndsi, ndvi, ndwi,
                       nir, red, shi, shi2, swir1, swir2, mask)

    else:

        concat_list = (blue, bsi, evi2, green, hot, kndvi,
                       log_bg, log_bn, log_br, log_bs1, log_bs2,
                       nbr, ndmi, ndsi, ndvi, ndwi,
                       nir, red, shi, shi2, swir1, swir2, mask)

    # Stack
    src_stack = xr.concat(concat_list, dim='band')\
                    .transpose('time', 'band', 'y', 'x')

    # src_stack = xr.where(mask.sel(band='mask') == 1, 0, src_stack)\
    #                 .transpose('time', 'band', 'y', 'x')

    return src_stack.assign_attrs(**attrs)


def saliency_map(image):

    from skimage.color import rgb2gray

    image = (image * 0.0001) * 255.0

    # Convert image to grayscale
    if len(image.shape) > 2:
        image = rgb2gray(image)
    else:
        image = image

    # Apply Gaussian Smoothing
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)

    image[image == 0] = np.nan

    # Apply Mean Smoothing
    image_mean = np.nanmean(image)

    # Generate Saliency Map
    return np.absolute(gaussian - image_mean)


def compute_cloud_crf(image_batch_list,
              dates_batch_list,
              chunks,
              nodata,
              resampling,
              future_files,
              crf_clf_clouds,
              crf_clf_shadows,
              lgb_clf,
              lcrf_clf,
              w=3,
              deep_crf=False,
              band_names=None,
              sensor=None,
              predict_labels=None,
              cloud_labels=None,
              shadow_labels=None,
              num_workers=None,
              cloud_proba_thresh=None,
              shadow_proba_thresh=None,
              pred_kwargs=None,
              dem_path=None,
              angle_src=None):

    # Order the class probability labels for the C dictionaries
    # predict_labels_ordered = [f'zproba{plab_idx:04d}' for plab_idx in range(0, len(predict_labels))]
    # band_names_concat = band_names[:-1] + predict_labels_ordered + [band_names[-1]]

    # nodata_layer = band_names.index('zxmask')

    # il = calc_il(data_src, dem_path=Path(dem_path), angles=angle_src, num_workers=num_workers)

    # Get the real time length
    with gw.open(image_batch_list,
                 time_names=dates_batch_list,
                 netcdf_vars=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
                 chunks=chunks,
                 num_threads=1,
                 nodata=nodata,
                 resampling=resampling) as data_src:

        # Stack the predictors
        src_stack = calc_features(data_src.transpose('time', 'band', 'y', 'x'))

        # Get the data
        sat_bands = src_stack.gw.compute(num_workers=num_workers)

    sat_bands[np.isnan(sat_bands) | np.isinf(sat_bands)] = 0

    if lgb_clf is not None:

        ##########################################
        # Add LightGBM probabilities as predictors

        new_x_data = []
        for X_data_layer in sat_bands:

            X_data_layer_pred = transform_data(X_data_layer[:-1], scale_factor=1.0)

            X_data_layer_probas = lgb_clf.predict_proba(nd_to_columns(X_data_layer_pred))
            X_data_layer_probas = columns_to_nd(X_data_layer_probas, len(predict_labels), *sat_bands.shape[2:])
            X_data_layer_stack = np.vstack((X_data_layer[:-1], X_data_layer_probas, X_data_layer[-1][np.newaxis, :, :]))

            new_x_data.append(X_data_layer_stack)

        sat_bands = np.array(new_x_data, dtype='float64')
        band_names = band_names + [f'zproba{plab_idx:03d}' for plab_idx in range(0, len(predict_labels))]
        ##################################################################################

    # Apply the model
    if deep_crf:

        pred = lcrf_clf.predict_sensor_probas(sat_bands[:, :-1, :, :],
                                              sensor,
                                              labels=predict_labels,
                                              scale_factor=1.0,
                                              add_indices=False,
                                              band_names=band_names,
                                              transform=True,
                                              remove_nodata=False,
                                              n_jobs=num_workers)

    else:

        pred_clouds = crf_clf_clouds.predict_sensor_probas(sat_bands,
                                                           sensor,
                                                           labels=cloud_labels + ['n'],
                                                           scale_factor=1.0,
                                                           add_indices=False,
                                                           band_names=band_names + ['zxmask'],
                                                           transform=True,
                                                           remove_nodata=True,
                                                           nodata_layer=-1,
                                                           n_jobs=num_workers)

        pred_shadows = crf_clf_shadows.predict_sensor_probas(sat_bands,
                                                             sensor,
                                                             labels=shadow_labels + ['n'],
                                                             scale_factor=1.0,
                                                             add_indices=False,
                                                             band_names=band_names + ['zxmask'],
                                                             transform=True,
                                                             remove_nodata=True,
                                                             nodata_layer=-1,
                                                             n_jobs=num_workers)

    # Write layers
    for j in range(0, pred_clouds.shape[0]):

        outfile = future_files[j]

        if not outfile.is_file():

            masks_to_file(sat_bands[j],
                          pred_clouds[j],
                          pred_shadows[j],
                          outfile,
                          cloud_labels + ['n'],
                          shadow_labels + ['n'],
                          pred_kwargs,
                          cloud_proba_thresh=cloud_proba_thresh,
                          shadow_proba_thresh=shadow_proba_thresh,
                          w=w,
                          num_workers=num_workers)

    return None
