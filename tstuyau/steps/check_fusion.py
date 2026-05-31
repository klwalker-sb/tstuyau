import shutil
from datetime import datetime
from pathlib import Path

from ..handler import logger
from .project import ProjectPaths
from .fusion_utils import BAP, get_medoid
from .io import ImageIO
from .date_utils import check_day_dist
from . import utils

import geowombat as gw
from geowombat.core import ndarray_to_xarray
from geowombat.radiometry._fusion import StarFM

import numpy as np
import xarray as xr
from tqdm import trange, tqdm


def fuse_sensors(params):

    """
    Fusion of images

    Args:
        params (dict)

    Returns:
        None
    """

    for grid in params['grids']:

        ppaths = ProjectPaths(params, grid=grid)

        if not ppaths.ts.is_dir():
            logger.warning(f'  The time series directory for grid {grid} does not exist.')
            continue

        if params['fusion']['overwrite']:
            for fn in ppaths.fusion.glob('*.tif'):
                if fn.is_file():
                    # Do not delete copied S2 files
                    if not fn.name.startswith('L1C_'):
                        fn.unlink()

        ppaths.clean_temp(ppaths.fusion)

        # The input time series path
        ts_dir = ppaths.ts / params['reconstruct']['si']
        
        s2_list = sorted(list(ts_dir.glob(f"{params['fusion']['start_year']}*.tif")))
        landsat_list = sorted(list(ts_dir.glob('*.tif')))

        # Move any dates greater than the start date
        s2_list += sorted([fn for fn in landsat_list if int(fn.name[:4]) > params['fusion']['start_year']])
        landsat_list = sorted(list(set(landsat_list).difference(s2_list)))

        # s2_list, landsat_list = utils.get_image_lists(ppaths.ms)

        io = ImageIO(s2_list, landsat_list)

        # Get the sorted lists
        s2_list = io.file_lists['hk']
        landsat_list = io.file_lists['mk']

        # Copy Sentinel images
        # with gw.config.update(sensor='l7', ref_res=float(params['fusion']['res'])):

        logger.info('  Copying Sentinel images ...')

        for fn in tqdm(s2_list, total=len(s2_list)):

            ms_path = ts_dir / fn.name
            sharp_image = ppaths.fusion / ms_path.name

            if sharp_image.is_file():
                continue

            shutil.copy(str(ms_path), str(sharp_image))

        stf = StarFM(window_size=params['fusion']['window_size'],
                     param_a=params['fusion']['param_a'],
                     param_n=params['fusion']['param_n'],
                     hres_uncert=params['fusion']['hres_uncert'],
                     mres_uncert=params['fusion']['mres_uncert'],
                     n_jobs=params['num_workers'])

        # Iterate over the landsat images,
        # beginning with the most recent
        for fn in landsat_list[::-1]:

            output_fusion = ppaths.fusion / fn.name

            if output_fusion.is_file():

                if params['fusion']['overwrite']:
                    output_fusion.unlink()
                else:
                    s2_list.append(output_fusion)
                    continue

            # if fn != landsat_list[::-1][1]:
            #     continue

            # Get the Landsat image date
            ldate_str = fn.stem
            ldate_dt = datetime.strptime(ldate_str, '%Y%j')

            # if (ldate_dt.year == 2019) and (ldate_dt.month == 2):
            #     pass
            # else:
            #     continue

            logger.info(f'  Adjusting moderate resolution image {Path(fn).name} at date {ldate_str}')

            # Get the mask image
            # lmask_image = ppaths.masks.joinpath(ldate_str + '.tif')

            # if not lmask_image.is_file():
            #     continue

            with gw.config.update(ignore_warnings=True):

                #, \gw.open(lmask_image) as lmask_src:
                with gw.open(ts_dir / fn, band_names=params['fusion']['wavelengths']) as mres_0_src:

                    # Check the clear area
                    # total_clear = xr.where(lmask_src <= params['masking']['min_mask'], 1, 0)\
                    #                     .sum().data\
                    #                     .compute(num_workers=params['num_workers'])

                    # pct_clear = (total_clear / (lmask_src.gw.nrows*lmask_src.gw.ncols)) * 100.0

                    # if pct_clear < params['fusion']['min_pct_thresh']:
                    #     continue

                    # attrs = mres_0_src.attrs.copy()

                    # mres_0_src = xr.where((lmask_src.sel(band=1) > params['masking']['min_mask']) | (mres_0_src.max(dim='band') == 0),
                    #                       params['nodata'],
                    #                       mres_0_src)\
                    #                 .transpose('band', 'y', 'x')\
                    #                 .assign_attrs(**attrs)

                    mres_0_src = mres_0_src.gw.set_nodata(0, 0, (0, 1), 'float64', scale_factor=0.0001)

                    mres_0_data = mres_0_src.data.compute(num_workers=params['num_workers'])
                    mres_0_data[np.isnan(mres_0_data)] = 0

            ####################################
            # Find the Sentinel reference images
            # closest to the Landsat date
            ####################################
            near_indices_hres_k = np.array([abs(ldate_dt - datetime.strptime(hks.stem, '%Y%j')) for hks in s2_list]).argsort()

            hres_k_init = False

            logger.info('  High-res k images:')

            hres_k_stack = []

            # Attempt to fill the high-res
            for imidx, near_hres_k_idx in enumerate(near_indices_hres_k):

                hkdate_str = s2_list[near_hres_k_idx].stem
                hkdate_dt = datetime.strptime(hkdate_str, '%Y%j')

                # hkmask_image = ppaths.masks.joinpath(hkdate_str + '.tif')

                # if not hkmask_image.is_file():
                #     continue

                if not check_day_dist(ldate_dt, hkdate_dt, params['fusion']['fill_max_days']):
                    continue

                if abs(hkdate_dt.year - ldate_dt.year) > params['fusion']['fill_max_years']:
                    continue

                logger.info(s2_list[near_hres_k_idx])

                # Load the data
                hres_k_data = io.load_data('hk', near_hres_k_idx, ppaths, params)

                # if not isinstance(hres_k_mdata, np.ndarray):
                #     continue

                if not hres_k_init:

                    # Best available pixel
                    bap = BAP(hres_k_data.shape,
                              max_cloud_dist=params['fusion']['max_cloud_dist'],
                              max_days=params['fusion']['fill_max_days'])

                    hres_k_init = True

                bap.calc_score(trg_data=hres_k_data,
                               ref_data=mres_0_data,
                               dta=ldate_dt,
                               dtb=hkdate_dt,
                               name=s2_list[near_hres_k_idx].name)

                hres_k_data[hres_k_data == 0] = np.nan
                res = ndarray_to_xarray(mres_0_src, hres_k_data, mres_0_src.band.values.tolist())
                hres_k_stack.append(res)

            # TODO: copy the moderate resolution image
            # if not hres_k_stack:
            #     continue

            logger.info('  Calculating medoids ...')

            # Get the medoid
            hres_k_data = get_medoid(hres_k_stack, params['fusion']['wavelengths'], params['num_workers'])

            hres_k_stack = None

            # hres_k_data = bap.finalize()
            hres_conf_weights = bap.max_score.copy()

            ###################################
            # Find the Landsat reference images
            # closest to the Landsat date
            ###################################
            near_indices_mres_k = np.array([abs(ldate_dt - datetime.strptime(mks.stem, '%Y%j')) for mks in landsat_list]).argsort()

            mres_k_init = False

            mres_k_stack = []

            logger.info('  Moderate-res k images:')

            # Attempt to fill the moderate-res
            for imidx, near_mres_k_idx in enumerate(near_indices_mres_k):

                mkdate_str = landsat_list[near_mres_k_idx].stem
                mkdate_dt = datetime.strptime(mkdate_str, '%Y%j')

                # mkmask_image = ppaths.masks.joinpath(mkdate_str + '.tif')

                if mkdate_dt == ldate_dt:
                    continue

                # if not mkmask_image.is_file():
                #     continue

                if not check_day_dist(ldate_dt, mkdate_dt, params['fusion']['fill_max_days']):
                    continue

                if abs(mkdate_dt.year - ldate_dt.year) > params['fusion']['fill_max_years']:
                    continue

                logger.info(landsat_list[near_mres_k_idx])

                # Load the data
                # <-- mask, reflectance, solar zenith angles
                mres_k_data = io.load_data('mk', near_mres_k_idx, ppaths, params)

                # if not isinstance(mres_k_mdata, np.ndarray):
                #     continue

                if not mres_k_init:

                    # Best available pixel
                    bap = BAP(mres_k_data.shape,
                              max_cloud_dist=params['fusion']['max_cloud_dist'],
                              max_days=params['fusion']['fill_max_days'])

                    mres_k_init = True

                bap.calc_score(trg_data=mres_k_data,
                               ref_data=mres_0_data,
                               dta=ldate_dt,
                               dtb=mkdate_dt,
                               name=landsat_list[near_mres_k_idx].name)

                mres_k_data[mres_k_data == 0] = np.nan
                res = ndarray_to_xarray(mres_0_src, mres_k_data, mres_0_src.band.values.tolist())
                mres_k_stack.append(res)

            logger.info('  Calculating medoids ...')

            # Get the medoid
            mres_k_data = get_medoid(mres_k_stack, params['fusion']['wavelengths'], params['num_workers'])

            mres_k_stack = None

            # mres_k_data = bap.finalize()
            mres_conf_weights = bap.max_score.copy()

            conf_weights = np.minimum(hres_conf_weights, mres_conf_weights)

            with gw.config.update(ignore_warnings=True):

                # gw.open(lmask_image) as lmask_src:
                with gw.open(ts_dir / fn, band_names=params['fusion']['wavelengths']) as mres_0_src:

                    attrs = mres_0_src.attrs.copy()

                    # mres_0_src = xr.where((lmask_src.sel(band=1) > params['masking']['min_mask']) | (mres_0_src.max(dim='band') == 0),
                    #                       params['nodata'],
                    #                       mres_0_src)\
                    #                     .transpose('band', 'y', 'x')\
                    #                     .assign_attrs(**attrs)

                    mres_0_src = mres_0_src.gw.set_nodata(0, 0, (0, 1), 'float64', scale_factor=0.0001)

                    w = params['fusion']['window_size']
                    hw = int(w / 2.0)

                    results = []

                    for bidx in trange(0, len(params['fusion']['wavelengths'])):

                        band = params['fusion']['wavelengths'][bidx]

                        mres_0_data = mres_0_src.sel(band=band).data.compute(num_workers=params['num_workers'])
                        mres_0_data[np.isnan(mres_0_data)] = 0

                        # res = imph.fit_transform(pad_array(hres_k_data[bidx], w, scale_factor=1),
                        #                          pad_array(mres_0_data, w, scale_factor=1))

                        res = stf.fit_transform(utils.pad_array(hres_k_data[bidx], hw, scale_factor=1),
                                                utils.pad_array(mres_k_data[bidx], hw, scale_factor=1),
                                                utils.pad_array(mres_0_data, hw, scale_factor=1),
                                                utils.pad_array(conf_weights[bidx], hw, scale_factor=1))[hw:-hw, hw:-hw]

                        res = ndarray_to_xarray(mres_0_src, res, [band])
                        results.append(res)

                    res = (xr.concat(results, dim='band') * 10000.0).clip(0, 10000).astype('uint16')
                    res = xr.where(res == 0, params['nodata'], res).assign_attrs(**attrs)

                    logger.info('  Computing fusion ...')

                    res.gw.to_raster(str(output_fusion),
                                     n_workers=1,
                                     n_threads=params['num_workers'],
                                     n_chunks=params['io']['n_chunks'],
                                     overwrite=True,
                                     nodata=params['nodata'],
                                     compress='lzw')

                    # ndarray_to_xarray(mres_0_src, bap.max_score,
                    #                   mres_0_src.band.values.tolist()).gw.to_raster(
                    #     str(output_fusion).replace('.tif', '_max_score.tif'),
                    #     n_workers=1,
                    #     n_threads=params['num_workers'],
                    #     n_chunks=params['io']['n_chunks'],
                    #     overwrite=True,
                    #     nodata=0,
                    #     compress='lzw')

                    # ndarray_to_xarray(mres_0_src.astype('uint8'), bap.count, mres_0_src.band.values.tolist()).gw.to_raster(
                    #     str(output_fusion).replace('.tif', '_count.tif'),
                    #     n_workers=1,
                    #     n_threads=params['num_workers'],
                    #     n_chunks=params['io']['n_chunks'],
                    #     overwrite=True,
                    #     nodata=0,
                    #     compress='lzw')
                    #
                    # ndarray_to_xarray(mres_0_src.astype('uint16'), bap.dates,
                    #                   mres_0_src.band.values.tolist()).gw.to_raster(
                    #     str(output_fusion).replace('.tif', '_dates.tif'),
                    #     n_workers=1,
                    #     n_threads=params['num_workers'],
                    #     n_chunks=params['io']['n_chunks'],
                    #     overwrite=True,
                    #     nodata=0,
                    #     compress='lzw')

                    # ndarray_to_xarray(mres_0_src, conf_weights, mres_0_src.band.values.tolist()).gw.to_raster(str(output_fusion).replace('.tif', '_weights.tif'),
                    #                  n_workers=1,
                    #                  n_threads=params['num_workers'],
                    #                  n_chunks=params['io']['n_chunks'],
                    #                  overwrite=True,
                    #                  nodata=0,
                    #                  compress='lzw')

                    # ndarray_to_xarray(mres_0_src, bap.spec_diff, mres_0_src.band.values.tolist()).gw.to_raster(
                    #     str(output_fusion).replace('.tif', '_spec.tif'),
                    #     n_workers=1,
                    #     n_threads=params['num_workers'],
                    #     n_chunks=params['io']['n_chunks'],
                    #     overwrite=True,
                    #     nodata=0,
                    #     compress='lzw')

                    # mres_k_data = ndarray_to_xarray(mres_0_src, mres_k_data, params['fusion']['wavelengths'])
                    # mres_k_data.gw.to_raster(str(output_fusion).replace('.tif', '_mres_k.tif'),
                    #                  n_workers=1,
                    #                  n_threads=params['num_workers'],
                    #                  n_chunks=params['io']['n_chunks'],
                    #                  overwrite=True,
                    #                  nodata=params['nodata'],
                    #                  compress='lzw')

                    # hres_k_data = ndarray_to_xarray(mres_0_src, hres_k_data, params['fusion']['wavelengths'])
                    # hres_k_data.gw.to_raster(str(output_fusion).replace('.tif', '_hres_k.tif'),
                    #                          n_workers=1,
                    #                          n_threads=params['num_workers'],
                    #                          n_chunks=params['io']['n_chunks'],
                    #                          overwrite=True,
                    #                          nodata=params['nodata'],
                    #                          compress='lzw')

                    s2_list.append(output_fusion)
