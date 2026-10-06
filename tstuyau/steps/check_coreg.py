import shutil
from datetime import datetime, timezone
from pathlib import Path

import geowombat as gw
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine

from ..db import TuyauDataBase
from ..handler import logger
from . import utils
from .lookup import SENSORS
from .processing_tracker import reconstruct_db
from .project import ProjectPaths

REFERENCE_BAND = 'nir'
LANDSAT_LIKE_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
REFERENCE_BAND_POSITION = LANDSAT_LIKE_BANDS.index(REFERENCE_BAND) + 1


def expand_time(dataset):
    """`open_mfdataset` preprocess function
    """
    attrs = dataset.attrs.copy()
    attrs['transform'] = Affine(*attrs['transform'])
    attrs['res'] = tuple(attrs['res'])
    ## Get the date
    file_date = datetime.strptime(Path(dataset.encoding['source']).stem.split('_')[3], '%Y%m%d').replace(tzinfo=timezone.utc)
    darray = (
        dataset
        .to_array()
        .rename({'variable': 'band'})
        .sel(band=REFERENCE_BAND)
        .assign_coords(time=file_date, y=dataset.y, x=dataset.x)
        .expand_dims('time')
        .transpose('time', 'y', 'x')
        .where(lambda x: x != x.nodatavals[0])  # set 'no data' values as nans
    )

    return darray.assign_attrs(**attrs)


def coregister(params):
    """
    Co-registers images in 'brdf' folder using AROSICS (https://pypi.org/project/arosic) via geowombat wrapper. 
    will exclude images with <params:reconstruct:skip_flag> str at end of file name before suffix (usually 'X'). Default is None.
              note: <params:reconstruct:skip_flag> can be multiple letters but names ending in any of them will be excluded
    Creates reference image from Landsat 8 and 9 images (temporal median) and coregisters Landsat 5, 7 & Sentinel2 images to this.
    Does not move Landsat 8 & 9 images (assumes they are all correctly aligned).
    Original image is sent to 's2_nocoreg' folder, and shifted image is put in the 'brdf' folder with same name but with 'coreg' appended
    Images that fail coreg are marked with an <params:reconstruct:skip_flag> (usually 'X') at the end of the file name 
         and the error is registered in the processing database 'coreg_error' column.
    Pixel shift (shift_x and shift_y) are added to the processing database for each image for future use (if need to shift cloud masks etc.)
    Can change AROSICS max_shift parameter (maximum shift distance before fail) with <params:coreg:max_shift> maximum shift. Default is 5 pixels
    """

    cells = utils.get_cell_list_from_grid_param(params['grids'])
    for cell in cells:
        ppaths = ProjectPaths(params, grid=cell)

        ## procssing_db is the full processing database tracking info about each image processed and steps taken
        processing_db_path = ppaths.ms.parent/'processing.info'
        if not processing_db_path.is_file():
            logger.warning(f'processing.info does not exist for cell {cell}. Making new db...')
            reconstruct_db(cell)
        processing_db = pd.read_pickle(processing_db_path)
        if 'coreg' not in processing_db:
             processing_db['coreg'] = np.nan
             processing_db['shift_x'] = np.nan
             processing_db['shift_y'] = np.nan
             processing_db['coreg_error'] = np.nan

        ## db is a simple cell-level processing database tracking which cells have been run (this is an sqlite db)
        db = TuyauDataBase(str(ppaths.ms.parent / f'{int(cell):06d}_tuyau.db'))

        if not db.table_exists:
            db.remove()
            db.create(exists_ok=True)
            db.insert(cell)

        check_download_db = False
        if params['status']['check_downloads']:
            check_download_db = True

        if check_download_db: ## NOTE-- this only works for dlMethod='GEE' currently
            msensors = params['image_type']
            if isinstance(msensors, str):
                msensors = [msensors] 
            if any (s in msensors for s in ['All', 'AllRaw', 'LS2']):
                msensors = ['S2','S2cp','LT05','LE07','LC08','LC09']
            elif 'L' in msensors:
                msensors = ['LT05','LE07','LC08','LC09']
            elif ('S' in msensors) or ('S2' in msensors):
                msensors = ['S2','S2cp']
        
            ## Check downloads
            ## No longer tracking sensor downloads in tuyau db. TODO: update or remove
            #for sen in msensors:
            #    senlab = SENSORS[sen]['sensor']
            #    senpath = SENSORS[sen]['GEE']
            #    if not db.eosvault_is_complete(senlab, senpath):
            #        logger.warning(f'  The {senlab} {senpath} downloads for grid {grid} are incomplete.')
            #        continue

            ## Check post-processing   NOTE-- this only works for dlMethod='GEE' currently
            if ppaths.gee.is_dir():
                for sen in msensors:
                    senunq = SENSORS[sen]['GEEunq']
                    if list(ppaths.gee.glob(f"{senunq}*[!s].nc")):
                        logger.warning(f'  The {senunq} post-processing for grid {cell} is incomplete.')
                        continue

        if params['status']['reset_db']:
            db.reset(cell, 'preprocess')

        if not ppaths.ms.is_dir():
            logger.warning(f'  The BRDF directory for cell {cell} does not exist.')
            continue

        nocoreg_path = ppaths.ms.parent.joinpath('s2_nocoreg')
        nocoreg_path.mkdir(parents=True, exist_ok=True)
        ref_dir = ppaths.ms.parent.joinpath('brdf_ref')
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_path = Path(ref_dir) / '_tmp_reference.tif'
        
        ## Get all images to coreg (sentinel + landsat 5 & 7)
        ##  Do not include files with basenames ending in <params:reconstruct:skip_flag> (in case some cleaning has been done)
        skip_flag = None
        if params['reconstruct']['skip_flag']:
            skip_flag = params['reconstruct']['skip_flag']
        ## angles files (ending in s.nc) should also be excluded, . but these are only in brdf folder if <dl_method> is 'gee'
        if params['dlMethod'] == 'GEE':
            skip_flag = (skip_flag or []) + ['s']
            
        match_str = '*.nc' if skip_flag is None else f'*[!{skip_flag}].nc'
        s2_list = utils.get_s2_list(ppaths.ms, pattern=match_str)
        l5_list = utils.get_l5_list(ppaths.ms, pattern=match_str)
        l7_list = utils.get_l7_list(ppaths.ms, pattern=match_str)
        image_list = s2_list + l5_list + l7_list

        if not image_list:
            logger.warning(f'  No images found for cell {cell}.')
            continue

        logger.info(f'  Checking grid {cell} ...')

        if (not ref_path.is_file()) or (params['coreg']['overwrite_ref']):
            logger.info('making reference image...')
            ref_path = str(ref_path)
            ## Get the median over all Landsat 8 and 9 images
            l8_list = utils.get_l8_list(ppaths.ms, pattern=match_str)
            l9_list = utils.get_l9_list(ppaths.ms, pattern=match_str)
            landsat_list = sorted(l8_list + l9_list)

            landsat_refs = [str(fn) for fn in landsat_list]
            logger.info(f'there are {len(landsat_refs)} landsat 8 & 9 images for reference')

            try:
                with xr.open_mfdataset(
                    landsat_refs,
                    concat_dim='time',
                    chunks={
                        'time':-1,
                        'band':-1,
                        'x': params['io']['n_chunks'],
                        'y': params['io']['n_chunks']
                    },
                    combine='nested',
                    engine='h5netcdf',
                    preprocess=expand_time,
                    parallel=True
                ) as src:
                    ## Calculate the temporal median, ignore nans
                    reference_med = (
                        (
                            src
                            .median(dim='time', skipna=True)
                            .chunk({
                                'y': params['io']['n_chunks'],
                                'x': params['io']['n_chunks']
                            })
                        )
                        .assign_attrs(**src.attrs)
                        .expand_dims(dim='band')
                    )
            #except (KeyError, OSError) as ex:
            except Exception as ex:
                logger.warning(ex)
                try:
                    for fn in landsat_list:
                        with xr.open_mfdataset(str(fn)) as src:
                            pass
                except Exception as ex:
                    logger.warning(f'corrupt image: {fn}')

            else:
                ## Save the reference image
                reference_med.gw.save(ref_path, overwrite=True)
            
            ## Use the same reference image for every target image
            for fn in image_list:
                ## Already co-registered
                if str(fn).endswith('coreg.nc'):
                    continue

                else:
                    tar_image = str(fn)
                    coreg_image = tar_image.replace('.nc', '_coreg.nc')

                    ## Open the reference image (i.e., one-band median)
                    try:
                        with gw.open(
                            ref_path,
                            chunks={
                                'band':-1,
                                'x': params['io']['n_chunks'],
                                'y': params['io']['n_chunks']
                            }
                        ) as reference, \
                            gw.open(
                                tar_image,
                                band_names=LANDSAT_LIKE_BANDS,
                                chunks= {
                                    'band': -1,
                                    'y' : params['io']['n_chunks'],
                                    'x': params['io']['n_chunks']
                                },
                                engine='h5netcdf'
                            ) as target:
                            ## This converts nodata values to nan
                            target = target.where(target != target.nodatavals[0])
                            reference = reference.where(reference != reference.nodatavals[0])
                            ## The fillna below converts nans to 0
                            
                            max_shift = 5
                            if params['coreg']['max_shift']:
                                max_shift = params['coreg']['max_shift']
                            
                            try:
                                data = gw.coregister(
                                    target=target.fillna(0).assign_attrs({'crs': target.crs}),
                                    reference=reference.fillna(0).assign_attrs({'crs':reference.crs}),
                                    band_names_reference=[REFERENCE_BAND],
                                    band_names_target=LANDSAT_LIKE_BANDS,
                                    ws=(256, 256),
                                    r_b4match=1,
                                    s_b4match=REFERENCE_BAND_POSITION,
                                    max_shift=max_shift,
                                    resamp_alg_deshift='nearest',
                                    resamp_alg_calc='cubic',
                                    out_gsd=[target.gw.celly, reference.gw.celly],
                                    q=True,
                                    nodata=(0, 0),
                                    CPUs=1
                                )
                                coreg_success = True

                            except Exception as ex:
                                logger.warning(f'  Could not co-register {tar_image} because -> {ex}.')
                                coreg_success = False
                                processing_db.loc[processing_db['brdf_id'].eq(fn.name),'coreg_error'] = ex
                                processing_db.loc[processing_db['brdf_id'].eq(fn.name),'coreg']= False
                    except (KeyError, OSError) as ex:
                        logger.warning(f'Could not co-register {tar_image} because -> image is corrupt')
                        coreg_success = False
                        processing_db.loc[processing_db['brdf_id'].eq(fn.name),'coreg_error'] = ex
                        processing_db.loc[processing_db['brdf_id'].eq(fn.name),'coreg']= False
                        processing_db.loc[processing_db['brdf_id'].eq(fn.name),'redownload']=True

                    
                    if coreg_success:
                        ## Write to file
                        data.gw.to_netcdf(coreg_image, zlib=True, complevel=5)
                        ## Move the original file
                        shutil.move(
                            tar_image,
                            str(nocoreg_path.joinpath(fn.name))
                        )
                        processing_db.loc[processing_db['brdf_id'].eq(fn.name),'coreg']= True
                        try:
                            ## shift_x and shift_y are coordinate shits of the coreged raster in pixels (to use if matching cloud masks later)
                            processing_db.loc[processing_db['brdf_id'].eq(fn.name),'shift_x']= data.attrs['x_shift_px']
                            processing_db.loc[processing_db['brdf_id'].eq(fn.name),'shift_y']= data.attrs['y_shift_px']
                        except Exception:
                            continue

                    else:
                        ## Rename the file that failed to coregister with an X at the end:
                        p = Path(tar_image)
                        p.rename(Path(p.parent, f"{p.stem}_{params['reconstruct']['skip_flag']}{p.suffix}"))

        pd.to_pickle(processing_db, ppaths.ms.parent/'processing.info')
        db.update(cell, 'preprocess')
