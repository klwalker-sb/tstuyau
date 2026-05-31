import shutil
from pathlib import Path
import tempfile

from ..handler import logger
from ..db import TuyauDataBase
from . import utils
from .project import ProjectPaths
from .lookup import SENSORS

import geowombat as gw

import xarray as xr
import pandas as pd
from affine import Affine
from datetime import datetime
import numpy as np

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
    file_date = datetime.strptime(Path(dataset.encoding['source']).stem.split('_')[3], '%Y%m%d')
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
    Co-registers images

    Args:
        params (dict)

    Returns:
        None
    """

    for grid in params['grids']:

        ppaths = ProjectPaths(params, grid=grid)

        processing_db = pd.read_pickle(ppaths.ms.parent/'processing.info')
        if 'coreg' not in processing_db:
             processing_db['coreg'] = np.nan
             processing_db['shift_x'] = np.nan
             processing_db['shift_y'] = np.nan
             processing_db['coreg_error'] = np.nan

        db = TuyauDataBase(str(ppaths.ms.parent / f'{int(grid):06d}.db'))

        if not params['status']['check_downloads']:
            check_download_db = False
        else:
            check_download_db = True

        msensors = params['image_type']
        if isinstance(msensors, str):
            msensors = [msensors] 
        if any (s in msensors for s in ['All', 'AllRaw', 'LS2']):
            msensors = ['S2','S2cp','LT05','LE07','LC08','LC09']
        elif 'L' in msensors:
            msensors = ['LT05','LE07','LC08','LC09']
        elif ('S' in msensors) or ('S2' in msensors):
            msensors = ['S2','S2cp']
        
        if check_download_db:
            ## Check downloads
            for sen in msensors:
                senlab = SENSORS[sen]['sensor']
                senpath = SENSORS[sen]['GEE']
                if not db.eosvault_is_complete(senlab, senpath):
                    logger.warning(f'  The {senlab} {senpath} downloads for grid {grid} are incomplete.')
                    continue

            ## Check post-processing
            if ppaths.gee.is_dir():
                for sen in msensors:
                    senunq = SENSORS[sen]['GEEunq']
                    if list(ppaths.gee.glob(f"{senunq}*[!s].nc")):
                        logger.warning(f'  The {senunq} post-processing for grid {grid} is incomplete.')
                        continue

        ## Open the `tstuyau` database
        db = TuyauDataBase(str(ppaths.ms.parent / f'{int(grid):06d}_tuyau.db'))

        if not db.table_exists:
            db.remove()
            db.create(exists_ok=True)
            db.insert(grid)

        if params['status']['reset_db']:
            db.reset(grid, 'preprocess')

        if not ppaths.ms.is_dir():
            logger.warning(f'  The BRDF directory for grid {grid} does not exist.')
            continue

        nocoreg_path = ppaths.ms.parent.joinpath('s2_nocoreg')
        nocoreg_path.mkdir(parents=True, exist_ok=True)
        ref_dir = ppaths.ms.parent.joinpath('brdf_ref')
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_path = Path(ref_dir) / '_tmp_reference.tif'
        
        ## Get all images to coreg (sentinel + landsat 5 & 7)
        ##  Do not include files with basenames ending in 's' (in case angles files in dir) or 'X' (in case some cleaning has been done)
        ##    note: angles files end in s.nc. but only in brdf folder if <dl_method> is 'gee'
        s2_list = utils.get_s2_list(ppaths.ms, pattern='*[!sX].nc')
        l5_list = utils.get_l5_list(ppaths.ms, pattern='*[!sX].nc')
        l7_list = utils.get_l7_list(ppaths.ms, pattern='*[!sX].nc')
        image_list = s2_list + l5_list + l7_list

        if not image_list:
            logger.warning(f'  No images found for grid {grid}.')
            continue

        logger.info(f'  Checking grid {grid} ...')

        if (not ref_path.is_file()) or (params['coreg']['overwrite_ref']):
            logger.info('making reference image...')
            ref_path = str(ref_path)
            ## Get the median over all Landsat 8 and 9 images
            l8_list = utils.get_l8_list(ppaths.ms, pattern='*[!sX].nc')
            l9_list = utils.get_l9_list(ppaths.ms, pattern='*[!sX].nc')
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
                except:
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
                            target = target.where(lambda x: x != target.nodatavals[0])
                            reference = reference.where(lambda x: x != reference.nodatavals[0])

                            ## The fillna below converts nans to 0
                            try:
                                data = gw.coregister(
                                    target=target.fillna(0).assign_attrs({'crs': target.crs}),
                                    reference=reference.fillna(0).assign_attrs({'crs':reference.crs}),
                                    band_names_reference=[REFERENCE_BAND],
                                    band_names_target=LANDSAT_LIKE_BANDS,
                                    ws=(256, 256),
                                    r_b4match=1,
                                    s_b4match=REFERENCE_BAND_POSITION,
                                    max_shift=params['coreg']['max_shift'],
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
                            processing_db.loc[processing_db['brdf_id'].eq(fn.name),'shift_x']= data.attrs['x_shift_px']
                            processing_db.loc[processing_db['brdf_id'].eq(fn.name),'shift_y']= data.attrs['y_shift_px']
                        except:
                            continue

                    else:
                        ## Rename the file that failed to coregister with an X at the end:
                        p = Path(tar_image)
                        p.rename(Path(p.parent, f"{p.stem}_X{p.suffix}"))

        pd.to_pickle(processing_db, ppaths.ms.parent/'processing.info')
        db.update(grid, 'preprocess')
