from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

from ..handler import logger
from .. import errors
from .project import ProjectPaths
from .image_utils import get_rbg_img
from .constants import FILENAME_DATE_INDEX, FILENAME_DATE_INDEX_GEE

from geowombat.core import sort_images_by_date


def precheck_reconstruct(grid, ppaths, params):

    """
    Checks if all masks are complete

    Args:
        grid (int)
        ppaths (object)
        params (dict)
    """
    if params['dlMethod'] == 'GEE':
        date_pos=FILENAME_DATE_INDEX_GEE
        prepend_str='netcdf:'
    else:
        date_pos=FILENAME_DATE_INDEX
        prepend_str=''
    
    image_dict = sort_images_by_date(getattr(ppaths, 'ms'), '*.tif', date_pos, 0, 8)

    image_names = list(image_dict.keys())
    time_names = list(image_dict.values())

    proc_names = []
    proc_times = []

    for fn, fnt in zip(image_names, time_names):

        if 'angles' not in fn:
            proc_names.append(fn)
            proc_times.append(fnt)

    for fn, date in zip(proc_names, proc_times):

        # Get the image date
        idate_str = datetime.strftime(date, '%Y%m%d')

        # Get the mask image
        imask_image = ppaths.masks.joinpath(idate_str + '.tif')

        if not imask_image.is_file():
            logger.warning(f'Mask file {imask_image.name} for grid {grid} does not exist.')
            raise errors.MaskError(imask_image)


def precheck_classify(params):

    """
    Checks if all variables are complete

    Args:
        params (dict)
    """

    for grid in params['grids']:

        ppaths = ProjectPaths(params, grid=grid)

        for image_var in params['classify']['image_bands']:

            ts_dir = ppaths.ts / image_var

            with open(ts_dir / f'{grid:06d}.window', mode='r') as pf:
                window_tracker = yaml.load(pf, Loader=yaml.FullLoader)

            if window_tracker[params['reconstruct']['chunks']]['latest'] != 1e9:
                logger.warning(f'Grid {grid}, variable {image_var} is not complete.')
                raise errors.TimeSeriesError(window_tracker[params['reconstruct']['chunks']]['latest'])

def make_thumbnail_batch(img_dir,thumbnail_dir,yr,params):

    gamma = params['plot']['gamma']
    reduct_factor = params['plot']['reduct_factor']
    include = params['reconstruct']['include']
    exclude=params['reconstruct']['exclude']

    if exclude and (exclude != 'None' and exclude !=''):
        imgs0 = list(img_dir.glob(f'*{include}[!{exclude}].nc')) + list(img_dir.glob(f'*{include}[!{exclude}].tif'))
    else:
        imgs0 = list(img_dir.glob(f'*{include}.nc')) +  list(img_dir.glob(f'*{include}.tif'))

    imgs = [im for im in imgs0 if im.stem.split('_')[3].startswith(str(yr))]
    
    logger.info(f'adding {len(imgs)} thumbnails for {yr} to {thumbnail_dir}')

    for i, img in enumerate(imgs):
        try:
            imgi = get_rbg_img(img,gamma)
            imgis = imgi[::reduct_factor, ::reduct_factor]
        except Exception as e:
            logger.warning(f'OOPS -- got this error for {img.stem}: {e}')
            continue
        fig = plt.figure(figsize=(20,20),dpi=80)
        plt.imsave(thumbnail_dir/f'{img.stem}.png', imgis)
        plt.close(fig)     
        
def make_thumbnails(params):

    image_type = params['image_type']
    yrs = [params['reconstruct']['start'][:4], params['reconstruct']['end'][:4]]

    if isinstance(params['grids'],int):
        params['grid'] = [params['grid']]
    
    for grid in params['grids']:
        ppaths = ProjectPaths(params, grid=grid) 

        logger.info(f'making thumbnails for cell {grid} from {yrs[0]} to {yrs[1]}')
    
        if (image_type == 'AllRaw') or (image_type == 'LS2'): ##note AllRaw is legacy and can be deleted at some point
            prefix=f"cell{grid}_brdf_FirstPassNoX"
            img_dir = ppaths.ms
            thumbnail_dir = ppaths.ms.parent / 'thumbnails/brdf'
        elif image_type.startswith('L'):
            prefix=f'cell{grid}_landsat_raw'
            img_dir = ppaths.ms.parent / 'landsat'
            thumbnail_dir = ppaths.ms.parent / 'thumbnails/dl'
        elif image_type == 'S2':
            prefix=f'cell{grid}_sentinel_raw'
            img_dir = ppaths.ms.parent,'sentinel2'
            thumbnail_dir = ppaths.ms.parent / 'thumbnails/dl'

        thumbnail_dir.mkdir(parents=True, exist_ok=True)

        if yrs and (yrs != 'None'):
        
            for yr in range(int(yrs[0]),int(yrs[1])+1): 
                for fn in list(thumbnail_dir.glob('*.png')):
                    if fn.stem.split('_')[3].startswith(str(yr)):
                        fn.unlink()
                make_thumbnail_batch(img_dir,thumbnail_dir,yr,params)
            
        else:
            ## remake all images
            for fn in list(thumbnail_dir.glob('*.png')):
                fn.unlink()

            make_thumbnail_batch(img_dir,thumbnail_dir,yr,params)