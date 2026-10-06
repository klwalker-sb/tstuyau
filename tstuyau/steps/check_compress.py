import os
from pathlib import Path

import geowombat as gw

from ..handler import logger
from . import utils
from .project import ProjectPaths


def compress(params):

    """
    Compresses GeoTiffs into NetCDF files

    Args:
        params (dict)

    Returns:
        None
    """

    cells = utils.get_cell_list_from_grid_param(params['grids'])
    for grid in cells:

        logger.info(f'  Compressing files for grid {grid} ...')

        ppaths = ProjectPaths(params, grid=grid)

        # [ppaths.ms, ppaths.ts]
        for walk_dir in [ppaths.ms]:

            for root, dirs, files in os.walk(str(walk_dir)):

                if files:

                    for fn in Path(root).glob('*.tif'):

                        outfile = fn.parent / fn.name.replace('.tif', '.nc')

                        if outfile.is_file():
                            outfile.unlink()

                        if fn.name.endswith('_angles.tif'):
                            band_names = ['sza', 'saa']
                        else:
                            band_names = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

                        with gw.open(fn, band_names=band_names) as src:
                            gw.to_netcdf(src, outfile, zlib=True, complevel=5)

                        if outfile.is_file():
                            fn.unlink()
