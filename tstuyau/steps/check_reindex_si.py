from ..handler import logger
from ..db import TuyauDataBase
from .project import ProjectPaths

import yaml


def reindex_si(params):

    """
    Reindexes vegetation indices from 60 weekly values to 30 bi-weekly values

    Args:
        params (dict)

    Returns:
        None
    """

    for grid in params['grids']:

        ppaths = ProjectPaths(params, grid=grid)

        db = TuyauDataBase(str(ppaths.ms.parent / f'{int(grid):06d}_tuyau.db'))

        # Check if the step is complete
        if not db.is_complete(grid, 'reconstruct'):
            logger.warning(f'  The reconstruction step is not complete.')
            continue

        # Check if the step is complete
        if db.is_complete(grid, 'reindex'):
            logger.warning(f'  The reindexing step is complete.')
            continue

        if not getattr(ppaths, 'ms').is_dir():
            logger.warning(f'  The input directory for grid {grid} does not exist.')
            continue

        ts_dir = ppaths.ts / params['reconstruct']['si']

        # Check if the file is complete
        if (ts_dir / f'{grid:06d}.window').is_file():

            with open(ts_dir / f'{grid:06d}.window', mode='r') as pf:
                window_tracker = yaml.load(pf, Loader=yaml.FullLoader)

            if int(window_tracker[params['reconstruct']['chunks']]['latest']) != 1e9:
                logger.warning(f'Grid {grid} is not complete.')
                continue

        file_list = list(ts_dir.glob('*.tif'))

        remove_list = file_list[1:][::2]

        for fn in remove_list:
            fn.unlink()

        db.update(grid, 'reindex')
