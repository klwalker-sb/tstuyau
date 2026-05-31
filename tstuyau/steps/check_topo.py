from pathlib import Path
from datetime import datetime

from ..handler import logger
from ..db import TuyauDataBase
from .project import ProjectPaths
from .web_utils import download_hgt

import geowombat as gw
from geowombat.core import sort_images_by_date
from geowombat.radiometry import Topo
from geowombat.radiometry.topo import calc_slope_delayed, calc_aspect_delayed

import geopandas as gpd
import dask.array as da
import xarray as xr


def delayed_to_xarray(delayed_data, data):

    dst_ = xr.DataArray(data=delayed_data,
                        coords={'y': data.y.values,
                                'x': data.x.values},
                        dims=('y', 'x'),
                        attrs=data.attrs)

    return dst_.assign_coords(band=1).expand_dims(dim='band')


def adjust_topo(params):

    """
    Normalizes topographic effects

    Args:
        params (dict)

    Returns:
        None
    """

    slope_kwargs = dict(format='MEM',
                        computeEdges=True,
                        alg='ZevenbergenThorne',
                        slopeFormat='degree')

    aspect_kwargs = dict(format='MEM',
                         computeEdges=True,
                         alg='ZevenbergenThorne',
                         trigonometric=False,
                         zeroForFlat=True)

    ppaths = ProjectPaths(params)

    srtm_file = ppaths.srtm / 'srtm30m_bounding_boxes.gpkg'
    samples_file = ppaths.calval / 'random_grids_v4_wgs84.gpkg'
    # nations_file = ppaths.political / 'ne_50m_admin_0_countries.shp'

    srtm_df = gpd.read_file(str(srtm_file))
    samples_df = gpd.read_file(str(samples_file))
    # nations_df = gpd.read_file(str(nations_file))

    topo = Topo()

    key_file = params['topo']['key_file']
    code_file = params['topo']['code_file']

    db = TuyauDataBase(params['database'])

    for grid in params['grids']:

        ppaths = ProjectPaths(params, grid=grid)

        ts_dir = ppaths.ts / params['reconstruct']['si']

        # Check if the file is complete
        # if not db.is_complete(grid, 'reindex'):
        #     continue

        # Intersect the SRTM grids with the sample grid
        srtm_df_int = srtm_df[srtm_df.geometry.intersects(samples_df.query(f"UNQ == {grid}").geometry.values[0])]

        # Intersect with the nations
        # nations_df_int = nations_df[['CONTINENT', 'geometry']][nations_df.geometry.intersects(srtm_df_int.geometry.values[0])]
        # hgt_url = f'{ppaths.hgt_url}/{HGT_CONTINENT_DICT[nations_df_int.CONTINENT.values[0]]}/hgt_merge'

        # Download the intersecting SRTM tiles
        hgt_files = download_hgt(params, ppaths, srtm_df_int, ppaths.hgt_url, key_file, code_file)

        if len(hgt_files) == 1:
            hgt_files = hgt_files[0]
            mosaic = False
        else:
            mosaic = True

        # Open the angle files
        image_dict = sort_images_by_date(ppaths.ms,
                                         '*.nc',
                                         date_pos=3,
                                         date_start=0,
                                         date_end=8,
                                         prepend_str='netcdf:')

        angle_image_names = list(image_dict.keys())
        angle_time_names = list(image_dict.values())

        # Open the si files
        image_dict = sort_images_by_date(ts_dir,
                                         '*.tif',
                                         date_pos=0,
                                         date_start=0,
                                         date_end=7,
                                         date_format='%Y%j')

        image_names = list(image_dict.keys())
        time_names = list(image_dict.values())

        open_kwargs = dict(chunks=512)

        with gw.open(angle_image_names,
                     time_names=angle_time_names,
                     netcdf_vars=['sza', 'saa'],
                     chunks=params['reconstruct']['chunks']) as src_ang:

            logger.debug(f"src_ang:{src_ang}")
            if params['log_lev']=='DEBUG':
                srcdata = src_ang.sel(band='sze').data.compute()
                logger.debug(f"src data: {srcdata}")

            # Iterate over each si image and flatten topograph
            for fn, dt in zip(image_names, time_names):

                fn_path = Path(fn)

                if fn_path.name != '2005121.tif':
                    continue

                fn_date = datetime.strptime(fn_path.stem, '%Y%j')

                # Get the angle nearest to the si date
                src_ang_slice = src_ang.sel(time=fn_date, method='nearest')

                with gw.config.update(ref_image=str(fn)):

                    with gw.open(fn,
                                 band_names=['vi'],
                                 dtype='float64',
                                 resampling='nearest',
                                 **open_kwargs) as src, \
                            gw.open(hgt_files,
                                    band_names=['elev'],
                                    mosaic=mosaic,
                                    dtype='float64',
                                    nodata=params['topo']['srtm_nodata'],
                                    resampling='average',
                                    **open_kwargs) as src_srtm:

                        bounds = src_srtm.gw.bounds_as_namedtuple

                        # Transform the SRTM to UTM @30m (i.e., native resolution)
                        src_srtm_res = src_srtm.gw.transform_crs(dst_crs=src_srtm.crs,
                                                                 dst_res=(30, 30),
                                                                 resampling='average',
                                                                 num_threads=params['num_workers'])

                        # Calculate slope and aspect
                        slope_deg = calc_slope_delayed(src_srtm_res.squeeze().data, **slope_kwargs)
                        slope_deg_fd = da.from_delayed(slope_deg, (src_srtm_res.gw.nrows, src_srtm_res.gw.ncols), dtype='float64')
                        src_slope = delayed_to_xarray(slope_deg_fd, src_srtm_res)

                        aspect_deg = calc_aspect_delayed(src_srtm_res.squeeze().data, **aspect_kwargs)
                        aspect_deg_fd = da.from_delayed(aspect_deg, (src_srtm_res.gw.nrows, src_srtm_res.gw.ncols), dtype='float64')
                        src_aspect = delayed_to_xarray(aspect_deg_fd, src_srtm_res)

                        # Transform back to 10m
                        src_slope = src_slope.gw.transform_crs(dst_crs=src_srtm.crs,
                                                               dst_bounds=bounds,
                                                               dst_res=src.gw.celly,
                                                               resampling='average',
                                                               num_threads=params['num_workers'])

                        src_aspect = src_aspect.gw.transform_crs(dst_crs=src_srtm.crs,
                                                                 dst_bounds=bounds,
                                                                 dst_res=src.gw.celly,
                                                                 resampling='average',
                                                                 num_threads=params['num_workers'])

                        attrs = src.attrs.copy()
                        src = (src * 0.0001).clip(0, 1)
                        src.attrs = attrs

                        out = topo.norm_topo(src,
                                             src_srtm,
                                             src_ang_slice.sel(band='sza'),
                                             src_ang_slice.sel(band='saa'),
                                             slope=src_slope,
                                             aspect=src_aspect,
                                             nodata=-999,
                                             n_jobs=params['num_workers'],
                                             slope_thresh=params['topo']['slope_thresh'],
                                             method=params['topo']['method'],
                                             angle_scale=0.01,
                                             min_samples=1000)  # 10% of a 10km x 10km grid

                        out = (out * 10000.0).astype('uint16')
                        out.attrs = attrs

                        out.gw.to_raster(fn_path.parent.joinpath(fn_path.stem + '_topo' + fn_path.suffix),
                                         n_workers=1,
                                         n_threads=int(params['num_workers']),
                                         n_chunks=int(params['io']['n_chunks']),
                                         overwrite=True,
                                         compress='lzw')

                        src_slope.attrs = attrs
                        src_slope.gw.to_raster(fn_path.parent.joinpath(fn_path.stem + '_topo_slope' + fn_path.suffix),
                                               n_workers=1,
                                               n_threads=int(params['num_workers']),
                                               n_chunks=int(params['io']['n_chunks']),
                                               overwrite=True,
                                               compress='lzw')

                        src_aspect.attrs = attrs
                        src_aspect.gw.to_raster(fn_path.parent.joinpath(fn_path.stem + '_topo_aspect' + fn_path.suffix),
                                                n_workers=1,
                                                n_threads=int(params['num_workers']),
                                                n_chunks=int(params['io']['n_chunks']),
                                                overwrite=True,
                                                compress='lzw')

                        return

