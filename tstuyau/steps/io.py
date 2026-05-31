import _pickle as cPickle
from datetime import datetime
from pathlib import Path
import concurrent.futures

from .spec_indices import calc_si_gw, SpecIndices
from ..handler import logger

import geowombat as gw
from geowombat.core import sort_images_by_date

import numpy as np
import pandas as pd
import h5py
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import dask.array as da
# from dask.distributed import performance_report, progress
# from dask.distributed import Client, LocalCluster
import ray
from ray.util.dask import ray_dask_get
from tqdm.auto import tqdm


ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor


class DataIO(object):

    def __init__(self, data_file):
        self.data_file = data_file

    def to_file(self, data):

        with open(self.data_file, mode='wb') as fh:
            cPickle.dump(data, fh)

    def from_file(self):

        with open(self.data_file, mode='rb') as fh:
            ldata = cPickle.load(fh)

        return ldata


class ImageIO(object):

    def __init__(self, hk_list, mk_list):

        hk_image_dict = sort_images_by_date(None, '*.tif', 0, 0, 7, file_list=hk_list, date_format='%Y%j')
        mk_image_dict = sort_images_by_date(None, '*.tif', 0, 0, 7, file_list=mk_list, date_format='%Y%j')

        hk_image_list = [Path(fn) for fn in list(hk_image_dict.keys())]
        mk_image_list = [Path(fn) for fn in list(mk_image_dict.keys())]

        self.file_lists = {'hk': hk_image_list,
                           'mk': mk_image_list}

    def load_data(self, which, near_idx, ppaths, params):

        """
        Returns:
            Mask, bands, angles
        """

        # kdate_str = self.file_lists[which][near_idx].stem

        # kmask_image = ppaths.masks.joinpath(kdate_str + '.tif')
        # angles_image = ppaths.ms.joinpath(self.file_lists[which][near_idx].name.replace('.tif', '_angles.tif'))

        # Ensure both files exist
        # if not angles_image.is_file():
        #
        #     # Remove the file to re-process
        #     if self.file_lists[which][near_idx].is_file():
        #         self.file_lists[which][near_idx].unlink()
        #
        #     return None, None, None

        # if not kmask_image.is_file():
        #
        #     # Remove the file to re-process
        #     if self.file_lists[which][near_idx].is_file():
        #         self.file_lists[which][near_idx].unlink()
        #
        #     return None, None, None

        # gw.open(kmask_image) as mask_k_src, \
        # gw.open(angles_image) as angles_k_src:
        with gw.open(self.file_lists[which][near_idx], band_names=params['fusion']['wavelengths']) as res_k_src:

            # attrs = res_k_src.attrs.copy()

            # res_k_src = xr.where((mask_k_src.sel(band=1) > params['masking']['min_mask']) | (res_k_src.max(dim='band') == 0),
            #                      params['nodata'],
            #                      res_k_src)\
            #                 .transpose('band', 'y', 'x')\
            #                 .assign_attrs(**attrs)

            res_k_src = res_k_src.gw.set_nodata(0, 0, (0, 1), 'float64', scale_factor=0.0001)

            res_k_data = res_k_src.data.compute(num_workers=params['num_workers'])
            # res_k_mdata = mask_k_src.sel(band=1).data.compute(num_workers=params['num_workers'])

            # res_k_sza = (angles_k_src.where(angles_k_src != -32768).sel(band=1)*0.01)\
            #                 .data.compute(num_workers=params['num_workers'])

        res_k_data[np.isnan(res_k_data)] = 0
        # res_k_mdata[res_k_data.min(axis=0) == 0] = 255
        # res_k_sza[np.isnan(res_k_sza)] = 90.0

        return res_k_data


def read_nc(filename, band_names, slicer, nodata, spec_index, extra_param=None):
    """
    Reads a single NetCDF file
    """

    with open(str(filename).replace('netcdf:', ''), mode='rb') as fo:
        try:
            f = h5py.File(fo, mode='r')
        except OSError as e:
            logger.warning(f'{e} on {filename}')
        # Stack the bands
        array = np.asarray([f.get(bd)[slicer] for bd in band_names])

    # Scale the array from [0,10000] -> [0,1]
    array = np.where(array == nodata, 0, array*0.0001)\
                .astype('float64')\
                .clip(min=0, max=1)

    return SpecIndices(spec_index)(array, extra_param)


def read_func(row, band_names, slicer, nodata, spec_index, dataframe, extra_param=None):

    if row.dupe == 'dupe1':

        yarr = np.stack([read_nc(dupe_row.name, band_names, slicer, nodata, spec_index, extra_param)
                         for dupe_row in dataframe.query(f"date == '{row.date}'").itertuples()]).max(axis=0)

    elif row.dupe == 'no':
        yarr = read_nc(row.name, band_names, slicer, nodata, spec_index, extra_param)
    else:
        yarr = None

    return yarr


class TimeSeriesLoader(object):
    def __init__(
        self,
        time_band_df_slice,
        start_pad_dt_slice,
        end_pad_dt_slice,
        band_names,
        slicer,
        params
    ):
        self.time_band_df_slice = time_band_df_slice
        self.start_pad_dt_slice = start_pad_dt_slice
        self.end_pad_dt_slice = end_pad_dt_slice
        self.band_names = band_names
        self.slicer = slicer
        self.params = params

    def load_on_cluster(self):
        with gw.open(
            self.time_band_df_slice.image_path.values.tolist()[0],
            time_names=self.time_band_df_slice.index.to_pydatetime().tolist()[0],
            netcdf_vars=self.band_names,
            chunks=self.params['reconstruct']['chunks']
        ) as src:
            attrs = src.attrs.copy()

        ray.shutdown()
        ray.init(num_cpus=self.params['num_workers'])

        with dask.config.set(scheduler=ray_dask_get):

            # with LocalCluster(n_workers=self.params['num_workers'],
            #                   threads_per_worker=1,
            #                   scheduler_port=0,
            #                   processes=True,
            #                   memory_limit=f'6GB') as cluster:
            #
            #     with Client(address=cluster) as client:

            def expand_time(dataset):
                """`open_mfdataset` preprocess function
                """

                # Convert the Dataset into a DataArray,
                # rename the band coordinate,
                # select the required si bands,
                # assign y/x coordinates from a reference,
                # add the time coordiante, and
                # get the sub-array slice
                darray = (
                    dataset.to_array()
                    .rename({'variable': 'band'})
                    .sel(band=self.band_names)
                    .assign_coords(y=src.y, x=src.x)
                    .expand_dims(dim='time')
                    .clip(0, self.params['nodata'])[self.slicer]
                )

                # Scale from [0-10000] -> [0,1]
                return (
                    xr.where(darray == self.params['nodata'], 0, darray*0.0001)
                    .astype('float64')
                    .clip(0, 1)
                )

            # Open all arrays and calculate the VI
            si_base = self.params['reconstruct']['si']
            if '.' in si_base:
                si_base = si_base.split('.')[0]
            elif any(m in si_base for m in ('-', '_')):
                si_base = si_base.split(m)[0]
            if (self.params['project_ver'] == 'Py_0') and (si_base in ['ndmi','ndvi','nbr']):
                si_base = f'{si_base}V0'
                logger.info(f'using legacy code to calculate {si}')
            ds = getattr(gw,si_base)(xr.open_mfdataset(self.time_band_df_slice.image_path.str.replace('netcdf:', '').values.tolist(),
                                                                                 concat_dim='time',
                                                                                 chunks={'y': 512, 'x': 512},
                                                                                 combine='nested',
                                                                                 engine='h5netcdf',
                                                                                 preprocess=expand_time,
                                                                                 parallel=True)\
                                                                    .assign_coords(time=self.time_band_df_slice.index.to_pydatetime().tolist())\
                                                                    .groupby('time').max()\
                                                                    .sel(time=slice(datetime.strftime(self.start_pad_dt_slice, '%Y-%m-%d'),
                                                                                    datetime.strftime(self.end_pad_dt_slice, '%Y-%m-%d')))
                                                          .assign_attrs(**attrs), nodata=0, scale_factor=1).squeeze()

            # Get the time series dates after grouping
            real_proc_times = ds.gw.pydatetime

            # Convert the DataArray into a NumPy array
            # ds.data.visualize(filename='graph.svg')
            # with performance_report(filename='dask-report.html'):
            with ProgressBar():
                y = ds.data.compute()

        ray.shutdown()

        return real_proc_times, y

    def load_netcdf(self):
        
        # Load the filenames and dates into a DataFrame
        df = pd.DataFrame(data=self.time_band_df_slice.image_path.values.tolist(),
                          columns=['name'],
                          index=self.time_band_df_slice.index.to_pydatetime().tolist())

        # Get the date as a string
        df['date'] = [datetime.strftime(dt, '%Y%m%d') for dt in self.time_band_df_slice.index.to_pydatetime().tolist()]

        # Locate duplicate dates
        df['dupe'] = 'no'
        df.loc[df.duplicated('date', keep='last'), 'dupe'] = 'dupe1'
        df.loc[df.duplicated('date', keep='first'), 'dupe'] = 'dupe2'

        # Get the processing dates after removing duplicates (don't need 2nd duplicate)
        real_proc_times = [datetime.strptime(date, '%Y%m%d')
                           for date in df.query("dupe == ['dupe1', 'no']").date.values.tolist()]

        si_arrays = []
        si = self.params['reconstruct']['si']
        ## sis may be passed in with parameters attached (e.g. savi.100 and/or with ts info (e.g. savi-raw or savi.100-raw). '_' is legacy only.
        if '.' in si:
            si = si.split('.')[0]
            si_extra = si.split('.')[1].split('-')[0]
        elif any(m in si for m in ('-', '_')):
            si = si.split(m)[0]
            si_extra = None
        else:
            si_extra = None
        ## legacy code for CELPy maps
        if (self.params['project_ver'] == 'Py_0') and (si in ['ndmi','ndvi','nbr']):
            si = f'{si}V0'
            logger.info(f'using legacy code to calculate {si}')
            
        data_gen = (
            (
                row,
                self.band_names,
                self.slicer[2:],
                self.params['nodata'],
                si,
                df,
                si_extra
            ) for row in df.itertuples()
        )

        with ThreadPoolExecutor(max_workers=self.params['num_workers']) as executor:
            for res in tqdm(executor.map(lambda f: read_func(*f), data_gen), total=df.shape[0]):
                if res is not None:
                    si_arrays.append(res)

        y = np.stack(si_arrays)

        return real_proc_times, y

    def load(self):

        '''
        Note this calls calc_si_gw, which uses methods with geowombat to calculate the si 
        (limited to only a few sis). 
        '''
        with gw.open(self.time_band_df_slice.image_path.values.tolist(),
                     time_names=self.time_band_df_slice.index.to_pydatetime().tolist(),
                     netcdf_vars=self.band_names,
                     chunks=self.params['reconstruct']['chunks']) as src:
                # gw.open(mask_image_list,
                #         time_names=mask_time_list,
                #         band_names=mask_band_names,
                #         netcdf_vars=mask_band_names,
                #         chunks=params['reconstruct']['chunks']) as lmask_src:

            attrs = src.attrs.copy()

            # if self.params['reconstruct']['use_masks']:
            #
            #     # Some mask files might not exist so extract the correct image files.
            #     if src.gw.ntime != lmask_src.gw.ntime:
            #
            #         src = src.sel(time=lmask_src.gw.pydatetime) \
            #             .transpose('time', 'band', 'y', 'x')

            # Get the time slice
            # src = src.sel(time=slice(datetime.strftime(start_pad_dt_slice, '%Y-%m-%d'),
            #                          datetime.strftime(end_pad_dt_slice, '%Y-%m-%d')))[slicer]

            # lmask_src = lmask_src.sel(time=slice(datetime.strftime(start_pad_dt_slice, '%Y-%m-%d'),
            #                                      datetime.strftime(end_pad_dt_slice, '%Y-%m-%d')))[slicer]

            src = src[self.slicer]

            # if self.params['reconstruct']['use_masks']:
            #
            #     lmask_src = lmask_src[slicer]
            #
            #     if not lmask_src.gw.has_time_dim:
            #         lmask_src = lmask_src.expand_dims(dim='time')
            #
            #     if not lmask_src.gw.has_time_coord:
            #         lmask_src = lmask_src.assign_coords(coords={'time': 1})
            #
            #     lmask_src = lmask_src.transpose('time', 'band', 'y', 'x')
            #
            #     src = src.astype('float64') \
            #         .where(lmask_src.sel(band='mask') <= params['masking']['min_mask']) \
            #         .where(xr.ufuncs.isfinite(src)) \
            #         .fillna(0) \
            #         .transpose('time', 'band', 'y', 'x') \
            #         .assign_attrs(**attrs)

            src = xr.where(src == self.params['nodata'], 0, src*0.0001).astype('float64').clip(0, 1)

            y = calc_si_gw(src, self.params).squeeze().data.compute(num_workers=self.params['num_workers'])

            real_proc_times = src.gw.pydatetime

        return real_proc_times, y

