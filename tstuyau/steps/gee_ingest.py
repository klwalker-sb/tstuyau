from abc import ABC, abstractmethod
from pathlib import Path
import concurrent.futures
import json
from tqdm import tqdm
import string
import random
import shutil
from datetime import datetime
from collections import namedtuple
import numpy as np
import pandas as pd
import geopandas as gpd

from .lookup import GEE_TRANSLATIONS
from ..handler import logger
from .web_utils import GEE, WebGCP
from .image_utils import get_grid_bounds

from geowombat.core.properties import get_sensor_info



class IngestAbstract(ABC):

    @abstractmethod
    def ingest(self):
        raise NotImplementedError

class IngestFromGoogle(IngestAbstract, GEE):

    def __init__(self, verbose=0):

        self.verbose = verbose
        GEE.__init__(self)

    def ingest(self):
        pass

    def ingest_from_gee(self, params, grid, ppaths):

        """
        """
        
        df = gpd.read_file(params['grid_file'])
        cell = df.query(f"UNQ == {grid}")

        bounds, proj_bounds, proj_crs = get_grid_bounds(
                    params['grid_file'],
                    params['grid_size'],
                    grid,
                    params['buffer'],
                    centroid_to_utm='n')

        ## note: dates for GEE are written YYYY-MM-DD or YYYY-M-D -- no leading zeros
        if params['masking']['date_range'][0] == 0:
            ## TODO: get full date range from db file
            start_date = 2017-1-1
            end_date = 2025-1-1
        else:
            start_date = params['masking']['date_range'][0]
            end_date = params['masking']['date_range'][1]
        
        if 'S2cp' in params['image_type']:
            out_dir = ppaths.ms.parent.joinpath('sentinel2')
            satellite = 'sentinel-2'
            asset_id = 'COPERNICUS/S2_CLOUD_PROBABILITY'
            bands = ['probability']
            prod = 'S2cp'
        elif 'S2' in params['image_type']:
            out_dir = ppaths.ms.parent.joinpath('sentinel2')
            satellite = 'sentinel-2'
            asset_id = 'COPERNICUS/S2'
            prod = 'S2'
        else:
            for sensor in ['LC08','LC09','LE07','LT05']:
                if sensor in params['image_type']:
                    out_dir = ppaths.ms.parent.joinpath('landsat')
                    satellite = 'landsat'
                    asset_id = f'LANDSAT/{sensor}/C01/T1_SR'
                    prod = sensor
        
        out_path = Path(out_dir).absolute()
        if satellite not in ['sentinel-2','landsat']:
            logger.warning('only sentinel-2 and landsat are supported for GEE download at this time. see eosvault for other options')
            wg = None
        else:
            logger.info(f'downloading files to {out_dir}...')
            
            wg = WebGCP(params['masking']['gee_index_dir'],
                    satellite=satellite,
                    verbose=self.verbose)

            ## Get the database if it doesn't exist
            wg.download_index()

            ## Read the metadata
            wg.read_index(satellite=satellite,
                          start_date=start_date,
                          end_date=end_date,
                          sensors=prod,
                          bounds=bounds,
                          collection='01',
                          batch_size=50000,
                          num_cpus=1)

            logger.info('authenticating connection to gee...')
            self.authenticate_gee(params['masking']['gee_key'])

            self.query_gee(asset_id=asset_id,
                       bounds=bounds,
                       satellite=satellite,
                       start_date=start_date,
                       end_date=end_date)

            def create_asset_id_sentinel2(row):
                return f"{row['PRODUCT_ID'].split('_')[2]}_{row['GRANULE_ID'].split('_')[3]}_{row['GRANULE_ID'].split('_')[1]}"

            def create_asset_id_landsat(row):
                return f"{asset_id}/{row['PRODUCT_ID'].split('_')[0]}_{int(row.WRS_PATH):03d}{int(row.WRS_ROW):03d}_{row.DATE_ACQUIRED.replace('-', '')}"

            if satellite == 'sentinel-2':
                wg.sat_index_df['ASSET_ID'] = wg.sat_index_df.apply(create_asset_id_sentinel2, axis=1)
            elif satellite == 'landsat':
                wg.sat_index_df['ASSET_ID'] = wg.sat_index_df.apply(create_asset_id_landsat, axis=1)

            future_results = []

            if self.query_content.decode().strip() == '{}':
                logger.warning('  The image query was empty.')

            if params['num_workers'] == 1:

                for asset in json.loads(self.query_content)['images']:
                    platform = asset.get('PLATFORM_ID')
                    logger.info(f'platform:{platform}')

                    future_results.append(self.stream_to_file(out_dir,
                                                          wg,
                                                          asset_id.split('/')[1],
                                                          asset,
                                                          satellite,
                                                          bands,
                                                          False,
                                                          params['res'],
                                                          bounds,
                                                          proj_bounds,
                                                          proj_crs,
                                                          True,
                                                          False,
                                                          True,
                                                          params))

            else:

                # Download individual scenes
                with concurrent.futures.ThreadPoolExecutor(max_workers=params['num_workers']) as executor:

                    # Submit futures
                    futures = [executor.submit(self.stream_to_file,
                                           out_dir,
                                           wg,
                                           asset_id.split('/')[1],
                                           asset,
                                           satellite,
                                           bands,
                                           False,
                                           params['res'],
                                           bounds,
                                           proj_bounds,
                                           proj_crs,
                                           False,
                                           False,
                                           True,
                                           params) for asset in json.loads(self.query_content)['images']]

                    for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                        future_results.append(f.result())

            # `future_results` should all be `True`
            if all(future_results): #or ignore_incomplete:
                #db.update(satellite, asset_id)
                logger.info(f'{len(future_results)} downloaded successfully.')
            else:
                logger.warning(f'{sum(future_results)} out of {len(future_results)} downloaded successfully.')

        
    @staticmethod
    def _prepare_scenes(scene_df, bands, satellite):

        url_dict = {}

        if satellite == 'landsat':

            for row in scene_df.itertuples():

                sensor = GEE_TRANSLATIONS['landsat']['gcp'][row.SENSOR_ID]
                wavelengths = get_sensor_info(key='wavelength', sensor=sensor)

                file_urls = [f"{row.BASE_URL}/{row.PRODUCT_ID}_B{getattr(wavelengths, band)}.TIF" for band in bands]

                file_urls.append(f'{row.BASE_URL}/{row.PRODUCT_ID}_ANG.txt')
                file_urls.append(f'{row.BASE_URL}/{row.PRODUCT_ID}_MTL.txt')

                url_dict[row.PRODUCT_ID] = file_urls

        elif satellite == 'sentinel-2':

            for row in scene_df.itertuples():
                sensor = row.PRODUCT_ID.split('_')[0].lower()
                wavelengths = get_sensor_info(key='wavelength', sensor=sensor)

                file_urls = [f"{row.BASE_URL}/GRANULE/{row.GRANULE_ID}/IMG_DATA/{row.GRANULE_ID.split('_')[1]}_{row.DATATAKE_IDENTIFIER.split('_')[1]}_B{getattr(wavelengths, band):02d}.jp2" for band in bands]

                file_urls.append(f"{row.BASE_URL}/GRANULE/{row.GRANULE_ID}/MTD_TL.xml")

                url_dict[f'{row.GRANULE_ID}_MTD'] = file_urls

        return url_dict