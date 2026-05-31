import sys
# import base64
import requests
import urllib
import zipfile
import gzip
from getpass import getpass
import math
from pathlib import Path
import yaml
from tqdm import tqdm
from cryptography.fernet import Fernet

from abc import ABC, abstractmethod

import subprocess
from datetime import datetime
import io
import json
import shutil
import geowombat as gw
from geowombat.core.properties import get_sensor_info
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.coords import BoundingBox
import affine
from retry import retry

from ..handler import logger
from .lookup import GEE_TRANSLATIONS, FILE_EXTENSIONS
from .utils import resample, tag_array, band_is_ok, get_qa_mask, check_missed_nodata, mask_data
from .image_utils import latlon_to_utm, polygon_from_bounds, geom_intersects



class PassKey(object):

    @staticmethod
    def create_key(key_file):

        key = Fernet.generate_key()

        with open(key_file, mode='w') as pf:
            yaml.dump({'key': key}, pf, default_flow_style=False)

    @staticmethod
    def create_passcode(key_file, passcode_file):

        """
        Args:
            key_file (str)
            passcode_file (str)
        """

        passcode = getpass()

        with open(key_file, mode='r') as pf:
            key = yaml.load(pf, Loader=yaml.FullLoader)

        cipher_suite = Fernet(key['key'])

        ciphered_text = cipher_suite.encrypt(passcode.encode())

        with open(passcode_file, mode='w') as pf:
            yaml.dump({'passcode': ciphered_text}, pf, default_flow_style=False)

    @staticmethod
    def load_passcode(key_file, passcode_file):

        with open(key_file, mode='r') as pf:
            key = yaml.load(pf, Loader=yaml.FullLoader)

        cipher_suite = Fernet(key['key'])

        with open(passcode_file, mode='r') as pf:
            ciphered_text = yaml.load(pf, Loader=yaml.FullLoader)

        return cipher_suite.decrypt(ciphered_text['passcode'])


class HTTPRedirectHandler(urllib.request.HTTPRedirectHandler):

    """
    Source:
        PyMODIS: www.pymodis.org
    """

    def http_error_302(self, req, fp, code, msg, headers):
        return urllib.request.HTTPRedirectHandler.http_error_302(self, req, fp, code, msg, headers)


class EarthDataDownloader(object):

    def __init__(self, username, key_file, code_file):

        self.username = username
        self.key_file = key_file
        self.code_file = code_file

        self.pk = PassKey()

    def download(self, url, outfile):

        outpath = Path(outfile)

        base64_password = self.pk.load_passcode(self.key_file, self.code_file).decode()

        chunk_size = 256 * 10240

        with requests.Session() as session:

            session.auth = (self.username, base64_password)

            # Open
            req = session.request('get', url)
            response = session.get(req.url, auth=(self.username, base64_password))

            if not response.ok:
                logger.exception('  Could not retrieve the page.')
                raise NameError

            if 'Content-Length' in response.headers:

                content_length = float(response.headers['Content-Length'])
                content_iters = int(math.ceil(content_length / chunk_size))
                chunk_size_ = chunk_size * 1

            else:

                content_iters = 1
                chunk_size_ = chunk_size * 1000

            logger.info(f'  Downloading into {outpath.parent} ...')

            with open(str(outfile), 'wb') as ofn:

                for data in tqdm(response.iter_content(chunk_size=chunk_size_), total=content_iters):
                    ofn.write(data)


def download_omi_toms(year):

    url = f'https://acdisc.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level3/OMTO3d.003/{year}'


def download_hgt(params, ppaths, dataframe, hgt_url, key_file, code_file):

    # pass_str = f"{params['topo']['username']}:{pk.load_passcode(key_file, code_file).decode()}".replace('\n', '')
    # base64_username = base64.urlsafe_b64encode(params['topo']['username'].encode()).decode()
    # base64_password = base64.urlsafe_b64encode(pk.load_passcode(key_file, code_file)).decode()

    edd = EarthDataDownloader(params['topo']['username'], key_file, code_file)

    hgt_files = []

    for dfn in dataframe.dataFile.values.tolist():

        zip_file = f"NASADEM_HGT_{dfn.split('.')[0].lower()}.zip"

        srtm_zip_file = ppaths.srtm / zip_file
        srtm_hgt_file = srtm_zip_file.parent.joinpath(f"{srtm_zip_file.name.replace('.zip', '').split('_')[-1]}.hgt")

        if srtm_zip_file.exists() or srtm_hgt_file.is_file():
            logger.info(f'  {srtm_zip_file.name} already exists.')
        else:

            url = f'{hgt_url}/{zip_file}'
            # url = 'https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/South_America/'

            edd.download(url, srtm_zip_file)

            zf = zipfile.ZipFile(str(srtm_zip_file))
            zf.extractall(path=str(srtm_zip_file.parent))

            srtm_zip_file.unlink()

        hgt_files.append(str(srtm_hgt_file))

    return hgt_files

class WebAbstract(ABC):

    @abstractmethod
    def download_index(self):
        raise NotImplementedError

    @abstractmethod
    def read_index(self,
                   satellite='landsat',
                   start_date=None,
                   end_date=None,
                   sensors=None,
                   bounds=None,
                   collection='01',
                   batch_size=1000,
                   num_cpus=1):

        raise NotImplementedError

    @abstractmethod
    def query_scenes(self,
                     dataframe,
                     satellite='landsat',
                     start_date=None,
                     end_date=None,
                     sensors=None,
                     bounds=None,
                     collection='01'):

        raise NotImplementedError


def _submit_download(com):

    # Attempt to get the stats of the file to check if it exists
    res = subprocess.run(com.replace('-q cp', 'stat').split(' ')[:3],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    if res.returncode == 0:        
        subprocess.call(com, shell=True)


def download_scene(url, out_dir, temp_dir=None, rename=None):

    if rename and isinstance(temp_dir, str):
        out_file = Path(temp_dir) / Path(url).name
    else:
        out_file = Path(out_dir) / Path(url).name

    if not out_file.is_file():

        if rename and isinstance(temp_dir, str):
            command = f'gsutil -q cp {url} {temp_dir}'
        else:
            command = f'gsutil -q cp {url} {out_dir}'

        _submit_download(command)

    if rename:

        for old_name, new_name in rename.items():

            if out_file.name == old_name:

                new_file = str(Path(out_dir) / new_name)
                shutil.copy(str(out_file), new_file)
                out_file.unlink()
                out_file = str(new_file)

    return str(out_file)


def _reader_generator(reader):

    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)


def _raw_newline_count_gzip(fname):

    """
    Source: https://stackoverflow.com/questions/48765610/reading-lines-from-gzipped-text-file-in-python-and-get-number-of-original-compre
    """

    with gzip.open(fname, 'rb') as f:

        f_gen = _reader_generator(f.read)
        total = sum(buf.count(b'\n') for buf in f_gen)

    return total


class GEEAuthenticate(object):

    def authenticate_gee(self, secret_key_file):

        from google.auth.transport.requests import AuthorizedSession
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(secret_key_file)
        scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])

        try:
            self.session = AuthorizedSession(scoped_credentials)
            logger.info("Google AuthorizedSession created successfully.")
        except Exception as e:
            logger.warning(f"Error creating AuthorizedSession: {e}")
    

class GEE(GEEAuthenticate):

    def __init__(self):

        self.gee_project = 'projects/earthengine-public'
        self.query_content = None

    def query_gee(self,
                  asset_id=None,
                  point=None,
                  bounds=None,
                  satellite=None,
                  start_date=None,
                  end_date=None,
                  cloud_thresh=90):

        """
        Queries a Google Earth Engine asset

        Args:
            asset_id (str): The Earth Engine collection to query.
            point (Optional[list]): Point coordinates to query.
            bounds (Optional[list | tuple]): A bounding box to query. Takes precedent over ``point``.
            satellite (str): The satellite to determine the cloud cover lookup. Choices are ['landsat', 'sentinel-2'].
            start_date (str): The start date.
            end_date (str): The end date.
            cloud_thresh (int): The cloud cover threshold (0-100). Images with cloud cover percentage less than
                ``cloud_thresh`` are returned in the query.
        """

        if satellite not in ['landsat', 'sentinel-2']:
            logger.warning(f'The satellite {satellite} is not supported.')

        if bounds:

            left, bottom, right, top = bounds

            # GeoJSON LineString of the bounding box
            geometry = str([[left, top],
                            [right, top],
                            [right, bottom],
                            [left, bottom],
                            [left, top]])

            region_codes = '{"type":"LineString", "coordinates":' + geometry + '}'

        else:

            geometry = str(point)
            region_codes = '{"type":"Point", "coordinates":' + geometry + '}'

        name = f'{self.gee_project}/assets/{asset_id}'

        if (satellite == 'sentinel-2') and (asset_id != 'COPERNICUS/S2_CLOUD_PROBABILITY'):
            cloud_var = 'CLOUDY_PIXEL_PERCENTAGE'
        elif satellite == 'landsat':
            cloud_var = 'CLOUD_COVER'
        else:
            cloud_var = None

        if cloud_var:

            url = 'https://earthengine.googleapis.com/v1alpha/{}:listImages?{}'.format(
                name, urllib.parse.urlencode({'startTime': f'{start_date}T00:00:00.000Z',
                                              'endTime': f'{end_date}T00:00:00.000Z',
                                              'region': region_codes,
                                              'filter': f'{cloud_var} < {cloud_thresh}'}))

        else:

            url = 'https://earthengine.googleapis.com/v1alpha/{}:listImages?{}'.format(
                name, urllib.parse.urlencode({'startTime': f'{start_date}T00:00:00.000Z',
                                              'endTime': f'{end_date}T00:00:00.000Z',
                                              'region': region_codes}))

        response = self.session.request('GET', url)
        
        if response.status_code == requests.codes.ok:  # or 200
            logger.info("API request successful.")
            
            self.query_content = response.content

        else:
            logger.warning(f"API request failed with status code: {response.status_code}")
        
        
    def stream_gee_stack(self,
                         asset_list,
                         ux=None,
                         uy=None,
                         width=None,
                         height=None,
                         res=None,
                         bands=None,
                         bounds=None,
                         return_as='numpy',
                         config=None):

        """
        Streams an image asset time stack from Google Earth Engine.

        Users should follow the Earth Engine quota limits found on https://developers.google.com/earth-engine/reference.

        Args:
            asset_dict (list): A list of Google Earth Engine image asset.
            ux (float): The upper left corner longitude of the image.
            uy (float): The upper left corner latitude of the image.
            width (int): The image width.
            height (int): The image height.
            res (float): The image cell resolution.
            bands (list): The image bands.
            bounds (list | tuple): The image bounds to return. Used to get the centroid, which is only needed if
                ``return_as='xarray'`.
            return_as (str): The data return type. Choices are ['numpy', 'xarray'].
            config (dict): Configuration settings.

        Returns:
            ``numpy.ndarray`` | ``xarray.DataArray``

}
        """
        
        res = xr.concat([self.stream_gee(asset_dict,
                                         ux=ux,
                                         uy=uy,
                                         width=width,
                                         height=height,
                                         res=res,
                                         bands=bands,
                                         bounds=bounds,
                                         return_as=return_as,
                                         config=config) for asset_dict in asset_list], dim='time')\
                        .transpose('time', 'band', 'y', 'x')

        attrs = res.attrs.copy()

        res.coords['time'] = [pd.Timestamp(np.datetime64(asset_dict['properties']['SENSING_TIME']))\
                                  .normalize()\
                                  .to_pydatetime()
                              for asset_dict in asset_list]

        return res.groupby('time').max().assign_attrs(**attrs)

    def stream_gee(self,
                   asset_dict,
                   asset_id_filter=None,
                   ux=None,
                   uy=None,
                   adjust_y=False,
                   crs=None,
                   width=None,
                   height=None,
                   res=None,
                   bands=None,
                   bounds=None,
                   return_as='numpy',
                   config=None,
                   orig_crs=None,
                   max_attempts=10,
                   delay_sec=5):

        """
        Streams an image asset from Google Earth Engine.

        Users should follow the Earth Engine quota limits found on https://developers.google.com/earth-engine/reference.

        Args:
            asset_dict (dict): A Google Earth Engine image asset.
            asset_id_filter (str): An asset id filter.
            ux (float): The upper left corner longitude of the image.
            uy (float): The upper left corner latitude of the image.
            adjust_y (bool): Whether to adjust projected y values. If ``True``, y values will be adjusted by
                10,000,000. This is intended to address Landsat scenes only.

                See https://www.usgs.gov/faqs/why-do-landsat-scenes-southern-hemisphere-display-negative-utm-values?qt-news_science_products=0#qt-news_science_products

            crs (str): A CRS in the correct projection.
            width (int): The image width.
            height (int): The image height.
            res (float): The image cell resolution.
            bands (list): The image bands.
            bounds (list | tuple): The image bounds to return. Used to get the centroid, which is only needed if
                ``return_as='xarray'`.
            return_as (str): The data return type. Choices are ['numpy', 'xarray'].
            config (dict): Configuration settings.
            orig_crs (str)
            max_attempts (int)
            delay_sec (int)
        """

        if asset_id_filter:

            if asset_id_filter not in asset_dict['id'].split('/')[-1]:
                return []

        name = f"{self.gee_project}/assets/{asset_dict['id']}"

        url = f'https://earthengine.googleapis.com/v1alpha/{name}:getPixels'

        if adjust_y:
            ## Transform the user bbox upper left from lat/lon to projected user coordinates
            ##     y values will be adjusted by 10,000,000. This is intended to address Landsat scenes only.
            ##     See https://www.usgs.gov/faqs/why-do-landsat-scenes-southern-hemisphere-display-negative-utm-values?qt-news_science_products=0#qt-news_science_products
            orig_grid_left, orig_grid_top = gw.lonlat_to_xy(ux, uy, crs)

            ## Adjust Landsat images in the Southern hemisphere
            if orig_crs.split(':')[1].startswith('326') and (uy < 0):
                orig_grid_top -= 10_000_000.0

        else:
            ## Transform the user bbox upper left from lat/lon to projected coordinates
            orig_grid_left, orig_grid_top = gw.lonlat_to_xy(ux, uy, orig_crs)

        out_array = []

        @retry(MemoryError, tries=max_attempts, delay=delay_sec)
        def stream_single_band(post_content):
            try:
                return np.load(io.BytesIO(post_content))
            except:
                raise MemoryError

        ## Stream each band to reduce size
        for bd in bands:

            body = json.dumps({'fileFormat': 'NPY',
                               'bandIds': [bd],
                               'grid': {'affineTransform': {'scaleX': res,
                                                            'scaleY': -res,
                                                            'translateX': orig_grid_left,
                                                            'translateY': orig_grid_top},
                                        'dimensions': {'width': width,
                                                       'height': height}}})

            pixels_response = self.session.post(url, body)

            band_array = stream_single_band(pixels_response.content)

            # Reshape
            band_names = band_array.dtype.names
            band_array = np.float64(band_array[band_names[0]])
            out_array.append(band_array)

        out_array = np.array(out_array)

        if return_as == 'numpy':
            return out_array
        else:
            # Get the CRS based on the lat/lon.
            if bounds:
                #logger.info(f'0:{centroid[0]}, 1:{centroid[1]}')
                bounds_geom = polygon_from_bounds(bounds)
                centroid = (bounds_geom.centroid.x, bounds_geom.centroid.y)
                crs = f'epsg:{latlon_to_utm(centroid[0], centroid[1])[-1]}'
                # grid_left, grid_top = gw.lonlat_to_xy(centroid[0], centroid[1], crs)

            else:
                #logger.info(f'0:{ux}, 1:{uy}')
                crs = f'epsg:{latlon_to_utm(ux, uy)[-1]}'

            if adjust_y:
                ## Adjust Landsat images in the Southern hemisphere (see above)
                if orig_crs.split(':')[1].startswith('326') and (uy < 0):
                    orig_grid_top -= 10_000_000.0

            attrs = {'orig_width': asset_dict['bands'][1]['grid']['dimensions']['width'],
                     'orig_height': asset_dict['bands'][1]['grid']['dimensions']['height'],
                     'orig_left': asset_dict['bands'][1]['grid']['affineTransform']['translateX'],
                     'orig_top': asset_dict['bands'][1]['grid']['affineTransform']['translateY'],
                     'orig_crs': asset_dict['bands'][1]['grid']['crsCode']}

            ## Tag the array with metadata
            return tag_array(bands,
                             crs,
                             orig_grid_left,
                             orig_grid_top,
                             config,
                             attrs,
                             'uint16',
                             out_array)

    def stream_to_file(self,
                       out_dir,
                       wg,
                       asset_id_sensor,
                       asset,
                       satellite,
                       bands,
                       mask_landsat,
                       out_res,
                       bounds,
                       proj_bounds,
                       proj_crs,
                       check_existing,
                       check_metadata,
                       force_redownload,
                       params):

        array_size_out = int(params['grid_size'] / out_res) + params['buffer']*2

        if satellite == 'sentinel-2':
            asset_df_info = wg.sat_index_df.query(f"ASSET_ID == '{asset['id'].split('/')[-1]}'")
        elif satellite == 'landsat':
            asset_df_info = wg.sat_index_df.query(f"ASSET_ID == '{asset['id']}'")
        else:
            asset_df_info = pd.DataFrame([1])

        if asset_df_info.empty:
            return True
        else:
            
            if satellite not in ['landsat', 'sentinel-2']:
                logger.warning('currently only supports landsat and sentinel-2. See eosvault for other options')
            else:
                # Needed to download metadata
                url_dict = self._prepare_scenes(asset_df_info, [], satellite)

                scene_id = list(url_dict.keys())[0]
                if (satellite == 'sentinel-2'):
                    platform = list(asset_df_info['PRODUCT_ID'])[0].split('_')[0]
                logger.info(f'keys:{url_dict.keys()}')

                if (satellite == 'sentinel-2') and (asset_id_sensor == 'S2_CLOUD_PROBABILITY'):
                    scene_stack = Path(out_dir) / f"{platform}_{scene_id.split('_')[1][1:]}_{scene_id.split('_')[3][:8]}_s2cloudless{FILE_EXTENSIONS[params['io']['file_format']]}"
                else:
                    scene_stack = Path(out_dir) / f"{scene_id}{FILE_EXTENSIONS[params['io']['file_format']]}"

            # Check if the file and metadata already exist
            if scene_stack.is_file():

                if force_redownload:
                    scene_stack.unlink()
                else:

                    meta_exists = False

                    if satellite == 'landsat':

                        if Path(str(scene_stack).replace(FILE_EXTENSIONS[params['io']['file_format']], '_MTL.txt')).is_file():
                            meta_exists = True

                    elif satellite == 'sentinel-2':

                        if Path(str(scene_stack).replace(FILE_EXTENSIONS[params['io']['file_format']], '_TL.xml')).is_file():
                            meta_exists = True

                    # Exit if both files exist
                    if meta_exists:

                        if check_existing:

                            if band_is_ok(f"netcdf:{str(scene_stack)}:{bands[-1]}", params['io']['n_chunks']):
                                return True

                        else:
                            return True

            else:
                ## if the image file doesn't exist, the metadata doesn't need to either. 
                ## We also don't care about files that don't exist when forcing a re-download.
                logger.info('skipping this for the moment -- TODO: put back')
                #if check_metadata or force_redownload:
                #    return True

            if (satellite in ['landsat', 'sentinel-2']) and (asset_id_sensor != 'S2_CLOUD_PROBABILITY'):

                # Download the metadata or metadata + angle file
                out_meta_files = [download_scene(url,
                                    str(out_dir),
                                    temp_dir=None,
                                    rename={'MTD_TL.xml': f"{scene_id}_TL{FILE_EXTENSIONS['sentinel-2_metadata']}"})
                                    for url in list(url_dict.values())[0]]

                # Check that the metadata files were downloaded
                all_meta_downloaded = all([Path(mfn).is_file() for mfn in out_meta_files])

                if not all_meta_downloaded:
                    return False

            band_pos = 1 if (satellite in ['landsat', 'sentinel-2']) and (asset_id_sensor != 'S2_CLOUD_PROBABILITY') else 0

            if 'crsCode' in asset['bands'][band_pos]['grid']:
                orig_crs = asset['bands'][band_pos]['grid']['crsCode']
            else:
                orig_crs = asset['bands'][band_pos]['grid']['crsWkt']

            bounds_geom = polygon_from_bounds(bounds)
            centroid = (bounds_geom.centroid.x, bounds_geom.centroid.y)
            logger.info(f'0:{bounds_geom.centroid.x}, 1:{bounds_geom.centroid.y}')
            crs = f'epsg:{latlon_to_utm(centroid[0], centroid[1])[-1]}'

            attrs = {'orig_width': asset['bands'][band_pos]['grid']['dimensions']['width'],
                    'orig_height': asset['bands'][band_pos]['grid']['dimensions']['height'],
                     'orig_left': asset['bands'][band_pos]['grid']['affineTransform']['translateX'],
                     'orig_top': asset['bands'][band_pos]['grid']['affineTransform']['translateY'],
                     'orig_crs': orig_crs}

            if satellite == 'landsat':

                resampling = 'nearest' if params['res']==30.0 else 'cubic'

                # Transform the user bbox upper left from lat/lon to projected user coordinates
                grid_left, grid_top = gw.lonlat_to_xy(bounds[0], bounds[3], orig_crs)

                # https://www.usgs.gov/faqs/why-do-landsat-scenes-southern-hemisphere-display-negative-utm-values?qt-news_science_products=0#qt-news_science_products
                if orig_crs.split(':')[1].startswith('326') and (bounds[3] < 0):
                    grid_top += 10_000_000.0

                sensor = GEE_TRANSLATIONS['landsat']['gcp'][asset_df_info.SENSOR_ID.values[0]]
                wavelengths = get_sensor_info(key='wavelength', sensor=sensor)
                stream_bands = [f"B{getattr(wavelengths, bd)}" for bd in bands]

                # Stream the array
                array_qa = self.stream_gee(asset,
                                           ux=bounds[0],
                                           uy=bounds[3],
                                           adjust_y=False,
                                           crs=crs,
                                           width=array_size_30m,
                                           height=array_size_30m,
                                           res=30,
                                           bands=['pixel_qa'],
                                           orig_crs=orig_crs)

                if not isinstance(array_qa, np.ndarray):
                    return False

                # Get the QA fill value
                qa_mask = get_qa_mask(array_qa, sensor='landsat')

                array_30m = self.stream_gee(asset,
                                            ux=bounds[0],
                                            uy=bounds[3],
                                            adjust_y=False,
                                            crs=crs,
                                            width=array_size_30m,
                                            height=array_size_30m,
                                            res=30,
                                            bands=stream_bands,
                                            orig_crs=orig_crs)

                if not isinstance(array_30m, np.ndarray):
                    return False

                if array_30m.max() == -9999:
                    return False

                # Check pixels missed by nodata
                array_30m = check_missed_nodata(array_30m, params['nodata'])

                if mask_landsat == 'y':

                    # Mask the data
                    array_30m = mask_data(array_30m, qa_mask, params['nodata'])

                # Tag the array with metadata
                data = tag_array(bands,
                                 array_size_out,
                                 crs,
                                 30.0,
                                 grid_left,
                                 grid_top,
                                 proj_crs.to_proj4(),
                                 proj_bounds,
                                 params,
                                 attrs,
                                 'uint16',
                                 resampling,
                                 array_30m)

            elif (satellite == 'sentinel-2') and (asset_id_sensor != 'S2_CLOUD_PROBABILITY'):

                resampling_10m = 'nearest' if params['res'] == 10.0 else 'cubic'
                resampling_20m = 'nearest' if params['res'] == 20.0 else 'cubic'

                # Transform the user bbox upper left from lat/lon to projected coordinates
                grid_left, grid_top = gw.lonlat_to_xy(bounds[0], bounds[3], orig_crs)

                wavelengths = get_sensor_info(key='wavelength', sensor='s2af')
                stream_bands_10m = [f"B{getattr(wavelengths, bd)}" for bd in bands if bd not in ['swir1', 'swir2']]

                # Skip back because of B8A
                stream_bands_20m = [f"B{getattr(wavelengths, bd)-1}" for bd in bands if bd in ['swir1', 'swir2']]

                # Stream the array
                array_10m = self.stream_gee(asset,
                                            ux=bounds[0],
                                            uy=bounds[3],
                                            crs=crs,
                                            width=array_size_out,
                                            height=array_size_out,
                                            res=10,
                                            bands=stream_bands_10m,
                                            orig_crs=orig_crs)

                if not isinstance(array_10m, np.ndarray):
                    return False

                # Check pixels missed by nodata
                array_10m = check_missed_nodata(array_10m, params['nodata'])

                array_20m = self.stream_gee(asset,
                                            ux=bounds[0],
                                            uy=bounds[3],
                                            crs=crs,
                                            width=array_size_20m,
                                            height=array_size_20m,
                                            res=20,
                                            bands=stream_bands_20m,
                                            orig_crs=orig_crs)

                # Resample the 20 m layer to match the 10 m
                # array_20m = resample(array_20m,
                #                      array_size_out,
                #                      array_size_out)

                if not isinstance(array_20m, np.ndarray):
                    return False

                # Check pixels missed by nodata
                array_20m = check_missed_nodata(array_20m, params['io']['nodata'])

                # Tag the array with metadata
                data_10m = tag_array(['blue', 'green', 'red', 'nir'],
                                     array_size_out,
                                     crs,
                                     10.0,
                                     grid_left,
                                     grid_top,
                                     proj_crs.to_proj4(),
                                     proj_bounds,
                                     params,
                                     attrs,
                                     'uint16',
                                     resampling_10m,
                                     array_10m)

                data_20m = tag_array(['swir1', 'swir2'],
                                     array_size_out,
                                     crs,
                                     20.0,
                                     grid_left,
                                     grid_top,
                                     proj_crs.to_proj4(),
                                     proj_bounds,
                                     params,
                                     attrs,
                                     'uint16',
                                     resampling_20m,
                                     array_20m)

                data = xr.concat((data_10m, data_20m), dim='band')\
                            .chunk(chunks={'band': 1,
                                           'y': data_10m.gw.row_chunks,
                                           'x': data_10m.gw.col_chunks})

                data.attrs['resampling'] = 'nearest (10 m);cubic (20 m)'

            elif (satellite == 'sentinel-2') and (asset_id_sensor == 'S2_CLOUD_PROBABILITY'):
                
                resampling = 'nearest' if params['res'] == 10.0 else 'cubic'

                # Transform the user bbox upper left from lat/lon to projected coordinates
                grid_left, grid_top = gw.lonlat_to_xy(bounds[0], bounds[3], orig_crs)

                stream_bands = ['probability']

                # Stream the array
                array = self.stream_gee(asset,
                                        ux=bounds[0],
                                        uy=bounds[3],
                                        crs=crs,
                                        width=array_size_out,
                                        height=array_size_out,
                                        res=10,
                                        bands=stream_bands,
                                        orig_crs=orig_crs)

                if not isinstance(array, np.ndarray):
                    return False

                if array.min() == 255:
                    return False

                params['io']['nodata'] = 255

                # Tag the array with metadata
                data = tag_array(['proba'],
                                 array_size_out,
                                 crs,
                                 10.0,
                                 grid_left,
                                 grid_top,
                                 proj_crs.to_proj4(),
                                 proj_bounds,
                                 params,
                                 attrs,
                                 'uint8',
                                 resampling,
                                 array)

            if scene_stack.is_file():
                scene_stack.unlink()

            # Save to file
            logger.info(f"saving{scene_stack}...")
            data.gw.save(str(scene_stack))

        return True


class WebGCP(WebAbstract):

    """
    Args:
        satellite (Optional[str]): Choices are ['landsat', 'sentinel-2'].
        verbose (Optional[int]): The level of verbosity. 
    """

    def __init__(self, out_dir, satellite='landsat', verbose=0):

        self.out_dir = out_dir
        self.satellite = satellite
        self.verbose = verbose
        #self.sat_index_gcp = f'gs://gcp-public-data-{satellite}/index.csv.gz'
        self.sat_index_gcp = f'https://storage.googleapis.com/gcp-public-data-{satellite}/index.csv.gz'
        self.sat_index = None
        self.sat_index_df = None

        self.out_sat_dir = Path(self.out_dir).absolute() / f'{self.satellite}_index'
        self.out_sat_dir.mkdir(parents=True, exist_ok=True)

    def download_index(self):

        if self.verbose > 0:
            logger.info(f"  Downloading the {self.satellite.title()} database to {self.out_sat_dir} ...")

        self.sat_index = self.out_sat_dir / f'index.csv.gz'

        if not self.sat_index.is_file():
            #subprocess.call(f"gsutil cp -r {self.sat_index_gcp} {str(self.out_sat_dir)}")
            with requests.get(self.sat_index_gcp, stream=True) as r:
                with open(self.sat_index, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=512):
                        f.write(chunk)
            logger.info(f"Successfully downloaded index")

    def read_index(self,
                   satellite='landsat',
                   start_date=None,
                   end_date=None,
                   sensors=None,
                   bounds=None,
                   collection='01',
                   batch_size=50000,
                   num_cpus=1):

        bounds_cache = self.out_sat_dir / f"bounds_{start_date}_{end_date}_{'_'.join([f'{float(coord):.06f}' for coord in list(map(str, bounds))])}.parquet.gzip"

        if bounds_cache.is_file():
            self.sat_index_df = pd.read_parquet(str(bounds_cache), engine='pyarrow')
        else:

            # To debug, use ``local_mode=True``
            # If Ray processes persist after shutdown, run:
            #     ps aux | grep ray::IDLE | grep -v grep | awk '{print $2}' | xargs kill -9
            # ray.init(local_mode=True)

            if self.verbose > 0:
                logger.info('  Getting index size ...')

            total = _raw_newline_count_gzip(self.sat_index)

            with open(self.sat_index, mode='rb') as gz_file:

                with gzip.open(gz_file, mode='rt') as file:

                    first_line = True
                    counter = 0
                    lines = []

                    if self.verbose > 0:
                        logger.info(f'  Querying the {satellite.title()} database ...')

                    with tqdm(total=total) as pbar:

                        for line in file:

                            if first_line:

                                header = line.replace('\n', '').split(',')
                                self.sat_index_df = pd.DataFrame(columns=header)
                                first_line = False

                            else:
                                lines.append(line.replace('\n', '').split(','))

                            counter += 1

                            if counter == batch_size:

                                lindex_df_ = pd.DataFrame(data=lines,
                                                          columns=header)

                                if satellite == 'landsat':
                                    # Convert the acquisition date to datetime objects
                                    lindex_df_['DATESTAMP'] = lindex_df_.apply(lambda x: datetime.strptime(x.DATE_ACQUIRED, '%Y-%m-%d'),
                                                                               axis=1)

                                elif satellite == 'sentinel-2':
                                    # Convert the acquisition date to datetime objects
                                    lindex_df_['DATESTAMP'] = lindex_df_.apply(lambda x: datetime.strptime(x.SENSING_TIME[:10], '%Y-%m-%d'),
                                                                               axis=1)

                                lindex_df_.index = lindex_df_.DATESTAMP.values

                                lindex_df_ = self.query_scenes(lindex_df_,
                                                               satellite=satellite,
                                                               start_date=start_date,
                                                               end_date=end_date,
                                                               sensors=sensors,
                                                               bounds=bounds,
                                                               collection='01')

                                self.sat_index_df = pd.concat((self.sat_index_df, lindex_df_), axis=0)

                                counter = 0
                                lines = []

                            if counter % 100 == 0:
                                pbar.update(100)

            self.sat_index_df.to_parquet(bounds_cache, compression='gzip')

    def query_scenes(self,
                     dataframe,
                     satellite='landsat',
                     start_date=None,
                     end_date=None,
                     sensors=None,
                     bounds=None,
                     collection='01'):

        if not sensors:
            sensors = ['TM', 'ETM', 'OLI_TIRS']

        user_geom = polygon_from_bounds(bounds)

        # Get the time slice
        dfs = dataframe.loc[start_date:end_date]

        # Get the intersecting scenes
        dfs = dfs.loc[dfs.apply(geom_intersects, args=(user_geom,), axis=1).values.flatten()]

        if satellite == 'landsat':
            # Get the requested sensors
            return dfs.query(f"(SENSOR_ID == {sensors}) & (COLLECTION_NUMBER == '{collection}')")

        elif satellite == 'sentinel-2':
            return dfs

    @staticmethod
    def shutdown():

        try:
            ray.shutdown()
        except ResourceWarning:
            pass