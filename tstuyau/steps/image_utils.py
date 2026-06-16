import shutil
import tempfile
import pathlib
from pathlib import Path
import geowombat as gw
from geowombat.core import sort_images_by_date
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio as rio
from skimage.exposure import rescale_intensity
from rasterio.coords import BoundingBox
from rasterio.windows import Window
from shapely.geometry import Polygon
from rasterio.mask import mask
import pyproj
from .utils import random_id
from ..handler import logger

_projections = {}

def img_to_bbox_offsets(gt, cell, grid_file, buffer=100, res=10.0):
    '''
    For aligning grids when rasterizing polygons. This is an older method (compared to image_to_snapped_bounds())),
    but gives cleaner results. The other is somethimes off by a pixel.
    '''
    if isinstance(grid_file, str):
        grid_file = gpd.read_file(grid_file)
    gridcell = grid_file[grid_file['UNQ'] == int(cell)]
    #buffer_geom = gridcell.buffer(params['buffer']+int(params['res']), cap_style='square',join_style='mitre')
    buffer_geom = gridcell.buffer(buffer+int(res), cap_style=3,join_style=2)
    grid_bound = buffer_geom.geometry.iloc[0]
    bounds = grid_bound.bounds  ## bounds returns (minx, miny, maxx, maxy)
    bbox = (float(bounds[0]), float(bounds[2]), float(bounds[1] ), float(bounds[3]))
    
    origin_x = gt[2]
    origin_y = gt[5]
    pixel_width = gt[0]
    pixel_height = gt[4]
    x1 = int(round((bbox[0] - origin_x) / pixel_width))
    x2 = int(round((bbox[1] - origin_x) / pixel_width))
    y1 = int(round((bbox[3] - origin_y) / pixel_height))
    y2 = int(round((bbox[2] - origin_y) / pixel_height))
    xsize = x2 - x1
    ysize = y2 - y1
    return [x1, y1, xsize, ysize]
    
def image_to_snapped_bounds(cell, grid_file, buffer=100, res=10.0, width=None, height=None):
    '''
    Gets grid bounds from bounding geography (cell bounds explicitly here) 
    sometimes need to enter width and height explicily (e.g 2021) to match to existing grids  
    '''
    if isinstance(grid_file, str):
        grid_file = gpd.read_file(grid_file)
    gridcell = grid_file[grid_file['UNQ'] == int(cell)]
    #buffer_geom = gridcell.buffer(params['buffer']+int(params['res']), cap_style='square',join_style='mitre')
    buffer_geom = gridcell.buffer(buffer+int(res), cap_style=3,join_style=2)
    grid_bound = buffer_geom.geometry.iloc[0]

    west, south, east, north = grid_bound.bounds  ## bounds returns (minx, miny, maxx, maxy)

    snapped_west = math.floor(west / res) * res - (0.5 * res)
    snapped_north = math.ceil(north / res) * res - (0.5 * res)

    if width and height:
        target_width = width
        target_height = height
    else:
        target_width = int(round((east - west) / res))
        target_height = int(round((north - south) / res))

    snapped_east = snapped_west + (target_width * res)
    snapped_south = snapped_north - (target_height * res)

    snapped_bounds = (snapped_west, snapped_south, snapped_east, snapped_north)
    
    return snapped_bounds


def rescale_band(img_in, maxval=255, profile=None, outpath=None):
    '''
    rescales single-band image to range 0-maxval and returns it as uint8
    '''

    if isinstance(img_in, str) or isinstance(img_in, pathlib.PurePath):
        with rio.open(img_in) as src:
            old_arr = src.read()
            profile = src.profile 
    else:
        old_arr = img_in
        profile = profile
        
    srcmin = np.amin(old_arr)
    srcmax = np.amax(old_arr)
    scaled_arr = (maxval * ((old_arr - srcmin) / (srcmax - srcmin))).astype('uint8')
    newmin = np.amin(scaled_arr)
    newmax = np.amax(scaled_arr)
    logger.info(f'data rescaled from {srcmin}-{srcmax} to {newmin}-{newmax}')
    logger.info(f'dtype changed from {old_arr.dtype} to {scaled_arr.dtype}')

    profile.update(dtype=scaled_arr.dtype)

    if outpath:
        out_file = Path(outpath)/'scaled255.tif'
        outpath.mkdir(parents=True, exist_ok=True)
        with rio.open(out_file, mode="w", **profile) as new_dataset:
            new_dataset.write(scaled_arr)

        return out_file

    else:
        return scaled_arr
        
def get_utm_zone(coordinates):

    """
    takes lon, lat coord and returns utm zone number
    ## Note int((lon+180)/6)+1, usually works, but there are some exceptions
    ##  TODO: check these exceptions for Africa vs. S. America
    """
    
    if 56 <= coordinates[1] < 64 and 3 <= coordinates[0] < 12:
        return 32

    elif 72 <= coordinates[1] < 84 and 0 <= coordinates[0] < 42:

        if coordinates[0] < 9:
            return 31
        elif coordinates[0] < 21:
            return 33
        elif coordinates[0] < 33:
            return 35
        return 37
        
    return int((coordinates[0] + 180.0) / 6.0) + 1

def get_utm_letter(coordinates):
    return 'CDEFGHJKLMNPQRSTUVWXX'[int((coordinates[1] + 80.0) / 8.0)]

def zone_to_epsg(lat, utm_zone):
    return int(f'326{utm_zone}') if lat > 0 else int(f'327{utm_zone}')
    
def latlon_to_utm(x, y):
    
    """
    takes lon,lat coordinate and finds full UTM projection.
    returns the zone number, zone letter, reprojected x coord, reprojected y coord, and epsg code
    """

    coordinates = (x, y)

    z = get_utm_zone(coordinates)
    l = get_utm_letter(coordinates)

    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')

    x, y = _projections[z](coordinates[0], coordinates[1])

    if y < 0:
        y += 10000000.0

    epsg = zone_to_epsg(coordinates[1], z)

    return z, l, x, y, epsg


def polygon_from_bounds(bounds):

    left, bottom, right, top = bounds

    return Polygon([(left, bottom),
                    (left, top),
                    (right, top),
                    (right, bottom),
                    (left, bottom)])
    
def get_grid_bounds(grid_file,
                    grid_size,
                    grid_cell,
                    buffer=0,
                    centroid_to_utm='n'):

    """
    Gets bounds for a grid cell <grid> within a larger grid (<grid_file>, input as .geojson or .gpkg file)
    Applies optional buffer to cell prior to calculating bounds -- NOTE: buffer units must be the same as grid_file crs 
    bounds are returned in lists [minx, miny, maxx, maxy]
    Returns list:
                [0] = the bounds (with optional buffer appplied) in latlon
                [1] = the bounds (with optional buffer applied) in the projection of the grid file
                [2] = the projection of the grid file
            
    if <centroid_to_utm> == 'y': returns projected products in appropriate UTM zone
    """
    
    ## TODO: use this temp file method for safer read
    #out_path = Path(grid_file).parent
    #with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
    #    temp_file = Path(temp_dir) / Path(grid_file).name
    #    shutil.copy(grid_file, temp_file)
    #    df = gpd.read_file(temp_file)
    df = gpd.read_file(grid_file)

    if grid_file.endswith('.geojson'):
        df['UNQ'] = grid_cell

    if centroid_to_utm == 'y':
        # Grid size, in meters
        grid_size = grid_size

        grid_size_half = int(grid_size / 2.0)

        # Get the centroid in lat/lon in order to get the UTM zone
        x = float(df.query(f'UNQ == {grid_cell}').to_crs('epsg:4326').geometry.centroid.x.values)
        y = float(df.query(f'UNQ == {grid_cell}').to_crs('epsg:4326').geometry.centroid.y.values)

        # Get the UTM zone as a CRS reference
        proj_crs = f'epsg:{latlon_to_utm(x, y)[-1]}'

        # Get the centroid in UTM coordinates
        cx = float(df.query(f'UNQ == {grid_cell}').to_crs(proj_crs).geometry.centroid.x.values)
        cy = float(df.query(f'UNQ == {grid_cell}').to_crs(proj_crs).geometry.centroid.y.values)

        left = cx - grid_size_half
        bottom = cy - grid_size_half
        right = cx + grid_size_half
        top = cy + grid_size_half

        # Get the UTM bounds
        proj_bounds = [left, bottom, right, top]

        # Transform the centroid to lat/lon
        def utm_to_latlon(xm, ym):
            return pyproj.Proj(proj_crs)(xm, ym, inverse=True)

        lleft = utm_to_latlon(left, bottom)
        uright = utm_to_latlon(right, top)

        # Get the lat/lon bounds
        bounds = [lleft[0],
                  lleft[1],
                  uright[0],
                  uright[1]]

        proj_crs = CRS.from_string(proj_crs)

    else:
        proj_crs = df.crs
        cell = df.query(f'UNQ == {grid_cell}')
        
        if buffer:
            proj_bounds = cell.buffer(buffer).bounds.values.flatten().tolist()
            cellbuf = polygon_from_bounds(proj_bounds)
            cellbuf2 = gpd.GeoDataFrame(index=[0], crs=proj_crs, geometry=[cellbuf])
            bounds = cellbuf2.to_crs('epsg:4326').bounds.values.flatten().tolist()
            
        else:
            proj_bounds = cell.bounds.values.flatten().tolist()
            bounds = cell.to_crs('epsg:4326').bounds.values.flatten().tolist()
    
    return bounds, proj_bounds, proj_crs

def clip_big_ras_to_small(small_ras, big_ras, big_clipped):
        
    '''## with gdal:
    src_small = gdal.Open(small_ras)
    ulx, xres, xskew, uly, yskew, yres  = src_small.GetGeoTransform()
    lrx = ulx + (src_small.RasterXSize * xres)
    lry = uly + (src_small.RasterYSize * yres)
    geometry = [[ulx,lry], [ulx,uly], [lrx,uly], [lrx,lry]]
    '''
    ## with geowombat:
    with gw.open(small_ras) as src:
        logger.debug(f'src: {src}')
    minx = src.x.min().item()
    maxx = src.x.max().item()
    miny = src.y.min().item()
    maxy = src.y.max().item()
    #bounds = src.bounds
    #ulx = bounds[0]
    #urx = bounds[2]
    #lry = bounds[1]
    #uly = bounds[3]
    geometry = [[maxx,miny], [maxx,maxy], [minx,maxy], [minx,miny]]
    roi = [Polygon(geometry)]
    with rio.open(small_ras) as src0:
        out_meta = src0.meta.copy()
        out_meta.update({"count":1})
    with rio.open(big_ras) as src:
        out_image, transformed = mask(src, roi, crop = True)
    with rio.open(big_clipped, 'w', **out_meta) as dst:
        dst.write(out_image)
                                
def geom_intersects(dfr, geom_b):

    bounds = (float(dfr.WEST_LON), float(dfr.SOUTH_LAT), float(dfr.EAST_LON), float(dfr.NORTH_LAT))

    geom_a = polygon_from_bounds(bounds)

    return geom_a.intersects(geom_b)

    
def reshape_array(data_src):

    """
    Renames and reshapes an array

    Args:
        data_src (DataArray): The data to reshape.

    Returns:
        ``xarray.DataArray`` shaped [time x bands x rows x columns]
    """

    return data_src.data.rename({'band': 'time'})\
                .expand_dims(dim='band')\
                .transpose('time', 'band', 'y', 'x')


def scale_stack(stack,
                vrt_files,
                params,
                method,
                selection_in,
                selection_out,
                image_index):

    """
    Opens a stack of image variables and scales to [0,1]

    Args:
        stack (object): A ``contextlib.ExitStack()`` context.
        vrt_files (list): A list of VRT composites (typically, one for each variable).
        params (dict): A dictionary of parameters.
        method (str): The step method.
        selection_in (str): The input bands.
        selection_out (str): The output bands.
        image_index (1d array-like): The time indices.

    Returns:
        ``xarray.DataArray`` shaped [time x bands x rows x columns]
    """

    src = xr.concat((stack.enter_context(reshape_array(gw.open(fn))) for fn in vrt_files), dim='band')\
                .assign_coords({'band': params[method][selection_in]})\
                .sel(time=image_index)\
                .sel(band=params[method][selection_out])

    attrs = src.attrs.copy()

    return (src.astype('float64') * 0.0001).assign_attrs(**attrs)


def clean_vrt_files(vrt_files):

    if vrt_files:
        for vrt_file in vrt_files:
            vrt_file.unlink()


def open_images(ppaths, params, method):

    """
    Opens all available variable VRT stacks

    Args:
        ppaths (object)
        params (dict)
        method (str)

    Returns:
        ``list``, ``list``, ``list``:
            vrt_files, image_names, time_names
    """

    if params['segment']['buffer'] > 0:

        out_path = Path(params['segment']['grid_file']).parent

        with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:

            temp_file = Path(temp_dir) / Path(params['segment']['grid_file']).name
            shutil.copy(params['segment']['grid_file'], temp_file)

            df_grids = gpd.read_file(temp_file)

        df = df_grids.query(f"UNQ == {ppaths.grid}")

        # Get all grids that intersect the center grid
        df_grids = df_grids.loc[[geom.intersects(df.buffer(params['segment']['buffer']).geometry.values[0])
                                 for geom in df_grids.geometry.values]]

        grid_list = df_grids.UNQ.values.tolist()

        read_bounds = df.buffer(params['segment']['buffer']).total_bounds.tolist()

    else:

        grid_list = [ppaths.grid]
        read_bounds = None

    vrt_files = []

    for image_var in params[method]['image_bands']:

        ts_dir = ppaths.ts / image_var

        image_dict = sort_images_by_date(ts_dir,
                                         '*.tif',
                                         date_pos=0,
                                         date_start=0,
                                         date_end=7,
                                         date_format='%Y%j')

        image_names = list(image_dict.keys())
        time_names = list(image_dict.values())

        out_vrt = ts_dir / f'stack_{random_id(9)}.vrt'

        if out_vrt.is_file():
            out_vrt.unlink()

        # Save the images as a VRT file
        with gw.open(image_names, persist_filenames=True) as src:

            src.attrs['crs'] = src.crs.replace('+init=', '')
            src.gw.to_vrt(str(out_vrt))

        vrt_files.append(out_vrt)

    return vrt_files, image_names, time_names


def scale_data(data, scale_factor=1.0):

    """
    Scales data from [0,10000] --> exp(data x 0.0001)[0,5]

    Args:
        data (ndarray)
        scale_factor (Optional[float])

    Returns:
        ``ndarray``
    """

    return np.clip(np.float64(rescale_intensity(np.exp(np.clip(np.array(data) * scale_factor, 0, 1)),
                                                in_range=(np.exp(0), np.exp(1)),
                                                out_range=(0, 5))), 0, 5)

def normalize(array):
    array_min, array_max = array.min(), array.max()
    if array_min < 0:
        array_min = 0
    return (array - array_min) / (array_max - array_min)
    
def gammacorr(band, gamma):
    return np.power(band, 1/gamma)
    
def get_rbg_img(image, gamma):
    if isinstance(image, str):
        image = Path(image)
    if image.suffix == '.tif':
        with rio.open(image) as src:
            red0 = src.read(3, masked=True)
            green0 = src.read(2, masked=True)
            blue0 = src.read(1, masked=True)
        red = np.ma.array(red0, mask=np.isnan(red0))
        red[red < 0] = 0
        green = np.ma.array(green0, mask=np.isnan(green0))
        green[green < 0] = 0
        blue = np.ma.array(blue0, mask=np.isnan(blue0))
        blue[blue < 0] = 0
        
    elif image.suffix == '.nc':
        with xr.open_dataset(image) as xrimg:
            red = xrimg['red'].where((xrimg['red'] > 0) & (xrimg['red'] < 10000))
            green = xrimg['green'].where((xrimg['green'] > 0) & (xrimg['green'] < 10000))
            blue = xrimg['blue'].where((xrimg['blue'] > 0) & (xrimg['blue'] < 10000))
            logger.debug(f'red: min={red.min()}, max={red.max()}')
   
    red_g=gammacorr(red, gamma)
    blue_g=gammacorr(blue, gamma)
    green_g=gammacorr(green, gamma)

    red_n = normalize(red_g)
    green_n = normalize(green_g)
    blue_n = normalize(blue_g)

    rgb = np.dstack([red_n, green_n, blue_n])
    
    return rgb
