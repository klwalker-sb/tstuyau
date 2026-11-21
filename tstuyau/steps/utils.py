import csv
import random
from pathlib import Path
import string
from collections import OrderedDict, namedtuple
import cv2
from datetime import datetime
import geowombat as gw
import numpy as np
from affine import Affine
from geowombat.core import ndarray_to_xarray

from .constants import (FILENAME_DATE_END_INDEX, 
                        FILENAME_DATE_INDEX, 
                        FILENAME_DATE_INDEX_GEE,
                        FILENAME_DATE_START_INDEX, 
                        FILENAME_SENSOR_INDEX)
from .lookup import SENSORS, SENSOR_FAMILIES
from ..handler import logger

_NON_IMAGE_TOKENS = ('angles', 'sharp')

def get_cell_list_from_grid_param(grid_param):
    """
    accepts grid parameter in multiple forms and converts is to list for uniform use  
    """
    cells = []
    if isinstance(grid_param, list):
        cells = grid_param
    elif isinstance(grid_param, str) and grid_param.endswith('.csv'): 
        with open(grid_param, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append(row[0])
    elif isinstance(grid_param, (int, str)): # if runing individual cells as array via bash script
        cells.append(grid_param) 
    else:
        logger.warning("ERROR: Problem parsing input as cell list. Needs to be list, .csv, or single int or string")

    return cells

#######################################################################################################  
#### sensor parsing helpers    --all sensor info should be in SENSORS in lookup.py
#######################################################################################################
def expand_sensors(tags):
    '''Sensor tag(s) -> ordered list of concrete SENSORS keys.'''
    tags = [tags] if isinstance(tags, str) else list(tags)
    out = []
    for tag in tags:
        entry = SENSORS.get(tag)
        if entry is None:
            logger.warning(f'  Unrecognized sensor tag {tag!r}; not in SENSORS (lookup.py).')
            continue
        out.extend(entry.get('members', [entry['unq']]))
    return list(dict.fromkeys(out))

def get_sensors(tags):
    '''Families covered by tags: subset of ['landsat', 'sentinel-2'].'''
    kinds = {SENSORS[k]['sensor'] for k in expand_sensors(tags) if 'sensor' in SENSORS[k]}
    return [f for f in SENSOR_FAMILIES if f in kinds]

def get_sensor_family(kind):
    '''Family entry (dir, metafile, prefix, ...) for 'landsat' or 'sentinel-2'.'''
    return SENSORS[SENSOR_FAMILIES[kind]]

def get_sensor_prefixes(tags):
    return tuple({p for k in expand_sensors(tags) for p in SENSORS[k]['matchstr']})

def get_sensor_code_from_filename(path, pos=FILENAME_SENSOR_INDEX, split_by='_'):
    parts = Path(path).name.split(split_by)
    return parts[pos] if len(parts) > pos else ''

def get_family_from_name(name):
    ''''landsat' / 'sentinel-2' from a file name or scene id (looks at the first two tokens).'''
    tokens = Path(name).name.split('_')[:2]
    for fam, key in SENSOR_FAMILIES.items():
        if any(t.startswith(tuple(SENSORS[key]['matchstr'])) for t in tokens):
            return fam
    return None

def get_sensor_param_for_mask(params, key, sensor=None):
    '''params['masking'][key], or '<prefix>_<key>' for the sensor family if set; None if unset.'''
    over = params['masking']
    unset = lambda v: v is None or (isinstance(v, str) and v.strip() in ('', 'None'))
    val = over.get(f"{get_sensor_family(sensor)['prefix']}_{key}") if sensor else None
    if unset(val):
        val = over.get(key)
    return None if unset(val) else val

def get_full_numpix(params):
    '''{family: pixels in a full scene clipped to the buffered cell}, from params['grid_size'], 
    params['buffer'] and params['res'] (all m).'''
    buffer_m = float(params.get('buffer') or 0)
    side_px = (float(params['grid_size']) + 2 * buffer_m) / float(params['res'])
    n = int(round(side_px ** 2))
    return {fam: n for fam in get_sensors('All')}      # {'sentinel-2': n, 'landsat': n}
###################################################################################################
### image list and filter helpers
###################################################################################################
def filter_images_by_sensor(image_dict, tags):
    ''' filters images after they are collected into an ordered dict'''
    prefixes = get_sensor_prefixes(tags)
    return OrderedDict((k, v) for k, v in image_dict.items() if get_sensor_code_from_filename(k).startswith(prefixes))


def filter_images_by_date(image_dict, start, end):
    ''' filters images after they are collected into an ordered dict
    start/end are YYYY-MM-DD strings (or datetimes)'''
    naive = lambda d: (datetime.strptime(d, '%Y-%m-%d') if isinstance(d, str) else d).replace(tzinfo=None)
    start, end = naive(start), naive(end)
    return OrderedDict((k, v) for k, v in image_dict.items() if start <= v <= end)

def skip_match(skip_flag, suffix='nc'):
    '''
    when filtering images within a directory, can skip files ending in <skip_flag> (just before suffix).
       <skip_flag> is a single letter (or multiple, but will skip files ending in ANY of the letters)
    '''
    return f'*.{suffix}' if not skip_flag else f'*[!{skip_flag}].{suffix}'

def get_image_list_for_type(ipath, image_type, pattern='*.nc'):
    prefixes = get_sensor_prefixes(image_type)
    files = [f for f in Path(ipath).glob(f'L3?_{pattern}')
             if 'angles' not in f.name and 'sharp' not in f.name]
    return sorted(f for f in files if get_sensor_code_from_filename(f).startswith(prefixes))


def get_pending_gee_files(gee_dir, tags, pattern='*[!_s].nc'):
    '''{SENSORS key: [files]} for sensors in tags that still have unprocessed GEE downloads in gee_dir.
    Sensors without a GEEunq (e.g. S2cp) are skipped. The glob excludes names ending in 's' (angles, cloudless).'''
    pending = {}
    for key in expand_sensors(tags):
        unq = SENSORS[key].get('GEEunq')
        if unq is None:
            continue
        files = list(Path(gee_dir).glob(f'{unq}{pattern}'))
        if files:
            pending[key] = files
    return pending

def get_brdf_sort_kwargs(params):
    '''Filename-date kwargs for sort_images_by_date; GEE file names have the date at a different position.'''
    if params['dlMethod'] == 'GEE':
        date_pos, prepend_str = FILENAME_DATE_INDEX_GEE, 'netcdf:'
    else:
        date_pos, prepend_str = FILENAME_DATE_INDEX, ''
    return dict(date_start=FILENAME_DATE_START_INDEX, date_end=FILENAME_DATE_END_INDEX,
                date_pos=date_pos, prepend_str=prepend_str)

def sort_images_by_date(image_path, image_wildcard, date_pos=0, date_start=0, date_end=8, split_by= '_', 
                        date_format='%Y%m%d', file_list=None, prepend_str=None):

    """Copied from geowombat.sort_images_by_date
    filters images in a directory and collects into an ordered dict, sorted by date

    Args:
        image_path (Path): The image directory.
        image_wildcard (str): The image search wildcard.
        date_pos (int): The date starting position in the file name.
        date_start (int): The date starting position in the split.
        date_end (int): The date ending position in the split.
        split_by (Optional[str]): How to split the file name.
        date_format (Optional[str]): The date format for :func:`datetime.datetime.strptime`.
        file_list (Optional[list of Paths]): A file list of names to sort. Overrides ``image_path``.
        prepend_str (Optional[str]): A string to prepend to each filename.

    Returns:
        ``collections.OrderedDict``
    """

    if file_list:
        fl = file_list
    else:
        fl = list(image_path.glob(image_wildcard))

    if prepend_str:
        fl = [Path(f'{prepend_str}{str(fn)}') for fn in fl]

    dates = [
        datetime.strptime(
            fn.name.split(split_by)[date_pos][date_start:date_end], date_format
        )
        for fn in fl
    ]

    return OrderedDict(
        sorted(
            dict(zip([str(fn) for fn in fl], dates)).items(),
            key=lambda x_: x_[1],
        )
    )

def get_sorted_images(directory, search_str, sort_kwargs, names=None):
    '''
    returns list of files that match <search_str> inside <directory>, sorted by date. 
    If <names> (set of file names) is given, keeps only those files.
          this allows single files to be passed along to function.
    '''
    image_dict = sort_images_by_date(directory, search_str, **sort_kwargs)
    if names is not None:
        image_dict = {k: v for k, v in image_dict.items() if Path(k).name in names}
    return image_dict

def get_sorted_brdf_images(in_dir, tags, skip_flag, sort_kwargs, names=None):
    '''Date-sorted brdf netcdfs in in_dir for the sensors in tags, minus angle / sharpened files.'''
    image_dict = get_sorted_images(in_dir, skip_match(skip_flag), sort_kwargs, names=names)
    image_dict = OrderedDict((k, v) for k, v in image_dict.items()
                             if not any(t in Path(k).name for t in _NON_IMAGE_TOKENS))
    return filter_images_by_sensor(image_dict, tags)
 
#####################################################################################################################
                                           
def random_id(string_length):
    """
    Generates a random string of letters and digits
    """

    letters_digits = string.ascii_letters + string.digits

    return ''.join(random.choice(letters_digits) for i in range(string_length))

                                           
def remove_angle_files(image_names, time_names):

    proc_names = []
    proc_times = []

    for fn, fnt in zip(image_names, time_names):

        if 'angles' not in fn:

            proc_names.append(fn)
            proc_times.append(fnt)

    return proc_names, proc_times


def pad_array(data, pad, scale_factor=1.0):
    return np.float64(cv2.copyMakeBorder(np.float32(data), pad, pad, pad, pad, cv2.BORDER_REFLECT)) * scale_factor


def check_missed_nodata(array, nodataval):

    for axis in [0, 1]:

        layer_vars = []

        for i, layer in enumerate(array):

            # Column-wise variance
            layer_vars.append(layer.var(axis=axis))

        # Check for 0 variance in all bands
        zero_vars = np.array(layer_vars).max(axis=0)

        idx = np.where(zero_vars == 0)[0]

        if idx.shape[0] > 0:

            for i in range(0, array.shape[0]):

                if axis == 0:
                    array[i, :, idx] = nodataval
                else:
                    array[i, idx, :] = nodataval

    return array


def resample(array, height, width, nodataval=None, mask=None):

    if len(array.shape) == 2:

        if isinstance(mask, np.ndarray):
            array[(array < 0) | (array > 10000) | (mask == 1)] = nodataval

        return np.float64(cv2.resize(np.float32(array),
                                     (height, width),
                                     interpolation=cv2.INTER_CUBIC))

    else:

        out = np.zeros((array.shape[0], height, width), dtype='float64')

        for i, layer in enumerate(array):

            if isinstance(mask, np.ndarray):
                layer[(layer < 0) | (layer > 10000) | (mask == 1)] = nodataval

            out[i] = np.float64(cv2.resize(np.float32(layer),
                                           (height, width),
                                           interpolation=cv2.INTER_CUBIC))

        return out


def band_is_ok(band, chunks):

    try:

        with gw.open(band, chunks=chunks) as src:
            _res = src.gw.read(band=1, num_workers=1)

        return True

    except (gw.backends.rasterio.rasterio.errors.RasterioIOError, OSError) as e:
        logger.warning(f'ERROR: {e}')
        return False
    
def tag_array(bands,
              dst_dim,
              src_crs,
              src_res,
              src_left,
              src_top,
              dst_crs,
              proj_bounds,
              params,
              extra_attrs,
              dtype,
              resampling,
              *arrays):

    # Combine the data
    if len(arrays) > 1:
        array_stack = np.vstack(arrays)
    else:
        array_stack = arrays[0]

    array_stack[(array_stack < 0) | (array_stack > 10000)] = params['nodata']

    _nbands, nrows, ncols = array_stack.shape

    src_res = params['storage']['res'] if not isinstance(src_res, float) else src_res

    attrs = {'transform': tuple(Affine(src_res,
                                       0.0,
                                       src_left,
                                       0.0,
                                       -src_res,
                                       src_top))[:6],
             'crs': src_crs,
             'res': (src_res, src_res),
             'is_tiled': 1,
             'nodatavals': tuple([params['nodata']]*array_stack.shape[0]),
             'scales': tuple([1]*array_stack.shape[0]),
             'offsets': tuple([0]*array_stack.shape[0])}

    if extra_attrs:
        attrs.update(extra_attrs)

    x = np.arange(src_left + src_res / 2.0, src_left + src_res / 2.0 + (src_res * ncols), src_res)[:ncols]
    y = np.arange(src_top - src_res / 2.0, src_top - src_res / 2.0 - (src_res * nrows), -src_res)[:nrows]

    # Store as an xarray and return
    res = ndarray_to_xarray(None,
                            array_stack.astype(dtype),
                            bands,
                            row_chunks=params['io']['n_chunks'],
                            col_chunks=params['io']['n_chunks'],
                            y=y,
                            x=x,
                            attrs=attrs).gw.transform_crs(dst_crs=dst_crs,
                                                          dst_bounds=proj_bounds,
                                                          dst_width=dst_dim,
                                                          dst_height=dst_dim,
                                                          src_nodata=params['nodata'],
                                                          dst_nodata=params['nodata'],
                                                          resampling=resampling,
                                                          num_threads=1)

    res.attrs['crs'] = dst_crs

    return res

#################################################################################################################                             
### old masking functions #######################################################################################
                                           
class BandQA:

    def __init__(self, sensor='landsat', collection='1'):
        
        # Bit flags for Landsat Tier 1 surface reflectance from Google Earth Engine
        bit_flags = {'landsat': {'fill': 1 << 0,
                                 'clear': 1 << 1,
                                 'water': 1 << 2,
                                 'shadow': 1 << 3,
                                 'snow': 1 << 4,
                                 'cloud': 1 << 5},
                     'd09a1': {'cloud': 1 << 0,
                               'shadow': 1 << 2}}
        
        self.sensor_flags = bit_flags[sensor]

    def mask(self, qa, mask_items=None):

        """
        Masks a QA array

        Args:
            qa (2d array): QA bit array.
            mask_items (list): QA bit flags.

        Returns:
            ``ndarray``:
                0: clear
                1: mask
        """

        if not mask_items:
            mask_items = ['cloud']

        mask_array = np.zeros(qa.shape, dtype='uint8')

        for mitem in mask_items:

            flag_mask = np.bitwise_and(qa, self.sensor_flags[mitem])
            mask_array = mask_array | flag_mask

        return np.uint8(np.where(mask_array > 0, 1, 0))


def mask_data(array, mask, nodataval):

    if len(array.shape) == 2:

        if isinstance(mask, np.ndarray):
            array[(array < 0) | (array > 10000) | (mask == 1)] = nodataval

        return array

    else:

        out = np.zeros(array.shape, dtype='float64')

        for i, layer in enumerate(array):

            if isinstance(mask, np.ndarray):
                layer[(layer < 0) | (layer > 10000) | (mask == 1)] = nodataval

            out[i] = layer

        return out


def get_qa_mask(array, sensor=None, mask_items=None):
    
    if not mask_items:
        mask_items = ['cloud', 'shadow', 'fill']

    lqa = BandQA(sensor=sensor)

    return lqa.mask(np.uint16(np.squeeze(array)),
                    mask_items=mask_items)

