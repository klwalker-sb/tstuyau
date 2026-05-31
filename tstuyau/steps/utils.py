import string
import random

import numpy as np
import cv2
import geowombat as gw
from geowombat.core import ndarray_to_xarray
from geowombat.radiometry import QAMasker
import xarray as xr
from affine import Affine

def random_id(string_length):

    """
    Generates a random string of letters and digits
    """

    letters_digits = string.ascii_letters + string.digits

    return ''.join(random.choice(letters_digits) for i in range(string_length))


def get_image_list(ipath, wildcard):
    return list(ipath.glob(wildcard))


def filter_list(image_list):

    image_list = [fn for fn in image_list if 'angles' not in fn.name]
    return [fn for fn in image_list if 'sharp' not in fn.name]


def get_s2_list(ipath, pattern='*.nc'):
    return filter_list(get_image_list(ipath, f'L3*_S2{pattern}'))


def get_l5_list(ipath, pattern='*.nc'):
    return filter_list(get_image_list(ipath, f'L3*_LT05{pattern}'))


def get_l7_list(ipath, pattern='*.nc'):
    return filter_list(get_image_list(ipath, f'L3*_LE07{pattern}'))


def get_l8_list(ipath, pattern='*.nc'):
    return filter_list(get_image_list(ipath, f'L3*_LC08{pattern}'))


def get_l9_list(ipath, pattern='*.nc'):
    return filter_list(get_image_list(ipath, f'L3*_LC09{pattern}'))


def get_image_lists(ms_brdf_path):

    s2_list = get_s2_list(ms_brdf_path)
    l5_list = get_l5_list(ms_brdf_path)
    l7_list = get_l7_list(ms_brdf_path)
    l8_list = get_l8_list(ms_brdf_path)

    landsat_list = l5_list + l7_list + l8_list

    return s2_list, landsat_list


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
            res = src.gw.read(band=1, num_workers=1)

        return True

    except:
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

    nbands, nrows, ncols = array_stack.shape

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

    class BandQA(object):

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

