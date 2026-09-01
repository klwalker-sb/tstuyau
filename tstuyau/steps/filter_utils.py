from dataclasses import dataclass, field
import numpy as np
import rasterio as rio
import xarray as xr
from rasterio.windows import Window, from_bounds
from scipy.ndimage import maximum_filter
from ..handler import logger
from .lookup import LC_CATS, LC_CATS_Py0

@dataclass
class FilterTsArgs:
    ts_files: list
    ts_yrs: list
    params: dict
    LC_CATS: dict
    base_rasters: dict = field(default_factory=dict)
    count_cache: dict = field(default_factory=dict)

def is_region_filter_active(apply_to=1):
    return apply_to not in [0, 'NA', 'all', 'All']
    
def get_regional_filter(idx,samp_ras,filter_list,filter_file):
    '''
    return 1s for portion of raster over which to apply filter, with 0 for non relevant areas
    input is an array of values for relevant strata/regions within an array of filters defined by <filter_file>
    0,'NA' or 'all' in the index location will return an array of ones
    '''
    with rio.open(samp_ras) as src:
        sample_bounds = src.bounds
        meta = src.meta.copy() 
    apply_to = filter_list[idx]
    if is_region_filter_active(apply_to):
        logger.info(f'applying priority filter to region(s) {apply_to}')     
        if isinstance(apply_to, str) or isinstance(apply_to, int):
            apply_to = [apply_to]
        with rio.open(filter_file) as region_src:
            window = from_bounds(
                sample_bounds.left,
                sample_bounds.bottom,
                sample_bounds.right,
                sample_bounds.top,
                src.transform,
            )
            reg = region_src.read(1,window=window)

        reg_filt = np.where(np.isin(reg,apply_to),1,0)
    else:
        reg_filt = np.ones((src.count, src.height, src.width), dtype=src.dtypes[0])
        
    return reg_filt

def apply_condition_to_timeseries(ts, cond, fill_value, cat_idx, samp_img, params,
                                   region_key='stable_regions', region_file_key='stable_region_file'):
    """Wherever cond is True, replace ts with fill_value; elsewhere leave ts unchanged."""
    apply_to = params['refine'][region_key][cat_idx]
    if is_region_filter_active(apply_to):
        region_filt = get_regional_filter(cat_idx, samp_img, params['refine'][region_key], params['refine'][region_file_key])
        cond = cond & (region_filt == 1)
    return ts.where(~cond, fill_value)

def store_count(name, count_cache, compute_fn):
    '''
    disctionary to store the total count for each desired category in a timeseries, so they only need to be calculated the first time needed
    '''
    if name not in count_cache:
        count_cache[name] = compute_fn()
    return count_cache[name]

def get_most_frequent_cat_in_timeseries(cats, ts, cat_dict=LC_CATS):
        '''
        takes a time series array opened in geowombat (or xarray?) and returns
        the most frequent observation from a set of choices <cat> defined in <cat_dict>
        to add a new set of choices, just add a new entry to lookup.LC_CATS (for CELPy classification) 
        or a custom dictionary for other products
        '''
        logger.info(f'getting stable base for {cats}')
        similar_cat_array = xr.DataArray(cat_dict[cats], dims=["lc"], coords={"lc": cat_dict[cats]})
        cat_counts = (ts == similar_cat_array).sum(dim="time").astype('uint8')
        drop_dims = [d for d in cat_counts.dims if d != "lc" and cat_counts.sizes[d] == 1]
        cat_counts = cat_counts.squeeze(dim=drop_dims)
        max_cat = cat_counts.idxmax(dim="lc")
        ##  if all cat_counts are 0, need to explicitly return 0 or will return first value in cat array
        winning_cat = max_cat.where(cat_counts.max(dim="lc") > 0, 0)
     
        return winning_cat

def mark_forest_edges(ts_single, params):
    '''reclassifies mature forest on forest edge as disturbed forest  
       works on a single raster
    '''
    logger.info('retouching forest edge...')
    if params['project_ver'] == 'Py_0':
        LC_CATS = LC_CATS_Py0
    first_mat_val = LC_CATS['first_mature']
    open_forest_val =  LC_CATS['open_for'][0]
    mature_forest = LC_CATS['dense_for']

    forest_mask = xr.where(ts_single > int(first_mat_val), 1, 9)
    edge_values = maximum_filter(forest_mask.values, size=3)
    edge = xr.DataArray(edge_values, coords=forest_mask.coords, dims=forest_mask.dims)
    final = ts_single.where((~ts_single.isin(mature_forest)) | (edge == 1), open_forest_val)

    return final
