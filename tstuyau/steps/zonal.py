import json
import math
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import features

#from rasterio.features import shapes
#from rasterstats import zonal_stats
#from rasterio.merge import merge
from rasterio.windows import Window, from_bounds
import warnings

from ..handler import logger
from . import utils
from .check_sample import get_polygons_in_grid
from .date_utils import get_date_range
from .image_utils import get_num_classes_in_raster
from .lookup import get_lc_cats, CONT_STATS
from .mask_utils import apply_binary_mask
from .project import ProjectPaths


def clip_ras_to_poly(ras_in, polys, out_dir,prod_name):
        
    out_path = Path(out_dir) / prod_name
    out_path.mkdir(parents=True, exist_ok=True)
        
    with fiona.open(polys, "r") as poly_src:
        logger.debug(f'poly_src: {poly_src}')
        #poly_crs = poly_src.crs
        shapes = [feature["geometry"] for feature in poly_src]

    for i, shape in enumerate(shapes):
        with rio.open(ras_in) as src:
            out_image, out_transform = rio.mask.mask(src, [shape], crop=True)
            out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

        with rio.open(out_path / f"{i}.tif", "w", **out_meta) as dest:
            dest.write(out_image)

def subtract_rasters(rasyr1, rasyr2, bands, printmap=False, out_path=None):
    
    with rio.open(rasyr1) as src1, rio.open(rasyr2) as src2:
        profile = src1.profile

        for i in range(1, src1.count + 1):
            data1 = src1.read(i)
            data2  =src2.read(i)
            prod_name = bands[i-1]
                        
            out_data = data2 - data1

            if printmap:
                profile.update(count=1)
                out = Path(out_path) / f'{prod_name}.tif'
                with rio.open(out, 'w', **profile) as dst:
                    dst.write(out_data, indexes = 1)
                return out
        
            else:
                return out_data

def make_reclass_dict(csv_path, old_col, new_col):
    '''
    0 stays as 0
    '''
    reclass_df = pd.read_csv(csv_path)
    old_new_dict = dict(zip(reclass_df[old_col], reclass_df[new_col]))
    old_new_dict[0] = 0   
    return old_new_dict
    
def reclassify_raster(params):
    '''
    <masking:ancillary_ras> is the input raster to reclassify. 
    the reclassified raster will be stored in the same directory as the input with <masking:to_vals> appended to name
        unless params masking:mask_path> is set to specify an alternative path and name.
    reclass_LUT = dictionary with from:to value as key:value pair. OR csv with <from_vals> column and <to_vals> column 
    no data value should be classed as 0; 0 will also always be classed as 0 
    '''
    
    reclass_LUT = params['masking']['reclass_LUT']
    if isinstance(reclass_LUT, str):
        reclass_dict = make_reclass_dict(csv_path=reclass_LUT, old_col=params['masking']['from_vals'], new_col=params['masking']['to_vals'])
    elif isinstance(reclass_LUT, dict):
        reclass_dict = reclass_LUT
    logger.debug(f"reclass_dict = {reclass_dict}")

    raster_path = params['masking']['ancillary_ras']
    if params['masking']['mask_path']:
        new_name = params['masking']['mask_path']
    else:
        new_name = Path(raster_path).parent / f"{params['masking']['to_vals']}{Path(raster_path).suffix}"
    with rio.open(raster_path) as src:
        old_arr = src.read(1)
        out_meta = src.meta.copy()
        if len(np.unique(old_arr)) > 1: ## if there are any values other than 0, nodata 
            new_arr = np.vectorize(reclass_dict.get)(old_arr)
            logger.info(f"{raster_path!s} old raster vals: {np.unique(old_arr)}  new raster vals: {np.unique(new_arr)}")
            out_meta.update({'nodata': 0})
            with rio.open(new_name, 'w', **out_meta) as dst:
                dst.write(new_arr, indexes=1)
            logger.info(f"masked raster saved to: {new_name} ")


def summarize_raster_cat(ras_in,map_dict=None, map_product=None, sum_classes=None, project_ver=None, mask_val=None):
    '''
    returns dictionary with summary percentages for a categorical raster <ras>. Especially designed for classes of 
        interest to tstuyau projects focused on crops, trees, burned area or grass. 
    If raster uses native classification system (or earlier system: <project_ver='Py_0'>),
         use model in lookup.SCHEMATIC_MODS to specify classes to summarize and pass key name with <sum_classes>. 
    If <ras> is an external map, define classes in dict with structure map_dict[map_product]['classes'][x,y,z] (with
       x,y,z being values that belong to the given class and pass in with <map_dict>, <map_product>. 
    If <ras> is a mask or a single class sum is desired, use 'mask' and <map_product> to get the percent masked. 
        The mask value defaults to 0, but can be passed in as any value or list of values with <mask_val> 
    If none of the optional parameters are supplied or lookup values are not found, will return a summary 
         of percent coverage for numerical class values

    Output dict includes an entry: 'missing_values' to list any values in the map that were not counted by this function.
       If 'missing_values' is not None, need to add values to correct LC_CAT key in lookup.py if using native classification
       or add values to map_dict otherwise.
    '''
    
    class_dict = {}
    
    with rio.open(ras_in) as ras:
        data = ras.read()
        nodata = getattr(ras, 'nodata', None)

    if 'mask' in map_product or (sum_classes is not None and 'mask' in sum_classes):
        if not mask_val:
            mask_val = [0]
        elif isinstance(mask_val, int):
            mask_val = [mask_val]

        ## Does not count nodata as masked or unmasked unless nodata and mask_val are == (then nodata is included in % masked)
        if nodata is None or nodata in mask_val:  
            valid = np.ones(data.shape, dtype=bool)
        elif np.isnan(nodata):
            valid = ~np.isnan(data)
        else:
            valid = data != nodata
        masked_count = int(np.count_nonzero(np.isin(data, mask_val) & valid))
        unmasked_count = int(np.count_nonzero(valid)) - masked_count
        pix_count = unmasked_count + masked_count

        class_dict['per_unmasked'] = round((100 * float(unmasked_count) / pix_count),1)
        class_dict['per_masked'] = round((100 * float(masked_count) / pix_count),1)
        class_dict['numpix'] = pix_count
        
        return class_dict
        
    else:
        unq_classes = None
        pix_count = 0
        crop_tot = 0
        burn_tot = 0
        grass_tot = 0
        tree_tot = 0
        lc_cats = None
        crop_edge_val = None
        
        if sum_classes and sum_classes in SCHEMATIC_MODS:
            mod_info = get_schematic_mod(sum_classes)
            unq_classes = mod_info.get('labels')
            if unq_classes:
                lc_cats = get_lc_cats(project_ver)
                ## crop edge is in both homogeneous crop and no crop sets and is halved in crop calcs to reduce double counting 
                crop_edge_val = lc_cats['Mixed_Crop-edge']     
        if not lc_cats:
            if map_dict:
                if isinstance(map_dict, (str, Path)):
                    with open(map_dict, 'r+') as map_dict_in:
                        map_dict = json.load(map_dict_in)
                    if isinstance(map_dict[map_product]['classes'], dict):
                        unq_classes = list(map_dict[map_product]['classes'].keys())
            
            if unq_classes is None:  ## no dictionary was passed for this map
                ## get the number and percent of each unique value in raster -- in this case, we have no other info about the class
                num_classes, unq_classes = get_num_classes_in_raster(ras_in)

        covered_vals = set()   ## raster values that belong to at least one class lookup, to warn about any that do not
        if any('crop' in item.lower() for item in unq_classes):
            edge_count = int(np.count_nonzero(data == crop_edge_val)) if crop_edge_val is not None else 0
        
        logger.info(f'summarizing classes: {unq_classes}')   
        for c in unq_classes:
            if isinstance(c, (int, np.integer)) or str(c).isdigit():
                c_count = int(np.count_nonzero(data == int(c)))
                covered_vals.add(int(c))
            else:
                if lc_cats:
                    if c not in lc_cats:
                        logger.warning(f"no LC_CATS entry for class '{c}'; skipping it")
                        continue
                    val_set = np.atleast_1d(lc_cats[c])   ## this handles single vals and lists
                    covered_vals.update(val_set.tolist())
                    if c.lower() in ['crop', 'homogeneous crop', 'no crop', 'nocrop'] and crop_edge_val is not None:
                        covered_vals.add(crop_edge_val)
                        val_set = val_set[val_set != crop_edge_val]
                        c_count = int(np.count_nonzero(np.isin(data, val_set))) + int(edge_count/2)
                    else:
                        c_count = int(np.count_nonzero(np.isin(data, val_set)))

                else:
                    val_set = map_dict[map_product]['classes'][c]
                    covered_vals.update(np.atleast_1d(val_set).tolist())
                    c_count = int(np.count_nonzero(np.isin(data, val_set)))
                if not c.lower().startswith('no'):
                    if ('crop' in c.lower()) and ('noncrop' not in c.lower()):
                        crop_tot = crop_tot + c_count
                    elif 'burn' in c.lower():
                        burn_tot = burn_tot + c_count
                    elif 'grass' in c.lower():
                        grass_tot = grass_tot + c_count
                    elif ('tree' in c.lower() and 'crop' not in c.lower()) or ('forest' in c.lower()):
                        ## note: tree crops (food crops, not plantations) are counted in crops, not trees
                        tree_tot = tree_tot + c_count
            
            class_dict[f'{c}']={'count': c_count}
            pix_count = pix_count + c_count

        final_class_dict = {}
        
        ## warn about raster values that no class lookup accounts for (nodata is not counted as missing)
        vals, counts = np.unique(data, return_counts=True)
        missing_vals = {int(v): int(n) for v, n in zip(vals, counts)
                        if np.isfinite(v) and int(v) not in covered_vals and (nodata is None or v != nodata)}
        if missing_vals:
            n_missing = sum(missing_vals.values())
            logger.warning(f'WARNING: {len(missing_vals)} value(s) in {ras_in} are not in any class lookup used for this summary '
                           f'(value: pixel count): {missing_vals}. These {n_missing} pixels ({round(100 * n_missing / data.size, 1)}% of the raster) '
                           f'are left out of the class percentages.')
            final_class_dict['missing_vals'] = list(missing_vals.values())
        else: final_class_dict['missing_vals'] = 'None'

        if pix_count == 0:
            logger.warning(f'ERROR: no pixels found in any of the classes for {ras_in}')
            return
        
        for c in unq_classes:
            final_class_dict[f'per_{c}'] = round((100 * class_dict[f'{c}']['count'] / pix_count),1)
        if crop_tot > 0:
            final_class_dict['per_crop'] = round(crop_tot/pix_count,1)
        if burn_tot > 0:
            final_class_dict['per_burned'] = round(burn_tot/pix_count,1)
        if grass_tot > 0:
            final_class_dict['per_grass'] = round(grass_tot/pix_count,1)    
        if tree_tot > 0:
            final_class_dict['per_tree'] = round(tree_tot/pix_count,1)
    
        final_class_dict['numpix'] = pix_count

    return final_class_dict


def summarize_raster_cont(ras_in, prod_name=None, aggstats=None, decimals=None):
    '''
    provides summary statistics for raster <ras_in> representing continuous measure. 
    statistics to summarize are passed in with <aggstats>. options are: 'avg','med','std','q75','q90','q25','q10',or'All'(default)
    <prod_name> is name of map product for labeling within dictionary.
    Only the raster's nodata value (if it has one) is treated as missing; zero is a real value unless it is the nodata value.
    '''
    entry = {}
    if prod_name is None:
        prod_name = 'map'
        logger.warning("no prod_name supplied. Using generic name 'map' for dictionary entries.")

    if aggstats is None or aggstats == 'None':
        aggstats = ['All']
    elif isinstance(aggstats, str):
        aggstats = [aggstats]
    unknown = [s for s in aggstats if s != 'All' and s not in _CONT_STATS]
    if unknown:
        logger.warning(f'ignoring unknown summary statistics {unknown}; options are {list(_CONT_STATS)} or All')

    with rio.open(ras_in) as ras:
        data = ras.read().astype(float)
        nodata = ras.nodata
    if nodata is None:
        logger.debug(f'{ras_in} has no nodata value set, so all pixel values are used in the summary')
    elif not np.isnan(nodata):   ## (a NaN nodata is already missing once the data is float)
        data[data == nodata] = np.nan

    ## Calculate statistics using NumPy functions (all-NaN rasters give 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for stat, func in _CONT_STATS.items():
            if 'All' in aggstats or stat in aggstats:
                value = func(data)
                if np.isnan(value):
                    value = 0
                elif decimals is None:
                    value = int(value)                  ## truncate (legacy behavior)
                elif decimals == 0:
                    value = int(round(float(value)))
                else:
                    value = round(float(value), decimals)
                entry[f'{prod_name}_{stat}'] = value

    return entry

def summarize_zones_cat(ras_in, polys, clip_dir, out_dir, prod_name, sum_classes=None, classes_out=None, print_out=False,
                        map_dict=None, project_ver=None):
    '''
    Summarizes categorical raster <ras_in> for each polygon in <polys>. Returns the polygon attribute table with one column per
    summary value (e.g. per_Crop, numpix). <classes_out> limits which are kept (list of output names, with or without the 'per_' prefix;
    default 'All'). <map_dict> can be the loaded dictionary or a path to it. <sum_classes> is a SCHEMATIC_MODS name.
    '''
    if classes_out is None or classes_out == 'None':
        classes_out = 'All'
    elif isinstance(classes_out, str):
        classes_out = [classes_out]
    
    clip_ras_to_poly(ras_in, polys, clip_dir, prod_name)
    plys = gpd.read_file(polys)
    plys.drop(['geometry'],axis=1,inplace=True)
    new_cols = []
    for i, row in plys.iterrows():
        clipped_ras = Path(clip_dir) / prod_name / f'{i}.tif'
        if not clipped_ras.exists():
            logger.warning(f'no clipped raster for polygon {i} ({clipped_ras}); it may be outside the raster extent. Skipping.')
            continue
        per_classes = summarize_raster_cat(clipped_ras, map_dict=map_dict, map_product=prod_name,  
                                           sum_classes=sum_classes, project_ver=project_ver)
        for key, value in per_classes.items():
            if classes_out == 'All' or key in classes_out or key.replace('per_', '', 1) in classes_out:
                logger.debug(f'class={key},val={value}')
                if value > 0:
                    plys.loc[i, f'{key}'] = value
                    if key not in new_cols:
                        new_cols.append(key)
    ## polygons that did not have a class get 0 rather than NaN
    plys[new_cols] = plys[new_cols].fillna(0)
 
    logger.debug(plys)
    if print_out == True:
        out_path = Path(out_dir) / f"zone_summary_{prod_name}.csv"
        pd.DataFrame.to_csv(plys, out_path, sep=',', index=True)
    
    return plys

    
def summarize_poly_features(params, ras_in=None):
    ''' 
    provides summary output for each polygon, output to dictionary <'feature_model':'poly_feat_dict'>
    This is most useful if polygons themselves are the subject of the model (e.g. farmers' fields, RCT units, etc.)
    '''
    
    polyfeat_dict = params['feature_model']['poly_feat_dict']  ## eg. "../data/poly_stats.json"
    if polyfeat_dict:
        with open(polyfeat_dict, 'r+') as poly_dict:
            dict_in = json.load(poly_dict)
            logger.debug(f"dict_in: {dict_in}")
    else:
        dict_in = {}

    plys = gpd.read_file(params['feature_model']['poly_vector_path'])
    plys.drop(['geometry'],axis=1,inplace=True)
    clip_dir = Path(params['scratch_dir'])
    
    with rio.open(ras_in) as dst:
        profile = dst.profile 
        for i in range(1, dst.count + 1):
            band_data = dst.read(i)
            prod_name = params['feature_model']['si_vars'][i-1]
            if params['masking']['mask_path']:
                logger.info(f"applying mask: {params['masking']['mask_path']}")
                profile.update(count=1)
                tmpras = Path(params['scratch_dir']) / f"{prod_name}.tif"
                ras = apply_binary_mask(band_data, params['masking']['mask_path'], printmap=True, out_path=tmpras, **profile)
            else:
                ras = band_data
            logger.info(f'clipping raster band {i} to polys...')
            clip_ras_to_poly(ras, params['feature_model']['poly_vector_path'],clip_dir,prod_name)
    
            for i, row in plys.iterrows():
                poly_stats = summarize_raster_cont(Path(clip_dir) / prod_name / f'{i}.tif',prod_name, params['feature_model']['aggstats'])
                logger.debug(f"poly stats: {poly_stats}")
                dict_in.setdefault(str(i),{})
                dict_in[str(i)].update(poly_stats)
            logger.debug(f"dict_in: {dict_in}")

    with open(polyfeat_dict, "w") as outfile:
        json.dump(dict_in, outfile)
    
    return dict_in


def summarize_zones_cont(ras_in, polys, clip_dir, out_dir, prod_name, aggstats=None, print_out=False, decimals=None):
    '''
    Summarizes continuous raster <ras_in> for each polygon in <polys> (statistics chosen with <aggstats>, see summarize_raster_cont).
    Returns the polygon attribute table with one column per statistic (<prod_name>_avg, etc.), the continuous counterpart of
    summarize_zones_cat. This is for getting summary statistics of large maps for exploration, etc. 
          For making modeling features, use summarize_poly_features()
    '''
    clip_ras_to_poly(ras_in, polys, clip_dir, prod_name)
    plys = gpd.read_file(polys)
    plys.drop(['geometry'],axis=1,inplace=True)
    for i, row in plys.iterrows():
        clipped_ras = Path(clip_dir) / prod_name / f'{i}.tif'
        if not clipped_ras.exists():
            logger.warning(f'no clipped raster for polygon {i} ({clipped_ras}); it may be outside the raster extent. Skipping.')
            continue
        poly_stats = summarize_raster_cont(clipped_ras, prod_name, aggstats, decimals=decimals)
        logger.debug(f'poly {i} stats: {poly_stats}')
        for key, value in poly_stats.items():
            plys.loc[i, key] = value
 
    if print_out == True:
        out_path = Path(out_dir) / f"zone_summary_{prod_name}.csv"
        pd.DataFrame.to_csv(plys, out_path, sep=',', index=True)
 
    return plys


def _resolve_map_inputs(params, polys, map_dict, map_prod):
    '''
    Works out what is being summarized, from the arguments and params.
    Returns (polys or None, out_dir (Path, created), map_prod, loaded map dict or None, raster path)
    '''
    if polys is None:
        polys = params['explore']['zones']
    if isinstance(polys, str) and polys in ('None', ''):
        polys = None
 
    out_dir = params['explore']['out_dir']
    if out_dir is None or out_dir == 'None':
        out_dir = params['backup_path'].parents[1]/'OutputData'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    if map_prod is None:
        map_prod = params['explore']['product_name']
        
    if map_dict is None:
        map_dict = params['feature_model']['ancillary_var_dict']
    if isinstance(map_dict, dict):
        dict_in = map_dict
    elif map_dict is not None and map_dict != 'None':
        with open(map_dict, 'r') as map_dict_in:
            dict_in = json.load(map_dict_in)
    else:
        dict_in = None
 
    if dict_in is not None:
        if map_prod not in dict_in:
            logger.warning(f"ERROR -- map product '{map_prod}' is not in the map dictionary {map_dict}")
        ras_path = dict_in[map_prod]['loc']
    else:
        ras_path = f'{map_prod}.tif'   ##TODO: accomodate other raster types (.nc)
    ras_path = Path(ras_path)
        
    if len(ras_path.parts) == 1:   ## just a file name, so look for it in the product directory
        map_dir = params['explore']['product_dir']
        if map_dir is None or map_dir == 'None':
            map_dir = params['backup_path'].parents[1]/'ancillary'
        ras_path = Path(map_dir)/ras_path
        if not ras_path.is_file():
            logger.warning(f'ERROR -- cannot find file {ras_path}. Maybe set params <explore><product_dir>')
 
    return polys, out_dir, map_prod, dict_in, ras_path
 
 
def _get_map_type(map_type, params, dict_in, map_prod, sum_classes):
    '''
    Returns 'cat' or 'cont'. In order of priority: the <map_type> argument, params['explore']['map_type'], a 'type' entry for the
    product in the map dictionary, then a guess (a product with a 'classes' entry, or a schematic summary model, is categorical;
    anything else is continuous). Anything but an explicit setting is logged, since the guess can be wrong
    (e.g. a categorical map listed without classes).
    '''
    aliases = {'cat': 'cat', 'categorical': 'cat', 'class': 'cat', 'classes': 'cat',
               'cont': 'cont', 'continuous': 'cont'}
    entry = dict_in[map_prod] if dict_in is not None else {}
    for source, value in [('argument', map_type), ('params explore:map_type', params['explore']['map_type']),
                          ("map dictionary 'type'", entry.get('type'))]:
        if value is not None and value != 'None' and value != '':
            if str(value).lower() not in aliases:
                logger.warning(f"ERROR -- unknown map type '{value}' from {source}; set param explore:map_type to 'cat' or 'cont'")
            return aliases[str(value).lower()]
    guess = 'cat' if (entry.get('classes') or sum_classes) else 'cont'
    logger.info(f"no map type was specified for '{map_prod}'; guessing '{guess}'. Set <map_type> to be explicit.")
    return guess
 
 
def _summarize_cat(params, polys, out_dir, map_prod, dict_in, ras_path, sum_classes):
    ## The list of classes comes from <sum_classes> (schematic model) or the map dict, inside summarize_raster_cat.
    ## <print_cats> only limits which summary values are kept in the output.
    print_cats = params['explore']['print_cats']
    if isinstance(print_cats, str) and print_cats in ('None', ''):
        print_cats = None
    if polys:
        clip_dir = Path(params['scratch_dir'])/'polys'
        clip_dir.mkdir(parents=True, exist_ok=True)
        return summarize_zones_cat(ras_path, polys, clip_dir, out_dir, map_prod, sum_classes=sum_classes,
                                   classes_out=print_cats, print_out=True, map_dict=dict_in, project_ver=params['project_ver'])
    return summarize_raster_cat(ras_path, map_dict=dict_in, map_product=map_prod, 
                                sum_classes=sum_classes, project_ver=params['project_ver'], mask_val=None)
 
 
def _summarize_cont(params, polys, out_dir, map_prod, ras_path):
    decimals = params['explore']['decimals'] ##if None, will truncate to integer, legacy behavior for Py_0
    aggstats = params['explore']['aggstats']   ## list of statistics, or None for all
    if isinstance(aggstats, str) and aggstats in ('None', ''):
        aggstats=None
    
    if polys:
        clip_dir = Path(params['scratch_dir'])/'polys'
        clip_dir.mkdir(parents=True, exist_ok=True)
        return summarize_zones_cont(ras_path, polys, clip_dir, out_dir, map_prod, aggstats=aggstats, print_out=True)
    return summarize_raster_cont(ras_path, map_prod, aggstats)


def summarize_raster(params, polys=None, map_dict=None, map_prod=None, map_type=None):
    '''
    Single entry point for summarizing a map product, categorical ('cat') or continuous ('cont'). See _get_map_type for how the
    type is chosen. For large-scale (multi-cell) map summarization.
    If polys are included, will create a summary for each polygon zone (returned as a DataFrame and written to
    zone_summary_<map_prod>.csv in the output dir). Else summarizes the full raster (returned as a dictionary).
       (note: can use file with single polygon as masking extent) 
    '''
    sum_classes = params['schematic_model']['summary_mod']   ## SCHEMATIC_MODS name (categorical only); if None the classes come from the map dict
    if isinstance(sum_classes, str) and sum_classes in ('None', ''):
        sum_classes = None
 
    polys, out_dir, map_prod, dict_in, ras_path = _resolve_map_inputs(params, polys, map_dict, map_prod)
    map_type = _get_map_type(map_type, params, dict_in, map_prod, sum_classes)
    logger.info(f"summarizing '{map_prod}' ({ras_path}) as a {'categorical' if map_type == 'cat' else 'continuous'} map")
 
    if map_type == 'cat':
        return _summarize_cat(params, polys, out_dir, map_prod, dict_in, ras_path, sum_classes)
    return _summarize_cont(params, polys, out_dir, map_prod, ras_path)


def parse_avar(avar):
    '''splits an ancillary variable name like "<avar0>-<stat>_<extra>" into (avar0, stat)'''
    return avar.split('-')[0], avar.split('-')[1].split('_')[0]
    

def _zonal_stat_gdf(polys, var_path, stat, categorical=False):
    '''
    Calculates <stat> of raster <var_path> within each polygon and joins it to the polygons as column <stat>.
    The index is reset first: zonal_stats returns a plain list, and the join matches on index labels, so a polygon table with
    a non-sequential index (e.g. a filtered subset) would get its statistics attached to the wrong polygons, or none.
    '''
    from rasterstats import zonal_stats
    polys = polys.reset_index(drop=True)
    stats_df = pd.DataFrame(zonal_stats(vectors=polys['geometry'], raster=var_path, stats=[stat], categorical=categorical))
    return polys.join(stats_df, how='left')


def get_ts_stats_within_polys(params, in_path=None, out_path=None):
    
    from rasterstats import zonal_stats

    poly_buf = params['refine']['buffer']
    if not out_path:
        out_dir = params['feature_model']['poly_var_path']
    else:
        out_dir = out_path
    tmp_out_dir= Path(params['scratch_dir']) /'tmp_poly_rasts'
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    ## saving raster grids with polygon features, using standard gridded procedures

    if params['feature_model']['ancillary_vars']:
        avar = params['feature_model']['ancillary_vars'][0]
        avar0, stat = parse_avar(avar)
        if in_path:
            var_path = in_path
            var_col = 'Value'
        else:
            var_dict = params['feature_model']['ancillary_var_dict']
            with open(var_dict, 'r+') as sfd:
                dic = json.load(sfd)
            if avar0 in dic: 
                var_path = dic[avar0]['path']
                var_col = dic[avar0]['col']
            else: logger.warning(f'no entry for {avar0} in dict at: {var_dict}')
        if var_path is None:
            logger.warning("ERROR -- need to supply variable path in ancillary var dict or directly as argument <in_path>")
        logger.info(f'working on {avar0}...')
        
    elif params['feature_model']['spec_indices']:
        si = params['feature_model']['spec_indices'][0]
        if '-' in si:
            si = si.split('-')[0]
        logger.info(f'working on {si}...')
        siv = params['feature_model']['si_vars'][0]
        season = siv.split('-')[1]
        stat = siv.split('-')[2]
        yrs = params['sample_model']['train_yrs'] ## should be single year here, but make sure:
        year = int(yrs[0]) if isinstance(yrs, (list, tuple)) else int(yrs)
        if isinstance(yrs,list):
            logger.warning(f"WARNING: <sample_model:train_yrs> was {yrs} -- this function takes just one year. clipped to: {year}")

        ## Check variable validity before running cells:
        ts_stat = siv.split('-')[0]
        ts_stat_ok = ts_stat in ('avg', 'cv', 'var')
        if ts_stat.startswith('per'):
            try:
                float(ts_stat.split('per')[1])
                ts_stat_ok = True
            except ValueError:
                pass
        if not ts_stat_ok:
            logger.warning(f"do not have a method for {siv} -- only have 'avg','cv','var' and 'perX' (e.g. 'per75'). No features were made.")
            return 
        
        use_dates = get_date_range(year,season,params,return_type='doy',padded=False)

    cells = utils.get_cell_list_from_grid_param(params['grids'])
    for cell in cells:
        logger.info(f'working on cell {cell}...\n')
        ppaths = ProjectPaths(params, grid=cell)
        grid_file = gpd.read_file(params['grid_file'])
        #snapped_bounds = image_to_snapped_bounds(cell, grid_file, buffer=params['buffer'], res=params['res'], width=2021, height=2021)
        out_tmp = None
        cell_var_path = None
        
        poly_path = params['feature_model']['poly_vector_path']
        if Path(poly_path).is_file(): 
            #polys_all = gpd.read_file(poly_path)
            polys = get_polygons_in_grid(grid_file, cell, poly_path, oldest=None, newest=None, obs_col=None)
        elif Path(poly_path).is_dir():
            cell_polys = list(Path(poly_path).glob(f'*{cell:04d}*.gpkg'))
            if not cell_polys:
                logger.warning(f'ERROR -- no polygon file for cell {cell} in {poly_path}; skipping this cell')
                continue
            polys = gpd.read_file(cell_polys[0])             
        else:
            logger.warning(f'not sure how to parse polys {poly_path}')
            return

        if params['feature_model']['ancillary_vars']:
            if poly_buf > 0:
                suffix = f'buf{poly_buf}'
            else:
                suffix = ''
            if out_path:
                out_file = out_path
            else:
                out_file = Path(out_dir)/f'{cell:06d}/{cell:06d}_{avar}_{suffix}.tif'
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)

            cell_var_path = var_path
            if (str(cell_var_path).startswith('relative')) or (f'{cell:04d}' in Path(cell_var_path).stem):
                ## if using classified outputs in the comp directory as ancillary inputs, the path in the dictionary should be:
                ##      "relative_<global_file_name> with relative in the place of the cell number at the beginning of the file name
                if Path(cell_var_path).is_file():
                    pass
                elif 'relative' in str(cell_var_path):
                    prepath = ppaths.ms.parent/'comp'/f'{cell:06d}'
                    #prepath = ppaths.comp/f'{cell:06d}'
                    cell_var_path = str(cell_var_path).replace('relative',str(prepath))
                
            logger.info(f'getting {avar0} at: {cell_var_path!s} \n')
            with rio.open(var_path) as src0:
                out_meta = src0.meta.copy()
                ''' if using image_to_snapped_bounds():
                raw_window = rio.windows.from_bounds(*snapped_bounds, transform=src0.transform)
                window = raw_window.round_lengths()
                out_shape = (int(window.height), int(window.width))
                new_gt = rio.windows.transform(window, src0.transform)
                '''
                new_gt = src0.transform
                #offset = img_to_bbox_offsets(gt, cell, grid_file, buffer=100, res=10.0)
                #new_gt = rio.Affine(gt[0], gt[1], (gt[2] + (offset[0] * gt[0])), 0.0, gt[4], (gt[5] + (offset[1] * gt[4])))
                out_shape = (src0.height, src0.width)
            out_meta.update({"count": 1, "height": out_shape[0], "width": out_shape[1], "transform": new_gt, "compress": "lzw", "tiled": True})

        elif params['feature_model']['spec_indices']:   ## calculating stats from time-series variables
            ## the following is only for smoothed indices. TODO: add in raw
            ts_dir = ppaths.ts / si
            logger.debug(f'looking in {ts_dir}')
            all_imgs = sorted(ts_dir.glob('*.tif'))
            rasts = sorted([r for r in all_imgs if int(r.stem) > use_dates[0] and int(r.stem) < use_dates[1]])
            logger.info(f'there are {len(rasts)} rasts between {use_dates[0]} and {use_dates[1]}')
            if len(rasts) == 0:  
                logger.warning(f'ERROR -- no images between {use_dates[0]} and {use_dates[1]} for cell {cell}; skipping this cell')
                continue

            if (params['project_ver'] == 'Py_0') and (siv == 'avg-NovDec-std'):
                out_file =  Path(out_dir) / f'AvgNovDec_FieldStd_{cell}.tif'
            else:
                out_file =  Path(out_dir) / f"Poly{siv.split('-')[2]}-{siv.split('-')[0]}{siv.split('-')[1]}_{cell:04d}.tif"
                    
            if polys.shape[0] == 0:
                logger.debug('there are no ploygon features in this cell')
                with rio.open(rasts[0]) as src:
                    out_meta = src.meta.copy()
                    out_meta.update(count=1, dtype=np.int16, compress="lzw", tiled=True)
                    samp_ras = src.read(1)
                    blank_ras = samp_ras*0
                with rio.open(out_file, 'w+', **out_meta) as dst:
                    dst.write_band(1, blank_ras)
                if params['segment']['make_blank_vars']:
                    if params['project_ver'] =='Py_0':
                        ## Make other blank filler files  This is a hacky fix for an old CELPy issue:
                        out_fn2 = Path(out_dir)/f"pred_APR_{cell}.tif"
                        if not out_fn2.exists():
                            with rio.open(str(out_fn2), 'w+', **out_meta) as dst:
                                dst.write_band(1, blank_ras)
                        out_fn3 = Path(out_dir)/f"pred_area_{cell}.tif"
                        if not out_fn3.exists():   
                            with rio.open(str(out_fn3), 'w+', **out_meta) as dst:
                                dst.write_band(1, blank_ras)
                        out_fn4 = Path(out_dir)/f"pred_APrEf_{cell}.tif"
                        if not out_fn4.exists():
                            with rio.open(str(out_fn4), 'w+', **out_meta) as dst:
                                dst.write_band(1, blank_ras)
                continue
                
            else:
                ## First calculate temporal stat for all images in indicated time period
                stack = []
                with rio.open(rasts[0]) as src0:
                    out_meta = src0.meta.copy()
                    new_gt = out_meta['transform']
                    out_shape = (out_meta['height'], out_meta['width'])
                logger.debug(f'out meta for ts features is: {out_meta}')
                out_meta.update(count=1, dtype=np.int16, compress="lzw", tiled=True)
                
                for rast in rasts:
                    with rio.open(rast) as src: 
                        arr = src.read(1)
                        stack.append(arr)
                            
                logger.info(f"getting {siv.split('-')[0]} for all images in period")
                if siv.split('-')[0] == 'avg':
                    arr = np.nanmean(stack, axis=0)
                elif siv.split('-')[0] == 'var':
                    arr = np.nanvar(stack, axis=0)
                elif siv.split('-')[0] == 'cv':
                    if params['project_ver'] in ['Py_0', 'Py_1']:
                        ## variance was accidentally calculates as 'cv' for CELPy methods
                        arr = np.nanvar(stack, axis=0)
                    else:
                        ## note cv is multiplied by 100 to accomodate int16 outupt 
                        mean_arr = np.nanmean(stack, axis=0)
                        arr = np.divide(np.nanstd(stack, axis=0) * 100, np.abs(mean_arr), out=np.zeros_like(mean_arr), where=(mean_arr != 0))

                elif siv.split('-')[0].startswith('per'):
                    num = float(siv.split('-')[0].split('per')[1])
                    arr = np.nanpercentile(stack, num, axis=0)
                else:
                    ## we chack this before now, so this should not be reached
                    logger.warning(f"OOPS -- do not have a method for {siv} -- only have 'avg','cv', 'var' and 'perX'")
                out_shape = arr.shape
                ## save intermediate mean raster 
                out_tmp = Path(tmp_out_dir) / f"{siv.split('-')[1]}{siv.split('-')[0]}_{cell:04d}.tif"
                with rio.open(out_tmp , "w", **out_meta) as dst:
                    dst.write(arr)
                cell_var_path = out_tmp

        ## within each polygon, calculate spatial stat for temporal stat ras
        logger.info(f'getting {stat} for all pixels in polygon...\n')
        if poly_buf > 0:
            polys["geometry"] = polys.buffer(poly_buf)
        if polys.shape[0] == 0:
            logger.info('there are no ploygon features in this cell')
            with rio.open(var_path, 'r') as src:
                ras_temp = src.read(1)
                blank_ras = ras_temp*0
            with rio.open(out_file, 'w+', **out_meta) as dst:
                dst.write_band(1, blank_ras)
        else:
            gdf = _zonal_stat_gdf(polys, cell_var_path, stat)

            out_shape_2d = out_shape[-2:]
            gdf[stat] = gdf[stat].fillna(0).astype(int)
            shapes = ((geom,value) for geom, value in zip(gdf.geometry, gdf[stat]))
            image = features.rasterize(shapes, out_shape=out_shape_2d, transform=new_gt, dtype=out_meta['dtype'])

            logger.info(f'final meta for poly stats is: {out_meta}')
            with rio.open(out_file, 'w+', **out_meta) as dst:
                dst.write_band(1, image)
            
            logger.info(f'wrote final file to: {out_file}')
                            
        ## delete intermediate mean raster
        out_tmp.unlink(missing_ok=True)
                            

def make_polygon_features(params, in_path=None, out_path=None):
    from rasterstats import zonal_stats
    '''
    If <feature_model:unit_of_analysis> == 'polygon', Provides summary output for each polygon, in dictionary <'feature_model':'poly_feat_dict'>

    If <feature_model:unit_of_analysis> == 'pixel', outputs rasters to be used for wall-to-wall classification, 
        using standard gridded procedures
    '''

    polys = params['feature_model']['poly_vector_path'] ## path to polygons
    uoa = params['feature_model']['unit_of_analysis']
    
    if uoa.lower().startswith('poly'):   ## making dictionary of polygon features
        #polyfeat_dict = params['feature_model']['poly_feat_dict']  ## eg. "../data/poly_stats.json"
        premask = params['mask']['mask_path']  ## eg. "/home/downspout-cel/biltong/mosaics/grass_obs_mask.tif"
        diff_feats = params['feature_model']['diff_feats']
        
        if premask: 
            mask_prefix = Path(premask).stem.split('_')[0]
        else:
            mask_prefix = ''

        if params['feature_model']['spec_indices']:  ## using ts data
            sis = params['feature_model']['spec_indices']   ## eg. ['kndvi', 'wi', 'ndmi']
            if isinstance(sis,str):
                sis = [sis]
            si_vars = params['feature_model']['si_vars']  ## eg. ['minv-wet', 'maxv-wet', 'minv-dry'] or ['avg-wet', 'cv-wet', 'cv-dry']
            yrs = params['sample_model']['train_yrs'] ## eg. [2020,2024]
            try:
                for idx in sis:
                    for yr in range(int(yrs[0]), int(yrs[-1]) + 1):
                        if (params['feature_model']['premade_composite'] is not False) and (params['feature_model']['premade_composite'] != 'False'):
                            ras_path = Path(params['backup_path'])/'mosaics'
                            ras_prefix = params['sample_model']['focus_area'] ##eg. 'cells_P1'
                            ras_in = Path(ras_path) / f"{ras_prefix}_{yr}_{idx}_{si_vars[0]}-{si_vars[1]}-{si_vars[2]}.tif"
                            bands = [f'{yr}_{idx}_{mask_prefix}_{si_vars[0]}',f'{yr}_{idx}_{mask_prefix}_{si_vars[1]}',f'{yr}_{idx}_{mask_prefix}_{si_vars[2]}']
                        else:
                            logger.info('TODO: finish this to make new composite')
                            continue
                
                        summarize_poly_features(params, ras_in)

                    if diff_feats and (yr > int(yrs[0])) and (yr <= int(yrs[-1])):
                            yr1 = int(yr)
                            yr0 = int(yr) - 1
                            yrstr = str(yr0)[2:] +'-'+ str(yr1)[2:]
                            logger.info(f"working on diff ras for {yrstr}...:")
                            bands = [f"delta{yrstr}_{idx}_{mask_prefix}_{si_vars[0]}",f"delta{yrstr}_{idx}_{mask_prefix}_{si_vars[1]}",
                                     f"delta{yrstr}_{idx}_{mask_prefix}_{si_vars[2]}"]
                            params['feature_model']['si_vars'] = bands
                            ras0 = Path(ras_path) / f"{ras_prefix}_{yr0}_{idx}_{si_vars[0]}-{si_vars[1]}-{si_vars[2]}.tif"
                            ras1 = Path(ras_path) / f"{ras_prefix}_{yr1}_{idx}_{si_vars[0]}-{si_vars[1]}-{si_vars[2]}.tif"
                            deltaras = subtract_rasters(ras0, ras1, bands, printmap=True, out_path=params['scratch_dir'])
                            summarize_poly_features(params, deltaras)
            finally:
                params['feature_model']['si_vars'] = si_vars
        elif params['feature_model']['ancillary_var_dict']:
            pass
            ##TODO consolidate summary methods
    
    else:  ## making raster outputs with polygon features
        ## If polys is a single file and no grid cells are specified, aassumes raster is already full extent and full-scale processing is desired. 
        ##    If grid cells are specified or polys is a directory, uses gridded structure instead (method below)
        if (not params['grids']) and (Path(polys).is_file()):
            polys = gpd.read_file(polys) 
            logger.debug(f' poly file looks like: {polys.head()} \n')
            ## single raster to pull data from. Either anscillary map or mosaicked classification outputs. ## TODO: add option to mosaic outputs VRT
            ## Need to add to <ancillary_var_dict> first
            if params['feature_model']['ancillary_vars']:
                orig_avars  = params['feature_model']['ancillary_vars']
                avars = orig_avars
                if isinstance(avars, str):
                    avars = [avars]
                try:
                    for avar in avars:
                        params['feature_model']['ancillary_vars'] = [avar]
                        avar0, stat = parse_avar(avar)
                        if in_path:
                            var_path = in_path
                            var_col = 'Value'
                        else:
                            var_dict = params['feature_model']['ancillary_var_dict']
                            with open(var_dict, 'r+') as sfd:
                                dic = json.load(sfd)
                            if avar0 in dic: 
                                var_path = dic[avar0]['path']
                                var_col = dic[avar0]['col']
                            else:
                                logger.warning(f'{avar0} is not in ancillary variable dict {var_dict}. Can supply in_path directly. \n')
                            if var_path is None:
                                logger.warning('ERROR -- ned to define variable parh')
                                continue
                            if not out_path: 
                                out_path = params['feature_model']['poly_var_path']
                            out_file = Path(out_path) / f'{avar}.tif'

                            logger.info(f'getting {avar0} at: {var_path} \n')
                            with rio.Env(GTIFF_SRS_SOURCE="EPSG"), rio.open(var_path) as src0:
                                ## note: this only works if var_path is already clipped to the grid cell. otherwise need one of the image_utils methods.
                                gt = src0.transform
                                out_shape=(src0.height, src0.width)
                                out_meta = src0.meta.copy()
                                out_meta.update(count=1, dtype=np.int16, compress="lzw", tiled=True)    
                            ## within each polygon, calculate spatial stat for ras
                            gdf = _zonal_stat_gdf(polys, var_path, stat, categorical=(stat == 'majority'))
                            gdf[stat] = gdf[stat].fillna(0)
                        
                            ## rasterize polygon using stat value
                            shapes = ((geom,value) for geom, value in zip(gdf.geometry, gdf[stat]))
                            if len(out_shape) == 3:
                                out_shape=out_shape[1:] 
                            image = features.rasterize( ((g, v) for g, v in shapes), out_shape=out_shape, transform=gt, fill=0, dtype=np.int16)
                            with rio.open(out_file, 'w+', **out_meta) as dst:
                                dst.write_band(1, image)
                            logger.debug(f'out_fn={out_file}')
                finally:
                    params['feature_model']['ancillary_vars'] = orig_avars
                    
            else: ## using ts data   NOTE -- this doesn't currently work without gridded structure (below).
                get_ts_stats_within_polys(params, in_path=in_path, out_path=out_path)
                
        else:  ## use gridded structure
            logger.info(' getting ts stats in polys...')
            if params['feature_model']['ancillary_vars']:
                avars = params['feature_model']['ancillary_vars']
                if isinstance(avars, str):
                    avars = [avars]
                try:
                    for avar in avars:
                        logger.info(f'working on {avar}...')
                        params['feature_model']['ancillary_vars'] = [avar]
                        get_ts_stats_within_polys(params, in_path=in_path, out_path=out_path)
                finally:
                    params['feature_model']['ancillary_vars'] = orig_avars
            
            elif params['feature_model']['spec_indices']:
                sis = params['feature_model']['spec_indices']   ## eg. ['kndvi', 'wi', 'ndmi']
                si_vars = params['feature_model']['si_vars']
                if isinstance(sis,str):
                    sis = [sis]
                if isinstance(si_vars,str):
                    si_vars = [si_vars]
                for i, si in enumerate (sis):
                    params['feature_model']['spec_indices'] = [si]
                    for ii, siv in enumerate (si_vars):
                        params['feature_model']['si_vars'] = [siv]
                        if i > 0 or ii > 0:
                            params['segment']['make_blank_vars'] = False
                        get_ts_stats_within_polys(params, in_path=in_path, out_path=out_path)

                params['feature_model']['spec_indices'] = sis
                params['feature_model']['si_vars'] = si_vars
                

                        

        
