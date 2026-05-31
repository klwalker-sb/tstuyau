import sys
import csv
import glob
import json
import math
from pathlib import Path
import datetime as dt
import rasterio as rio
#from rasterio.merge import merge
from rasterio.windows import Window
from rasterio import features
#from rasterio.features import shapes
import pandas as pd
import numpy as np
import geowombat as gw
import geopandas as gpd
import xarray as xr
import fiona
from shapely.geometry import Polygon
#from rasterstats import zonal_stats
from .project import ProjectPaths
from ..handler import logger
from .mask_utils import apply_binary_mask, combine_binary_masks
from .date_utils import get_date_range
from .check_sample import get_polygons_in_grid


def img_to_bbox_offsets(gt, bbox):
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
    
def clip_ras_to_poly(ras_in, polys, out_dir,prod_name):
        
    out_path = Path(out_dir) / prod_name
    out_path.mkdir(parents=True, exist_ok=True)
        
    with fiona.open(polys, "r") as poly_src:
        logger.debug(f'poly_src: {poly_src}')
        poly_crs = poly_src.crs
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
    
    with rio.open(rasyr1) as src1:
        with rio.open(rasyr2) as src2:
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
            logger.info(f"{str(raster_path)} old raster vals: {np.unique(old_arr)}  new raster vals: {np.unique(new_arr)}")
            out_meta.update({'nodata': 0})
            with rio.open(new_name, 'w', **out_meta) as dst:
                dst.write(new_arr, indexes=1)
            logger.info(f"masked raster saved to: {new_name} ")
            
def summarize_raster_cat(ras_in,map_dict,map_product):
    with rio.open(ras_in) as ras:
        data = ras.read()

    class_dict = {}
    other_tot = 0
    
    if '8Cat' in map_product:
        grass_class = 70 
        gs_count = np.count_nonzero(data == grass_class)
        pix_count = gs_count
        
        tree_class = 200
        ts_count = np.count_nonzero(data == tree_class)
        pix_count = pix_count + ts_count
        
        shrub_class = 160
        ss_count = np.count_nonzero(data == shrub_class)
        pix_count = pix_count + ss_count

        crop_class = 100
        cs_count = np.count_nonzero(data == crop_class)
        pix_count = pix_count + cs_count
        
        other_classes = [10,20,30,40,64]
        for oc in other_classes:
            oc_count = np.count_nonzero(data == oc)
            other_tot = other_tot + oc_count
            pix_count = pix_count + oc_count
        
        class_dict['per_grass'] = round((100 * float(gs_count) / pix_count),1)
        class_dict['per_tree'] = round((100 * float(ts_count) / pix_count),1)
        class_dict['per_shrub'] = round((100 * float(ss_count) / pix_count),1)
        class_dict['per_crop'] = round((100 * float(cs_count) / pix_count),1)
        class_dict['per_other'] = round((100 * float(other_tot) / pix_count),1)
        class_dict['numpix'] = pix_count

    elif 'mask' in map_product:
        unmasked_count = np.count_nonzero(data == 1)
        pix_count = unmasked_count
        masked_count = np.count_nonzero(data == 0)
        pix_count = pix_count + masked_count

        class_dict['per_unmasked'] = round((100 * float(unmasked_count) / pix_count),1)
        class_dict['per_masked'] = round((100 * float(masked_count) / pix_count),1) 
        
    return class_dict
    
def summarize_raster_cont(ras_in,prod_name, aggstats):

    entry = {}
    
    with rio.open(ras_in) as ras:
        data = ras.read()

        data = data.astype(float)
        if ras.nodata:
            data[data == ras.nodata] = np.nan
        data[data == 0] = np.nan
        
        ## Calculate statistics using NumPy functions
        if aggstats == 'All':
            aggstats = ['All']
        if 'All' in aggstats or 'avg' in aggstats:
            entry[f'{prod_name}_avg'] = 0 if math.isnan(np.nanmean(data)) else int(np.nanmean(data))
        if 'All' in aggstats or 'med' in aggstats:
            entry[f'{prod_name}_med'] = 0 if math.isnan(np.nanmedian(data)) else int(np.nanmedian(data))
        if 'All' in aggstats or 'std' in aggstats:
            entry[f'{prod_name}_std'] = 0 if math.isnan(np.nanstd(data)) else int(np.nanstd(data))
        if 'All' in aggstats or 'q75' in aggstats:
            entry[f'{prod_name}_q75'] = 0 if math.isnan(np.nanquantile(data, 0.75)) else int(np.nanquantile(data, 0.75))
        if 'All' in aggstats or 'q90' in aggstats:
            entry[f'{prod_name}_q90'] = 0 if math.isnan(np.nanquantile(data, 0.9)) else int(np.nanquantile(data, 0.9))
        if 'All' in aggstats or 'q25' in aggstats:
            entry[f'{prod_name}_q25'] = 0 if math.isnan(np.nanquantile(data, 0.75)) else int(np.nanquantile(data, 0.25))
        if 'All' in aggstats or 'q10' in aggstats:
            entry[f'{prod_name}_q10'] = 0 if math.isnan(np.nanquantile(data, 0.9)) else int(np.nanquantile(data, 0.1))
        
        return(entry)

def summarize_zones_cat(params, print_out=False, out_dir=None):

    polys = params['feature_model']['poly_vector_path']
    map_dict = params['feature_model']['ancillary_var_dict']
    clip_dir = params['scratch_dir']['polys']
    map_product = params['feature_model']['ancillary_vars']
    if isinstance(map_product,list):
        map_product = map_product[0]

    if not out_dir:
        ppaths=ProjectPaths(params)
        out_dir = ppaths.datasum
        out_dir.mkdir(parents=True, exist_ok=True)

    if map_dict:  
        with open(map_dict, 'r+') as map_dict_in:
            dict_in = json.load(map_dict_in)
        file_name = dict_in[map_product]['loc']
    else:
        file_name = f'{map_product}.tif' 
    ras_in = Path(map_dir) / file_name
    clip_ras_to_poly(ras_in, polys,clip_dir,map_product)
    plys = gpd.read_file(polys)
    plys.drop(['geometry'],axis=1,inplace=True)
    for i, row in plys.iterrows():
        per_classes = summarize_raster_cat(Path(clip_dir) / map_product / f'{i}.tif',map_dict,map_product)
        for key, value in per_classes.items():
            logger.debug(f'class={key},val={value}')
            if value > 0:
                plys.loc[i, f'{key}'] = value

    logger.debug(plys)
   
    if print_out == True:
        out_path = Path(out_dir) / f"zone_summary_{map_product}.csv"
        pd.DataFrame.to_csv(plys, out_path, sep=',', index=True)
    
    return plys

    
def summarize_zones_cont(params, ras_in=None):
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
            if params['maskING']['mask_path']:
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


def get_ts_stats_within_polys(params):
    from rasterstats import zonal_stats
    
    out_path = params['feature_model']['poly_var_path']
    tmp_out_dir= Path(params['scratch_dir']) /'tmp_poly_rasts'
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
        
    ## saving raster grids with polygon features, using standard gridded procedures
    cells = []
    if isinstance(params['grids'], list):
        cells = params['grids']
    elif isinstance(params['grids'], str) and params['grids'].endswith('.csv'): 
        with open(params['grids'], newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append(row[0])
    elif isinstance(params['grids'], int) or isinstance(params['grids'], str): # if runing individual cells as array via bash script
        cells.append(params['grids']) 
    
    for cell in cells:
        logger.info(f'working on cell {cell}...')
        ppaths = ProjectPaths(params, grid=cell)
        logger.debug(f'working on cell {cell}... \n')
        grid_file = gpd.read_file(params['grid_file'])
        gridcell = grid_file[grid_file['UNQ'] == int(cell)]
        #buffer_geom = gridcell.buffer(params['buffer']+int(params['res']), cap_style='square',join_style='mitre')
        buffer_geom = gridcell.buffer(params['buffer']+int(params['res']), cap_style=3,join_style=2)
        grid_bound = gridcell.buffer(params['buffer']+int(params['res']),cap_style=3,join_style=2).geometry.iloc[0]
        bounds = grid_bound.bounds ## bounds returns (minx, miny, maxx, maxy)
        boundary = (float(bounds[0]), float(bounds[2]), float(bounds[1] ), float(bounds[3]))

        poly_path = params['feature_model']['poly_vector_path']
        if Path(poly_path).is_file(): 
            #polys_all = gpd.read_file(polys)
            polys = get_polygons_in_grid(grid_file, cell, polys, oldest=None, newest=None, obs_col=None)
        else:
            polys = gpd.read_file([Path(poly_path)/i for i in list(Path(poly_path).glob(f'*{cell:04d}*.gpkg'))][0])

        sis = params['feature_model']['spec_indices']   ## eg. ['kndvi', 'wi', 'ndmi']
        if isinstance(sis,str):
            sis = [sis]
        for si in sis:
            if '-' in si:
                si = si.split('-')[0]
            logger.info(f'working on {si}...')
            ## the following is only for smoothed indices. TODO: add in raw
            ts_dir = ppaths.ts / si
            logger.debug(f'looking in {ts_dir}')
            all_imgs = sorted(list(ts_dir.glob('*.tif')))
            
            si_vars = params['feature_model']['si_vars']
            for siv in si_vars:
                season = siv.split('-')[1]
                stat = siv.split('-')[2]
                year = int(params['sample_model']['train_yrs']) ## should be single year here
                use_dates = get_date_range(year,season,params,return_type='doy',padded=False)
                rasts = sorted([r for r in all_imgs if int(r.stem) > use_dates[0] and int(r.stem) < use_dates[1]])
                logger.info(f'there are {len(rasts)} rasts between {use_dates[0]} and {use_dates[1]}')
                if (params['project_ver'] == 'Py_0') and (siv == 'avg-NovDec-std'):
                    out_file =  Path(out_path) / f'AvgNovDec_FieldStd_{cell}.tif'
                else:
                    out_file =  Path(out_path) / f"Poly{siv.split('-')[2]}-{siv.split('-')[0]}{siv.split('-')[1]}_{cell:04d}.tif"
                    
                if polys.shape[0] == 0:
                    logger.debug('there are no ploygon features in this cell')
                    with rio.open( ts_dir / rasts[0]) as src:
                        out_meta = src.meta.copy()
                        out_meta.update({"count": 1, "dtype":np.int16})
                        samp_ras = src.read(1)
                        blank_ras = samp_ras*0
                    with rio.open(out_file, 'w+', **out_meta) as dst:
                        dst.write_band(1, blank_ras)
                    if params['segment']['make_blank_vars']:
                        ## Make other blank filler files:
                        out_fn2 = Path(out_path)/f"pred_APR_{cell}.tif"
                        if not out_fn2.exists:
                            with rio.open(str(out_fn2), 'w+', **out_meta) as dst:
                                dst.write_band(1, blank_ras)
                        out_fn3 = Path(out_path)/f"pred_area_{cell}.tif"
                        if not out_fn3.exists:   
                            with rio.open(str(out_fn3), 'w+', **out_meta) as dst:
                                dst.write_band(1, blank_ras)
                        out_fn4 = Path(out_path)/f"pred_APrEf_{cell}.tif"
                        if not out_fn4.exists:
                            with rio.open(str(out_fn4), 'w+', **out_meta) as dst:
                                dst.write_band(1, blank_ras)
                        
                else:
                    ## First calculate temporal stat for all images in indicated time period
                    stack = []
                    with rio.open(rasts[0]) as src0:
                        out_meta = src0.meta.copy()
                        out_meta.update({"count": 1, "dtype":np.int16})
                    for rast in rasts:
                        with rio.open(rast) as src:
                            gt = src.transform
                            offset = img_to_bbox_offsets(gt, boundary)
                            new_gt = rio.Affine(gt[0], gt[1], (gt[2] + (offset[0] * gt[0])), 0.0, gt[4], (gt[5] + (offset[1] * gt[4])))
                            arr = src.read(window=Window(offset[0], offset[1], offset[2], offset[3]))      
                            stack.append(arr)
                            
                    logger.info(f"getting {siv.split('-')[0]} for all images in period")
                    if siv.split('-')[0] == 'avg':
                        arr = np.nanmean(stack, axis=0)
                    elif siv.split('-')[0] == 'cv':
                        arr = np.nanvar(stack, axis=0)
                    elif siv.split('-')[0].startswith('per'):
                        num = siv.split('-')[0].split('per')[1]
                        arr = np.nanpercentile(stack, num, axis=0)
                    else:
                        logger.warning(f"OOPS -- do not have a method for {siv} -- only have 'avg','cv',and 'perX'")
                    out_shape = arr.shape
                    ## save intermediate mean raster 
                    out_tmp = Path(tmp_out_dir) / f"{siv.split('-')[1]}{siv.split('-')[0]}_{cell:04d}.tif"
                    with rio.open(out_tmp , "w", **out_meta) as dst:
                        dst.write(arr)

                    ## within each polygon, calculate spatial stat for temporal stat ras
                    logger.info(f'getting {stat} for all pixels in polygon...\n')
                    gdf = polys.join(pd.DataFrame(zonal_stats(
                        vectors=polys['geometry'], raster=out_tmp, stats=[stat])), how='left' )
                    ## save raster 
                    with rio.open(out_file, 'w+', **out_meta) as dst:
                        tmp_arr = dst.read(1)
                        ## rasterize polygon using stat value
                        shapes = ((geom,value) for geom, value in zip(gdf.geometry, gdf[stat]))
                        image = features.rasterize( ((g, v) for g, v in shapes), out_shape=out_shape[1:], transform=new_gt)
                        dst.write_band(1, image)
                    logger.info(f'wrote final file to: {out_file}')
                            
                    ## delete intermediate mean raster
                    out_tmp.unlink()
                            

def make_polygon_features(params):
    from rasterstats import zonal_stats
    '''
    If <feature_model:unit_of_analysis> == 'polygon', Provides summary output for each polygon, in dictionary <'feature_model':'poly_feat_dict'>

    If <feature_model:unit_of_analysis> == 'pixel', outputs rasters to be used for wall-to-wall classification, 
        using standard gridded procedures
    '''

    sis = params['feature_model']['spec_indices']   ## eg. ['kndvi', 'wi', 'ndmi']
    if isinstance(sis,str):
        sis = [sis]
    si_vars = params['feature_model']['si_vars']  ## eg. ['minv-wet', 'maxv-wet', 'minv-dry'] or ['avg-wet', 'cv-wet', 'cv-dry']
    yrs = params['sample_model']['train_yrs'] ## eg. [2020,2024]
    polys = params['feature_model']['poly_vector_path'] ## path to polygons

    uoa = params['feature_model']['unit_of_analysis']
    
    if uoa.lower().startswith('poly'):
        ## making dictionary of polygon features
        polyfeat_dict = params['feature_model']['poly_feat_dict']  ## eg. "../data/poly_stats.json"
        premask = params['mask']['mask_path']  ## eg. "/home/downspout-cel/biltong/mosaics/grass_obs_mask.tif"
        diff_feats = params['feature_model']['diff_feats']
        
        if premask: 
            mask_prefix = Path(premask).stem.split('_')[0]
        else:
            mask_prefix = {}
    
        for idx in sis:
            for yr in range(int(yrs[0]), int(yrs[-1]) + 1):
                if (params['feature_model']['premade_composite'] is not False) and (params['feature_model']['premade_composite'] != 'False'):
                    ras_path = Path(params['backup_path'])/'mosaics'
                    ras_prefix = params['sample_model']['focus_area'] ##eg. 'cells_P1'
                    ras_in = Path(ras_path) / f"{ras_prefix}_{yr}_{idx}_{si_vars[0]}-{si_vars[1]}-{si_vars[2]}.tif"
                    bands = [f'{yr}_{idx}_{mask_prefix}_{si_vars[0]}',f'{yr}_{idx}_{mask_prefix}_{si_vars[1]}',f'{yr}_{idx}_{mask_prefix}_{si_vars[2]}']
                else:
                    logger.info('finish this to make new composite')
                
                summarize_zones_cont(params, ras_in)

                if diff_feats:
                    if yr > yrs[0] and yr < yrs[-1]:
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
                        summarize_zones_cont(params, deltaras)

    else:
        ## if polys is a single file, aassumes raster is already full extent. If polys is a directory, uses gridded structure (method below)
        out_path = params['feature_model']['poly_var_path']
        if Path(polys).is_file(): 
            polys = gpd.read_file(polys) 
            bounds = polys.bounds ## bounds returns (minx, miny, maxx, maxy)
            boundary = (float(bounds[0]), float(bounds[2]), float(bounds[1] ), float(bounds[3]))
            ## single raster to pull data from. Either anscillary map or mosaicked classification outputs. ## TODO: add option to mosaic outputs VRT
            ## Need to add to <ancillary_var_dict> first
            if params['feature_model']['ancillary_vars']:
                var_dict = params['feature_model']['ancillary_var_dict']
                for avar in params['feature_model']['ancillary_vars']:
                    stat = avar.split('-')[1]
                    out_file = Path(out_path) / f'{avar}.tif'
                    with open(var_dict, 'r+') as sfd:
                        dic = json.load(sfd)
                        if avar in dic: 
                            var_path = dic[avar]['path']
                            var_col = dic[avar]['col']
                            logger.info(f'getting {avar} from {var_path} \n')
                            with rio.open(var_path) as src0:
                                gt = src0.transform
                                offset = img_to_bbox_offsets(gt, boundary)
                                out_shape=src0.shape
                                new_gt = rio.Affine(gt[0], gt[1], (gt[2] + (offset[0] * gt[0])), 0.0, gt[4], (gt[5] + (offset[1] * gt[4])))
                                #arr = src.read(window=Window(offset[0], offset[1], offset[2], offset[3]))  
                                out_meta = src0.meta.copy()
                                out_meta.update({"count": 1, "dtype":np.int16})    
                            ## within each polygon, calculate spatial stat for ras
                            gdf = polys.join(pd.DataFrame(zonal_stats(
                                vectors=polys['geometry'], raster=var_path, stats=[stat])), how='left' )
                            with rio.open(out_file, 'w+', **out_meta) as dst:
                                tmp_arr = dst.read(1)
                            ## rasterize polygon using stat value
                            shapes = ((geom,value) for geom, value in zip(gdf.geometry, gdf[stat]))
                            image = features.rasterize( ((g, v) for g, v in shapes), out_shape=out_shape[1:], transform=new_gt)
                            dst.write_band(1, image)
                            logger.debug(f'out_fn={out_file}')
                    
            else: ## using ts data
                get_ts_stats_within_polys(params)
                
        elif Path(polys).is_dir():
            ## directory with polygon files, also indicating gridded structure
            get_ts_stats_within_polys(params)
            
        else:
            logger.warning(f'not sure how to parse polys {polys}')

                        

        
