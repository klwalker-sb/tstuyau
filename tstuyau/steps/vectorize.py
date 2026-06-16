import sys
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio as rio
from rasterio import features
from rasterio.features import shapes
from rasterio.windows import Window, from_bounds
import geopandas as gpd
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import shapely
from rasterio.windows import Window
from rasterio.features import shapes
import gc
import math
import shutil
from ..handler import logger
from .project import ProjectPaths
#from .image_utils import img_to_bbox_offsets

#############################################################################################
 
def instance_to_poly(input_raster, mmu=900.0):
    with rio.open(input_raster, 'r') as tmp:
        rast = tmp.read(1)
        rast_crs = tmp.crs
    mask = None
    instance_shapes = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes(rast, mask=mask, transform=tmp.transform)))
    #geoms = list(instance_shapes)
    vectorized_all  = gpd.GeoDataFrame.from_features(list(instance_shapes))
    vectorized = vectorized_all[vectorized_all['raster_val'] > 0.0] 
    # gdf = gpd.GeoDataFrame(pd.DataFrame(list(range(0,len(vectorized.geometry), 1)), columns=['pred_id']), 
    #                                  geometry=vectorized.geometry)
    vectorized['area'] = vectorized.area
    vectorized = vectorized[vectorized['area'] >= mmu]

    vectorized = vectorized.set_crs(rast_crs)

    return vectorized

    
def cut_fields(vectorized_gdf, proj_crs):
    vectorized_gdf['pred_id'] = list(range(0,len(vectorized_gdf), 1))

    ##  cut: negative buffer by X. if it's a polygon, rebuffer. if it's a multipolygon, split parts then rebuffer 
    eroded_geom = vectorized_gdf.buffer(distance=-10, resolution=1,  join_style=1)
    erd_clean = gpd.GeoDataFrame.from_features(eroded_geom.buffer(0))

    ## if it's an empty polygon (eroded away into nothing), take old geometry
    logger.info(erd_clean)
    if not len(erd_clean) > 0:
        return erd_clean
    emptyTF = erd_clean[erd_clean['geometry'].isna()].reset_index()
    emptyTF.columns=['pred_id','empty']
    empty_old = vectorized_gdf.set_index('pred_id').join(emptyTF.set_index('pred_id'), how='inner')
    empty_old = empty_old[ empty_old['geometry'] !=  None]
    empty_old = empty_old.reset_index()
    empty_old = empty_old.drop(columns=['empty'])

    ## if it has a real geometry remaining but is a polygon or multipolygon 
    multi_polygons=[]
    polygons=[]
    not_empty = erd_clean[~erd_clean['geometry'].isna()].reset_index()
    for i, row in not_empty.iterrows():
        if row.geometry.geom_type.startswith("Multi") or row.geometry.geom_type.startswith("MULTI"):
            multi_polygons.append(row.geometry)
        elif row.geometry.geom_type.startswith("Polygon") or row.geometry.geom_type.startswith("POLYGON"):
            polygons.append(row.geometry)

    ## if it's a polygon (didn't cut), use old geom (orig shape) 
    polys_gdf = gpd.GeoDataFrame(gpd.geoseries.GeoSeries(polygons), 
                                 columns=['geometry'], 
                                 crs=proj_crs)
    old_polys = vectorized_gdf.sjoin(polys_gdf, how="inner", predicate="intersects")
    ## if it's a multipolygon, split parts (explode), then rebuffer to old geom 
    if len(multi_polygons)>0:
        multi_geoS = gpd.geoseries.GeoSeries(multi_polygons).explode(index_parts=True)
        multi_geoS=multi_geoS.reset_index()
        multi_geoS=multi_geoS.drop(columns=["level_0", "level_1"])
        logger.info(multi_geoS)
        multi_explode = gpd.GeoDataFrame(geometry=multi_geoS[0])#
        multi_explode = multi_explode.set_crs(proj_crs)
        multi_explode_reBuff = multi_explode.buffer(10, join_style=1)
        multi_explode_reBuff = gpd.GeoDataFrame(geometry=multi_explode_reBuff)#
    else:
        multi_explode_reBuff = None
    ## combine 
    new_cut_geom = pd.concat([empty_old, old_polys, multi_explode_reBuff], axis=0)
    logger.info(new_cut_geom)
    
    ## dissolve shapes that touch 
    logger.info(f'length of new gemetry object is {len(new_cut_geom)}')
    if len(new_cut_geom) <= 2:
        vectorized_gdf = vectorized_gdf.reset_index()
        dissolved_gdf = vectorized_gdf.drop(columns=["raster_val","area","level_1","level_0"])
    else:
        dissolved_geom = gpd.geoseries.GeoSeries([geom for geom in new_cut_geom.unary_union.geoms])
        dissolved_gdf = gpd.GeoDataFrame(pd.DataFrame(list(range(0,len(dissolved_geom), 1)), columns=['pred_id']), geometry=dissolved_geom)    

    dissolved_gdf = dissolved_gdf.set_crs(proj_crs)
    
    return dissolved_gdf


def boundary_extent_thresh(bound_arr, ext_arr, bound_thresh, ext_thresh, maxval):
    
    ## THRESHOLD BOUNDARY MASK 
    bound_mask = np.copy(bound_arr).astype(np.uint8)
    bound_mask=np.where(bound_arr > bound_thresh*maxval, 1, 0).astype(np.uint8)
    ## THRESHOLD EXTENT MASK # double mask extent w/ boundary mask
    extent_mask = np.copy(ext_arr).astype(np.uint8)
    extent_mask=np.where(ext_arr > ext_thresh*maxval, 1, 0).astype(np.uint8)
    ## add boundary mask to crop mask (make boundary pixels 0, even if they're crop pixels)
    extent_mask=np.where(bound_mask == 1, 0, extent_mask).astype(np.uint8)
        
    return extent_mask
    

def watershed_segmentation(dist_rast_masked, extent_mask, seed_size):
    ftp_xy = peak_local_max(dist_rast_masked,  
                            footprint=np.ones((int(seed_size), int(seed_size))),  
                            labels=extent_mask) 
    mask = np.zeros(extent_mask.shape, dtype=bool)
    mask[tuple(ftp_xy.T)] = True
    markers, _ = ndi.label(mask)
    instances = watershed(-dist_rast_masked,   
                          markers,  
                          mask=extent_mask) 
    return instances

def eo_instance(ext_arr, bound_arr, eo_thresh):
    
        extent = ext_arr/1000
        boundary=bound_arr/1000
        test_arr = 1 + extent - boundary
        out_arr = test_arr.copy()
        out_arr[out_arr < eo_thresh ] = 0
        out_arr[out_arr >= eo_thresh ] = 1
        
        return out_arr

    
def single_semantic2instance(params):

    pred_dir = params['feature_model']['poly_var_path']  ## rasterized polygon features
    poly_dir = params['feature_model']['poly_vector_path']
    prefix = params['segment']['prefix']
    grid_file = gpd.read_file(params['grid_file'])
    proj_crs = grid_file.crs
    instance_method = params['segment']['instance_method']
    single_band = params['vectorize']['single_band']  #True
    mmu =  params['vectorize']['mmu']

    if not poly_dir.is_dir():
        poly_dir.mkdir(parents=True, exist_ok=True)

    files = []
    pred_rasts = sorted(list(pred_dir.glob('*.tif')))
    
    for pred_rast in pred_rasts:
        grid = pred_rast.name.split(".")[-2][-4:] ## end file with grid number 
        gridcell = grid_file[grid_file['UNQ'] == int(grid)]
        #buffered = gridcell.buffer(params['buffer']+int(params['res']), cap_style='square',join_style='mitre')
        buffered = gridcell.buffer(params['buffer']+int(params['res']), cap_style=3,join_style=2)
        bounds = buffered.geometry.iloc[0].bounds ## bounds returns (minx, miny, maxx, maxy)
        boundary = (float(bounds[0]), float(bounds[2]), float(bounds[1]), float(bounds[3]))

        if single_band==True:
            name = Path(pred_rast).stem            
            dist_pba_name = Path(pred_dir) / f"{name.replace(f'pred_{prefix}_', 'pred_dst_')}"
            extent_pba_name = Path(pred_dir) / f"{name.replace(f'pred_{prefix}_', 'pred_ext_')}"            
            with rio.open(pred_rast) as src:
                out_meta = src.meta.copy()
                band_count = src.count
                if band_count == 4:
                    dist_arr, bound_arr, ext_arr, _ = src.read()
                elif band_count ==3:
                    dist_arr, bound_arr, ext_arr = src.read()
                else:
                    logger.warning(f'got {band_count} bands. Expecting 3 or 4')
            if not Path(dist_pba_name).exists():
                with rio.open(dist_pba_name, "w", **out_meta) as dst1:
                    dst1.write(dist_arr, indexes=1)
            if not Path(extent_pba_name).exists():
                with rio.open(extent_pba_name, "w", **out_meta) as dst2:
                    dst2.write(ext_arr, indexes=1) 

        if "EO" in instance_method:
            eo_thresh = params['vectorize']['eo_thresh'] # 7
            eo_name = Path(poly_dir) / f"{prefix}_EO_{str(grid)}_{str(eo_thresh).replace('.', 'pt')}th.tif"
            files.append(eo_name)
            fname = files[-1]  
        elif "thresh" in instance_method:
            bound_thresh = params['vectorize']['bound_thresh'] #0.4
            ext_thresh = params['vectorize']['ext_thresh'] #0.6
            thresh_name = Path(poly_dir) / f"{prefix}_thresh_{str(grid)}_b{str(bound_thresh)}0_e{(ext_thresh)}th.tif"
            thresh_name=f"{thresh_name.replace('.','pt',2)}.tif"
            files.append(thresh_name)
            fname = files[-1]
        else:
            bound_thresh = params['vectorize']['bound_thresh'] #0.4
            ext_thresh = params['vectorize']['ext_thresh'] #0.6
            seed_size = params['vectorize']['seed_size'] #15
            water_name = Path(poly_dir) / f"{prefix}_water_{str(grid)}_b{str(bound_thresh)}0_e{(ext_thresh)}th_s{str(seed_size)}.tif"
            eater_name=f"{thresh_name.replace('.','pt',2)}.tif"
            files.append(water_name)
            fname = files[-1]
            
        logger.info(f'fname={fname}')    
        
        if not Path(fname).exists():
            with rio.open(pred_rast) as src:
                gt = src.transform
                '''
                ## old method. Seems overkill but TODO: make sure this is working fine without
                offset = img_to_bbox_offsets(gt, cell, grid_file, buffer=100, res=10.0)
                new_gt = rio.Affine(gt[0], gt[1], (gt[2] + (offset[0] * gt[0])), 0.0, gt[4], (gt[5] + (offset[1] * gt[4])))
                dist_arr, bound_arr, ext_arr, _ = src.read(window=Window(offset[0], offset[1], offset[2], offset[3]))      
                '''
                window = from_bounds(*boundary, transform=src.transform)
                new_gt = src.window_transform(window)
                dist_arr, bound_arr, ext_arr, _ = src.read(window=window)
                out_meta = src.meta.copy()
                ## read in the 2k x 2k grid shape window to remove cultionet inference edge-effects 
                out_meta.update({"count": 1, "dtype":np.int16, "transform":new_gt, "height":int(window.height), "width":int(window.width)})
        
                if "EO" in instance_method:
                    ## EO THRESHOLD METHOD (3)
                    eo_arr = eo_instance(ext_arr, bound_arr, eo_thresh)
                    ## SAVE SINGLE-BAND INSTANCE RASTER ** 3
                    with rio.open(eo_name, "w", **out_meta) as dst:
                        dst.write(eo_arr, indexes=1)         
                
                if "thresh" in instance_method  or "water" in instance_method:
                    ## THRESHOLD BOUNDARY AND EXTENT RASTERS 
                    extent_mask = boundary_extent_thresh(bound_arr, ext_arr, bound_thresh, ext_thresh, params['masking']['maxval'])
                    ## SAVE SINGLE-BAND INSTANCE RASTER only if it's not the watershed method 
                    if not "water" in instance_method: 
                        with rio.open(thresh_name, 'w', **out_meta) as dst:
                            dst.write(extent_mask, indexes=1)     
                        
                if "water" in instance_method:
                    
                    ## MASK DISTANCE RASTER
                    dist_rast_masked = np.copy(dist_arr)
                    dist_rast_masked = np.where(extent_mask == 0, 0, dist_rast_masked) 

                    ## WATERSHED SEGMENTATION 
                    instances = watershed_segmentation(dist_rast_masked, extent_mask, seed_size)

                    ## SAVE SINGLE-BAND INSTANCE RASTER ** 2
                    with rio.open(water_name, "w", **out_meta) as dst:
                        dst.write(instances, indexes=1)     

        ## instance to polys
        og_polys = instance_to_poly(fname, mmu)  
        if "water" in instance_method:
            fname = f"Wtrshd_pred_polys_b{str(bound_thresh)[-1]}0_e{str(ext_thresh)[-1]}0_s{str(seed_size)}_{str(grid)}.gpkg"
        elif "thresh" in instance_method:
            fname = f"{instance_method}_pred_polys_b{str(bound_thresh)[-1]}0_e{str(ext_thresh)[-1]}0_{str(grid)}.gpkg"
        elif "EO" in instance_method:
            fname = f"{instance_method}_pred_polys_{str(eo_thresh).replace('.', 'pt')}th_{str(grid)}.gpkg"
        
        if not (Path(poly_dir)/fname).exists():
            merged_polygons = og_polys.dissolve().explode(index_parts=True)
            merged_polygons.to_file(Path(poly_dir)/fname, mode="w")

            ## cut fields 
            cut_polys = cut_fields(merged_polygons, proj_crs)
            print(cut_polys)
            if len(cut_polys) > 0:
                cut_polys.to_file(Path(poly_dir)/f"{fname.replace('.gpkg', '_cut.gpkg')}", mode='w')

    
def vectorize_seg_results(params):
    '''
    vectorized polys saved as .gpkg files for each grid in main project space in directory <segmentation>/<model>/<yr>/*_infer_polys
       and final merged .gpkg product in same directory.
    also creates area-based raster features for each grid in  <segmentation>/<model>/*_infer_polys*yr
    
    if <segment:temp_inputs> and <segment:clean_temp_data> are True, will clean out the data/predict folder in the temp drive, (this can get huge!)
    '''
    ppaths=ProjectPaths(params)
    grid_file = gpd.read_file(params['grid_file'])
    proj_crs = grid_file.crs
    instance_method = params['segment']['instance_method']
    single_band = params['vectorize']['single_band']  ## always True for now
    mmu =  params['vectorize']['mmu']

    ## only works one year at a time
    yr = params['sample_model']['train_yrs']
    if isinstance(yr,list):
        yr = yr[0]

    ## saving output into project dir (usually from temp dir)
    seg_dir_main = params['segment']['seg_dir_main']
    if not seg_dir_main:
        seg_dir_main = ppaths.segmentation
    #seg_dir_main.mkdir(parents=True, exist_ok=True)
    seg_dir_out = Path(seg_dir_main)/f"{params['segment']['seg_dir_mod']}"
    seg_dir_out.mkdir(parents=True, exist_ok=True)
    logger.info(f'saving outputs to {seg_dir_out} \n')
                
    if params['segment']['temp_inputs']:
        seg_dir_in = ppaths.segdir_temp/f"{params['segment']['seg_dir_mod']}"
    else: 
        seg_dir_in = Path(seg_dir_main)/f"{params['segment']['seg_dir_mod']}"
    logger.info(f'getting inputs from {seg_dir_in} \n')
   
    pred_prefix = params['segment']['prefix']

    if instance_method == 'EO':
        eot = params['vectorize']['eo_thresh']
        threshs = str(eot).replace('.', 'pt')
    elif "thresh" in instance_method:
        bt = params['vectorize']['bound_thresh']
        et = params['vectorize']['ext_thresh']
        threshs = f"{str(bt).replace('.', 'pt')}_{str(et).replace('.', 'pt')}"
    elif "water" in instance_method:
        bt = params['vectorize']['bound_thresh']
        et = params['vectorize']['ext_thresh']
        ss = params['vectorize']['seed_size']
        threshs = f"{str(bt).replace('.', 'pt')}_{str(et).replace('.', 'pt')}_{str(ss).replace('.', 'pt')}"
    
    prepred_dir = Path(seg_dir_in)/'composites_probas'
    out_dir = Path(seg_dir_out)/f"infer_polys_{str(instance_method)}_{threshs}_{yr}"
    params['feature_model']['poly_vector_path'] = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    perm_feat_dir = Path(seg_dir_out)/f"feats_{str(instance_method)}_{threshs}_{yr}"
    perm_feat_dir.mkdir(parents=True, exist_ok=True)  

    params['feature_model']['poly_var_path'] = prepred_dir

    single_semantic2instance(params)
    
    ## save extent band and distance to boundary band. also write instance raster -> polys + cut

    ## merge polys so area calculations aren't cut off at edges
    in_dir = out_dir

    ## save single merged gpkg to file for calculating stats (field size stats per admin area) from vectors
    merged_fi = Path(out_dir)/f"{params['segment']['prefix']}_{yr}_polys_{str(instance_method)}_{threshs}_merged.gpkg"
            
    if params['vectorize']['overwrite_merged']:
        merged_fi.unlink(missing_ok=True)
    if not merged_fi.exists():
        files = sorted(list(in_dir.glob('*_cut.gpkg')))
        logger.debug(f'files:{files}')
        gdfs = [gpd.read_file(f) for f in files]
        field_shp = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True)).dissolve().explode()

        ## add field size attributes
        field_shp['area_m'] = field_shp.area
        field_shp['APR'] = field_shp['area_m']/field_shp.length
        field_shp['area'] = field_shp.area*0.01
        field_shp['perimeter'] = field_shp.length
        field_shp['APrEf'] = 100*field_shp['perimeter']/(4*np.sqrt(field_shp['area_m']))

        ## delete fields < 30m x 30m
        mmu = params['vectorize']['mmu']
        field_shp = field_shp[field_shp['area_m'] >= mmu]

        field_shp = field_shp.set_crs(proj_crs)
        field_shp.to_file(merged_fi, mode="w")
    else:
        field_shp = gpd.read_file(merged_fi)

    ## write grid cell per file based on grid shape
    files = sorted(list(in_dir.glob('*_cut.gpkg')))
    grids = [i.stem.replace('_cut', '')[-4:] for i in files]
    for grid in grids:
        ## Need to buffer cell bounds to match other raster products for cell
        gridcell = grid_file[grid_file['UNQ'] == int(grid)]
        grid_bound = gridcell.buffer(params['buffer']+int(params['res']),cap_style=3,join_style=2).geometry.iloc[0]
        polys_per_grid = gpd.clip(field_shp, grid_bound) ## making area raster from merged shape
        print(polys_per_grid.bounds)
        rst_fn = Path(out_dir)/f"{pred_prefix}_{str(instance_method)}_{str(grid)}_{threshs}th.tif"

        ## raster to use as template
        with rio.open(rst_fn) as rst:
            meta = rst.meta.copy()
        meta.update({'dtype':np.uint16,'crs':proj_crs})
        logger.debug(f'meta = {meta}')
        
        for attrib in ['area', 'APR', 'APrEf']:
            logger.debug(f'working on {attrib}...\n')
            out_fn = Path(perm_feat_dir)/ f"pred_{attrib}_{str(grid)}.tif"
            with rio.open(out_fn, 'w+', **meta) as src:
                tmp_arr = src.read(1)
                shapes = ((geom,value) for geom, value in zip(polys_per_grid.geometry, polys_per_grid[attrib]))
                image = features.rasterize(((g, v) for g, v in shapes), out_shape=src.shape, transform=src.transform)
                src.write_band(1, image)

    ## copy feature_vars to main dir if in temp:
    if params['segment']['temp_inputs']:
        logger.info(f'copying original files from temp drive to {perm_feat_dir}')
        for tif_file in prepred_dir.glob('*.tif'):
            shutil.copy2(tif_file, perm_feat_dir)
            tif_file.unlink()
            Path(str(tif_file).replace('.tif','')).unlink()
        if params['segment']['clean_temp_data']:
            logger.info('cleaning predict data files from temp dir')
            data_folder = ppaths.segdir_temp/f"{params['segment']['seg_dir_mod']}/data/predict"
            shutil.rmtree(data_folder)
            ts_folder = ppaths.segdir_temp/f"{params['segment']['seg_dir_mod']}/time_series_vars"
            shutil.rmtree(ts_folder)
    
    logger.info('done!')