from pathlib import Path
import csv
import rasterio as rio
from rasterio.windows import Window
import numpy as np
#from scipy.ndimage import generic_filter
from scipy.ndimage import uniform_filter, maximum_filter ,distance_transform_edt
from .lookup import CROP_CATS_Py0, CROP_CATS
from .project import ProjectPaths
from ..handler import logger
from .zonal import make_polygon_features
from .aggregate import mosaic_cells


def reclass_small_fields(params, class_ras, poly_ras, ras_out): 
    
    if params['project_ver'] == 'Py_0':
        crop_dict = CROP_CATS_Py0
    else:
        crop_dict = CROP_CATS
        
    smallholder_class = crop_dict['smallcrop_main']
    bigcrop_classes = crop_dict['bigcrops']
    crops = crop_dict['all_crops']
    
    with rio.open(class_ras, 'r') as maj_src:
        maj = maj_src.read(1)
    with rio.open(poly_ras, 'r') as size_src:
        profile = size_src.profile
        polysize = size_src.read(1)
    cond_big_crop = np.isin(maj, bigcrop_classes) & (polysize < 500)
    cond_other_crop = np.isin(maj, crops) & (polysize < 200)
    condition = cond_big_crop | cond_other_crop
    maj_reclassed = np.where(condition, smallholder_class, maj)
    with rio.open(ras_out, 'w', **profile) as rcsmall_dst:
        rcsmall_dst.write(maj_reclassed,1)

        
def make_filter_layers_for_cell(params, filter_set):
    
    out_yr = params['classify']['out_yrs']

    if 'polys_buf' in filter_set:
        ## get majority class for polygons with buffer applied
        buf = params['refine']['buffer']
        params['feature_model']['ancillary_vars'] = [f'CELseg{out_yr}-majority']
        if 'rcsmall' in filter_set['polys_buf']['cell_final']:
            polys_buf_out = str(filter_set['polys_buf']['cell_final']).replace('rcsmall.tif', '.tif')
        else: 
            polys_buf_out = filter_set['polys_buf']['cell_final']
        make_polygon_features(params, in_path=filter_set['polys_buf']['cell_base'], out_path=polys_buf_out)
            
    if 'polys_maj' in filter_set:
        ## get majority class for polygons without buffer applied
        params['feature_model']['ancillary_vars'] = [f'CELseg{out_yr}-majority']
        params['refine']['buffer'] = 0
        if 'rcsmall' in filter_set['polys_maj']['cell_final']:
            polys_maj_out = str(filter_set['polys_buf']['cell_final']).replace('rcsmall.tif', '.tif')
        else: 
            polys_maj_out = filter_set['polys_maj']['cell_final']
        make_polygon_features(params, in_path=filter_set['polysmaj']['cell_base'], out_path=polys_maj_out)
            
    if 'polys_area' in filter_set or 'area_focal' in filter_set:
        ## get field size raster (this should already exist from vectorize_seg_results(), but may need to buffer)
        if 'polys_area' in filter_set:
            poly_area_in = filter_set['polys_area']['cell_base']
        else:
            poly_area_in = filter_set['area_focal']['cell_base']
            
        ## If no buffer, the output is the same as the input, so no need to process this
        if buf == 0:
            poly_area_out = poly_area_in
        ## if following up with... need to apply buffer to match polys_buf
        else:
            params['refine']['buffer'] = buf
            ## get polygon area with buffer applied (buffer is not calculated in area)
            if 'polys_area' in filter_set:
                poly_area_out = Path(filter_set['polys_area']['cell_final']).parent/f'polyarea{out_yr}-majority.tif'
            else:
                poly_area_out = Path(filter_set['area_focal']['cell_final']).parent/f'polyarea{out_yr}-majority.tif'
            
            params['feature_model']['ancillary_vars'] = [f'polyarea{out_yr}-majority']
            make_polygon_features(params, in_path=poly_area_in,  out_path=poly_area_out)
        
        if 'area_focal_paths' in filter_set:
            ## Run moving neighborhood calc on field size raster
            nbhd = params['refine']['neighborhood'] 
            with rio.open(poly_area_out, 'r') as size_src:
                profile = size_src.profile
                polysize = size_src.read(1)
            polysizef = polysize.astype(np.float32)
            focal_mean0 = uniform_filter(polysizef, size=nbhd, mode="reflect")
            ## convert to int and convert null to 0
            focal_mean = np.nan_to_num(np.round(focal_mean0), nan=0).astype(np.uint16)
            profile.update(dtype=rio.uint16)
            with rio.open(filter_set['area_focal']['cell_final'], "w", **profile) as dst:
                dst.write(focal_mean, 1)

    if params['refine']['post_filter'] == 'smCrop':
        ## Reclass majclass polygons to smallholder if area <5 ha (and majclass is big crop) or <2ha (and majclass is any crop)
        reclass_small_fields(params, polys_buf_out, filter_set['area_focal']['cell_final'], filter_set['polysmaj']['cell_final'])
        reclass_small_fields(params, polys_maj_out, filter_set['area_focal']['cell_final'], filter_set['polys_buf']['cell_final'])

    if 'highveg_nbhd' in filter_set: 
        ## make high_veg mask from original classed map
        if params['project_ver'] == 'Py_0':
            first_highveg = 50
        else:
            first_highveg = 180
        
        with rio.open(filter_set['highveg_nbhd']['cell_base'], 'r') as lc_src:
            profile = lc_src.profile
            lc = lc_src.read(1)
        highveg = np.where(lc >= first_highveg, 1, 0)
        highveg_nbhd0 = maximum_filter(highveg, size=5)
        ## convert null to 0
        highveg_nbhd = np.nan_to_num(np.round(highveg_nbhd0), nan=0).astype(np.uint16)
        profile.update(dtype=rio.uint16)
        with rio.open(filter_set['highveg_nbhd']['cell_final'], "w", **profile) as dst:
            dst.write(highveg_nbhd, 1)
        
    if ('sm_nbhd' in filter_set) or ('sm_nbhd9_mask' in filter_set) or ('sm_nbhd_dist' in filter_set):
        ## make smallholder mixed mask from original classed map
        if params['project_ver'] == 'Py_0':
            smallholder_class = 35
        else:
            smallholder_class = 137
        with rio.open(filter_set['sm_nbhd']['cell_base'], 'r') as lc_src:
            profile = lc_src.profile
            lc = lc_src.read(1)
        sm = np.where(lc == smallholder_class, 1, 0)
        #sm_nbhd = generic_filter(sm, np.sum, size=3, mode='reflect', cval=0)
        smf = sm.astype(np.float32)
        sm_nbhd_mean = uniform_filter(smf, size=3, mode='reflect', cval=0.0)
        sm_nbhd = np.round(sm_nbhd_mean * 9).astype(np.uint16)
        if 'sm_nbhd' in filter_set:
            with rio.open(filter_set['sm_nbhd']['cell_final'], "w", **profile) as dst:
                dst.write(sm_nbhd, 1)
        if 'sm_nbhd9_mask' in filter_set:
            sm_nbhd9_mask = np.where(sm_nbhd == 9, 1, np.nan)
            profile.update(dtype=rio.uint16)
            with rio.open(filter_set['sm_nbhd9_mask']['cell_final'], "w", **profile) as dst:
                dst.write(sm_nbhd9_mask, 1)
        if 'sm_nbhd_dist' in filter_set:
            ## need to invert mask and change null to 0 for scipy distance to work
            sm_nbhd_mask2 = np.where(sm_nbhd9_mask == 1, 0, 1)
            if np.any(sm_nbhd_mask2 == 0):   ## this is crucial for cells with no smallholder pixels
                sm_nbhd_dist = distance_transform_edt(sm_nbhd_mask2)
                ## clip max distance to avoid huge rasters. Note distance is measures in pixels with this method.
                sm_nbhd_dist = np.clip(sm_nbhd_dist, 0, 10)
            else:
                sm_nbhd_dist = np.full(sm_nbhd9_mask.shape, 10)
            with rio.open(filter_set['sm_nbhd_dist']['cell_final'], "w", **profile) as dst:
                dst.write(sm_nbhd_dist, 1)
            
            
def make_full_filter_layers(params, filter_set):
    params['classify']['save_mosaic'] = True
    treat_final = params['classify']['test']
    params['classify']['test'] = True

    for fs in filter_set:
        if not fs['full'].is_file():
            logger.info(f'making {fs} mosaic')
            params['classify']['name'] = fs['full'].stem.split('.')[0]
            mosaic_cells(params)      
            
    params['classify']['test'] = treat_final

def post_classification_spatial_filter_smallholder(params, filter_set):

    if params['classify']['test']:
        out_dir = Path(params['scratch_dir']) / 'classified'
    else:
        out_dir = Path(params['backup_path']) / 'mosaics' 
        
    filter1_path = Path(out_dir)/'sm_filter1.tif'
    filterfinal_path = Path(out_dir)/f"{params['classify']['name']}_SmallholderF.tif"
    
    if params['project_ver'] == 'Py_0':
        crop_dict = CROP_CATS_Py0
    else:
        crop_dict = CROP_CATS
        
    smallholder_class = crop_dict['smallcrop_main']
    bigcrop_classes = crop_dict['bigcrops']
    lowcrops = crop_dict['low_crops']
    crops = crop_dict['all_crops']

    first_filter_rasters = {
        'base_map' :  filter_set['base_map']['full'],
        'polysmaj' : filter_set['polysmaj']['full'],
        'polys_buf' :  filter_set['polys_buf']['full'],
        'area_focal' : filter_set['area_focal']['full']
        }

    with rio.open(first_filter_rasters['base_map']) as src:
        meta = src.meta.copy()

        with rio.open(filter1_path, 'w', **meta) as dst:
            readers = {k: rio.open(v) for k, v in first_filter_rasters.items()}

            for ji, window in src.block_windows(1):
                data = {k: r.read(1, window=window) for k, r in readers.items()}

                ## 1. Inside original segmented polygons:
                ##   a. Reclass all crop to main crop(classification (without segmenation, if possible), else retain current classification:
                poly_a = np.where(np.isin(data['polysmaj'],lowcrops), data['polysmaj'], data['base_map'])
                ##   b. classification with segmentation classifies a pixel as sugar or mixed crop, keep it that way
                #poly_b = np.where(np.isin(data['maj_class6'], [smallholder_class, crop_dict['sugar']]), data['maj_class6'], poly_a)
                ##   (the above is only useful if majority class is based on pre-segmentation map):
                poly_b = poly_a
                ##   c. convert sugar to mixed if inside area where avg field size <= 3 ha:
                poly_c = np.where((data['area_focal'] <= 300) & (data['polysmaj'] == crop_dict['sugar']), smallholder_class, poly_b)
                ##   d. convert all other crop to mixed if inside area where avg field size <= 5 ha:
                polys_treated = np.where(((data['area_focal'] < 500) & np.isin(data['polysmaj'],lowcrops)) 
                                     & (data['polysmaj'] != crop_dict['sugar']), smallholder_class, poly_c)
            
                ## 2. Inside buffered Polygons:
                ##   a. If the majority class is mixed crop, reclassify all crop pixels in buffer area as mixed as well:
                polys_treated2 = np.where((data['polys_buf'] == smallholder_class) & (np.isin(data['base_map'],lowcrops)), smallholder_class, polys_treated)

                ## 3. Inside buffer area only:
                buf_a = np.where(np.isin(data['base_map'],lowcrops), smallholder_class, data['base_map'])
                ##   b. Crops in polygon buffers in areas with fields > 5ha are not converted to smallholder 
                buf_b = np.where((data['area_focal'] > 500) & (np.isin(data['base_map'],lowcrops)), data['polys_buf'], buf_a)
                ##   c. Pixels classified as sugar in a polygon buffer are retained as sugar unless in areas with no field > 3ha (sugar fields look smaller) 
                buf_treated = np.where((data['area_focal'] > 300) & (data['polys_buf'] ==  crop_dict['sugar']),  crop_dict['sugar'], buf_b) 

                is_null_seg = np.isnan(data['polysmaj'])
                seg_treated = np.where(is_null_seg, buf_treated, polys_treated2)

                ## 4. Outside crop polygons and buffers:
                #   a. If away from large fields and classified as any crop, reclassify as smallholder, otherwise retain current classification 
                outside_treated = np.where((data['area_focal'] <= 500) & (np.isin(data['base_map'],lowcrops)), smallholder_class, data['base_map'])    
            
                is_null_buf = np.isnan(data['polys_buf'])
                filter_result = np.where(is_null_buf, outside_treated, seg_treated)

                dst.write(filter_result.astype(rio.uint16), 1, window=window)

        for r in readers.values():
            r.close()

    ## Final filter to remove mixed crop halos around high vegetation:
    ## Anything classified as smallholder and within 2 pixels of highVeg will be reset to mixed veg IF
    ##  IF it is > 1 pixel from a 3x3 block of smallholder crop (This ensures that edges of the actual smallholder fields 
    ##    will be preserved even if they touch high veg. (smallholder pixels are assumed to be real if a 9-pixel block is all classified as smallholder)
                                       
    second_filter_rasters = {
        'base_map' :  filter1_path,
        'highveg_nbhd' : filter_set['highveg_nbhd']['full'],
        'sm_nbhd' :  filter_set['sm_nbhd']['full'],
        'sm_nbhd9_mask' :  filter_set['sm_nbhd9_mask']['full'],
        'sm_nbhd_dist' : filter_set['sm_nbhd_dist']['full']
        }

    with rio.open(second_filter_rasters['base_map']) as src:
        meta = src.meta.copy()

        with rio.open(filterfinal_path, 'w', **meta) as dst:
            readers2 = {k: rio.open(v) for k, v in second_filter_rasters.items()}

            for ji, window in src.block_windows(1):
                data2 = {k: r.read(1, window=window) for k, r in readers.items()}
                                       
                rc_mixed_veg = np.where((data2['sm_nbhd'] > 0) & (data2['sm_nbhd'] <9) & (data['base_map'] == smallholder_class), crop_dict['mixed_edge'], data['base_map'])
                rc_mixed_veg2 = np.where((data2['sm_nbhd'] > 0) & (data2['sm_nbhd'] <9) & (data['sm_nbhd_dist'] > 1) & 
                         (data['base_map'] == smallholder_class), crop_dict['mixed_edge'],  rc_mixed_veg)
                outside_sm_block = np.where(np.isnan(data['sm_nbhd9_mask'],rc_mixed_veg2,data['base_map']))
                final = np.where(data2['highveg_nbhd'] == 0, data['base_map'], outside_sm_block)

                dst.write(filter_result.astype(rio.uint16), 1, window=window)

        for r in readers2.values():
            r.close()


def post_aggregation_filter(params):
    
    params['refine']['post_filter'] = 'smCrop'   ##currently only option. Could expand later
    params['feature_model']['spec_indices'] = ['']
    params['feature_model']['si_vars'] = ['']
    params['feature_model']['unit_of_analysis'] = 'pixel'
    out_yr = params['classify']['out_yrs']
    if isinstance (out_yr,list):
        out_yr = out_yr[0]
        params['classify']['out_yrs'] = out_yr
    mod_base = params['classify']['name']
    if mod_base.startswith('relative'):
        comp_mod_name = mod_base.replace('relative_','')

    ##  Reminder that the segmentation polys are names by the last year in the time sequence but the models / classified products are by the first  
    if not params['feature_model']['poly_vector_path'].endswith(str(int(out_yr)+1)):
        logger.info(f"changing segmentation paths from {params['feature_model']['poly_vector_path']} to match your model year: {out_yr}")
        params['feature_model']['poly_vector_path'] = params['feature_model']['poly_vector_path'][:-4] + str(int(out_yr)+1)
        logger.info(f"new segmentation path is: {params['feature_model']['poly_vector_path']}")
    
    ## Get majority class within segmentation polygons (in map version without polygons applied here, but maybe not necessary)
    poly_var_path_orig = params['feature_model']['poly_var_path']
    if not poly_var_path_orig.endswith(str(int(out_yr)+1)):
        poly_var_path_orig = params['feature_model']['poly_var_path'][:-4] + str(int(out_yr)+1)

    buf = params['refine']['buffer']
       
    if params['grids']:
        all_cells = params['grids']
        ## new poly vars will be sent to the temp drive
        tmp_poly_var_path = Path(params['scratch_dir']) /'tmp_poly_rasts'
        params['feature_model']['poly_var_path'] = tmp_poly_var_path
        
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
            #params['grids'] = [cell]
            ppaths = ProjectPaths(params, grid=cell)
            logger.info(f'working on cell {cell}...\n')

            if params['classify']['comp_dir'] == 'input_dir':
                comp_dir_in = ppaths.ms.parent/'comp'
            elif params['classify']['comp_dir'] == 'backup':
                comp_dir_in = ppaths.comp
       
            comp_dir_out = ppaths.scratch /'comp'

            if params['refine']['post_filter'] == 'smCrop':
                filter_set = { 
                    'base_map' : {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                 'cell_final':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}', 
                                 'full':f"{params['backup_path'].parents[1]}/mosaics/CELPyTileAll_{comp_mod_name}"},
                    'polysmaj' : {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}', 
                                 'cell_final':f'{tmp_poly_var_path}/{cell:06d}/CELseg{out_yr}-majority_rcsmall.tif', 
                                 'full':f'{comp_dir_out}/CELseg{out_yr}-majority_rcsmall.tif'},
                    'polys_buf' : {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}', 
                                  'cell_final':f'{tmp_poly_var_path}/{cell:06d}/CELseg{out_yr}-majority_buf{buf}_rcsmall.tif', 
                                  'full':f'{comp_dir_out}/CELseg{out_yr}-majority_buf{buf}_rcsmall.tif'},
                    'polys_area' : {'cell_base':f'{poly_var_path_orig}/pred_area_{cell:04d}.tif', 
                                   'cell_final':f'{tmp_poly_var_path}/{cell:06d}/polyarea{out_yr}-majority.tif', 
                                   'full':f'{comp_dir_out}/polyarea{out_yr}-majority.tif'},
                    'area_focal' : {'cell_base':f'{poly_var_path_orig}/pred_area_{cell:04d}.tif', 
                                   'cell_final':f'{tmp_poly_var_path}/{cell:06d}/field_area_focal100avg_int0.tif', 
                                   'full':f'{comp_dir_out}/field_area_focal100avg_int0.tif'},
                    'highveg_nbhd': {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                    'cell_final':f'{tmp_poly_var_path}/{cell:06d}/highveg_nbhd.tif', 
                                     'full':f'{comp_dir_out}/highveg{out_yr}_nbhd.tif'},
                    'sm_nbhd': {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                    'cell_final':f'{tmp_poly_var_path}/{cell:06d}/sm_nbhd.tif', 
                                     'full':f'{comp_dir_out}/sm_nbhd{out_yr}.tif'},
                    'sm_nbhd9_mask': {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                    'cell_final':f'{tmp_poly_var_path}/{cell:06d}/sm_nbhd9_mask.tif', 
                                     'full':f'{comp_dir_out}/sm_nbhd9_{out_yr}_mask.tif'},
                    'sm_nbhd_dist': {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                    'cell_final':f'{tmp_poly_var_path}/{cell:06d}/sm_nbhd_dist.tif', 
                                     'full':f'{comp_dir_out}/sm_nbhd{out_yr}_dist.tif'}
                    }  
                      
            make_filter_layers_for_cell(params, filter_set=filter_set)
        
        all_cells = params['grids']
        make_full_filter_layers(params, filter_set=filter_set)

    else:
        if params['refine']['post_filter'] == 'smCrop':
            filter_set = { 
                    'base_map' : {'full':f"{params['backup_path'].parents[1]}/mosaics/CELPyTileAll_{comp_mod_name}"},
                    'polysmaj' : {'full':f'{comp_dir_out}/CELseg{out_yr}-majority_rcsmall.tif'},
                    'polys_buf' : {'full':f'{comp_dir_out}/CELseg{out_yr}-majority_buf{buf}_rcsmall.tif'},
                    'polys_area' : {'full':f'{comp_dir_out}/polyarea{out_yr}-majority.tif'},
                    'area_focal' : {'full':f'{comp_dir_out}/field_area_focal100avg_int0.tif'},
                    'highveg_nbhd': {'full':f'{comp_dir_out}/highveg{out_yr}_nbhd.tif'},
                    'sm_nbhd9_mask': {'full':f'{comp_dir_out}/sm_nbhd9_{out_yr}_mask.tif'},
                    'sm_nbhd_dist': {'full':f'{comp_dir_out}/sm_nbhd{out_yr}_dist.tif'}
                    }           
        make_full_filter_layers(params, filter_set=filter_set)
    
    if params['refine']['post_filter'] == 'smCrop':
        post_classification_spatial_filter_smallholder(params, filter_set=filter_set)
                      
        
            