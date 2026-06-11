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
from .image_utils import clip_big_ras_to_small

        
def reclass_small_fields(params, class_ras, poly_ras, area_ras, ras_out): 
    '''
    reclassifies crop polygons smaller than 5ha as smallholder (or <2ha for other crops like sugar)
    if in an area that is not dominated by large crops (average field size < 10 ha)
    
    Note that areas for the input poly_ras are expected in ha*10
       (to represent variance withough causing errors in 16-byte outputs)
    '''
    if params['project_ver'] == 'Py_0':
        crop_dict = CROP_CATS_Py0
    else:
        crop_dict = CROP_CATS
        
    with rio.open(class_ras, 'r') as maj_src:
        maj = maj_src.read(1)
        with rio.open(poly_ras, 'r') as size_src:
            profile = size_src.profile
            polysize = size_src.read(1)
            with rio.open(area_ras, 'r') as area_src:
               croparea = area_src.read(1)
        big_crop = np.isin(maj, crop_dict['bigcrops']) & (croparea > 100)
        maj_reclassed = np.where((np.isin(maj, crop_dict['all_crops'])) & (polysize < 20) & (~big_crop), crop_dict['smallcrop_main'], maj)
        profile.update(compress='lzw', tiled=True)
        with rio.open(ras_out, 'w', **profile) as rcsmall_dst:
            rcsmall_dst.write(maj_reclassed,1)

        
def make_filter_layers_for_cell(params, filter_set):

    out_yr = params['classify']['out_yrs']
    buf = params['refine']['buffer']
    
    if params['project_ver'] == 'Py_0':
        crop_dict = CROP_CATS_Py0
    else:
        crop_dict = CROP_CATS

    if ('polys_area' in filter_set) or ('area_focal' in filter_set):
        ## get field size raster (this should already exist from vectorize_seg_results(), but may need to buffer)
        if 'polys_area' in filter_set:
            poly_area_in = filter_set['polys_area']['cell_base']
        else:
            poly_area_in = filter_set['area_focal']['cell_base']
        with rio.open(poly_area_in, 'r') as area_src:
            profile = area_src.profile
            polyarea = area_src.read(1)
        polyha = np.round(polyarea/10).astype(np.uint16)
        areaha_inter = Path(filter_set['area_focal']['cell_final']).parent / 'area_ha.tif'
        profile.update(dtype=rio.uint16, compress='lzw', tiled=True)
        logger.info(f'profile = {profile}')
        with rio.open(areaha_inter, "w", **profile) as dst:
                dst.write(polyha, 1)

        ## get polygon area with buffer applied (buffer is not calculated in area)
        if 'polys_area' in filter_set:
            poly_area_out = Path(filter_set['polys_area']['cell_final']).parent/f'polyarea{out_yr}-majority.tif'
        else:
            poly_area_out = Path(filter_set['area_focal']['cell_final']).parent/f'polyarea{out_yr}-majority.tif'
            
        params['feature_model']['ancillary_vars'] = [f'polyarea{out_yr}-majority']
        logger.info(f'saving polys_area to {poly_area_out} \n')
        make_polygon_features(params, in_path=areaha_inter,  out_path=poly_area_out)
        
        if 'area_focal' in filter_set:
            ## Run moving neighborhood calc on field size raster
            nbhd = params['refine']['neighborhood'] 
            with rio.open(poly_area_out, 'r') as size_src:
                profile = size_src.profile
                polysize = size_src.read(1)
            polysizef = polysize.astype(np.float32)
            focal_mean0 = uniform_filter(polysizef, size=nbhd, mode="reflect")
            ## convert to int and convert null to 0
            focal_mean = np.nan_to_num(np.round(focal_mean0), nan=0).astype(np.uint16)
            profile.update(dtype=rio.uint16, compress='lzw', tiled=True)
            logger.info(f"saving area_focal to {filter_set['area_focal']['cell_final']} \n")
            with rio.open(filter_set['area_focal']['cell_final'], "w", **profile) as dst:
                dst.write(focal_mean, 1)
                
    if 'polys_buf' in filter_set:
        ## get majority class for polygons with buffer applied
        params['feature_model']['ancillary_vars'] = [f'CELseg{out_yr}-majority']
        if 'rcsmall' in filter_set['polys_buf']['cell_final']:  ##this is the final version after reclassing very small polygons to smallholder
            ## make an intermediate version reclass_small_fields()
            polys_buf_out = str(filter_set['polys_buf']['cell_final']).replace('rcsmall.tif', '.tif')
        else: 
            polys_buf_out = filter_set['polys_buf']['cell_final']
        logger.info(f'saving polys_buf to {polys_buf_out} \n')
        params['refine']['buffer'] = buf
        make_polygon_features(params, in_path=filter_set['polys_buf']['cell_base'], out_path=polys_buf_out)
            
    if 'polys_maj' in filter_set:
        ## get majority class for polygons without buffer applied
        params['feature_model']['ancillary_vars'] = [f'CELseg{out_yr}-majority']
        params['refine']['buffer'] = 0
        if 'rcsmall' in filter_set['polys_maj']['cell_final']: ## this is the final version after reclassing very small polygons to smallholder
            ## make an intermediate version to feed to reclass_small_fields()
            polys_maj_out = str(filter_set['polys_maj']['cell_final']).replace('rcsmall.tif', '.tif')
        else: 
            polys_maj_out = filter_set['polys_maj']['cell_final']
        logger.info(f'saving polys_maj to {polys_maj_out} \n')
        make_polygon_features(params, in_path=filter_set['polys_maj']['cell_base'], out_path=polys_maj_out)
        params['refine']['buffer'] = buf

    if params['refine']['post_filter'] == 'smCrop':
        logger.info('reclassifying small polys to smallholder...\n')
        polys_buf_temp = Path(filter_set['polys_buf']['cell_final']).parent/'poly_buffers.tif'
        ## Reclass majclass polygons to smallholder if area <5 ha (and majclass is big crop) or <2ha (and majclass is any crop)
        reclass_small_fields(params, polys_buf_out, filter_set['polys_area']['cell_final'], filter_set['area_focal']['cell_final'], polys_buf_temp)
        reclass_small_fields(params, polys_maj_out, filter_set['polys_area']['cell_final'], 
                             filter_set['area_focal']['cell_final'], filter_set['polys_maj']['cell_final'])
         
        ## reclass buffer areas based on inner polygons for tasks relevant to smallholder area:
        with rio.open(polys_buf_temp, 'r') as buf_src:
            profile = buf_src.profile
            polysbuf = buf_src.read(1)
            with rio.open(filter_set['polys_maj']['cell_final'], 'r') as in_src:
                polysinner = in_src.read(1)
                with rio.open(filter_set['base_map']['cell_final']) as class_src:
                    lc = class_src.read(1)
                #is_null_buf = np.isnan(polysbuf)
                #is_null_poly = np.isnan(polysinner)
                ## reclass smallholder crop in buffer zone of large crops to crop_edge
                buf_out = np.where((polysinner == 0) & (polysbuf != 0) & (np.isin(polysbuf, crop_dict['bigcrops'])) 
                                        & (lc == crop_dict['smallcrop_main']), crop_dict['mixed_edge'], polysbuf) 
                profile.update(compress='lzw', tiled=True)
                with rio.open(filter_set['polys_buf']['cell_final'], 'w', **profile) as buf_dst:
                    buf_dst.write(buf_out, 1)

    if 'chaco' in filter_set:
        logger.info(f'clipping chaco \n')
        clip_big_ras_to_small(filter_set['area_focal']['cell_final'], filter_set['chaco']['cell_base'], filter_set['chaco']['cell_final'])
         
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
        highveg_nbhd = maximum_filter(highveg, size=5).astype(np.uint16)
        ## convert null to 0
        profile.update(dtype=rio.uint16, compress='lzw', tiled=True, nodata=None)
        with rio.open(filter_set['highveg_nbhd']['cell_final'], "w", **profile) as dst:
            dst.write(highveg_nbhd, 1)
        
    if ('sm_neighbors' in filter_set) or ('sm_nbhd9_mask' in filter_set) or ('sm_nbhd_dist' in filter_set):
        ## make smallholder mixed mask from original classed map combined with refined buffer layer

        with rio.open(filter_set['base_map']['cell_final'], 'r') as lc_src:
            profile = lc_src.profile
            profile.update(dtype=rio.uint16, compress='lzw', tiled=True,  nodata=None)
            lc = lc_src.read(1)
            with rio.open(filter_set['polys_buf']['cell_final'], 'r')  as buf_src:
                poly_lc = buf_src.read(1)
            sm = np.where((lc == crop_dict['smallcrop_main']) | (poly_lc == crop_dict['smallcrop_main']), 1, 0)
            #sm_nbhd = generic_filter(sm, np.sum, size=3, mode='reflect', cval=0)
            smf = sm.astype(np.float32)
            sm_nbhd_mean = uniform_filter(smf, size=3, mode='reflect', cval=0.0)
            sm_nbhd = np.round(sm_nbhd_mean * 9).astype(np.uint16)
            if 'sm_neighbors' in filter_set:
                with rio.open(filter_set['sm_neighbors']['cell_final'], "w", **profile) as dst:
                    dst.write(sm_nbhd, 1)
            if 'sm_nbhd9_mask' in filter_set:
                sm_nbhd9_mask = np.where(sm_nbhd == 9, 1, np.nan)
                with rio.open(filter_set['sm_nbhd9_mask']['cell_final'], "w", **profile) as dst:
                    dst.write(sm_nbhd9_mask, 1)
            if 'sm_nbhd_dist' in filter_set:
                ## need to invert mask and change null to 0 for scipy distance to work
                sm_nbhd_mask2 = np.where(sm_nbhd9_mask == 1, 0, 1)
                if np.any(sm_nbhd_mask2 == 0):   ##  without this will get infinate hang if cell has no smallholder pixels
                    sm_nbhd_dist = distance_transform_edt(sm_nbhd_mask2)
                    ## clip max distance to avoid large numbers. Note distance is measures in pixels with this method.
                    sm_nbhd_dist = np.clip(sm_nbhd_dist, 0, 10)
                    sm_nbhd_dist = np.nan_to_num(sm_nbhd_dist, nan=10).astype(rio.uint16)
                else:  ## there were no smallholder pixels. Fill with max distance (which is 10 here)
                    sm_nbhd_dist = np.full(sm_nbhd9_mask.shape, 10)
                with rio.open(filter_set['sm_nbhd_dist']['cell_final'], "w", **profile) as dst:
                    dst.write(sm_nbhd_dist, 1)
            
         
def make_full_filter_layers(params, filter_set):
    params['classify']['save_mosaic'] = True
    treat_final = params['classify']['test']
    params['classify']['test'] = True
    comp_dir_orig = params['classify']['comp_dir']
    
    for key, fs in filter_set.items():
        logger.debug(f'key = {key}, fs = {fs}')
        if not Path(fs['full']).is_file():
            logger.info(f'making {key} mosaic')
            if key == 'base_map':
                params['classify']['comp_dir'] = comp_dir_orig
                params['classify']['name'] = Path(fs['cell_final']).stem.split('_', 1)[1]
            else:
                params['classify']['name'] = Path(fs['cell_final']).stem
                params['classify']['comp_dir'] = Path(fs['cell_final']).parent
            mosaic_cells(params, out_path=Path(fs['full']))    
        else:
            logger.info(f" {key} mosaic already exists at {fs['full']}")
            
    params['classify']['test'] = treat_final
    params['classify']['comp_dir'] = comp_dir_orig


def post_classification_spatial_filter_smallholder(params, filter_set):

    if params['classify']['test']:
        out_dir = Path(params['scratch_dir']) / 'classified'
    else:
        out_dir = Path(params['backup_path']) / 'mosaics' 
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filter1_path = Path(params['scratch_dir']) /'comp/first_smallholder_filter.tif'
    filter2_path = Path(params['scratch_dir']) /'comp/second_smallholder_filter.tif'
    filterfinal_path = Path(out_dir)/f"{Path(filter_set['base_map']['full']).stem}_SmallholderF.tif"

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
        'polys_maj' : filter_set['polys_maj']['full'],
        'polys_buf' :  filter_set['polys_buf']['full'],
        'area_focal' : filter_set['area_focal']['full']
        }

    if 'paraguay' in str(params['backup_path']):
        paraguay_extra = {'chaco' :  filter_set['chaco']['full']}
        first_filter_rasters.update(paraguay_extra )
        
        
    logger.info('running final filters.... \n')
    with rio.open(first_filter_rasters['base_map']) as src:
        meta = src.meta.copy()
        meta.update(dtype='uint8', compress='lzw',tiled=True)
        with rio.open(filter1_path, 'w', **meta) as dst:
            readers = {k: rio.open(v) for k, v in first_filter_rasters.items()}
            for ji, window in src.block_windows(1):
                data = {k: r.read(1, window=window) for k, r in readers.items()}

                ## 1. Inside original segmented polygons:
                ##   a. Reclass all crop to main crop(classification (without segmenation, if possible), else retain current classification:
                poly_a = np.where(np.isin(data['polys_maj'],lowcrops), data['polys_maj'], data['base_map'])
                ##   b. classification with segmentation classifies a pixel as sugar or mixed crop, keep it that way
                #poly_treated = np.where(np.isin(data['maj_class6'], [smallholder_class, crop_dict['sugar']]), data['maj_class6'], poly_a)
                ##   (the above is only useful if majority class is based on pre-segmentation map):
                polys_treated = poly_a
                
                ## 2. Inside buffered Polygons:
                ##   a. If the majority class is mixed crop, reclassify all crop pixels in buffer area as mixed as well:
                polys_treated2 = np.where((data['polys_buf'] == smallholder_class) & (np.isin(data['base_map'],lowcrops)), smallholder_class, polys_treated)

                ## 3. Inside buffer area only:
                buf_a = np.where(np.isin(data['base_map'],lowcrops), smallholder_class, data['base_map'])
                ##   b. Crops in polygon buffers in areas with fields > 5ha are not converted to smallholder 
                buf_b = np.where((data['area_focal'] > 50) & (np.isin(data['base_map'],lowcrops)), data['polys_buf'], buf_a)
                ##   c. Pixels classified as sugar in a polygon buffer are retained as sugar unless in areas with no field > 3ha (sugar fields look smaller) 
                buf_treated = np.where((data['area_focal'] > 30) & (data['polys_buf'] ==  crop_dict['sugar']),  crop_dict['sugar'], buf_b) 

                seg_treated = np.where(data['polys_maj'] == 0, buf_treated, polys_treated2)

                ## 4. Outside crop polygons and buffers:
                #   a. If away from large fields and classified as any crop, reclassify as smallholder, otherwise retain current classification 
                outside_treated = np.where((data['area_focal'] <= 50) & (np.isin(data['base_map'],lowcrops)), smallholder_class, data['base_map'])    
            
                filter_result = np.where(data['polys_buf'] == 0, outside_treated, seg_treated)

                ## vegetation seems noisier in the Chaco -- do not convert veg to smallholder crop outside of segmented polygons there
                filter_result = np.where((data['polys_buf'] == 0) & (data['chaco'] == 1), data['base_map'], filter_result) 

                dst.write(filter_result.astype(rio.uint16), 1, window=window)

        for r in readers.values():
            r.close()
        dst.close()

    ## Final filter to remove mixed crop halos around high vegetation:
    ## Anything classified as smallholder and within 2 pixels of highVeg will be reset to mixed veg IF
    ##  IF it is > 1 pixel from a 3x3 block of smallholder crop (This ensures that edges of the actual smallholder fields 
    ##    will be preserved even if they touch high veg. (smallholder pixels are assumed to be real if a 9-pixel block is all classified as smallholder)
                                       
    second_filter_rasters = {
        'base_map' :  filter1_path,
        'highveg_nbhd' : filter_set['highveg_nbhd']['full'],
        'sm_nbhd' :  filter_set['sm_neighbors']['full'],
        'sm_nbhd9_mask' :  filter_set['sm_nbhd9_mask']['full'],
        'sm_nbhd_dist' : filter_set['sm_nbhd_dist']['full']
        }

    with rio.open(first_filter_rasters['base_map']) as src2:
        meta = src2.meta.copy()
        meta.update(dtype='uint8', compress='lzw',tiled=True)
        with rio.open(filter2_path, 'w', **meta) as dstf:
            readers2 = {k2: rio.open(v2) for k2, v2 in second_filter_rasters.items()}
            for yz, window in src2.block_windows(1):
                data2 = {k2: r2.read(1, window=window, masked=True) for k2, r2 in readers2.items()}
                        
                rc_mixed_veg = np.where(((data2['sm_nbhd'] > 0) & (data2['sm_nbhd'] <9) & (data2['sm_nbhd_dist'] > 1) & 
                         (data2['base_map'] == smallholder_class)), crop_dict['mixed_edge'], data2['base_map'])
                rc_sm_strays = np.where(data2['sm_nbhd9_mask'] == 0,rc_mixed_veg,data2['base_map'])
                rc_highveg = np.where(data2['highveg_nbhd'] == 0, data2['base_map'], rc_sm_strays)
                ## reclass any crop_edge (if exists) to mixed_edge (will change actual crop_edge back in next step)
                final = np.where(rc_highveg == crop_dict['crop_edge'], crop_dict['mixed_edge'], rc_highveg)

                dstf.write(final.astype(rio.uint16), 1, window=window)

        for r2 in readers2.values():
            r2.close()

    ## crop edge filter with single-pixel padding to avoid edge effects along read blocks
        PAD_SIZE = 1 
        with rio.open (filter2_path, 'r') as src3:
            profile = src3.profile
            profile.update(dtype='uint8', compress='lzw',tiled=True)
            with rio.open(filterfinal_path, 'w', **profile) as dstf:
                for ij, window in src3.block_windows(1):
                    read_window = Window(
                        col_off = window.col_off - PAD_SIZE,
                        row_off = window.row_off - PAD_SIZE,
                        width = window.width + (PAD_SIZE * 2),
                        height = window.height + (PAD_SIZE * 2)
                        )
                    lcf_padded = src3.read(1, window=read_window, boundless=True, fill_value=0)
                    crop_mask = np.where(np.isin(lcf_padded, crops), 1, 0)
                    crop_nbhd = maximum_filter(crop_mask, size=3)
                    lcff_padded = np.where((crop_nbhd == 1) & (lcf_padded == crop_dict['mixed_edge']), crop_dict['crop_edge'], lcf_padded)
                    lcff = lcff_padded[PAD_SIZE:-PAD_SIZE, PAD_SIZE:-PAD_SIZE]
        
                    dstf.write(lcff.astype(rio.uint16), 1, window=window)
    

    logger.info(f' ALL DONE!  saved final output at: {filterfinal_path}')

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
    else:
        comp_mod_name = mod_base
    if params['classify']['test']:
        final_dir = Path(params['scratch_dir']) / 'classified'
    else:
        final_dir = Path(params['backup_path']).parents[1] / 'mosaics' 
    final_dir.mkdir(parents=True, exist_ok=True)

    ## Get segmentation polygons -- seg polys are named by the last year in the time sequence but the models / classified products are by the first  
    if not params['feature_model']['poly_vector_path'].endswith(str(int(out_yr)+1)):
        logger.info(f"changing segmentation paths from {params['feature_model']['poly_vector_path']} to match your model year: {out_yr}")
        params['feature_model']['poly_vector_path'] = params['feature_model']['poly_vector_path'][:-4] + str(int(out_yr)+1)
        logger.info(f"new segmentation path is: {params['feature_model']['poly_vector_path']}")
    
    ## Get seg variable path (for precalculated area) (in map version without polygons applied here, but maybe not necessary)
    poly_var_path_orig = params['feature_model']['poly_var_path']
    if not poly_var_path_orig.endswith(str(int(out_yr)+1)):
        poly_var_path_orig = params['feature_model']['poly_var_path'][:-4] + str(int(out_yr)+1)

    ## new variables will be sent to the temp drive
    tmp_poly_var_path = Path(params['scratch_dir']) /'tmp_poly_rasts'
    tmp_poly_var_path.mkdir(parents=True, exist_ok=True)
    params['feature_model']['poly_var_path'] = tmp_poly_var_path
        
    buf = params['refine']['buffer']
       
    if params['grids']:
        
        all_cells = params['grids']
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
            cell = int(cell)
            params['grids'] = [cell]
            ppaths = ProjectPaths(params, grid=cell)
            logger.info(f'working on cell {cell}...\n')

            if params['classify']['comp_dir'] == 'input_dir':
                comp_dir_in = ppaths.ms.parent/'comp'
            elif params['classify']['comp_dir'] == 'backup':
                comp_dir_in = ppaths.comp
            
            comp_dir_out = Path(params['scratch_dir']) /'comp'
            comp_dir_out.mkdir(parents=True, exist_ok=True)
            
            cell_var_dir = Path(tmp_poly_var_path)/f'{cell:06d}'
            cell_var_dir.mkdir(parents=True, exist_ok=True)
            
            if params['refine']['post_filter'] == 'smCrop':
                map_filters = { 
                    'base_map' : {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                 'cell_final':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}', 
                                 'full':f'{final_dir}/CELPyTileAll_{mod_base}'},
                    'polys_maj' : {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}', 
                                 'cell_final':f'{cell_var_dir}/CELseg{out_yr}-majority_rcsmall.tif', 
                                 'full':f'{comp_dir_out}/CELseg{out_yr}-majority_rcsmall.tif'},
                    'polys_buf' : {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}', 
                                  'cell_final':f'{cell_var_dir}/CELseg{out_yr}-majority_buf{buf}_rcsmall.tif', 
                                  'full':f'{comp_dir_out}/CELseg{out_yr}-majority_buf{buf}_rcsmall.tif'},
                    'polys_area' : {'cell_base':f'{poly_var_path_orig}/pred_area_{cell:04d}.tif', 
                                   'cell_final':f'{cell_var_dir}/polyarea{out_yr}-majority.tif', 
                                   'full':f'{comp_dir_out}/polyarea{out_yr}-majority.tif'},
                    'area_focal' : {'cell_base':f'{poly_var_path_orig}/pred_area_{cell:04d}.tif', 
                                   'cell_final':f'{cell_var_dir}/field_area_focal100avg_int0.tif', 
                                   'full':f'{comp_dir_out}/field_area_focal100avg_int0.tif'},
                    'highveg_nbhd': {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                    'cell_final':f'{cell_var_dir}/highveg_nbhd.tif', 
                                     'full':f'{comp_dir_out}/highveg{out_yr}_nbhd.tif'},
                    'sm_neighbors': {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                    'cell_final':f'{cell_var_dir}/sm_neighbors.tif', 
                                     'full':f'{comp_dir_out}/sm_neighbors{out_yr}.tif'},
                    'sm_nbhd9_mask': {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                    'cell_final':f'{cell_var_dir}/sm_nbhd9_mask.tif', 
                                     'full':f'{comp_dir_out}/sm_nbhd9_{out_yr}_mask.tif'},
                    'sm_nbhd_dist': {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                    'cell_final':f'{cell_var_dir}/sm_nbhd_dist.tif', 
                                     'full':f'{comp_dir_out}/sm_nbhd{out_yr}_dist.tif'}
                    }  

                paraguay_extra = {'chaco': {
                                  'cell_base' : f"{Path(params['backup_path']).parents[1]}/ancillary/Chaco.tif",
                                  'cell_final':f'{tmp_poly_var_path}/{cell:06d}/Chaco.tif', 
                                  'full':f'{comp_dir_out}/Chaco.tif'}
                                    }

                if 'paraguay' in str(params['backup_path']):
                    map_filters.update(paraguay_extra)
                    
            make_filter_layers_for_cell(params, filter_set=map_filters)
        
        params['grids'] = all_cells
        logger.info(f"mosaicking {params['grids']}")
        make_full_filter_layers(params, filter_set=map_filters)

    else:
        if params['refine']['post_filter'] == 'smCrop':
            map_filters = { 
                    'base_map' : {'full':f'{final_dir}/CELPyTileAll_{mod_base}'},
                    'polys_maj' : {'full':f'{comp_dir_out}/CELseg{out_yr}-majority_rcsmall.tif'},
                    'polys_buf' : {'full':f'{comp_dir_out}/CELseg{out_yr}-majority_buf{buf}_rcsmall.tif'},
                    'polys_area' : {'full':f'{comp_dir_out}/polyarea{out_yr}-majority.tif'},
                    'area_focal' : {'full':f'{comp_dir_out}/field_area_focal100avg_int0.tif'},
                    'highveg_nbhd': {'full':f'{comp_dir_out}/highveg{out_yr}_nbhd.tif'},
                    'sm_neighbors': {'full':f'{comp_dir_out}/sm_neighbors{out_yr}.tif'},
                    'sm_nbhd9_mask': {'full':f'{comp_dir_out}/sm_nbhd9_{out_yr}_mask.tif'},
                    'sm_nbhd_dist': {'full':f'{comp_dir_out}/sm_nbhd{out_yr}_dist.tif'}
                    }           
        make_full_filter_layers(params, filter_set=map_filters)
    
    if params['refine']['post_filter'] == 'smCrop':
        post_classification_spatial_filter_smallholder(params, filter_set=map_filters)
                      
        
            