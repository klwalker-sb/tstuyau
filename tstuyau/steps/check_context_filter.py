from pathlib import Path
import csv
import rasterio as rio
from rasterio.windows import Window
import numpy as np
import xarray as xr
import geowombat as gw
#from scipy.ndimage import generic_filter
from scipy.ndimage import uniform_filter, maximum_filter ,distance_transform_edt
from .project import ProjectPaths
from .zonal import make_polygon_features
from .aggregate import mosaic_cells
from .image_utils import clip_big_ras_to_small
from .filter_utils import FilterTsArgs, store_count, get_most_frequent_cat_in_timeseries, mark_forest_edges, is_region_filter_active, get_regional_filter
from .time_series_post_utils import CORRECTIONS
from .lookup import LC_CATS_Py0, LC_CATS
from ..handler import logger
        
def reclass_small_fields(params, class_ras, poly_ras, area_ras, ras_out): 
    '''
    reclassifies crop polygons smaller than 5ha as smallholder (or <2ha for other crops like sugar)
    if in an area that is not dominated by large crops (average field size < 10 ha)
    
    Note that areas for the input poly_ras are expected in ha*10
       (to represent variance withough causing errors in 16-byte outputs)
    '''
    if params['project_ver'] == 'Py_0':
        LC_CATS = LC_CATS_Py0

    sm_class = LC_CATS['smallcrop_main']
    big_crops = LC_CATS['bigcrops']
    lowcrops = LC_CATS['low_crops']
        
    with rio.open(class_ras, 'r') as maj_src:
        maj = maj_src.read(1)
        with rio.open(poly_ras, 'r') as size_src:
            profile = size_src.profile
            polysize = size_src.read(1)
            with rio.open(area_ras, 'r') as area_src:
               croparea = area_src.read(1)
        big_crop = np.isin(maj, big_crops) & (croparea > 200)
        maj_reclassed0 = np.where((np.isin(maj, lowcrops)) & (polysize < 50) & (~big_crop), sm_class, maj)
        maj_reclassed = np.where((np.isin(maj, big_crops)) & (polysize <= 20) & (croparea < 200), sm_class, maj_reclassed0)
        profile.update(compress='lzw', tiled=True)
        with rio.open(ras_out, 'w', **profile) as rcsmall_dst:
            rcsmall_dst.write(maj_reclassed,1)


def refine_polygon_area_ras(params, poly_area_in,poly_area_out,buf=0):
    '''
    Preps rasterized area polygons for neighborhood operation. Buffers if buf >0, Reduces area for badly split polygons 
      and divides original output (100 ha) by 10 so that neighborhood results will not exceed dtype limits
      
    inputs: poly_area_in is path to rasterized polygons with field size
        expects files with same name but 'APrEf' in place of 'area' to break up polygons that did not segment well
            if APrEF >= 200, split area in half (two polygons). If APrEF > 400, split in 3 
            (if that file doesn't exist, this step will be skipped)
    '''
    
    if buf == 0:
        ras1 = poly_area_out
    else:
        ras1 = Path(poly_area_out).parent/'poly_area_tmp.tif'

    
    with rio.open(poly_area_in, 'r') as area_src:
        profile = area_src.profile
        polyarea = area_src.read(1)

        ## high values for APrEF (area-perimeter efficiency) suggest that multiple plots are in one polygon
        ##    if APrEF >= 200, split area in halp (two polygons). If APrEF > 400, split in 3 
        poly_apref = poly_area_in.replace('area','APrEf')
        if Path(poly_apref).is_file():
            with rio.open(poly_apref, 'r') as ef_src:
                apref = ef_src.read(1)
                doubles = np.where(apref >= 200, polyarea/2, polyarea)
                polyarea = np.where(apref >= 400, polyarea/3, doubles)
        else:
            logger.info('WARNING: can not find poly_APrEf. Keeping areas as is.')

        ## divide by 10 to reduce load (original area values are 100 hectares, now are 10 hectares)
        polyarea_simp = np.round(polyarea/10).astype(np.uint16)
            
        profile.update(dtype=rio.uint16, compress='lzw', tiled=True)
        logger.debug(f'profile = {profile}')
        with rio.open(ras1, "w", **profile) as dst:
            dst.write(polyarea_simp, 1)

    if buf > 0:
        pix_res = abs(profile['transform'][0]) 
        pixel_radius = int(np.round(buf / pix_res))
        window_size = (2 * pixel_radius) + 1
        expanded_polyarea = maximum_filter(polyarea_simp, size=window_size)
        with rio.open(poly_area_out, "w", **profile) as dst:
            dst.write(expanded_polyarea, 1)
        #make_polygon_features(params, in_path=ras1,  out_path=poly_area_out)
            
def get_avg_fieldsize_ras_from_polys(params, poly_area_in, nbhd_out, nbhd=100):

     with rio.open(poly_area_in, 'r') as size_src:
        profile = size_src.profile
        polysize = size_src.read(1)
        
        polysizef = polysize.astype(np.float32)
        focal_mean0 = uniform_filter(polysizef, size=nbhd, mode="reflect")
        ## convert to int and convert null to 0
        focal_mean = np.nan_to_num(np.round(focal_mean0), nan=0).astype(np.uint16)
        profile.update(dtype=rio.uint16, compress='lzw', tiled=True)
        logger.info(f"saving area_focal to {nbhd_out} \n")
        with rio.open(nbhd_out, "w", **profile) as dst:
            dst.write(focal_mean, 1)


def make_filter_layers_for_cell(params, filter_set):

    def make_blank_layer(params, filter_set, out_file):
        with rio.open(filter_set['base_map']['cell_base']) as src:
            profile = src.profile
            data = src.read()
        zeros_data = np.zeros_like(data)
        profile.update(count=zeros_data.shape[0])
        with rio.open(out_file, "w", **profile) as dst:
            dst.write(zeros_data)
        return out_file
        
    out_yr = params['classify']['out_yrs']
    buf = params['refine']['buffer']
    
    if params['project_ver'] == 'Py_0':
        LC_CATS = LC_CATS_Py0
    
    if ('polys_area' in filter_set) or ('area_focal' in filter_set):
        ## get field size raster (this should already exist from vectorize_seg_results(), but may need to buffer)
        
        if 'polys_area' in filter_set:
            poly_area_in = filter_set['polys_area']['cell_base']
        else:
            poly_area_in = filter_set['area_focal']['cell_base']
        
        ## if no file for poly_area in, assume that it has no (segmentable) crops and create zero placeholder raster 
        polys_exist=True
        if not Path(poly_area_in).is_file():
            polys_exist = False
            logger.info(f"WARNING: there is no polygon layer for this cell.  Making 0 placeholder...") 
            
            poly_area_in = f"{params['scratch_dir']}/tmp_poly_rasts/{params['grids'][0]:06d}/poly_area.tif"
            make_blank_layer(params, filter_set, poly_area_in)
            if 'polys_area' in filter_set:
                filter_set['polys_area']['cell_base'] = poly_area_in
            else:
                filter_set['area_focal']['cell_base'] = poly_area_in
            
        ## get polygon area with buffer applied (buffer is not calculated in area)
        if 'polys_area' in filter_set:
            poly_area_out = Path(filter_set['polys_area']['cell_final']).parent/f'polyarea{out_yr}-majority.tif'
        else:
            poly_area_out = Path(filter_set['area_focal']['cell_final']).parent/f'polyarea{out_yr}-majority.tif'
            
        params['feature_model']['ancillary_vars'] = [f'polyarea{out_yr}-majority']
        logger.info(f'saving polys_area to {poly_area_out} \n')
        refine_polygon_area_ras(params, poly_area_in,poly_area_out,buf=0)
        
        if 'area_focal' in filter_set:
            ## Run moving neighborhood calc on field size raster
            nbhd = params['refine']['neighborhood'] 
            nbhd_out = filter_set['area_focal']['cell_final']
            get_avg_fieldsize_ras_from_polys(params, poly_area_out, nbhd_out, nbhd=nbhd)
           
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
        if polys_exist == False:
            make_blank_layer(params, filter_set, polys_buf_out)
        else:
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
        if polys_exist == False:
            make_blank_layer(params, filter_set, polys_maj_out)
        else:
            make_polygon_features(params, in_path=filter_set['polys_maj']['cell_base'], out_path=polys_maj_out)
        params['refine']['buffer'] = buf

    if params['refine']['spatial_filter'] == 'smCrop':
        
        logger.info('reclassifying small polys to smallholder...\n')
       
        ## Reclass majclass polygons to smallholder if area <5 ha (and majclass is big crop) or <2ha (and majclass is any crop)
        # polys_buf_temp = Path(filter_set['polys_buf']['cell_final']).parent/'poly_buffers.tif'
        reclass_small_fields(params, polys_maj_out, filter_set['polys_area']['cell_final'],
                             filter_set['area_focal']['cell_final'], filter_set['polys_maj']['cell_final'])
        
        ## do same for buffered polygons -- need to buffer area raster first
        poly_area_buf = Path(filter_set['polys_area']['cell_final']).parent/'poly_area_buf.tif'
        refine_polygon_area_ras(params, filter_set['polys_area']['cell_base'], poly_area_buf, buf=buf)
        reclass_small_fields(params, polys_buf_out, poly_area_buf, filter_set['area_focal']['cell_final'], 
                             filter_set['polys_buf']['cell_final'])
     
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
        
    if ('sm_neighbors' in filter_set) or ('sm_nbhd_mask' in filter_set) or ('sm_nbhd_dist' in filter_set):
        ## make smallholder mixed mask from original classed map combined with refined buffer layer

        with rio.open(filter_set['base_map']['cell_final'], 'r') as lc_src:
            profile = lc_src.profile
            profile.update(dtype=rio.uint16, compress='lzw', tiled=True,  nodata=None)
            lc = lc_src.read(1)
            with rio.open(filter_set['polys_buf']['cell_final'], 'r')  as buf_src:
                poly_lc = buf_src.read(1)
            sm = np.where((np.isin(lc, LC_CATS['smallcrops'])) | (poly_lc == LC_CATS['smallcrop_main']), 1, 0)
            #sm_nbhd = generic_filter(sm, np.sum, size=3, mode='reflect', cval=0)
            smf = sm.astype(np.float32)

            sm_nbhd_size = params['refine']['sm_neighborhood']
            if sm_nbhd_size > 3:
                majsm = int(round(.8 * sm_nbhd_size * sm_nbhd_size))
            sm_nbhd_mean = uniform_filter(smf, size=sm_nbhd_size, mode='reflect', cval=0.0)
            sm_nbhd = np.round(sm_nbhd_mean * sm_nbhd_size*sm_nbhd_size).astype(np.uint16)
            if 'sm_neighbors' in filter_set:
                with rio.open(filter_set['sm_neighbors']['cell_final'], "w", **profile) as dst:
                    dst.write(sm_nbhd, 1)
            if 'sm_nbhd_mask' in filter_set:
                sm_nbhd_mask = np.where(sm_nbhd >= majsm, 1, np.nan)
                with rio.open(filter_set['sm_nbhd_mask']['cell_final'], "w", **profile) as dst:
                    dst.write(sm_nbhd_mask, 1)
            if 'sm_nbhd_dist' in filter_set:
                ## need to invert mask and change null to 0 for scipy distance to work
                sm_nbhd_mask2 = np.where(sm_nbhd_mask == 1, 0, 1)
                if np.any(sm_nbhd_mask2 == 0):   ##  without this will get infinate hang if cell has no smallholder pixels
                    sm_nbhd_dist = distance_transform_edt(sm_nbhd_mask2)
                    ## clip max distance to avoid large numbers. Note distance is measures in pixels with this method.
                    sm_nbhd_dist = np.clip(sm_nbhd_dist, 0, 10)
                    sm_nbhd_dist = np.nan_to_num(sm_nbhd_dist, nan=10).astype(rio.uint16)
                else:  ## there were no smallholder pixels. Fill with max distance (which is 10 here)
                    sm_nbhd_dist = np.full(sm_nbhd_mask.shape, 10)
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
    '''
    Applies a post-classification filter to the raw classified data to:
        -- reclassify crop polygons to majority class
        -- remove mixed crop halos from edge of larger fields so that mixed crop pixels can be classified as smallholder crop
        -- reclassify crop in very small fields (except sugar) to smallholder
        -- reclassify forest pixels bordered by lower LCs as disturbed forest  TODO -- add back in here
    Note, this is very specific to CELPy maps for Paraguay. Other tailored filters can be applied by defining them in 
        post_aggregation_filter() below and building a similar function as this one.
    Output products will have 'filtsmh-sp' in the file name
    '''

    if params['classify']['test']:
        out_dir = Path(params['scratch_dir']) / 'classified'
    else:
        out_dir = Path(params['backup_path']) / 'mosaics' 
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filter1_path = Path(params['scratch_dir']) /'comp/second_smallholder_filter.tif'
    filterfinal_path = Path(out_dir)/f"{Path(filter_set['base_map']['full']).stem}_filtsmh-sp.tif"

    PAD_SIZE = 1 
    is_paraguay = 'paraguay' in str(params['backup_path'])

    if params['project_ver'] == 'Py_0':
        LC_CATS = LC_CATS_Py0

    smallholder_class = LC_CATS['smallcrop_main']
    smallholder_classes = LC_CATS['smallcrops']
    bigcrop_classes = LC_CATS['bigcrops']
    lowcrops = LC_CATS['low_crops']
    crops = LC_CATS['all_crops']
    sugar_val = LC_CATS['sugar']
    mixed_edge_val = LC_CATS['mixed_edge']
    crop_edge_val = LC_CATS['crop_edge']

    sm_edge_dist = params['refine']['sm_neighborhood']

    raster_keys = ['base_map', 'polys_maj', 'polys_buf', 'area_focal', 'highveg_nbhd', 'sm_neighbors', 'sm_nbhd_mask', 'sm_nbhd_dist']
    all_rasters = {k: filter_set[k]['full'] for k in raster_keys if k in filter_set}
    if is_paraguay:
        all_rasters['chaco'] = filter_set['chaco']['full']

    with rio.open(all_rasters['base_map']) as src:
        meta = src.meta.copy()
        meta.update(dtype='uint8', compress='lzw', tiled=True)
    
        readers = {k: rio.open(v) for k, v in all_rasters.items()}
    
        with rio.open(filterfinal_path, 'w', **meta) as dstf:
            for ji, window in src.block_windows(1):
                read_window = Window(
                    col_off = window.col_off - PAD_SIZE,
                    row_off = window.row_off - PAD_SIZE,
                    width = window.width + (PAD_SIZE * 2),
                    height = window.height + (PAD_SIZE * 2)
                )
                data = {k: r.read(1, window=read_window, boundless=True, fill_value=0) for k, r in readers.items()}
            
                orig = data['base_map']
                p_maj = data['polys_maj']
                p_buf = data['polys_buf']
                a_focal = data['area_focal']
                smdist = data['sm_nbhd_dist']
                #smnbhd = data['sm_neighbors']
                highvegnbhd = data['highveg_nbhd']
                area_focal = data['area_focal']
                chaco = data['chaco'] if is_paraguay else np.zeros_like(orig)
            
                ## 1. Inside original segmented polygons
                ## reclass to majority crop if crop pixel in crop polygon
                polys_treated = np.where(np.isin(p_maj,crops) & np.isin(orig, crops), p_maj, orig)
            
                ## 2. Inside buffered Polygons
                ## leave alone if not low crop, doesn't border a low crop polygon, is outside smallholder area (avg field size > 5 ha), or is in Chaco 
                keep_asis = (a_focal > 50) | (chaco == 1) | (~np.isin(p_buf, lowcrops)) | (~np.isin(orig, lowcrops))
                ##  otherwise, convert low crop pixels to smallholder class if in polygon buffer
                buf = np.where(keep_asis, orig, smallholder_class)
                ## sugar exception: if any crop in the buffer of a sugar field where field sizes are > 3 ha, convert to sugar 
                buf_treated = np.where((a_focal > 30) & (p_buf == sugar_val) & (np.isin(orig, crops)), sugar_val, buf)
                
                ## if is in a segmented polygon, treat according to #1, otherwise treat according to #2 (if in polygon buffer area)
                polys_wbuf_treated = np.where(p_maj == 0, buf_treated, polys_treated)
            
                ## 3. Outside polygons and buffers
                ## low crop pixels in smallholder areas (except in Chaco) are converted to smallholder class if outside polygon & buffer area
                
                maybe_smallholder = (a_focal <= 50) & np.isin(orig, lowcrops) & (chaco == 0)
                outside_poly_treatment = np.where(maybe_smallholder, smallholder_class, orig)
                in_out = np.where(p_buf == 0, outside_poly_treatment, polys_wbuf_treated) 

                ## 4. Mixed crop halo removal -- 
                ## Anything classified as smallholder and within 2 pixels of highVeg will be reset to mixed veg IF
                ##  IF it is > 1 pixel from a block of smallholder crop (bolck size determined with <refine:sm_neighborhood> 4)
                ##       (This ensures that edges of the actual smallholder fields will be preserved even if they touch high veg) 
                
                in_sm_classes = np.isin(in_out, smallholder_classes)

                ## remove high veg halos: reclassify smallholder crop to mixed edge if borders high veg and > 1 pixel from contiguous smallholder block
                wo_highveg_halo = np.where((smdist > 1) & (highvegnbhd == 1) & in_sm_classes, mixed_edge_val, in_out)
                ## remove other halos: reclassify smallholder pixels to mixed edge if in buffer of main field 
                no_halo = np.where((smdist > 1) & in_sm_classes & (p_maj == 0) & np.isin(p_buf, bigcrop_classes), 
                                   mixed_edge_val, wo_highveg_halo)
            
                no_specs = np.where((smdist > sm_edge_dist) & in_sm_classes & (p_buf == 0) & ~np.isin(orig, lowcrops), mixed_edge_val, no_halo)
                
                ## convert all crop edge to mixed edge (to convert back in next step if actually on crop edge)
                postfilt = np.where(no_specs == crop_edge_val, mixed_edge_val, no_specs)
                
                ## convert mixed edge pixels to crop edge if border crops 
                ##      -- uses neighborhood with padding to avoid edge effects along read blocks
                crop_mask = np.where(np.isin(postfilt, crops), 1, 0)
                crop_nbhd = maximum_filter(crop_mask, size=3)
                final_padded = np.where((crop_nbhd == 1) & (postfilt == mixed_edge_val), crop_edge_val, postfilt)
                # Slice off the 1-pixel boundary padding to return to the original window size
                final = final_padded[PAD_SIZE:-PAD_SIZE, PAD_SIZE:-PAD_SIZE]

                logger.debug(f"Unique classes in orig: {np.unique(orig)}")
                logger.debug(f"Unique classes in polys_wbuf_treated: {np.unique(polys_wbuf_treated)}")
                logger.debug(f"Unique classes in in_out: {np.unique(in_out)}")
                logger.debug(f"Unique classes in final: {np.unique(final)}")
                
                dstf.write(final.astype(rio.uint8), 1, window=window)
        
        for r in readers.values():
            r.close()
    
    logger.info(f' ALL DONE! saved final output at: {filterfinal_path}')

def post_aggregation_filter(params):
    '''
    Applies filter based on spatial context
    Specific filter is indicatied with <params:refine:spatial_filter> -- Currently only option is 'smCrop' for smallholder mapping
    '''
    
    params['refine']['spatial_filter'] = 'smCrop'   ##currently only option. Could expand later
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
        final_dir = Path(params['scratch_dir']) / f'classified'
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
            comp_prefix = 'comp'
        elif isinstance(params['grids'], str) and params['grids'].endswith('.csv'):
            comp_prefix = Path(params['grids']).stem.split('.')[0]
            with open(params['grids'], newline='') as cell_file:
                for row in csv.reader(cell_file):
                    cells.append(row[0])
        elif isinstance(params['grids'], int) or isinstance(params['grids'], str): # if runing individual cells as array via bash script
            comp_prefix = params['grids']
            cells.append(params['grids']) 

        comp_dir_out = Path(params['scratch_dir']) /f'comp/{comp_prefix}'
        comp_dir_out.mkdir(parents=True, exist_ok=True)
        
        for cell in cells:
            cell = int(cell)
            params['grids'] = [cell]
            ppaths = ProjectPaths(params, grid=cell)
            logger.info(f'working on cell {cell}...\n')

            if params['classify']['comp_dir'] == 'input_dir':
                comp_dir_in = ppaths.ms.parent/'comp'
            elif params['classify']['comp_dir'] == 'backup':
                comp_dir_in = ppaths.comp
            
            cell_var_dir = Path(tmp_poly_var_path)/f'{cell:06d}'
            cell_var_dir.mkdir(parents=True, exist_ok=True)
            
            if params['refine']['spatial_filter'] == 'smCrop':
                if not params['refine']['sm_neighborhood']:
                    params['refine']['sm_neighborhood'] = 3
                sm_nbhd_size = params['refine']['sm_neighborhood']
                
                map_filters = { 
                    'base_map' : {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                 'cell_final':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}', 
                                 'full':f'{final_dir}/{comp_prefix}_{mod_base}'},
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
                    'sm_nbhd_mask': {'cell_base':f'{str(comp_dir_in)}/{cell:06d}_{mod_base}',
                                    'cell_final':f'{cell_var_dir}/sm_nbhd_mask.tif', 
                                     'full':f'{comp_dir_out}/sm_nbhd{sm_nbhd_size}_{out_yr}_mask.tif'},
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
        if params['refine']['spatial_filter'] == 'smCrop':
            if not params['refine']['sm_neighborhood']:
                params['refine']['sm_neighborhood'] = 3
            sm_nbhd_size = params['refine']['sm_neighborhood']
            
            map_filters = { 
                    'base_map' : {'full':f'{final_dir}/CELPyTileAll_{mod_base}'},
                    'polys_maj' : {'full':f'{comp_dir_out}/CELseg{out_yr}-majority_rcsmall.tif'},
                    'polys_buf' : {'full':f'{comp_dir_out}/CELseg{out_yr}-majority_buf{buf}_rcsmall.tif'},
                    'polys_area' : {'full':f'{comp_dir_out}/polyarea{out_yr}-majority.tif'},
                    'area_focal' : {'full':f'{comp_dir_out}/field_area_focal100avg_int0.tif'},
                    'highveg_nbhd': {'full':f'{comp_dir_out}/highveg{out_yr}_nbhd.tif'},
                    'sm_neighbors': {'full':f'{comp_dir_out}/sm_neighbors{out_yr}.tif'},
                    'sm_nbhd_mask': {'full':f'{comp_dir_out}/sm_nbhd{sm_nbhd_size}_{out_yr}_mask.tif'},
                    'sm_nbhd_dist': {'full':f'{comp_dir_out}/sm_nbhd{out_yr}_dist.tif'}
                    }           
        make_full_filter_layers(params, filter_set=map_filters)
    
    if params['refine']['spatial_filter'] == 'smCrop':
        post_classification_spatial_filter_smallholder(params, filter_set=map_filters)


def filter_temporal_noise_from_stable_cats(ts_files, LC_CATS, cat, cat_idx, params, count_cache):
    '''
    Returns the class with the majority count from a set of classes for each pixel, unless one of the defined
    exceptions below applies. Treats sets of classes that are theoretically stable (unlikely short-term
    fluctuations) but easily confused with one another due to slight positional or temporal fluctuations by
    applying the most frequent occurrence from the set (e.g. water/wet_veg or wet vs dry grass).
    '''
    logger.info(f'getting stable base for {cat}')
    with gw.open(ts_files) as ts:
        attrs = ts.attrs.copy()

    ##########  Exceptions to simple majority rule:
    ## shrub forest if ever classified as shrub forest, or if classified as both dense forest & shrub
    ## and (optionally) in a specified region (e.g. Chaco in Paraguay)
    if cat in ['shrub_forest', 'shrub_for']:
        nshrubfor = store_count('shrub_for', count_cache, lambda: (ts.isin(LC_CATS['shrub_for'])).sum(dim="time").astype('uint8'))
        nshrub = store_count('shrub', count_cache, lambda: (ts.isin(LC_CATS['shrub'])).sum(dim="time").astype('uint8'))
        ndensefor = store_count('dense_for', count_cache, lambda: (ts.isin(LC_CATS['dense_for'])).sum(dim="time").astype('uint8'))

        region_cond = (ndensefor > 0) & (nshrub > 0)
        apply_to = params['refine']['stable_regions'][cat_idx]
        if is_region_filter_active(apply_to):
            region_filt = get_regional_filter(cat_idx, ts_files[0], params['refine']['stable_regions'], params['refine']['stable_region_file'])
            region_cond = region_cond & (region_filt == 1)

        base_cat = np.where((nshrubfor > 0) | region_cond, LC_CATS['shrub_for'][0], 0)

    elif cat in ['palm_forest', 'palm_for']:
        ## palm forest if majority out of palm-forest, grass-tree-mix and med-crop
        npalmfor = store_count('palm_for', count_cache, lambda: (ts.isin(LC_CATS['palm_for'])).sum(dim="time").astype('uint8'))
        ngtmix = store_count('gtmix', count_cache, lambda: (ts.isin(LC_CATS['gtmix'])).sum(dim="time").astype('uint8'))
        nmedcrop = store_count('med_crops', count_cache, lambda: (ts.isin(LC_CATS['med_crops'])).sum(dim="time").astype('uint8'))
        base_cat = np.where((npalmfor > ngtmix) & (npalmfor > nmedcrop), LC_CATS['palm_for'][0], 0)

    elif cat in ['open_forest', 'open_for']:
        ## open forest if open_forest or shrub for at least 25% of ts and dense forest for the rest
        nopenfor = store_count('open_for', count_cache, lambda: (ts.isin(LC_CATS['open_for'])).sum(dim="time").astype('uint8'))
        nshrub = store_count('shrub', count_cache, lambda: (ts.isin(LC_CATS['shrub'])).sum(dim="time").astype('uint8'))
        ndensefor = store_count('dense_for', count_cache, lambda: (ts.isin(LC_CATS['dense_for'])).sum(dim="time").astype('uint8'))
        base_cat = np.where(
            ((nopenfor + nshrub) >= 2) & (ndensefor == (len(ts.time) - (nopenfor + nshrub))),
            LC_CATS['open_for'][0], 0
        )

    ##########  simple majority rule:
    else:
        base_cat = get_most_frequent_cat_in_timeseries(cat, ts, cat_dict=LC_CATS)

    return np.squeeze(base_cat) if isinstance(base_cat, np.ndarray) else base_cat.squeeze()

def get_stable_base(ts_files, ts_yrs, params, count_cache):
    '''
    Get base LC for time series (best representation for a given input array, assuming LC is actually stable
    across time). In the final filter, this is only applied to certain classes that should be stable logically.
    '''
    logger.info('getting base lc for time series...')
    if params['project_ver'] == 'Py_0':
        LC_CATS = LC_CATS_Py0

    if params['project_name'].lower().startswith('paraguay'):
        params['refine']['stable_group'] = ['shrub_for', 'palm_for', 'open_for', 'dense_for', 'wet', 'grass_Py36']
        params['refine']['stable_regions'] = [1, 'All', 'All', 'All', 'All', 'All']

    ## Get max class for theoretically stable classes:
    base_rasters = {}
    for i, cat in enumerate(params['refine']['stable_group']):
        ras_path = Path(params['scratch_dir']) / f"tsfilters/{params['refine']['mod_prescript']}/{cat}.tif"
        if ras_path.exists():
            logger.info(f'found existing {cat} file')
            with rio.open(ras_path) as src:
                ras = src.read(1)
            base_rasters[cat] = ras
        else:
            base_rasters[cat] = filter_temporal_noise_from_stable_cats(ts_files, LC_CATS, cat, i, params, count_cache)

    ## if >1 type of forest listed, base value for pixels with >1 forest type in ts is assigned based on
    ## order of forest types listed (not counts). If most-frequent should win instead, use a more generic
    ## forest category that includes all types (e.g. ['forest'] rather than ['shrub_forest', 'palm_forest'])
    forest_path = Path(params['scratch_dir']) / f"tsfilters/{params['refine']['mod_prescript']}/forest.tif"
    if forest_path.exists():
        logger.info(f'found existing for_cat file at {forest_path}')
        with rio.open(forest_path) as src:
            ras = src.read(1)
        base_rasters['forest'] = ras
    else:
        forest_priority = [c for c in params['refine']['stable_group'] if (c.endswith('_for') or c.endswith('_forest'))]
        if len(forest_priority) == 1:
            base_for = base_rasters[forest_priority[0]]
        elif len(forest_priority) > 1:
            base_for = np.where(base_rasters[forest_priority[0]] > 0, base_rasters[forest_priority[0]], base_rasters[forest_priority[1]])
            for p in range(2, len(forest_priority)):
                base_for = np.squeeze(base_for)
                other_for = np.squeeze(base_rasters[forest_priority[p]])
                base_for = np.where(base_for > 0, base_for, other_for)
        base_rasters['forest'] = base_for

    if params['refine']['save_inter'] == True:
        with rio.open(ts_files[0]) as src0:
            meta = src0.meta.copy()
        meta.update(count=1, compress='LZW', tiled=True)

        for b in base_rasters.keys():
            base_dir = Path(params['scratch_dir']) / f"tsfilters/{params['refine']['mod_prescript']}"
            base_dir.mkdir(parents=True, exist_ok=True)
            base_ras = Path(base_dir) / f'{b}.tif'
            arr = np.squeeze(base_rasters[b])
            with rio.open(base_ras, 'w', **meta) as dst:
                dst.write(arr, 1)

    logger.info(f'created the following base rasters for ts refinement: {list(base_rasters.keys())}')
    return base_rasters

def filter_ts_rasters(ts_files, ts_yrs, base_rasters, params, count_cache):

    if params['project_ver'] == 'Py_0':
        LC_CATS = LC_CATS_Py0
    if params['project_name'].lower().startswith('paraguay'):
        params['refine']['illogical'] = [
            'sugar-palm', 'sugar-grass', 'banana-wet', 'palm_for-grass-aggressive', 'palm_for-wetgrass',
            'grass-forest-aggressive', 'tree_plant-med_crop', 'rice-water', 'rice-built',
            'forest-treeplant', 'forest-brieflow', 'grass-to-forest', 'grass-to-palmforest',
            'twmix-medwet', 'noplant-plant',
        ]
        params['refine']['illogical_regions'] = [0, 0, [2,3], [2,3], 0, 0, 0, 0, 0, 0, 0, 0, [1, 4], 0, 0]
        params['refine']['illogical_region_file'] = params['refine']['stable_region_file']
        params['refine']['group_suffix'] = 'Py36'

    ctx = FilterTsArgs(ts_files, ts_yrs, params, LC_CATS, base_rasters, count_cache)
    
    with gw.open(ts_files, time_names=ts_yrs, stack_dim='time') as ts:
        attrs = ts.attrs.copy()

    group_suffix = ""
    if params['refine']['group_suffix']:
        group_suffix = params['refine']['group_suffix']

    ####### Stabilize land covers within logical groups:
    ## if pixel is forest, set to base forest type:
    if 'forest' in base_rasters:
        ts = ts.where(
            (~ts.isin(ctx.LC_CATS[f'forest_{group_suffix}'])) | (base_rasters['forest'] < ctx.LC_CATS['first_mature']),
            base_rasters['forest']
        )
    ## if pixel is grass, set to base grass type:
    if f'grass_{group_suffix}' in base_rasters:
        ts = ts.where(
            (~ts.isin(ctx.LC_CATS[f'grass_{group_suffix}'])) | (base_rasters[f'grass_{group_suffix}'] == 0),
            base_rasters[f'grass_{group_suffix}']
        )
    ## if pixel is wet, set to base wet type IF change is NOT stable for 2 yrs:
    if 'wet' in base_rasters:
        ts = ts.where(
            (~ts.isin(ctx.LC_CATS['wet'])) | (base_rasters['wet'] == 0) |
            (ts.shift(time=1).fillna(0) == ts) | (ts.shift(time=-1).fillna(0) == ts),
            base_rasters['wet']
        )

    for idx, problem_pair in enumerate(params['refine']['illogical']):
        fix_fn = CORRECTIONS.get(problem_pair)
        if fix_fn is not None:
            ts = fix_fn(ts, idx, ctx)
            #ts = ts.compute()  # TEMP: force eager eval to localize the failure
            #ts = ts.chunk()    # re-chunk if downstream code expects dask array
            #logger.info(f'{problem_pair} OK')
        else:
            logger.warning(f'no correction defined for illogical pair: {problem_pair}')

    return ts


def ts_filter(params):
    '''
    Applies time-series filter to multiple years of classified data
    uses logical rules to stabilize certiain land cover classes
    These rules are implemented through get_stable_base() and filter_ts_rasters()
    Prints out filtered time series as single yearly rasters ending in '-tsfilt.tif'
    '''
    logger.info('applying ts filters...')
    if params['classify']['mod_dir']:
        comp_dir_in = params['classify']['mod_dir']
    else:
        comp_dir_in = Path(params['scratch_dir']) / f'classified'
    
    if params['classify']['test']:
        final_dir = Path(params['scratch_dir']) / f'classified'
    else:
        final_dir = Path(params['backup_path']).parents[1] / 'mosaics' 
    final_dir.mkdir(parents=True, exist_ok=True)

    if params['refine']['spatial_filter']:
        if params['refine']['spatial_filter'] == 'smCrop':
            params['refine']['mod_postscript'] = 'filtsmh-sp'
    postscript = params['refine']['mod_postscript']
    prescript = params['refine']['mod_prescript']

    out_yrs = list(range(params['classify']['out_yrs'][0], params['classify']['out_yrs'][1]+1))
    logger.info(f'filtering time series based on years: {out_yrs}')               
    
    ts_files = sorted([str(f) for f in comp_dir_in.glob(f"{prescript}*{postscript}.tif")])
    found_yrs = []
    for file in ts_files: 
        found_yrs.append(int(file.split('_')[-2])) 
    logger.info(f'found inputs for {found_yrs}')
    complete = True
    for yr in out_yrs:
        if yr not in found_yrs:
            logger.warning(f'OOPS -- did not find raster for {yr} in {comp_dir_in}')
            complete = False
    if complete == True:
        for file in ts_files: 
            if int(file.split('_')[-2]) not in out_yrs:
                ts_files.remove(file)
        count_cache = {}
        base_rasters = get_stable_base(ts_files, out_yrs, params, count_cache)
        refined_ts = filter_ts_rasters(ts_files,out_yrs,base_rasters,params, count_cache)

        for y in out_yrs:
            logger.info(f'getting final raster for {y}...')
            out_file = Path(final_dir)/f'{prescript}_{y}_{postscript}-tsfilt.tif'
            ts_single = refined_ts.sel(time=y).squeeze().fillna(0).astype('uint8')
            ts_single = mark_forest_edges(ts_single, params)
            with gw.config.update(ref_image=ts_files[0], nodata=0):
                ts_single.gw.save(out_file,num_workers=params['num_workers'],compression='lzw',overwrite=True)
    
    
