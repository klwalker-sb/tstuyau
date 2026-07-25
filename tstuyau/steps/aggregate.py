import sys
from pathlib import Path
import datetime as dt
import rasterio as rio
from rasterio.merge import merge
from rasterio.io import MemoryFile
import numpy as np
import geowombat as gw
import xarray as xr
import pandas as pd
import csv
import glob
import math
#import re
#from shapely.geometry import box
from .project import ProjectPaths, get_tsdir_name
from ..handler import logger
from .date_utils import get_date_range
from .check_reconstruction import reconstruct
from .pheno import prep_pheno_bands, prep_ts_variable_bands
from .texture import make_glcm


def get_monthly_ts(si_vars, img_dir, start_yr, start_mo, comp_band_names, ras_list, ts_type=None):

    '''
    This is just to pull single image per month from time series. It is called if the month is the variable (the first component of <si_var>,
    not the aggregation factor (the second component of si_var). To calculate monthly statistics, we get the dates with get_date_range in date_utils.
    This currently pulls the image from the 20th of the month (or the closest to it if a raw ts). TODO: allow for other values, like the 1st. 
    '''

    MONTH_DEFS = {
    'Jan': {'mo_num': 1, 'doy_20': ('020',)},
    'Feb': {'mo_num': 2, 'doy_20': ('051',)},
    'Mar': {'mo_num': 3, 'doy_20': ('079', '080')},
    'Apr': {'mo_num': 4, 'doy_20': ('110', '111')},
    'May': {'mo_num': 5, 'doy_20': ('140', '141')},
    'Jun': {'mo_num': 6, 'doy_20': ('171', '172')},
    'Jul': {'mo_num': 7, 'doy_20': ('201', '202')},
    'Aug': {'mo_num': 8, 'doy_20': ('232', '233')},
    'Sep': {'mo_num': 9, 'doy_20': ('263', '264')},
    'Oct': {'mo_num': 10, 'doy_20': ('293', '294')},
    'Nov': {'mo_num': 11, 'doy_20': ('324', '325')},
    'Dec': {'mo_num': 12, 'doy_20': ('354', '355')}
    }
    
    current_year_str = str(start_yr)
    next_year_str = str(int(start_yr) + 1)

    all_imgs = sorted(list(img_dir.glob('*.tif')))
    
    months_to_run = {}
    ## Check whether each month is in any of the si_vars before running it
    for mo_name, mo_defs in MONTH_CONFIGS.items():       
        if any(var.startswith(mo_name) for var in si_vars):
            months_to_run[mo_name] = mo_defs

    for mo_name, mo_defs in months_to_run.items():
        ## Determine the target year prefix for this month (in case the year doesn't start Jan 1st)
        target_year_str = current_year_str if start_mo <= mo_defs['mo_num'] else next_year_str
        candidate_images = [img for img in all_imgs if img.stem.startswith(target_year_str)]
        if not candidate_images:
            logger.warning(f"No images found in {mo_name} for year {target_year_str}")
        else:
            if f'{mo_name}-20' in si_vars:
                ## just grab the image that corresponds to the 20th of the month (or closest if using raw time series)
                if (not ts_type) or (ts_type.startswith('sm')):
                    for img in all_imgs:
                        im = img.stem
                        if im.startswith(target_year_str) and im.endswith(mo_defs['doy_20']):
                            ras_list.append(img)
                            comp_band_names.append(f'{mo_name}-20')
                            logger.info(f"added {mo_name}-20")   
                elif ts_type == 'raw':
                    best_img = None
                    smallest_distance = float('inf')

                    # Evaluate candidates to find the one closest to the 20th doy
                    for img in candidate_images:
                        try:
                            img_suffix_num = int(img.stem[len(target_year_str):])
                        except ValueError:
                            continue # Skip files without clean numbers at the end
            
                        ## Find the absolute distance to the nearest target suffix
                        ## (e.g. if suffixes are 79 and 80, and file is 82, distance is 2)
                        this_dist = abs(img_suffix_num - mo_defs['doy_20'][0])
                        # If this image is closer than anything we've seen before, save it
                        if this_dist < smallest_distance:
                            smallest_distance = this_dist
                            best_img = img

                    if best_img:
                        ras_list.append(best_img)
                        comp_band_names.append(mo_name)
                        logger.info(f"added {mo_name} ({smallest_distance} days from 20th)")
    
    return comp_band_names,ras_list

def get_image_stack(params, temp, img_dir):
        
    model_yr = int(params['feature_model']['start_yr'])
    ts_stack = []
    ds_stack = []
    if (params['feature_model']['use_pheno'] and params['feature_model']['pheno_pad_days'] and params['feature_model']['pheno_pad_days'] != [0,0]):
        padding = True
        pad_days = params['feature_model']['pheno_pad_days']
        ts_stack_padded = []
        ds_stack_padded = []
    else:
        padding = False
    start, end = get_date_range(model_yr,temp,params,return_type='doy',padded=False)
    logger.info(f"using images for {temp} from {start} to {end} \n")

    logger.info(f"looking in {img_dir}...")
    for img in sorted(img_dir.rglob("*.tif")):
        ## ts images are named YYYYdoy.tif
        imgdt = int(img.stem)
        logger.debug(f'imgdt: {imgdt}')
        img_date = pd.to_datetime(img.stem,format='%Y%j')
            
        if (imgdt >= start) and (imgdt <= end):
            ts_stack.append(str(img))
            ds_stack.append(img_date)
                
    logger.debug(f'stack length for {temp}: {len(ds_stack)}')
    if padding:
        padded_start, padded_end = get_date_range(model_yr,temp,params,return_type='doy',padded=True)
        if (imgdt >= padded_start) and (imgdt <= padded_end):
            ts_stack_padded.append(str(img))
            ds_stack_padded.append(img_date)
        
    return ts_stack, ds_stack, ts_stack_padded, ds_stack_padded
     
def make_ts_composite_single(ppaths, params):
    '''
    makes ts composite for a single year
    '''

    si = params['feature_model']['spec_indices'][0]  ##Only working for first index for now. TODO: loop through all sis in list
    si_vars = params['feature_model']['si_vars']

    if (len(si_vars) == 1):
        si_stat = si_vars[0].split('-')[0]
        if 'Monthly' in si_vars[0] or 'monthly' in si_vars[0]:
            logger.info(f'Making monthly {si_stat} composite')
            all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            offset = params['calendar']['first_mo'] - 1
            ## reorder so that band sequence will start at start_mo if making monthly composite
            reordered_months = all_months[offset:] + all_months[:offset]
            si_vars = [f'{si_stat}-{mon}' for mon in reordered_months]
        elif 'Quarterly' in si_vars[0] or 'quarterly' in si_vars[0]:
            logger.info(f'Making quarterly {si_stat} composite')
            si_vars = [f'{si_stat}-Q1',f'{si_stat}-Q2',f'{si_stat}-Q3',f'{si_stat}-Q4']
    
    model_yr = int(params['feature_model']['start_yr'])
    start_mo = params['calendar']['first_mo']
    start_doy = int(30.5 * int(start_mo)) - 30
    nodata_in = params['reconstruct']['nodata']
    tmpout_dir = ppaths.scratch/si/str(model_yr)
    tmpout_dir.mkdir(parents=True, exist_ok=True)

    ## the following parameters are only needed for pheno variables, and only for some cases
    sigdif = None
    if params['feature_model']['pheno_sigdif']:
        sigdif = params['feature_model']['pheno_sigdif']
    basethresh_pre = None    
    if params['feature_model']['pheno_basethresh_pre'] is not None:
        basethresh_pre = params['feature_model']['pheno_basethresh_pre']
    if params['feature_model']['pheno_basethresh_post'] is not None:
        basethresh_post = params['feature_model']['pheno_basethresh_post'] 
    imgbuf = None
    if params['feature_model']['pheno_imgbuf'] is not None:
        imgbuf = params['feature_model']['pheno_imgbuf']
        
    if '-' in si:
        tst = si.split('-')[1]
        if 'raw' in tst:
            ts_type = 'raw'
        elif tst.startswith('sm'):
            ts_type = 'smooth'
        else:
            logger.warning(f"ERROR: problem parsing si_var. ts_type should br 'raw' or 'sm...'. Got {tst}")
    elif 'raw' in si:
        ts_type = 'raw'
    else:
        ts_type = 'smooth'

    ### DO THIS
    ###if __ == 'monthly'...
    ###    params['reconstruct']['overwrite']
    
    ras_list = []
    comp_band_names = []
    gw_args = {'verbose':1,'n_workers':4,'n_threads':1,'n_chunks':200, 'gdal_cache':64,'overwrite':True}

    ts_root = get_tsdir_name(params)
    if ts_type == 'raw':
        ## If time series is raw, will run reconstruction to calculate si for all images in time period (with pheno pad if set) 
        ##    and output them to a tmp folder. all images in this tmp folder are then used for the calculation of the statistic.
        for i, var in enumerate(si_vars):
            params['feature_model']['si_vars'] = [var]
            logger.info(f"working on unsmoothed brdf images for {var}")
            ts_stack = []
            ds_stack = []
            season = var.split('-')[1]
            img_dir = ppaths.scratch / 'raw' / ts_root / si / str(model_yr)
            params['reconstruct']['si'] = si
            params['reconstruct']['start'] = params['feature_model']['start_yr']
            if params['calendar']['first_mo']>1:
                params['reconstruct']['end'] = params['feature_model']['start_yr']+1
            else:
                params['reconstruct']['end'] = params['feature_model']['start_yr']
            
            ## only remake indices if using different temporal period for aggregate statistic
            ##    currently only works for single year. TODO: make work with multiple yrs in 'classify''yrs_out'
            if params['reconstruct']['overwrite']:
                if i > 0:
                    former_season = si_vars[i-1].split('-')[1]
                    season = var.split('-')[1]
                    if season == former_season:
                        params['reconstruct']['overwrite'] = False
            logger.debug(f"overwrite = {params['reconstruct']['overwrite']}")
        
            if params['reconstruct']['overwrite'] or (not img_dir.is_dir()) or (not any(img_dir.glob('*.tif'))):
                ## calculate the indices for selected time period -- these are sent to ppaths.scratch/raw/sis
                logger.info(f'calculating new raw time series for {si} index in {img_dir}')
                reconstruct(params)
            
            ## read the images in the img_dir to get the aggregated bands for the composite
            
            images = list(img_dir.rglob("*.tif"))
            
            if len(images) == 0:
                logger.warning("OOPS -- no images have been created in the temp index directory")

            else:
                logger.info(f"there are {len(images)} images in {img_dir}") 
                
                for img in sorted(images):
                    img_date = pd.to_datetime(img.stem,format='%Y%j')
                    ts_stack.append(str(img))
                    ds_stack.append(img_date)

                logger.debug(f'the set of images being used for this band are from: {ds_stack}')

                if params['feature_model']['use_pheno']:
                    comp_band_names,ras_list = prep_pheno_bands(var, ts_stack, ds_stack, ts_stack, ds_stack,tmpout_dir,model_yr,
                        season, start_doy, comp_band_names, ras_list, sigdif=sigdif, 
                        basethresh_pre=basethresh_pre, basethresh_post=basethresh_post, imgbuf=imgbuf, **gw_args)

                else:
                    comp_band_names,ras_list = prep_ts_variable_bands(
                        var, ts_stack, ds_stack, tmpout_dir,season,start_doy, comp_band_names, ras_list, nodata_in, ppaths, **gw_args)

    elif ts_type == 'smooth':
        ## get stack from images in smoothed time-series directory that match temporal period of interest
        img_dir = ppaths.ts / si

        ## Calculate statistics for each time period

        ## if temp not specified in si_var, treat as year:
        yr_bands = [b for b in si_vars if ("-" not in b) and ("_" not in b)]
        if len(yr_bands) > 0:
            logger.info(f"calculating {annual} bands: {yr_bands}...")
            ts_stack, ds_stack, ts_stack_padded, ds_stack_padded = get_image_stack(params,'yr',img_dir)
            if params['feature_model']['use_pheno']:
                comp_band_names,ras_list = prep_pheno_bands(annual_bands, ts_stack, ds_stack, ts_stack_padded, ds_stack_padded, 
                    tmpout_dir,model_yr,'yr', start_doy, comp_band_names, ras_list, sigdif=sigdif, basethresh_pre=basethresh_pre, 
                    basethresh_post=basethresh_post, imgbuf=imgbuf, **gw_args)
            else:
                comp_band_names,ras_list = prep_ts_variable_bands(annual_bands, ts_stack, ds_stack, 
                                                                  tmpout_dir,temp,start_doy, comp_band_names, ras_list, nodata_in, ppaths, **gw_args)

        all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        offset = params['calendar']['first_mo'] - 1
        ## reorder so that band sequence will start at start_mo if making monthly composite
        reordered_months = all_months[offset:] + all_months[:offset]
        logger.info(f'month order will be {reordered_months}')
        all_quarters = ['Q1','Q2','Q3','Q4']  ## nothe these are already defined starting at start_mo
        all_temps = ['yr','wet','dry'] + reordered_months + all_quarters
        
        for temp in all_temps:
            bands = [b for b in si_vars if ("_" in b and b.split("_")[1] == temp) or ("-" in b and b.split("-")[1] == temp)]
            if len(bands) > 0:
                logger.info(f"calculating {temp} bands: {bands}...")
                ts_stack, ds_stack, ts_stack_padded, ds_stack_padded = get_image_stack(params,temp, img_dir)
                if params['feature_model']['use_pheno']:
                    comp_band_names,ras_list = prep_pheno_bands(bands, ts_stack, ds_stack, ts_stack_padded, ds_stack_padded, 
                        tmpout_dir,model_yr,temp, start_doy, comp_band_names, ras_list, sigdif=sigdif, basethresh_pre=basethresh_pre, 
                        basethresh_post=basethresh_post, imgbuf=imgbuf, **gw_args)
                else:
                    comp_band_names,ras_list = prep_ts_variable_bands(bands, ts_stack, ds_stack, 
                                                                  tmpout_dir,temp,start_doy, comp_band_names, ras_list, nodata_in, ppaths, **gw_args)

        ## to get an example image of each month (from the 20th), the SI variable is written with month first (as statistic) followed by -20 (e.g. Jan-20) 
        mo_bands = [b for b in si_vars if ("-" in b and b.split("_")[1] == '20') or ('-' in b and b.split("-")[1] == '20')]
        if len(mo_bands) > 0:
            comp_band_names,ras_list = get_monthly_ts(mo_bands, img_dir, model_yr, start_mo, comp_band_names, ras_list)

    glcm_vars = [siv for siv in si_vars if 'glcm' in siv]
    if len(glcm_vars) > 0:
        logger.info('making glcm variables...\n')
        for gv in glcm_vars: 
            no_glcm = gv.split('.')[0]
            if '-' in gv:
                no_glcm = f"{no_glcm}-{gv.split('-',1)[1]}"
            list_loc = comp_band_names.index(no_glcm)
            logger.info(f'found base variable {no_glcm} at location {list_loc} in stack')
            out_dir=Path(params['scratch_dir'])/'temp_glcms'
            out_dir.mkdir(parents=True, exist_ok=True)
            out = Path(out_dir)/f"{gv.split('.')[0]}_{gv.split('.')[2]}.tif"
            glcm_band = make_glcm(base_img=ras_list[list_loc], params=params, si_var=gv, print_out=True, out_path=out)
            ras_list[list_loc] = glcm_band
            comp_band_names[list_loc] = gv
            logger.info(f'new band list: {comp_band_names}')

        
    return comp_band_names,ras_list

    
def make_ts_composite(params):

    '''
    Makes aggregated statistics (e.g. mean, stdv, cv, amp, etc.) for all images in specified time period ('yr', 'wet', or 'dry') 
    
    Statistics to be generated are set with the <feature_model:si_vars> parameter or <feature_model:pheno_vars> parameter
        in combination with <feature_model:spec_indices> or <feature_model:spec_indices_pheno> to set the vegetation index 
        (Note this function does not work with multiple spec indices; if a list is given, only the first item will be used)
        The <feature_model:start_yr> parameter is used to set the nominal year (first year if mapping period spans two nominal years). 
        The specific dates for the mapping year and seasons are set with the <calendar> parameters
        The <feature_model:pheno_pad_days> param can be set to allow for some overlap between seasons when fitting curve tails.
             this is a list [l,r] with l as padding (in days) into left side and r as padding into right side.

    Can make a composite of the same variable over multiple years by setting the <classify:out_yrs> parameter to multiple years
    Can make a composite from unsmoothed images in the brdf folder by adding -raw to the si_var (e.g. nbr-raw)
    To make composites from the time-series folders, should add -sm.. to the si_var (e.g. nbr-smwh)
    '''
    
    if params['feature_model']['use_pheno']:
        params['feature_model']['si_vars'] = params['feature_model']['pheno_vars']
        logger.info(f"making pheno variables {params['feature_model']['si_vars']}...")
        if isinstance(params['feature_model']['spec_indices_pheno'],str):
            params['feature_model']['spec_indices'] = [params['feature_model']['spec_indices_pheno']]
        else:
            params['feature_model']['spec_indices'] = params['feature_model']['spec_indices_pheno']
    else:
        if isinstance(params['feature_model']['spec_indices'],str):
            params['feature_model']['spec_indices'] = [params['feature_model']['spec_indices']]

    if len(params['feature_model']['spec_indices']) > 1:
        logger.warning("WARNING: all but the first spec index will be ignored")

    si = params['feature_model']['spec_indices'][0]

    ## sipre is usually '' unless compare params are set to compare models with different image_type, resolution and/or procseq
    sipre = get_tsdir_name(params)
    if sipre == '' or sipre == 'ms':
        si_full = si
    else:
        si_full = f'{sipre}_{si}'
        
    si_vars0 = params['feature_model']['si_vars']
    if isinstance(si_vars0, str):
        si_vars0 = [si_vars0]
    mod_yr = params['feature_model']['start_yr']
    logger.debug(f"si_vars = {si_vars0}")
    
    if isinstance(params['grids'],int):
        params['grids'] = [params['grids']]
    for cell in params['grids']:
        ppaths = ProjectPaths(params, grid=cell)
        if params['feature_model']['treat_out'] == 'archive':
            out_dir = ppaths.comp
        elif params['feature_model']['treat_out'] == 'tmp':
            out_dir = ppaths.scratch / 'comp'

        out_dir.mkdir(parents=True, exist_ok=True)

        if not params['classify']['out_yrs'] or (isinstance(params['classify']['out_yrs'],int)) or (
            len(params['classify']['out_yrs']) == 1):
            logger.info(f'making single year composite for cell {cell}...')
            
            comp_band_names, ras_list = make_ts_composite_single(ppaths, params)
            logger.info(f' ras_list: {ras_list}')
            logger.info(f' band_names: {comp_band_names}')

            if len(ras_list) < len(si_vars0):
                logger.warning('OOPS -- got an unknown band')
            elif (len(si_vars0) == 1) and ('Monthly' in si_vars0[0]) and(len(ras_list)) < 12:
                logger.warning(f'UH_OH -- montlhy series only has {len(ras_list)} bands')
            elif (len(si_vars0) == 1) and ('Quarterly' in si_vars0[0]) and(len(ras_list)) < 4:
                logger.warning(f'UH_OH -- quarterly series only has {len(ras_list)} bands')
                
            else:
                if (len(si_vars0) == 1) and (('Monthly' in si_vars0[0]) or ('Quarterly' in si_vars0[0])):
                    out_ras = f"{out_dir}/{int(cell):06d}_{mod_yr}_{si_full}_{si_vars0[0]}.tif"
                    ## simplify band names
                    comp_band_names = [s.split('.')[0] + '-' + s.split('-')[-1] if '.' in s else s for s in comp_band_names]
                elif params['feature_model']['use_pheno']:
                    out_ras = f"{out_dir}/{int(cell):06d}_{mod_yr}_{si_full}_{'-'.join(comp_band_names)}_Phen.tif"
                else:
                    if len(ras_list)>12:
                        out_ras = f"{out_dir}/{int(cell):06d}_{mod_yr}_{si_full}_ModVars.tif"
                    elif len(ras_list)==12:
                        out_ras = f"{out_dir}/{int(cell):06d}_{mod_yr}_{si_full}_monthly20.tif"
                    else:
                        out_ras = f"{out_dir}/{int(cell):06d}_{mod_yr}_{si_full}_{'-'.join(comp_band_names)}.tif"
                        
            ## Start writing output composite
            with rio.open(ras_list[0]) as src0:
                meta = src0.meta
                meta.update(count = len(ras_list))

            with rio.open(out_ras, 'w', **meta) as dst:
                for id, layer in enumerate(ras_list, start=1):
                    with rio.open(layer) as src1:
                        dst.write(src1.read(1),id)
                    dst.descriptions = tuple(comp_band_names)

            logger.info(f'final composite written to {out_ras}')
                
        else:
            logger.info(f'making multi year composite for cell {cell}...')
            full_ras_list = []
            #full_comp_band_names = []

            if len(si_vars) > 1:
                logger.warning(' OOPS -- can currently only create multi-year composites with single si variable and index')

            else:
                for yr in params['classify']['out_yrs']:
                    params['feature_model']['start_yr'] = yr
                    band_names, ras_list = make_ts_composite_single(ppaths, params)
                    full_ras_list.extend(ras_list)
                    #full_comp_band_names.extend(band_names)

                if len(full_ras_list) != len(params['classify']['out_yrs']):
                    logger.warning('  OOPS -- band number does not match')
                        
                else:
                    logger.debug(f'ras_list:{full_ras_list}')
                    comp_band_names = [str(yr) for yr in params['classify']['out_yrs']]
                    #logger.debug(f'comp_band_names:{full_comp_band_names}')
                    logger.info(f'writing stack for si_vars:{si_vars}')

                    ## Start writing output composite
                    with rio.open(full_ras_list[0]) as src0:
                        meta = src0.meta
                        meta.update(count = len(full_ras_list))

                    var_name = comp_band_names[0]
                    out_ras = f"{out_dir}/{int(cell):06d}_{si_full}_{var_name}_{'-'.join(comp_band_names)}.tif"

                    with rio.open(out_ras, 'w', **meta) as dst:
                        for id, layer in enumerate(ras_list, start=1):
                            with rio.open(layer) as src1:
                                dst.write(src1.read(1),id)
                        dst.descriptions = tuple(comp_band_names)

    return out_ras, comp_band_names
    

def mosaic_cells(params, out_path=None):
    '''
    Mosaics classified results for a set of cells defined with a list or path to a .csv file with a cell number on each line.
    if a .csv file, the file basename will be start the basename of the output mosaic. 
    The <classify:name> parameter is the common string in the filenmaes of the cell products to be mosaicked (usually the full model name)
    and will comprise the rest of the output mosaic name. The output mosaic is saved to the 'classified' directory on the backup drive,
    unless <classify:test> is set to true, in which case it will be saved to the scratch drive. 

    If a buffer distance was used when creating data for each grid cell (<buffer> > 0), will unbuffer cells before mosaicking to remove edge effects
      make sure <buffer> param is set to original value
    '''
    if out_path:
        out_dir = Path(out_path).parent
    elif params['classify']['test']:
        out_dir = Path(params['scratch_dir']) / 'classified'
    else:
        out_dir = Path(params['backup_path']).parents[1] / 'mosaics'
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(params['grids'], int):
        params['grids'] = [params['grids']]
    if isinstance(params['grids'], list):
        cells = params['grids']
        if not out_path:
            out_path = Path(out_dir) / f"{params['classify']['name']}_mosaic.tif"
    elif params['grids'].endswith('.csv'):
        if not out_path:
            out_path = Path(out_dir) / f"{Path(params['grids']).stem}_{params['classify']['name']}.tif"
        cells = []
        with open(params['grids'], newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])
    else:
        logger.warning('cell_list needs to be a list or path to .csv file with list')

    logger.debug(f"mosaicking cells:{cells}")
    ras_list = []
    for cell in cells:
        ppaths = ProjectPaths(params, grid=int(cell))
        if params['classify']['comp_dir'] == 'input_dir':
            comp_path = ppaths.ms.parent / 'comp'
        elif params['classify']['comp_dir'] == 'backup':
            comp_path = ppaths.comp / params['classify']['local_dir']
        elif params['classify']['comp_dir'] == 'tmp':
            comp_path = ppaths.scratch  / 'comp'
        elif params['classify']['comp_dir'].is_dir():
            comp_path = Path(params['classify']['comp_dir']).parent / f'{int(cell):06d}'
        else: 
            logger.warning("comp_dir must be main, backup, temp, or an actual directory. You put {params['classify']['comp_dir']}")
        
        logger.debug(f'Looking in {comp_path} for individual inputs')
        if not comp_path.is_dir():
            logger.warning(f"there is no comp folder {comp_path} for cell {cell}.")
        else:
            matches = glob.glob(str(comp_path) + f"/*{params['classify']['name']}*.tif")
            if len(matches) == 0:
                logger.warning(f"no raster was created for cell {cell} for model {params['classify']['name']}.")
            elif len(matches) > 1:
                logger.warning(f"more than one raster containing {params['classify']['name']} was found for cell {cell}. -- using first match")
                ras_list.append(matches[0])
            else:
                ras_list.append(matches[0])
                
    logger.debug(f' ras_list = {ras_list} \n')
    logger.info(f"mosaicking {len(ras_list)} images...\n")
        
    with rio.open(ras_list[0], 'r') as src_exmp:
        out_meta = src_exmp.meta.copy()
        band_names = src_exmp.descriptions
    logger.debug(f"these are {src_exmp.meta['count']}-band rasters")
            
    if params['buffer']:
        ## unbuffer to remove edge effects
        #res = params['res']
        src_datasets = []
        mem_files = []
        
        for ras in ras_list:
            with rio.open(ras) as src:
                res = src.res[0]
                extrapix_x = math.ceil(params['buffer'] / res)
                extrapix_y = math.ceil(params['buffer'] / res)
                
                window = rio.windows.Window(
                    col_off=extrapix_x,
                    row_off=extrapix_y,
                    width=src.width - (2 * extrapix_x),
                    height=src.height - (2 * extrapix_y),
                )
                kwargs = src.meta.copy()
                crop_transform = rio.windows.transform(window, src.transform)
                data = src.read(window=window)

                kwargs.update({
                    "height": window.height,
                    "width": window.width,
                    "transform": crop_transform,
                })

                mem_file = MemoryFile()
                mem_ds = mem_file.open(**kwargs)
                mem_ds.write(data)

                mem_files.append(mem_file)
                src_datasets.append(mem_ds)
        
        mosaic, output = merge(src_datasets)

    else:
        mosaic, output = merge(ras_list)
            
    if params['classify']['save_mosaic']:
        out_meta.update(
            {"driver": "GTiff",
                "compress":'lzw',
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": output,
            })
    
        with rio.open(out_path, 'w', **out_meta) as m:
            m.write(mosaic)
            m.descriptions = band_names
            logger.info(f"writing mosaic to: {out_path}")

    else:
        logger.warning("OOPS -- Sorry -- this script has not been finished! - you can save the mosaic for now by setting classify:save_mosaic = True")
        ## TODO: add vrt method

    if params['buffer']:
        for ds in src_datasets:
            ds.close()
        for mf in mem_files:
            mf.close()

    return out_path
