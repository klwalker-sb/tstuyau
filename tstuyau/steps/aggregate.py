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

    
def get_monthly_ts(si_vars, img_dir, start_yr, start_mo, comp_band_names, ras_list):
    for img in sorted(list(img_dir.glob('*.tif'))):
        im = str(img.stem)
        if (start_mo == 1 and im.startswith(str(start_yr))) or (start_mo > 1 and im.startswith(str(int(start_yr)+1))):
            logger.debug(f'looking at {im}')
            if im.endswith('020') and ('Jan-20' in si_vars):
                ras_list.append(img)
                comp_band_names.append('Jan-20')
                logger.info('added Jan')
        if (start_mo <= 2 and im.startswith(str(start_yr))) or (start_mo > 2 and im.startswith(str(int(start_yr)+1))):
            if im.endswith('051') and 'Feb-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('Feb-20')
                logger.info('added Feb')
        if (start_mo <= 3 and im.startswith(str(start_yr))) or (start_mo > 3 and im.startswith(str(int(start_yr)+1))):
            if (im.endswith('079') | im.endswith('080')) and 'Mar-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('Mar-20')
                logger.info('added Mar')
        if (start_mo <= 4 and im.startswith(str(start_yr))) or (start_mo > 4 and im.startswith(str(int(start_yr)+1))):
            if (im.endswith('110') | im.endswith('111')) and 'Apr-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('Apr-20')
                logger.info('added Apr')
        if (start_mo <= 5 and im.startswith(str(start_yr))) or (start_mo > 5 and im.startswith(str(int(start_yr)+1))):
            if (im.endswith('140') | im.endswith('141')) and 'May-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('May-20')
                logger.info('added May')
        if (start_mo <= 6 and im.startswith(str(start_yr))) or (start_mo > 6 and im.startswith(str(int(start_yr)+1))):
            if (im.endswith('171') | im.endswith('172')) and 'Jun-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('Jun-20')
                logger.info('added Jun')
        if (start_mo <= 7 and im.startswith(str(start_yr))) or (start_mo > 7 and im.startswith(str(int(start_yr)+1))):
            if (im.endswith('201') | im.endswith('202')) and 'Jul-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('Jul-20')
                logger.info('added July')
        if (start_mo <= 8 and im.startswith(str(start_yr))) or (start_mo > 8 and im.startswith(str(int(start_yr)+1))):
            if (im.endswith('232') | im.endswith('233')) and 'Aug-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('Aug-20')
                logger.info('added Aug')
        if (start_mo <= 9 and im.startswith(str(start_yr))) or (start_mo > 9 and im.startswith(str(int(start_yr)+1))):
            if (im.endswith('263') | im.endswith('264')) and 'Sep-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('Sep-20')
                logger.info('added Sep')
        if (start_mo <= 10 and im.startswith(str(start_yr))) or (start_mo > 10 and im.startswith(str(int(start_yr)+1))):
            if (im.endswith('293') | im.endswith('294')) and 'Oct-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('Oct-20')
                logger.info('added Oct')
        if (start_mo <= 11 and im.startswith(str(start_yr))) or (start_mo > 11 and im.startswith(str(int(start_yr)+1))):
            logger.debug(f'looking at {im}')
            if (im.endswith('324') | im.endswith('325')) and 'Nov-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('Nov-20')
                logger.info('added Nov')
        if (start_mo <=12 and im.startswith(str(start_yr))):
            if (im.endswith('354') | im.endswith('355')) and 'Dec-20' in si_vars:
                ras_list.append(img)
                comp_band_names.append('Dec-20')
                logger.info('added Dec')

    return comp_band_names,ras_list

def make_ts_composite_single(ppaths, params):

    si = params['feature_model']['spec_indices'][0]  ##Only working for first index for now. TODO: loop through all sis in list
    si_vars = params['feature_model']['si_vars']
       
    model_yr = int(params['feature_model']['start_yr'])
    start_mo = params['calendar']['first_mo']
    start_doy = int(30.5 * int(start_mo)) - 30
    nodata_in = params['reconstruct']['nodata']
    tmpout_dir = ppaths.scratch/si/str(model_yr)
    tmpout_dir.mkdir(parents=True, exist_ok=True)

    ## the following parameters are only needed for pheno variables, and only for some cases
    if params['feature_model']['pheno_sigdif']:
        sigdif = params['feature_model']['pheno_sigdif']
    else:
        sigdif = None
    if params['feature_model']['pheno_basethresh'] is not None:
        basethresh = params['feature_model']['pheno_basethresh']
    else:
        basethresh = None
    if params['feature_model']['pheno_imgbuf'] is not None:
        imgbuf = params['feature_model']['pheno_imgbuf']
    else:
        imgbuf = None

    if '-' in si:
        tst = si.split('-')[1]
        if 'raw' in tst:
            ts_type = 'raw'
        elif tst.startswith('sm'):
            ts_type = 'smooth'
        else:
            logger.warning(f"ERROR: problem parsing si_var. ts_type should br 'raw' or 'sm...'. Got {tst}")
    else:   
        if 'raw' in si:
            ts_type = 'raw'
        else:
            ts_type = 'smooth'

    ras_list = []
    comp_band_names = []
    gw_args = {'verbose':1,'n_workers':4,'n_threads':1,'n_chunks':200, 'gdal_cache':64,'overwrite':True}

    ts_root = get_tsdir_name(params)
    if ts_type == 'raw':
        for i, var in enumerate(si_vars):
            params['feature_model']['si_vars'] = [var]
            logger.info(f"working on unsmoothed brdf images for {var}")
            ts_stack = []
            ds_stack = []
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
            if params['reconstruct']['overwrite'] or (not any(img_dir.iterdir())):
                ## calculate the indices for selected time period -- these are sent to ppaths.scratch/raw/sis
                logger.info(f'calculating new raw time series for {si} index in {img_dir}')
                reconstruct(params)
            
            ## read the images in the img_dir to get the aggregated bands for the composite
            season = var.split('-')[1]
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
                        season, start_doy, comp_band_names, ras_list, sigdif=sigdif, basethresh=basethresh, imgbuf=imgbuf, **gw_args)

                else:
                    comp_band_names,ras_list = prep_ts_variable_bands(
                        var, ts_stack, ds_stack, tmpout_dir,season,start_doy, comp_band_names, ras_list, nodata_in, ppaths, **gw_args)
    
    elif ts_type == 'smooth':
        ## get stack from images in smoothed time-series directory that match temporal period of interest
        img_dir = ppaths.ts / si
        ts_stack = []
        ds_stack = []
        ts_stack_wet = []
        ts_stack_dry = []
        ds_stack_wet = []
        ds_stack_dry = []
        
        if (params['feature_model']['use_pheno'] and params['feature_model']['pheno_pad_days'] and params['feature_model']['pheno_pad_days'] != [0,0]):
            padding = True
            pad_days = params['feature_model']['pheno_pad_days']
            ts_stack_wet_padded = []
            ds_stack_wet_padded = []
            ts_stack_dry_padded = []
            ds_stack_dry_padded = []
            logger.info(f"padded will add {pad_days[0]} days on left and {pad_days[1]} days on right \n")
        else:
            padding = False

        yr_start, yr_end = get_date_range(model_yr,'yr',params,return_type='doy',padded=False)
        wet_start, wet_end = get_date_range(model_yr,'wet',params,return_type='doy',padded=False)
        dry_start, dry_end = get_date_range(model_yr,'dry',params,return_type='doy',padded=False)
        logger.info(f"using images from {yr_start} to {yr_end} \n")
        logger.info(f"wet season is from {wet_start} to {wet_end} and dry season is from {dry_start} to {dry_end} \n")
        
        logger.info(f"looking in {img_dir}...")
        for img in sorted(img_dir.rglob("*.tif")):
            ## ts images are named YYYYdoy.tif
            imgdt = int(img.stem)
            logger.debug(f'imgdt: {imgdt}')
            img_date = pd.to_datetime(img.stem,format='%Y%j')
            
            if (imgdt >= yr_start) and (imgdt <= yr_end):
                ts_stack.append(str(img))
                ds_stack.append(img_date)
                
            ## if in wet season add to wet season subset
            if (imgdt >= wet_start) and (imgdt <= wet_end):
                ts_stack_wet.append(str(img))
                ds_stack_wet.append(img_date)
                    
            ## if in dry season add to dry season subset
            if (imgdt >= dry_start) and (imgdt <= dry_end):
                ts_stack_dry.append(str(img))
                ds_stack_dry.append(img_date)

            logger.debug(f'images stack lengths: annual:{len(ds_stack)}, wet:{len(ds_stack_wet)}, dry:{len(ds_stack_dry)}')
            
            if padding:
                wet_padded_start, wet_padded_end = get_date_range(model_yr,'wet',params,return_type='doy',padded=True)
                if (imgdt >= wet_padded_start) and (imgdt <= wet_padded_end):
                    ts_stack_wet_padded.append(str(img))
                    ds_stack_wet_padded.append(img_date)
                    
                dry_padded_start, dry_padded_end = get_date_range(model_yr,'dry',params,return_type='doy',padded=True)
                if (imgdt >= dry_padded_start) and (imgdt <= dry_padded_end):
                    ts_stack_dry_padded.append(str(img))
                    ds_stack_dry_padded.append(img_date)
                    
        ## Calculate statistics for each time period
        
        annual_bands = [b for b in si_vars if (("-" not in b) and ("_" not in b)) or 'yr' in b]
        logger.info(f"calculating annual bands: {annual_bands}...")
        if len(annual_bands) > 0:
            if params['feature_model']['use_pheno']:
                comp_band_names,ras_list = prep_pheno_bands(annual_bands, ts_stack, ds_stack, None, None, 
                    tmpout_dir,model_yr,'yr',start_doy, comp_band_names, ras_list, sigdif=sigdif, basethresh=basethresh, imgbuf=imgbuf, **gw_args)
            else:
                comp_band_names,ras_list = prep_ts_variable_bands(annual_bands, ts_stack, ds_stack, 
                                                                  tmpout_dir,'yr',start_doy, comp_band_names, ras_list, nodata_in, ppaths, **gw_args)

        wet_bands = [b for b in si_vars if ("_" in b and b.split("_")[1] == 'wet') or ("-" in b and b.split("-")[1] == 'wet')]
        logger.info(f"calculating wet bands: {wet_bands}...")
        if len(wet_bands) > 0:
            if params['feature_model']['use_pheno']:
                comp_band_names,ras_list = prep_pheno_bands(wet_bands, ts_stack_wet, ds_stack_wet, ts_stack_wet_padded, ds_stack_wet_padded,
                    tmpout_dir,model_yr,'wet', start_doy, comp_band_names, ras_list, sigdif=sigdif, basethresh=basethresh, imgbuf=imgbuf, **gw_args)
            else:
                comp_band_names,ras_list = prep_ts_variable_bands(wet_bands, ts_stack_wet, ds_stack_wet, 
                                                                  tmpout_dir,'wet', start_doy, comp_band_names, ras_list, nodata_in, ppaths, **gw_args)
            
        dry_bands = [b for b in si_vars if ("_" in b and b.split("_")[1] == 'dry') or ("-" in b and b.split("-")[1] == 'dry')]
        logger.info(f"calculating dry bands: {dry_bands}...")
        if len(dry_bands) > 0:
            if params['feature_model']['use_pheno']:
                comp_band_names,ras_list = prep_pheno_bands(dry_bands, ts_stack_dry, ds_stack_dry, ts_stack_dry_padded, ds_stack_dry_padded,
                    tmpout_dir,model_yr,'dry', start_doy, comp_band_names, ras_list, sigdif=sigdif, basethresh=basethresh, imgbuf=imgbuf, **gw_args)
            else:
                comp_band_names,ras_list = prep_ts_variable_bands(dry_bands, ts_stack_dry, ds_stack_dry, 
                                                                  tmpout_dir,'dry', start_doy, comp_band_names, ras_list, nodata_in, ppaths, **gw_args)
    
        mo_bands = [b for b in si_vars if ("_" in b and b.split("_")[1] == '20') or ('-' in b and b.split("-")[1] == '20')]
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
    
    sipre = get_tsdir_name(params)
    if sipre == '' or sipre == 'ms':
        si_full = si
    else:
        si_full = f'{sipre}_{si}'
        
    si_vars = params['feature_model']['si_vars']
    mod_yr = params['feature_model']['start_yr']
    logger.debug(f"si_vars = {si_vars}")
    
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
            
            if len(ras_list) < len(si_vars):
                logger.warning('OOPS -- got an unknown band')

            else:
                ## Start writing output composite
                with rio.open(ras_list[0]) as src0:
                    meta = src0.meta
                    meta.update(count = len(ras_list))

                if params['feature_model']['use_pheno']:
                    out_ras = f"{out_dir}/{int(cell):06d}_{mod_yr}_{si_full}_{'-'.join(comp_band_names)}_Phen.tif"
                
                else:
                    if len(ras_list)>12:
                        out_ras = f"{out_dir}/{int(cell):06d}_{mod_yr}_{si_full}_ModVars.tif"
                    elif len(ras_list)==12:
                        out_ras = f"{out_dir}/{int(cell):06d}_{mod_yr}_{si_full}_monthly.tif"
                    else:
                        out_ras = f"{out_dir}/{int(cell):06d}_{mod_yr}_{si_full}_{'-'.join(comp_band_names)}.tif"

                with rio.open(out_ras, 'w', **meta) as dst:
                    for id, layer in enumerate(ras_list, start=1):
                        with rio.open(layer) as src1:
                            dst.write(src1.read(1),id)
                    dst.descriptions = tuple(comp_band_names)
                
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
                    #logger.debug(f'comp_band_names:{full_comp_band_names}')
                    logger.info(f'writing stack for si_vars:{si_vars}')

                    ## Start writing output composite
                    with rio.open(full_ras_list[0]) as src0:
                        meta = src0.meta
                        meta.update(count = len(full_ras_list))

                    var_name = comp_band_names[0]
                    comp_band_names = [str(yr) for yr in params['classify']['out_yrs']]
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
        output_path = out_path
    elif params['classify']['test']:
        out_dir = Path(params['scratch_dir']) / 'classified'
    else:
        out_dir = Path(params['backup_path']).parents[1] / 'mosaics'
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(params['grids'], list):
        cells = params['grids']
        if not output_path:
            output_path = Path(out_dir) / f"{params['classify']['name']}_mosaic.tif"
    elif params['grids'].endswith('.csv'):
        if not output_path:
            output_path = Path(out_dir) / f"{Path(params['grids']).stem}_{params['classify']['name']}.tif"
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
            comp_path = Path(params['classify']['comp_dir']).parent / f'{cell:06d}'
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
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": output,
                })
    
            with rio.open(output_path, 'w', **out_meta) as m:
                m.write(mosaic)
                logger.info(f"writing mosaic to: {output_path}")

        else:
            logger.warning("OOPS -- Sorry -- this script has not been finished! - you can save the mosaic for now by setting classify:save_mosaic = True")
            ## TODO: add vrt method

        if params['buffer']:
            for ds in src_datasets:
                ds.close()
            for mf in mem_files:
                mf.close()

    return output_path
