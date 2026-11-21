import copy
import csv
import calendar
from collections import namedtuple
import datetime
import glob
import math
from pathlib import Path
import re
import statistics

import geowombat as gw
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.io import MemoryFile
from rasterio.merge import merge

from ..handler import logger
from .date_utils import get_date_range
from .pheno import prep_pheno_bands, prep_ts_variable_bands
from .processing_tracker import get_processing_db, match_brdf_files_to_record
from .project import ProjectPaths, get_tsdir_name, get_raw_ts_dir
from .texture import make_glcm
from .utils import get_cell_list_from_grid_param, get_sorted_brdf_images, get_full_numpix

ALL_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONTH_NUMS = {m: i + 1 for i, m in enumerate(ALL_MONTHS)}
Scene = namedtuple('Scene', ['dt', 'size', 'numpix', 'cover'])
ALL_QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']  ## defined starting at start_mo


def get_month_order(first_mo):
    '''Month names reordered so the sequence starts at first_mo.'''
    offset = int(first_mo) - 1
    return ALL_MONTHS[offset:] + ALL_MONTHS[:offset]


def parse_si_var(var):
    '''
    Split an si_var into (stat, temp). Accepts 'mean-wet' or 'mean_wet';
    a var with no separator (e.g. 'mean') is an annual stat, so temp = 'yr'.
    Also covers the monthly-sample vars: 'Jan-20' -> ('Jan', '20').
    '''
    for sep in ('-', '_'):
        if sep in var:
            stat, temp = var.split(sep)[:2]
            return stat, temp
    return var, 'yr'

def parse_ts_type(si):
    '''
    'nbr-raw' / 'nbrraw' -> 'raw';  'nbr-smwh' / 'nbr' -> 'smooth'.
    Raises (instead of falling through to an undefined variable) on e.g. 'nbr-foo'.
    '''
    tst = si.split('-')[1] if '-' in si else si
    if 'raw' in tst:
        return 'raw'
    if '-' in si and not tst.startswith('sm'):
        raise ValueError(f"problem parsing si. ts_type should be 'raw' or 'sm...'. Got {tst}")
    return 'smooth'

def expand_si_vars(si_vars, first_mo):
    '''Expand a single 'X-Monthly' / 'X-Quarterly' si_var into the full band list.'''
    if isinstance(si_vars, str):
        si_vars = [si_vars]
    if len(si_vars) != 1:
        return si_vars
 
    stat = si_vars[0].split('-')[0]
    low = si_vars[0].lower()
    if 'monthly' in low:
        logger.info(f'Making monthly {stat} composite')
        return [f'{stat}-{mon}' for mon in get_month_order(first_mo)]
    if 'quarterly' in low:
        logger.info(f'Making quarterly {stat} composite')
        return [f'{stat}-{q}' for q in ALL_QUARTERS]
    return si_vars

def get_pheno_kwargs(params):
    '''Pheno-only params, gathered once.'''
    fm = params['feature_model']
    return {
        'sigdif': fm['pheno_sigdif'] or None,
        'basethresh_pre': fm['pheno_basethresh_pre'],
        'basethresh_post': fm['pheno_basethresh_post'],
        'imgbuf': fm['pheno_imgbuf'],
    }

def get_month_day_target(var, start_yr, start_mo):
    '''<Month>-<day> (e.g. 'Mar-5') -> datetime.date in the right mapping year, or None if unparseable.
    Days past the end of a month are clamped (Feb-30 -> Feb 28/29).'''
    mo_name, day_str = parse_si_var(var)
    mo_num = MONTH_NUMS.get(mo_name)
    if mo_num is None or not day_str.isdigit() or not 1 <= int(day_str) <= 31:
        logger.warning(f"can't parse {var}; expected <Month>-<day>, e.g. 'Jan-20' (day 1-31)")
        return None
    ## target year for this month (in case the year doesn't start Jan 1st)
    yr = int(start_yr) if start_mo <= mo_num else int(start_yr) + 1
    return datetime.date(yr, mo_num, min(int(day_str), calendar.monthrange(yr, mo_num)[1]))
 
 
def get_sample_image_at_date(si_vars, img_dir, start_yr, start_mo, comp_band_names, ras_list):
    '''
    Pulls a single image per requested month-day var from the ts folder (if running on smoothed series) or brdf folder (if running on raw).
    Each var is <Month>-<day>, e.g. 'Jan-20' or 'Mar-5' (month is the variable, day is any 1-31, clamped to month end).
    
    Takes the image closest in time to that date, so the series can have any interval (10-day, weekly, ...).
    A tie goes to the earlier image. A warning is logged if the closest image is further away than the series' usual
    interval (a sign of a gap in the series). Band names are the si_vars.
    
    '''
    dated = []  ## (date, path) for every image named YYYYdoy.tif
    for img in sorted(Path(img_dir).glob('*.tif')):
        try:
            dated.append((datetime.datetime.strptime(img.stem, '%Y%j').date(), img))
        except ValueError:
            continue  ## skip files not named YYYYdoy
    if not dated:
        logger.warning(f"no YYYYdoy.tif images found in {img_dir}")
        return comp_band_names, ras_list
 
    dates = [d for d, _ in dated]
    interval = statistics.median((b - a).days for a, b in zip(dates, dates[1:])) if len(dates) > 1 else None
 
    for var in si_vars:
        target = get_month_day_target(var, start_yr, start_mo)
        if target is None:
            continue
        d, img = min(dated, key=lambda x: abs((x[0] - target).days))
        gap = abs((d - target).days)
        if interval is not None and gap > interval:
            logger.warning( f"{var}: closest image ({img.stem}) is {gap} days from {target};"
                           f" series interval is ~{interval:g} days, so there may be a gap")
        ras_list.append(img)
        comp_band_names.append(var)
        logger.info(f"added {var} from {img.stem} ({gap} days from {target})")
 
    return comp_band_names, ras_list


def get_sensor_family(name):
    '''Sensor family ('S2', 'LANDSAT', ...) from a brdf file name like L3B_S2B_..._coreg.nc; 'OTHER' if unrecognized.'''
    for family, pattern in FAMILY_PATTERNS:
        if re.search(pattern, Path(name).name):
            return family
    return 'OTHER'

    
def get_scene_candidates(ppaths, params, cell=None, min_valid_pct=90, min_filesize_pct=80, full_numpix=None):
    '''
    brdf .nc files that can be used for month-day samples -> {file name: Scene(dt, size in bytes, numpix)}.
    Sensors come from params['image_type'] (see get_scene_list). Files are what the list helpers return (skip_flag
    applied, plus whatever filter_list does); dates are parsed the same way reconstruct does (sort_images_by_date +
    the FILENAME_DATE constants).

    Scenes that only partly cover the cell are dropped, in order of preference:
      1. processing.info: keep scenes with numpix > full_numpix * min_valid_pct / 100 (files are matched to records by
         brdf_id with match_brdf_files_to_record; files with no matching record are dropped since their coverage is unknown)
      2. if there is no processing.info or it has no numpix column: keep scenes whose file size is at least
         min_filesize_pct percent of the largest file of that family in the cell
         A value of None or 0 turns the corresponding filter off.
    <full_numpix> = (param['grid_size'] / param['res'])**2 unless a different number is passed in
    '''
    
    if full_numpix is None:
        full_numpix = get_full_numpix(params)
    elif isinstance(full_numpix, (int, float)):
        full_numpix = {fam: full_numpix for fam in get_full_numpix(params)}

    skip_flag = params['reconstruct']['skip_flag'] or None
    dated = get_sorted_brdf_images(ppaths.ms, params['image_type'], skip_flag, get_brdf_sort_kwargs(params))
    cands = {Path(p).name: Scene(dt, (Path(ppaths.ms) / Path(p).name).stat().st_size, np.nan, np.nan)
             for p, dt in dated.items()}
    if not cands:
        return cands

    processing_db = get_processing_db(ppaths, cell)
    numpix_col = None
    if processing_db is not None:
        if 'numpix' not in processing_db.columns:
            logger.warning("processing.info has no numpix column; using file size to screen scenes instead")
        else:
            numpix_col = pd.to_numeric(processing_db['numpix'], errors='coerce')
 
    scene_of = {}  ## file name -> processing.info record
    if numpix_col is not None:
        file_to_scene = match_brdf_files_to_record(ppaths.ms, processing_db, exclude=skip_flag,
                                                   method='GEE' if params['dlMethod'] == 'GEE' else 'stac')
        scene_of = {Path(f).name: scene for f, scene in file_to_scene.items() if Path(f).name in cands}
 
        def lookup(name):
            if name not in scene_of:
                return np.nan
            v = numpix_col.loc[scene_of[name]]
            return v.max() if isinstance(v, pd.Series) else v  ## duplicate record ids -> best
 
        cands = {n: c._replace(numpix=lookup(n)) for n, c in cands.items()}
 
    families = {}
    for n in cands:
        families.setdefault(get_sensor_family(n), []).append(n)
 
    kept = {}
    for fam, names in families.items():
        full = full_numpix.get(fam) if numpix_col is not None else None
        if numpix_col is not None:
            ## cover = numpix as a fraction of a full scene (if the full count isn't known, of the best scene in the cell)
            vals = [cands[n].numpix for n in names if pd.notna(cands[n].numpix)]
            ref = full or (max(vals) if vals else None)
            if ref:
                for n in names:
                    cands[n] = cands[n]._replace(cover=cands[n].numpix / ref)
        if full:
            if not min_valid_pct:
                logger.info(f"{fam} coverage filter is off (min_valid_pct is 0 or None); using every {fam} scene")
                keep = names
            else:
                min_numpix = full * min_valid_pct / 100
                keep = [n for n in names if cands[n].numpix > min_numpix]  ## NaN (no record) fails
                n_unmatched = sum(n not in scene_of for n in names)
                logger.info(f"{fam} coverage filter: keeping {len(keep)} of {len(names)} scenes with numpix > {min_numpix:g}"
                            f"{f' ({n_unmatched} had no processing.info record)' if n_unmatched else ''}")
                if not keep:
                    vals = [cands[n].numpix for n in names if pd.notna(cands[n].numpix)]
                    logger.warning(f"no {fam} scene has numpix > {min_numpix:g} (best is {max(vals) if vals else None})")
        else:
            if numpix_col is not None:
                logger.info(f"no full-scene numpix is set for {fam}; using file size to screen {fam} scenes")
            if not min_filesize_pct:
                logger.info(f"{fam} size filter is off (min_size_pct is 0 or None); using every {fam} scene")
                keep = names
            else:
                max_size = max(cands[n].size for n in names)
                cutoff = max_size * min_filesize_pct / 100
                keep = [n for n in names if cands[n].size >= cutoff]
                logger.info(f"{fam} size filter: keeping {len(keep)} of {len(names)} scenes (>= {cutoff/1e6:.1f} MB = {min_filesize_pct}% of the largest, {max_size/1e6:.1f} MB)")
        kept.update({n: cands[n] for n in keep})
    return kept

 

def pick_closest_scene(cands, target, day_flex=0):
    '''
    Picks a scene for the target date from get_scene_candidates -> (file name, datetime), or (None, None).
 
    day_flex == 0: the scene closest in time to the target date; if more than one falls on that date, the largest file.
        (A tie between two dates equally close to the target goes to the earlier date.)
    day_flex > 0: of all scenes within day_flex days either side of the target, the one with the largest numpix
        (ties: closer to the target, then larger file). If numpix is not available (no column or no record), file size
        stands in for it. f no scene is within the window, falls back to the closest eligible scene (as for
        day_flex == 0) and logs a warning.
    '''
    if not cands:
        return None, None
 
    def days_off(n):
        return abs((cands[n].dt.date() - target).days)
 
    def closest():
        best_date = min({c.dt.date() for c in cands.values()}, key=lambda d: (abs((d - target).days), d))
        same_day = [n for n, c in cands.items() if c.dt.date() == best_date]
        if len(same_day) > 1:
            logger.info(f"{len(same_day)} images on {best_date}; using the largest file")
        return max(same_day, key=lambda n: cands[n].size)
 
    if not day_flex:
        name = closest()
        return name, cands[name].dt
 
    window = [n for n in cands if days_off(n) <= day_flex]
    if not window:
        name = closest()
        logger.warning(f"no eligible scene within {day_flex} days of {target}; using the closest eligible scene instead: "
                       f"{name} ({days_off(name)} days away)")
        return name, cands[name].dt

    if any(pd.notna(cands[n].numpix) for n in window):
        name = max(window, key=lambda n: (cands[n].numpix if pd.notna(cands[n].numpix) else -np.inf, -days_off(n), cands[n].size))
        logger.info(f"{len(window)} scenes within {day_flex} days of {target}; using {name} "
                    f"(numpix {cands[name].numpix:g}, {cands[name].cover:.0%} of a full scene)")
    else:
        name = max(window, key=lambda n: (cands[n].size, -days_off(n)))
        logger.info(f"{len(window)} scenes within {day_flex} days of {target}; no numpix available, using the largest file: {name}")
    return name, cands[name].dt


def get_monthly_raw(si_vars, ppaths, params, cell, start_yr, start_mo, comp_band_names, ras_list, out_dir, si,
                    min_valid_pct=None,min_filesize_pct=None, day_flex=None):
    '''
    Raw series: for each <Month>-<day> var, finds an image (pick_closest_scene) and runs just that file through
    reconstruct's normal raw index calculation (reconstruct(files=...)), writing YYYYdoy.tif into out_dir.
    Band names are the si_vars. Images already in out_dir are reused unless params['reconstruct']['overwrite'].
    Partial-coverage scenes are screened out first (see get_scene_candidates):
    min_valid_pct: scenes need numpix (processing.info) above this percent of FULL_SCENE_NUMPIX
        (default: params['feature_model']['min_valid_pct'] if set, otherwise 80; 0 turns it off).
    min_fileize_pct: only used if processing.info is missing or has no numpix column: scenes smaller than this percent of the
        largest file in the cell are ignored (default: params['feature_model']['min_filesize_pct'] if set, otherwise 75;
        0 turns it off).
    day_flex: days either side of the date that can be used (default: params['feature_model']['day_flex'] if set, otherwise 0).
        0 = the closest scene to the date. >0 = the scene with the largest numpix (processing.info) within that many days.

    '''
    if day_flex is None:
        day_flex = params['feature_model']['day_flex']
    if day_flex is None:
        day_flex = 0
    if min_valid_pct is None:
        min_valid_pct = params['feature_model']['min_valid_pct']
    if min_valid_pct is None or min_valid_pct in ('None', ''):
        min_valid_pct = 90
    if min_filesize_pct is None:
        min_filesize_pct = params['feature_model']['min_filesize_pct']
    if min_filesize_pct is None or min_filesize_pct in ('None', ''):
        min_filesize_pct = 80
    logger.info(f"Image selection settings: min_valid_pct={min_valid_pct}, min_filesize_pct={min_filesize_pct}, day_flex={day_flex}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overwrite = params['reconstruct']['overwrite']
    run_params = copy.deepcopy(params)
    run_params['reconstruct']['si'] = si
    cands = get_scene_candidates(ppaths, params, cell, min_valid_pct, min_filesize_pct)  ## listed once for all the dates
    done = set()
 
    for var in si_vars:
        target = get_month_day_target(var, start_yr, start_mo)
        if target is None:
            continue
        name, dt = pick_closest_scene(cands, target, day_flex)
        if name is None:
            logger.warning(f"no usable image found in {ppaths.ms} for {var}")
            continue
 
        out = out_dir / f"{dt.year}{dt.timetuple().tm_yday:03d}.tif"
        if (overwrite or not out.exists()) and name not in done:
            reconstruct(run_params, cells=[cell], files=[name], out_dir=out_dir)
            done.add(name)
        if not out.exists():
            logger.warning(f"reconstruct did not produce {out.name} for {var}")
            continue
 
        ras_list.append(out)
        comp_band_names.append(var)
        logger.info(f"added {var} from {name} ({abs((dt.date() - target).days)} days from {target})")
 
    return comp_band_names, ras_list


def get_image_stack(params, temp, img_dir):
        
    model_yr = int(params['feature_model']['start_yr'])
    ts_stack = []
    ds_stack = []
    pad_days = params['feature_model']['pheno_pad_days']
    if ((params['feature_model']['use_pheno']) and pad_days and (pad_days != [0,0])):
        padding = True
        
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


def make_ts_composite_single(ppaths, params, cell=None):
    '''
    makes ts composite for a single year
    cell is only used internally. Variables should be passed in via params
    '''
    fm = params['feature_model']
    si = fm['spec_indices'][0]  ##Only working for first index for now. TODO: loop through all sis in list
    start_mo = params['calendar']['first_mo']
    si_vars = expand_si_vars(fm['si_vars'], start_mo)
 
    model_yr = int(fm['start_yr'])
    start_doy = int(30.5 * int(start_mo)) - 30
    nodata_in = params['reconstruct']['nodata']
    tmpout_dir = ppaths.scratch / si / str(model_yr)
    tmpout_dir.mkdir(parents=True, exist_ok=True)
 
    pheno_kw = get_pheno_kwargs(params)
    ts_type = parse_ts_type(si)
 
    ## month-day samples (e.g. Jan-20, Mar-5) are handled separately from the aggregate stats (and come last)
    day_bands = [b for b in si_vars if parse_si_var(b)[1].isdigit()]
    stat_vars = [b for b in si_vars if b not in day_bands]
 
    ras_list = []
    comp_band_names = []
    gw_args = {'verbose': 1, 'n_workers': 4, 'n_threads': 1, 'n_chunks': 200, 'gdal_cache': 64, 'overwrite': True}
 
    def build_bands(bands, temp, stacks):
        '''Single dispatch point for pheno vs. plain stat bands (used by both raw and smooth).'''
        ts_stack, ds_stack, ts_stack_padded, ds_stack_padded = stacks
        if fm['use_pheno']:
            return prep_pheno_bands(bands, ts_stack, ds_stack, ts_stack_padded, ds_stack_padded,
                                    tmpout_dir, model_yr, temp, start_doy, comp_band_names, ras_list,
                                    params=params, **pheno_kw, **gw_args)
        return prep_ts_variable_bands(bands, ts_stack, ds_stack, tmpout_dir, temp, start_doy,
                                      comp_band_names, ras_list, nodata_in, ppaths, **gw_args)
 
    if ts_type == 'raw':
        ## If time series is raw, will run reconstruction to calculate si for all images in time period (with pheno pad if set)
        ##    and output them to a tmp folder. all images in this tmp folder are then used for the calculation of the statistic.
        img_dir = get_raw_ts_dir(ppaths, params, si, model_yr)
        rc = params['reconstruct']
        prev_season = None
 
        for i, var in enumerate(stat_vars):
            _, season = parse_si_var(var)
            fm['si_vars'] = [var]
            logger.info(f"working on unsmoothed brdf images for {var}")
 
            rc['si'] = si
            rc['start'] = fm['start_yr']
            rc['end'] = fm['start_yr'] + 1 if start_mo > 1 else fm['start_yr']
 
            ## only remake indices if using different temporal period for aggregate statistic
            ##    currently only works for single year. TODO: make work with multiple yrs in 'classify''yrs_out'
            if rc['overwrite'] and i > 0 and season == prev_season:
                rc['overwrite'] = False
            prev_season = season
            logger.debug(f"overwrite = {rc['overwrite']}")
 
            if rc['overwrite'] or not any(img_dir.glob('*.tif')):
                ## calculate the indices for selected time period -- these are sent to ppaths.scratch/raw/sis
                logger.info(f'calculating new raw time series for {si} index in {img_dir}')
                reconstruct(params)
 
            ## read the images in the img_dir to get the aggregated bands for the composite
            images = sorted(img_dir.rglob("*.tif"))
            if not images:
                logger.warning("OOPS -- no images have been created in the temp index directory")
                continue
 
            logger.info(f"there are {len(images)} images in {img_dir}")
            ts_stack = [str(img) for img in images]
            ds_stack = [pd.to_datetime(img.stem, format='%Y%j') for img in images]
            logger.debug(f'the set of images being used for this band are from: {ds_stack}')
 
            ## note the padding from <feature_model:pheno_pad_days> is already added to the time series during reconstruction,
            ##      so ts_stack and ts_stack_padded are the same. Need to access params to strip buffer from end product if desired.
            comp_band_names, ras_list = build_bands(var, season, (ts_stack, ds_stack, ts_stack, ds_stack))
 
    else:  ## smooth
        ## get stack from images in smoothed time-series directory that match temporal period of interest
        img_dir = ppaths.ts / si
 
        ## group bands by temporal period (if no temp in si_var, it is treated as 'yr')
        groups = {}
        for b in stat_vars:
            groups.setdefault(parse_si_var(b)[1], []).append(b)
 
        month_order = get_month_order(start_mo)
        logger.debug(f'month order will be {month_order}')
        for temp in ['yr', 'wet', 'dry'] + month_order + ALL_QUARTERS:
            bands = groups.get(temp)
            if bands:
                logger.info(f"calculating {temp} bands: {bands}...")
                stacks = get_image_stack(params, temp, img_dir)
                comp_band_names, ras_list = build_bands(bands, temp, stacks)
 
    if day_bands:
        if ts_type == 'raw':
            ## closest image to each date, index calculated for just that image
            if cell is None:
                raise ValueError("make_ts_composite_single needs cell= to make raw month-day variables")
            comp_band_names, ras_list = get_monthly_raw(day_bands, ppaths, params, cell, model_yr, start_mo, comp_band_names,
                                                        ras_list, tmpout_dir / 'monthly_raw', si)
        else:
            comp_band_names, ras_list = get_monthly_ts(day_bands, ppaths.ts / si, model_yr, start_mo, comp_band_names, ras_list)
 
    glcm_vars = [siv for siv in si_vars if 'glcm' in siv]
    if len(glcm_vars) > 0:
        logger.info('making glcm variables...\n')
        for gv in glcm_vars:
            no_glcm = gv.split('.')[0]
            if '-' in gv:
                no_glcm = f"{no_glcm}-{gv.split('-',1)[1]}"
            list_loc = comp_band_names.index(no_glcm)
            logger.info(f'found base variable {no_glcm} at location {list_loc} in stack')
            out_dir = Path(params['scratch_dir']) / 'temp_glcms'
            out_dir.mkdir(parents=True, exist_ok=True)
            out = Path(out_dir) / f"{gv.split('.')[0]}_{gv.split('.')[2]}.tif"
            glcm_band = make_glcm(base_img=ras_list[list_loc], params=params, si_var=gv, print_out=True, out_path=out, rgb=False)
            ras_list[list_loc] = glcm_band
            comp_band_names[list_loc] = gv
            logger.info(f'new band list: {comp_band_names}')
 
    return comp_band_names, ras_list


    
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
    
    cells = get_cell_list_from_grid_param(params['grids'])
    for cell in cells:
        ppaths = ProjectPaths(params, grid=cell)
        if params['feature_model']['treat_out'] == 'archive':
            out_dir = ppaths.comp
        elif params['feature_model']['treat_out'] == 'tmp':
            out_dir = ppaths.scratch / 'comp'

        out_dir.mkdir(parents=True, exist_ok=True)

        if not params['classify']['out_yrs'] or (isinstance(params['classify']['out_yrs'],int)) or (
            len(params['classify']['out_yrs']) == 1):
            logger.info(f'making single year composite for cell {cell}...')
            
            comp_band_names, ras_list = make_ts_composite_single(ppaths, params, cell=cell)
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

            if len(si_vars0) > 1:
                logger.warning(' OOPS -- can currently only create multi-year composites with single si variable and index')

            else:
                for yr in params['classify']['out_yrs']:
                    params['feature_model']['start_yr'] = yr
                    _band_names, ras_list = make_ts_composite_single(ppaths, params, cell=cell)
                    full_ras_list.extend(ras_list)
                    #full_comp_band_names.extend(band_names)

                if len(full_ras_list) != len(params['classify']['out_yrs']):
                    logger.warning('  OOPS -- band number does not match')
                        
                else:
                    logger.debug(f'ras_list:{full_ras_list}')
                    comp_band_names = [str(yr) for yr in params['classify']['out_yrs']]
                    #logger.debug(f'comp_band_names:{full_comp_band_names}')
                    logger.info(f'writing stack for si_vars:{si_vars0}')

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
    Mosaics classified results for a set of cells or tiles
    Cells are defined by <grids> as individual cells, defined with a list, a path to a .csv file, or an integer indicating a tile.
       If mosaicking to a tile (or chunk of cells), a corresponding .csv file needs to be 
           in <backup_path>/tiles named <project_prefix>TileX.csv with each cell number in the tile on a new line. 
           Can also use a custom list of cells with same format.
    A set of tiles can be mosaicked into a final map by removing <grid> param and setting <classify:chunk_tiles> = True 
    
    The output mosaic is saved to the 'classified' directory on the backup drive,
          unless <classify:test> is set to true, in which case it will be saved to the scratch drive.
    The <classify:name> parameter is the common string in the filenmaes of the cell products to be mosaicked 
         (usually the full model name) and will comprise the rest of the output mosaic name. 
         The output mosaic is saved to the 'classified' directory on the backup drive,
          unless <classify:test> is set to true, in which case it will be saved to the scratch drive. 
          If a .csv file, the file basename will be start of the basename of the output mosaic. 

    If a buffer distance was used when creating data for each grid cell (<buffer> > 0), 
          will unbuffer cells before mosaicking to remove edge effects
      make sure <buffer> param is set to original value. This does not apply to mosaics of tiles (chunks of cells).
    '''
    ppaths = ProjectPaths(params, grid=None)
    if out_path:
        out_dir = Path(out_path).parent
    elif params['classify']['test']:
        out_dir = Path(params['scratch_dir']) / 'classified'
    else:
        out_dir = ppaths.mosaics
    out_dir.mkdir(parents=True, exist_ok=True)

    ras_list = []
    
    prefix = ''
    if params['project_prefix']:
        prefix = params['project_prefix']
                    
    if params['grids']:
        mosaic_cells = True
        
        if isinstance(params['grids'], int):
            params['grids'] = [params['grids']]
        elif isinstance(params['grids'], list):
            cells = params['grids']
            if not out_path:
                out_path = Path(out_dir) / f"{params['classify']['name']}_mosaic.tif"
        elif isinstance(params['grids'], str):
            if 'tile' in params['grids'].lower():
                tilenum = int(params['grids'].lower().split('/')[-1].split('tile')[1].split('.')[0])
                cell_list = ppaths.tiles/f'{prefix}Tile{tilenum}.csv'
                if not out_path:
                    out_path = Path(out_dir) / f"{prefix}Tile{tilenum}_{params['classify']['name']}.tif"
            elif params['grids'].endswith('.csv'):
                cell_list = params['grids']
                if not out_path:
                    out_path = Path(out_dir) / f"{Path(params['grids']).stem}_{params['classify']['name']}.tif"
            else:
                logger.warning("if using params['grids'], needs to be a list, a string like 'Tile9'," 
                               f"or the path of a .csv file containing the cell list")
            cells = []
            with open(cell_list, newline='') as cell_file:
                for row in csv.reader(cell_file):
                    cells.append (row[0])
    elif params['classify']['chunked_map']:
        logger.info(f"chunk_tile = {params['classify']['chunk_tile']}")
        try:
            tilenum = int(params['classify']['chunk_tile'])
        except (ValueError, TypeError):
            mosaic_cells = False
        else:
            cell_list = ppaths.tiles/f'{prefix}Tile{tilenum}.csv'
            cells = []
            with open(cell_list, newline='') as cell_file:
                for row in csv.reader(cell_file):
                    cells.append (row[0])
            mosaic_cells = True
        
    if mosaic_cells == True:
        logger.debug(f"mosaicking cells:{cells}")
        
        for cell in cells:
            ppaths = ProjectPaths(params, grid=int(cell))
            if params['classify']['comp_dir'] == 'input_dir':
                comp_path = ppaths.ms.parent / 'comp'
            elif params['classify']['comp_dir'] == 'backup':
                comp_path = ppaths.comp / params['classify']['local_dir']
            elif params['classify']['comp_dir'] == 'tmp':
                comp_path = ppaths.scratch  / 'comp'
            elif Path(params['classify']['comp_dir']).is_dir():
                comp_path = Path(params['classify']['comp_dir']).parent / f'{int(cell):06d}'
            else: 
                comp_path = ppaths.ms.parent / params['classify']['comp_dir']
            if not comp_path.is_dir:
                logger.warning(f"comp_dir {comp_dir} does not exist. Use 'input_dir' for 'comp' folder in main directory,"
                               f" 'backup' for 'comp' folder in backup directory (with optional subfolder set as local_dir)"
                               f" 'tmp' for 'comp' folder in scratch drive, any other string for sub folder in the main directory"
                               f" or the directory that is the parent of all cell files"
                               f" You put {params['classify']['comp_dir']}")
        
            logger.debug(f'Looking in {comp_path} for individual inputs')
            if not comp_path.is_dir():
                logger.warning(f"there is no comp folder {comp_path} for cell {cell}.")
            else:
                matches = glob.glob(str(comp_path) + f"/*{params['classify']['name']}*.tif")
                if len(matches) == 0:
                    logger.warning(f"no raster was created for cell {cell} for model {params['classify']['name']}.")
                elif len(matches) > 1:
                    logger.warning(f"more than one raster containing {params['classify']['name']}" 
                                   f"was found for cell {cell}. -- using first match")
                    ras_list.append(matches[0])
                else:
                    ras_list.append(matches[0])
                
    else:
        if isinstance(params['classify']['chunk_tile'], list):
            tiles = params['classify']['chunk_tile']
        elif 'to' in params['classify']['chunk_tile']:
            start, end = params['classify']['chunk_tile'].split('to')
            tiles = list(range(int(start), int(end) + 1))
        params['buffer'] = None
        ## in_dir will be whatever the out_dir was for the cell to tile mosaics
        in_dir = out_dir
        out_dir = ppaths.mosaics
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(out_dir) / f"{prefix}All_{params['classify']['name']}.tif"
        logger.info(f"mosaicking tiles:{tiles}")
        
        for n in tiles:
            tile_path = Path(in_dir) / f"{prefix}Tile{n}_{params['classify']['name']}.tif"
            ras_list.append(tile_path)
        
    logger.debug(f' ras_list = {ras_list} \n')
    logger.info(f"mosaicking {len(ras_list)} images...\n")
        
    with rio.open(ras_list[0], 'r') as src_exmp:
        out_meta = src_exmp.meta.copy()
        band_names = src_exmp.descriptions
    logger.debug(f"these arie {src_exmp.meta['count']}-band rasters")
            
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
                 "BIGTIFF": "IF_SAFER",
                "tiled": True, "blockxsize": 512, "blockysize": 512,
                "SPARSE_OK": "TRUE",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": output,
            })
    
        with rio.open(out_path, 'w', **out_meta) as m:
            m.write(mosaic)
            m.descriptions = band_names
        logger.info(f"wrote mosaic to: {out_path}")

    else:
        logger.warning("OOPS -- Sorry -- this script has not been finished! - "
                       f"you can save the mosaic for now by setting classify:save_mosaic = True")
        ## TODO: add vrt method

    if params['buffer']:
        for ds in src_datasets:
            ds.close()
        for mf in mem_files:
            mf.close()

    return out_path
