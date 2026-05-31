from pathlib import Path
import csv
import math
import datetime
import geowombat as gw
from geowombat.core import sort_images_by_date
import pandas as pd
import rasterio as rio
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt

from .project import ProjectPaths, get_tsdir_name
#from .spec_indices import calc_si_gw
from .constants import FILENAME_DATE_INDEX, FILENAME_DATE_INDEX_GEE, FILENAME_DATE_START_INDEX, FILENAME_DATE_END_INDEX
from .check_sample import get_pts_in_grid, get_polygons_in_grid, get_ran_pts_in_polys
from .date_utils import get_date_range, get_img_date
from .mod_utils import get_train_yrs_str
from .spec_indices import calculate_raw_index, calculate_char_index, SI_DICT
from .separability import get_separability
from .lookup import SENSORS
from ..handler import logger

LANDSAT_LIKE_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    
def get_index_vals_at_pts(ts_stack, ts_type, img_type, polys, spec_index, npts, 
                          seed, load_samp=False, ptgdb=None, printfile=False, out_dir=None, params=None):
    '''
    Gets values for all sampled points <'numpts'> in all polygons <'polys'> for all images in <'ts_stack'>
    OR gets values for points in a previously generated dataframe <ptgdb> using loadSamp=True.

    Noet this is different from the sample method "check_sample.get_variables_at_pts" in that the calculations are done on the fly for only
        the points in question. This is intened for smaller samples for pre-modeling exploration. The other method collects training data at 
        sample points after features have been created at the raster level.
    If <ts_type> == 'smooth' (or one of the smoothed collections, e.g. 'sm-wh'): indices are assumed to already be calculated and
          <'ts_stack'> is a list of image paths, with basenames = YYYYDDD of image acquisition (DDD is Julien day, 1-365)
    If <ts_type> is raw: images are still in raw .nc form (6 bands) and indices are calculated here
          <ts_stack'> is a list of image paths from which YYYYDDD info can be extracted
    output is a dataframe with a pt (named polygonID_pt#)
    on each row and an image index value(named YYYYDDD) in each column
    '''
    logger.info('getting index values...')
    
    maxval=10000
    if params:
        if params['masking']['maxval']:
            maxval = params['masking']['maxval']
    
    if not load_samp:
        if polys:
            ptsgdb = get_ran_pts_in_polys(polys, npts, seed)
        else:
            logger.info('There are no polygons or points to process in this cell')
            return None
    elif load_samp:
        ptsgdb = ptgdb.reset_index(names='pt')

    xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
    coords = list(map(list, zip(*xy)))
    
    pt_dict={}
    for img in ts_stack:
        img_date = get_img_date(img, ts_type, img_type)
        if ts_type == 'smooth':
            ## smoothed time-series should already be calculated
            with rio.open(img, 'r') as src:
                pt_dict[img_date] = [sample[0] for sample in src.sample(coords)]
        elif ts_type == 'raw':
            spec_index = spec_index.split('-')[0]
            available_indices = list(SI_DICT)
            if spec_index not in available_indices:
                 logger.warning(f"ERROR: {spec_index} is not specified or does not have current method")
            else:
                xrimg = xr.open_dataset(img)
                xr_nir = xrimg['nir'].where(xrimg['nir'] < maxval)
                #xr_nir = xrimg['nir'].map({>=maxval: np.nan, < maxval: xrimg['nir']})
                if spec_index in ['evi2','msavi','ndvi','savi','wi','kndvi','red','char','bai','gemi']:
                    xr_red = xrimg['red'].where(xrimg['red'] < maxval)
                    #xr_red = xrimg['red'].map({>=maxval: np.nan, < maxval: xrimg['red']})
                if spec_index in ['ndmi','wi','nbr2','cai','mirbi','swir1']:
                    xr_swir1 = xrimg['swir1'].where(xrimg['swir1'] < maxval)
                    #xr_swir1 = xrimg['swir1'].map({>=maxval: np.nan, < maxval: xrimg['swir1']})
                if spec_index in ['ndwi','gcvi','green','char']:
                    xr_green = xrimg['green'].where(xrimg['green'] < maxval)
                    #xr_green = xrimg['green'].map({>=maxval: np.nan, < maxval: xrimg['green']})
                if spec_index in ['nbr','nbr2','swir2','cai','mirbi','baim']:
                    xr_swir2 = xrimg['swir2'].where(xrimg['swir2'] < maxval)
                    #xr_swir2 = xrimg['swir2'].map({>=maxval: np.nan, < maxval: xrimg['swir2']})
                if spec_index in ['blue','char']:
                    xr_blue = xrimg['blue'].where(xrimg['blue'] < maxval)
                    
                pt_vals = []
                for index, row in ptsgdb.iterrows():
                    
                    if spec_index in ['char']:
                        thispt_red = xr_red.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                        red_val = thispt_red.values
                        thispt_blue = xr_blue.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                        blue_val = thispt_blue.values
                        thispt_green = xr_green.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                        green_val = thispt_green.values

                        index_val = calculate_char_index(red_val, green_val, blue_val, scale_factor=maxval)

                    else:
                        if spec_index in ['wi','nbr2','cai','mirbi']:  ## note thisptnir is actually swir1 here
                            thispt_nir = xr_swir1.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                    method='nearest', tolerance=30)
                        else:
                            thispt_nir = xr_nir.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                            method='nearest', tolerance=30)
                        nir_val = thispt_nir.values 
                    
                        if spec_index in ['evi2','msavi','ndvi','savi','wi','kndvi','bai','gemi','red']:
                            thispt_b2 = xr_red.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                            b2_val = thispt_b2.values
                        elif spec_index in ['ndmi','swir1']:
                            thispt_b2 = xr_swir1.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                            b2_val = thispt_b2.values
                        elif spec_index in ['ndwi','gcvi','green']:
                            thispt_b2 = xr_green.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                            b2_val = thispt_b2.values
                        elif spec_index in ['nbr','nbr2','cai','mirbi','swir2','bai2']:
                            thispt_b2 = xr_swir2.sel(x=ptsgdb['geometry'].x[index],y=ptsgdb['geometry'].y[index],
                                method='nearest', tolerance=30)
                            b2_val = thispt_b2.values
                        elif spec_index in ['nir']:
                            b2_val = nir_val

                        logger.debug(f"b2_val = {b2_val} for image {str(img_date)}.")

                        index_val = calculate_raw_index(nir_val, b2_val, spec_index, params=params)
                    
                    pt_vals.append(index_val)
                pt_dict[str(img_date)] = pt_vals
                logger.debug(f" got {len(pt_vals)} values for {str(img_date)}")
        
    ptdf = pd.DataFrame.from_dict(pt_dict, orient='columns')
    ptsgdb = pd.concat([ptsgdb,ptdf], axis=1)     
    if printfile==True:
        outfile = Path(out_dir)/'ptsgdb.csv'               
        logger.debug(f'printing resulting db to {outfile}')     
        pd.DataFrame.to_csv(ptsgdb, outfile, sep=',', index=True)
    
    return ptsgdb

def sample_timeseries(params):
    '''
    Returns datetime dataframe of values for sampled pts (n=<npts>) for each polygon in <polys>}
    OR for previously generated points with <load_samp>=True and <pt_file>=path to .csv file
     (.csv file needs 'XCoord' and 'YCoord' fields (in this case, <polyfile>, <oldest>, <newest>, <npts> and <seed> are not used))
    for all images of <image_type> acquired between <start_yr> and <end_yr> in <img_dir>
    Output format is a datetime object with date (YYYY-MM-DD) on each row and sample name (polygonID_pt#) or pt(OID_) in columns
    Inputs for smoothed data needs to already be calculated and in ts folder.  Will calculate raw data on the fly.
    '''
    ## Args
    grids = params['grids']  ## int or list
    grid_file = params['grid_file']  ## str
    img_type = params['image_type'] ## str 'LS2'| 'S2' | 'L'| 'LT05'| 'LE07'| 'LC08'| 'LC09'| 'PS'| 'CHIRPS_'
    spec_index = params['feature_model']['spec_indices'] ## str | list (takes first item)
    start_yr = params['sample_model']['train_yrs'][0] ## int(YYYY)
    end_yr = params['sample_model']['train_yrs'][1] ## int(YYYY)
    load_samp = params['sample_model']['load_samp']  ## True | False
    ptfile =  params['sample_model']['point_file'] ## str  -- needed only if load_samp==True
    filter_col = params['sample_model']['filter_col'] ## str -- optional
    filter_class = params['sample_model']['filter_class'] ## str -- optional
    poly_file = params['sample_model']['poly_file'] ## str -- optional

    spectsdf_dir = params['explore']['spectsdf_dir'] ## str -- optional
    else:
        ppaths=ProjectPaths(params)
        spectsdf_dir = ppaths.tssigs
    ## maxval passed to get_index_value:
    #if params['masking']['maxval']:
    #    maxval = params['masking']['maxval']  ## int -- optional (defaults to 10000) usually 1 for unprocessed data

    ## the following parameters are only needed if poly_file is not None
    if poly_file:
        npts = params['sample_model']['npts']  ## int
        seed = params['sample_model']['poly_samp_seed']  ## int -- optional
        oldest = params['sample_model']['oldest'] ## int -- optional
        newest = params['sample_model']['newest'] ## int -- optional
        obs_col = params['sample_model']['obs_col'] ## str -- optional
    ## the following uses the calendar parameters:
    dt_start, dt_end1 = get_date_range(start_yr,'yr',params,return_type='doy',padded=params['feature_model']['pheno_pad_days'])
    dt_end = str(end_yr) + str(dt_end1)[4:]
    
    if isinstance(spec_index,str):
        params['feature_model']['spec_indices'] = [spec_index]
    si = params['feature_model']['spec_indices'][0]
    if '-' in si:
        sidx = si.split('-')[0]
    else:
        sidx = si
    allpts = pd.DataFrame()

    if load_samp:
        polys = None
            
        if isinstance(ptfile, gpd.GeoDataFrame) | isinstance(ptfile, pd.DataFrame):
            point_df = ptfile
        elif ptfile.endswith('.csv'):
            point_df = pd.read_csv(ptfile, index_col='OID_')
        elif ptfile.endswith('.shp') or ptfile.endswith('.gpkg'):
            point_df = gpd.read_file(ptfile)
            
        if filter_class:
            pts = point_df[point_df[filter_col]==filter_class]
        else:
            pts = point_df

    cells = []
    if isinstance(grids, list):
        cells = grids
    elif isinstance(params['grids'], str) and grids.endswith('.csv'): 
        with open(grids, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append(row[0])
    elif isinstance(grids, int) or isinstance(grids, str): # if runing individual cells as array via bash script
        cells.append(grids) 
    
    for cell in cells:
        ppaths = ProjectPaths(params, grid=int(cell))
        logger.info(f"working on cell {cell}")
        ts_stack = []
        ds_stack = []
        
        if load_samp:
            logger.debug(f'pts is type: {type(pts)} and looks like this: \n {pts.head()}')
            points = get_pts_in_grid(grid_file, cell, pts)
        else:
            polys = get_polygons_in_grid(grid_file, cell, poly_file, oldest=oldest, newest=newest, obs_col=obs_col)
            points = None

        if isinstance(points, gpd.GeoDataFrame) or polys:
            if 'raw' not in si:
                ts_type = 'smooth'
                cell_dir = ppaths.ts / si
                logger.info(f'looking for images in {cell_dir}')
                for img in list(cell_dir.glob('*tif')):
                    img_dt = int(img.stem)
                    if (img_dt >= int(dt_start)) and (img_dt <= int(dt_end)):
                        ts_stack.append(str(img))
                        ds_stack.append(img_dt)
                        ts_stack.sort()
                if len(ts_stack) == 0:
                    logger.warning(f' there are no images in {cell_dir} between {dt_start} and {dt_end}.')
                else:
                    logger.info(f' there are {len(ts_stack)} images in {cell_dir} between {dt_start} and {dt_end}')
            
            else:
                ts_type = 'raw'
                cell_dir = ppaths.proc
                matchstr = SENSORS[img_type]['matchstr']
                for img in list(cell_dir.glob('*[!X].nc')):
                        imgtyp = img.stem.split('_')[1][:4] if img.stem.split('_')[1].startswith('L') else img.stem.split('_')[1][:3]
                        if imgtyp in matchstr:
                            img_dt =  get_img_date(img.stem, 'raw', img_type)
                            logger.debug(f'image date: {img_dt}')
                            if (int(img_dt) >= int(dt_start)) and (int(img_dt) <= int(dt_end)):
                                ts_stack.append(str(img))
                                ds_stack.append(img_dt)
                ts_stack = [ts for ds, ts in sorted(zip(ds_stack, ts_stack))]
                if len(ts_stack) == 0:
                    logger.warning(f' there are no images in {cell_dir} between {dt_start} and {dt_end} that match {matchstr}.')
                else:
                    logger.info(f' there are {len(ts_stack)} images in {cell_dir} between {dt_start} and {dt_end} for {img_type}')
                
            logger.debug(f"ts_stack: {ts_stack}")
            
            if params['log_level']=='DEBUG':
                logme=True
                outdir=ppaths.test
                outdir.mkdir(parents=True, exist_ok=True)
            else:
                logme=False
                outdir=None

            if load_samp:
                polys=None
                ptvals = get_index_vals_at_pts(ts_stack, ts_type, img_type, polys, sidx, npts=1, seed=88, 
                                            load_samp=True, ptgdb=points,printfile=logme,out_dir=outdir,params=params)
            else:
                ptvals = get_index_vals_at_pts(ts_stack, ts_type, img_type, polys, sidx, npts, seed=seed, 
                                            load_samp=False, ptgdb=None,printfile=logme,out_dir=outdir,params=params)

            ptvals.drop(columns=['geometry'], inplace=True)
            allpts = pd.concat([allpts, ptvals])
            logger.info(f"there are now {allpts.shape[0]} pts processed in the pts db")

        else:
            logger.info('skipping this cell')
            pass

    allpts.set_index('pt', inplace=True, drop=True)
    ts = allpts.transpose()
    ts['date'] = [pd.to_datetime(e[:4]) + pd.to_timedelta(int(e[4:]) - 1, unit='D') for e in ts.index]
    ##Note columns are all object due to mask. Need to change to numeric or any NA will result in  NA in average.
    logger.debug(ts.dtypes)
    cols = ts.columns[ts.dtypes.eq('object')]
    for c in cols:
        ts[c] = ts[c].astype(float)
    logger.debug(ts.dtypes)
    ##There are a lot of 9s...
    #ts = ts.replace(9, np.nan)
    ts.set_index('date', inplace=True)
    #ts.index.names = ['date']
    ts=ts.sort_index()

    ts['ALL'] = ts.median(axis=1)
    ts['std'] = ts.std(axis=1)

    procprefix = get_tsdir_name(params)
    if procprefix == '' or procprefix == 'ms':
        siall = si
    else:
        siall = f'{procprefix}_{si}'

    if filter_class:
        prefix = f"{siall}_{filter_col.replace('_','-')}-{filter_class.replace('_','-')}"
    else:
        prefix = f"{siall}"
        
    if load_samp and (isinstance(ptfile,gpd.GeoDataFrame) or ptfile.endswith('Coords')):
        if len(cells) == 1:
            outfile = Path(spectsdf_dir) / f"Cell{cells}Coords_{prefix}_{img_type}_{start_yr}-{end_yr}.csv"
        else:
            outfile = Path(spectsdf_dir) / f"Coords_{prefix}_{img_type}_{start_yr}-{end_yr}.csv"
    elif len(cells) == 1:
        outfile = Path(spectsdf_dir) / f"Cell{cell}_{prefix}_{img_type}_{start_yr}-{end_yr}.csv"
    else:
        if params['explore']['sig_prefix']:
            outfile = Path(spectsdf_dir) / f"{sig_prefix}_{prefix}_{img_type}_{start_yr}-{end_yr}.csv"
        else:    
            outfile = Path(spectsdf_dir) / f"{prefix}_{img_type}_{start_yr}-{end_yr}.csv"
    
    pd.DataFrame.to_csv(ts, outfile, sep=',', na_rep='NaN', index=True)
    logger.info(f" Printing final dataframe to: {outfile}")
    
    return ts
    
def get_time_slice(names, times, start_slice, end_slice):
    '''
    Takes two correspoinding lists of file paths (<names>) and dates (<times>), merges into a daframe and slices by 
        designated start and stop times (<start_slice> and <end_slice>, in format 'YYYY-mm-dd').
    <times> column already contains datetime objects
    Works well with geowombat sort_images_by_date method (TODO: replace this method with local code)
    start and end dates need to start with the first of the month, or strange errors occur
    '''
    df = pd.DataFrame(data=names,
                      columns=['names'],
                      index=times)
    logger.info(f"there are {len(df)} images for this cell......")
    df = df[start_slice:end_slice]
    df['time'] = df.index
    logger.info(f"there are {len(df)} images in the time slice from {start_slice} to {end_slice}")
    
    ## note this should not matter unless using completely raw data because duplicate dates are removed in brdf processing
    dupe_first = df['time'].duplicated(keep='first')
    dupe_last = df['time'].duplicated(keep='last')
    # All unique dates -> ~(first | last)
    # or one of the duplicates -> | last
    df = df[~(dupe_first | dupe_last) | dupe_last]
    logger.info(f"     ......and{len(df)} images after")
    
    image_names = df['names'].values.tolist()
    time_names = df.index.tolist()

    return image_names, time_names, df

def load_ts_from_file(ts_file):
    ts = pd.read_csv(ts_file)
    ts.set_index('date', drop=True, inplace=True)
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()

    return ts

def prep_ts_for_plotting(si, params):
    found_ts = False
    params['feature_model']['spec_indices'] = [si]
    filter_col = params['sample_model']['filter_col']
    filter_class = params['sample_model']['filter_class'] ## str -- optional
    img_type = params['image_type']
    start_yr = params['sample_model']['train_yrs'][0]
    end_yr = params['sample_model']['train_yrs'][1]
    prefix = params['explore']['sig_prefix']

    plot_start, plot_end1 = get_date_range(start_yr,'yr',params,return_type='dt',padded=params['feature_model']['pheno_pad_days'])
    plot_end = str(end_yr) + str(plot_end1)[4:]
    logger.info(f"plot should be from {plot_start} to {plot_end}") 

    spectsdf_dir = params['explore']['spectsdf_dir'] ## str -- optional
    if not spectsdf_dir:
        ppaths=ProjectPaths(params)
        spectsdf_dir = ppaths.tssigs

    ## can use list of coordinates as input with <plot:coords>. Otherwise, uses <ptfile> or polygons based on other parameters
    if params['plot']['coords']: ## coords should be in format [Lon1, lat1, Lon2, Lat2...]
        lons = []
        lats = []
        params['sample_model']['filter_col'] = ''
        params['sample_model']['filter_class'] = ''
        params['sample_model']['load_samp'] = True
        with gpd.read_file(params['grid_file']) as gf:
            crs_grid = gf.crs    
        for ci in range(0, len(params['plot']['coords']), 2):
            lons.append(ci)
            lats.append(ci+1)
        ptdf = pd.DataFrame({'Lat': lats, 'Lon': lons})
        params['sample_model']['point_file'] = gpd.GeoDataFrame(geometry=gpd.points_from_xy(ptdf.Lon, ptdf.Lat),crs=crs_grid)
        ts = sample_timeseries(params) 

    ## will look for default file name below, but can load ts with a different name using <plot:existing_ts>
    elif params['plot']['existing_ts']:
        ts_in = load_ts_from_file(params['plot']['existing_ts'])
        ts = ts_in[(ts_in.index >= plot_start) & (ts_in.index <= plot_end)]
    
    else:       
        if isinstance (params['grids'],int) or len(params['grids'] == 1):
            if isinstance (params['grids'],int):
                cell = params['grids']
            else:
                cell = params['grids'][0]   
            if filter_class:
                ts_file = Path(spectsdf_dir) / f"Cell{cell}_{si}_{filter_col.replace('_','-')}-{filter_class.replace('_','-')}_{img_type}_{start_yr}-{end_yr}.csv"
                alt_files = list(spectsdf_dir.glob(f"Cell{cell}_{si}_{filter_col.replace('_','-')}-{filter_class.replace('_','-')}_{img_type}_*.csv"))
            else:
                ts_file = Path(spectsdf_dir) / f"Cell{cell}_{si}_{img_type}_{start_yr}-{end_yr}.csv"
                alt_files = list(spectsdf_dir.glob(f"Cell{cell}_{si}_{img_type}_*.csv"))

        elif filter_class:
            ts_file = Path(spectsdf_dir) / f"{prefix}_{si}_{filter_col.replace('_','-')}-{filter_class.replace('_','-')}_{img_type}_{start_yr}-{end_yr}.csv"
            alt_files = list(spectsdf_dir.glob(f"{prefix}_{si}_{filter_col.replace('_','-')}-{filter_class.replace('_','-')}_{img_type}_*.csv"))
        else:
            ts_file = Path(spectsdf_dir) / f"{prefix}_{si}_{img_type}_{start_yr}-{end_yr}.csv"
            alt_files = list(spectsdf_dir.glob(f"{prefix}_{si}_{img_type}_*.csv"))
        
        if ts_file.is_file():
            found_ts = True
            logger.info(f'using existing ts file: {ts_file}')
            ts = load_ts_from_file(ts_file)    
        elif len(alt_files) > 0:
            for f in alt_files:
                yrs = f.name.split('_')[-1].split('.')[0]
                ts_start_yr = int(yrs.split('-')[0])
                ts_end_yr = int(yrs.split('-')[1])
                if (ts_start_yr <= start_yr) & (ts_end_yr >= end_yr):
                    found_ts = True
                    logger.info(f'found existing ts file that covers time period: {f}')
                    ts_in = load_ts_from_file(f) 
                    ts = ts_in[(ts_in.index >= plot_start) & (ts_in.index <= plot_end)]
        if not found_ts:
            logger.info('making new ts file...\n')
            ts = sample_timeseries(params) 
        ts = ts[ts['ALL'] != 0]
        if params['plot']['multi_coords'] and (params['plot']['multi_coords']=='median' or params['plot']['multi_coords'].endswith('w_med')):
            ts = ts
        else:
            ts = ts.drop(columns=['ALL','std'])
            
        if params['plot']['limit_coords']:
            if isinstance(params['plot']['limit_coords'],list):
                ts.columns = ts.columns.astype(int)
                ts_cols = ts.columns.to_list()
                match_cols = [p for p in params['plot']['limit_coords'] if p in ts_cols]
                ts=ts[match_cols]
            elif params['plot']['limit_coords'].startswith('ran'):
                numsamp = int(params['plot']['limit_coords'].split('_')[1])
                logger.info(f"sampling {numsamp} random points to plot from {ts.shape[1]} results")
                if numsamp < ts.shape[1]:
                    ts = ts.sample(n=numsamp, axis=1)
                ## save sampled pts as list so that a new sample is not pulled for each iteration
                new_pts = ts.columns.astype(int).to_list()
                params['plot']['limit_coords'] = new_pts
    return ts

    
def plot_timeseries(params):
    '''
    Plots a single or multiple time series for the period <plot:start_yr>, <plot:end_yr> (with calendar applied)
    input is a timeseries already saved from "sample_timeseries" above, a list of coordinates with format [Lon1, lat1, Lon2, Lat2...],
          or parameters set to create new times series with the "sample_timeseries" method.
          Smoothed data needs to already be calculated and in ts folder.  Will calculate raw data on the fly.
          Can limit points to use based on <filter_col> with <filter_class> and with <plot:limit_coords> to reduce the number being plotted
              <plot:limit_coords> can be a list of pt ids to include (ids = OID_) or 'ran_X', where X is sample size). Default is 'all'
    output options:
       -- Single plot for single point with multiple smoothed indices, either of different spectral indices or different smoothing methods
       -- Single plot for multiple polints for single index (<plot:multi_coords> != False). Points can be shown as: 
            -- separate lines for all points in time series (<plot:multi_coords> == 'individual')
            -- single median line with error bars showing standard deviation (plot:multi_coords> == 'median')
            -- plot with both individual lines and median plot:multi_coords> == 'individual_w_med'
        -- Plot(s) for single point comparing smoothed and raw estimates
        -- Plot(s) comparing multiple classes on single index plot (<plot:filter_class> given as list). Can be single line for each or multi.  
        -- Plot(s) for single point comparing observations from different sensors (set with <image_type>)
               (for different Landsat sensors, use [''LT05','LE07','LC08','LC09']. For Sentinel vs Landsat, use ['L','S2'],etc.)
        Last two options will create a separate plot for each point and/or each spectral index
        Point labels are the OID_ value
    '''

    def get_row_col(n, numcols):
        row = n // numcols 
        col = n % numcols
        return row, col

    def get_grid_layout(params,ts0):
        '''
        define plot layout -- if <plot:multicoords> == True, all points are plotted on same plot. Otherwise, make subplot for each point
        each index base will also be plotted on new subplot unless <plot:multi_idx> == True (to compare multiple indices in one plot)
        raw indices or different variations of smoothed indices will automatically share a plot with any other versions of the same base
        '''
        if params['plot']['multi_coords']:
            if params['plot']['multi_idx']:
                numplots,numcols,numrows = 1,1,1
            else:
                numplots = len(idx_bases)
                numcols = 2
                numrows = math.ceil(numplots/2)
        else:
            numplots = ts0.shape[1]
            if numplots <= 3:
                numcols = numplots
                numrows = 1
            elif numplots == 4:
                numcols = 2
                numrows = 2
            else:
                numcols = 3
                numrows = math.ceil(numplots/3)

        return numcols, numrows

    def get_namedet(params):    
        if isinstance (params['grids'],int) or len(params['grids'] == 1):
            if isinstance (params['grids'],int):
                cell = params['grids']
            else:
                cell = params['grids'][0]
        
        if params['sample_model']['load_samp'] and (isinstance(params['sample_model']['point_file'],gpd.GeoDataFrame) 
                                                    or params['sample_model']['point_file'].endswith('Coords')):
            coordtext = 'SampleCoords'
        else:
            coordtext = ''
        
        if params['sample_model']['filter_class']:
            if isinstance(params['sample_model']['filter_class'],list):
                if len(params['sample_model']['filter_class']) == 2:
                    filterclass = f"{params['sample_model']['filter_class'][0]}_vs_{params['sample_model']['filter_class'][1]}"
                elif len(params['sample_model']['filter_class']) == 3:
                    filterclass = f"{params['sample_model']['filter_class'][0]}_vs_{params['sample_model']['filter_class'][1]}vs_{params['sample_model']['filter_class'][2]}"
                else: filterclass = 'multiple_classes'
            else:        
                filterclass = f"{(params['sample_model']['filter_col']).replace('_','-')}-{params['sample_model']['filter_class'].replace('_','-')}"

            if cell:
                namedet = f'{filterclass}_{coordtext}{cell}'
            else:
                namedet = filterclass
        else: 
            namedet = f'{coordtext}{cell}'

        return namedet
        
    def format_figure(params,fig,title,out_path,lines=None, labels=None):
        fig.suptitle(title)
        if params['plot']['legend'] and 'sub' not in params['plot']['legend']:
            fig.legend(lines, labels, loc=params['plot']['legend'])
        #fig.tight_layout(pad=0.1)
        plt.savefig(out_path)
        logger.info(f'saved figure {out_path}')
        return fig

    def format_subplot(axs, params):
        if params['plot']['wet_dry']:
            dt_start, dt_end1 = get_date_range(params['sample_model']['train_yrs'][0],'yr',params,return_type='doy',padded=False)
            if params['calendar']['start_wet'] >= dt_start:
                start_wet = pd.to_datetime(params['calendar']['start_wet'], unit='D', origin=str(params['plot']['start_yr']))
            else:
                start_wet = pd.to_datetime(params['calendar']['start_wet'], unit='D', origin=str(params['plot']['start_yr'] + 1))
            if params['calendar']['end_wet'] >= dt_start:
                end_wet = pd.to_datetime(params['calendar']['end_wet'], unit='D', origin=str(params['plot']['start_yr']))
            else:
                end_wet = pd.to_datetime(params['calendar']['end_wet'], unit='D', origin=str(params['plot']['start_yr'] + 1))
            if params['calendar']['start_dry'] >= dt_start:    
                start_dry = pd.to_datetime(params['calendar']['start_dry'], unit='D', origin=str(params['plot']['start_yr']))
            else:
                start_dry = pd.to_datetime(params['calendar']['start_dry'], unit='D', origin=str(params['plot']['start_yr'] + 1))
            if params['calendar']['end_dry'] >= dt_start:    
                end_dry = pd.to_datetime(params['calendar']['end_dry'], unit='D', origin=str(params['plot']['start_yr']))
            else:
                end_dry = pd.to_datetime(params['calendar']['end_dry'], unit='D', origin=str(params['plot']['start_yr'] + 1))    
            axs.plot([start_wet,start_wet], [params['plot']['ylim'][0],params['plot']['ylim']][1], color='skyblue')
            axs.plot([end_wet,end_wet], [params['plot']['ylim'][0],params['plot']['ylim']][1], color='skyblue')
            axs.plot([start_dry,start_dry], [params['plot']['ylim'][0],params['plot']['ylim']][1], color='gold')
            axs.plot([end_dry,end_dry], [params['plot']['ylim'][0],params['plot']['ylim']][1], color='gold')
        
        idx_bases = list(set([si.split('-')[0] for si in params['feature_model']['spec_indices']]))
        if len(idx_bases) > 1:
            axs.set_ylabel('index value')
        else:
            axs.set_ylabel(idx_bases[0].upper())
        axs.set_xlabel('Date')
        if params['plot']['ylim']:
            axs.set_ylim(params['plot']['ylim'][0], params['plot']['ylim'][1])
        axs.tick_params(axis='x', labelrotation=45)
        if (params['plot']['legend']) and (params['plot']['legend'].startswith('sub')):
            legend_loc = (params['plot']['legend']).split('-')[1]
            axs.legend(loc=legend_loc)
        return axs
        
    if params['plot']['out_path']:
        figdir = Path(params['plot']['out_path'])
    else:
        ppaths=ProjectPaths(params)
        figdir = ppaths.figs
    logger.info(f'figdir = {figdir}')
    figdir.mkdir(parents=True, exist_ok=True)
    
    sensors = params['image_type']
    if isinstance(sensors,str):
        sensors = [sensors]
    ## set image type to first item in list if list was given because most methods expect single image type. Use 'sensors' to iterate later
    params['image_type'] = sensors[0]
    plot_multiidx = False
    plot_multicoords = False
    
    ## get temporal range of plot -- uses <sample_model':train_yrs> by default, but can alter with <plot:start_yr> and <plot:end_yr>
    orig_train_yrs = params['sample_model']['train_yrs']
    if params['plot']['start_yr']:
        params['sample_model']['train_yrs'] = [params['plot']['start_yr']]
    if params['plot']['end_yr'] and (params['plot']['end_yr'] != params['plot']['start_yr']):
        params['sample_model']['train_yrs'].append(params['plot']['end_yr'])
    
    if len(params['sample_model']['train_yrs']) > 1:
        startyr, endyr = params['sample_model']['train_yrs'][0], params['sample_model']['train_yrs'][1]
    else: 
        startyr = params['sample_model']['train_yrs'][0]

    namedet_main = get_namedet(params)
    if params['sample_model']['filter_class']:
        if isinstance(params['sample_model']['filter_class'],list):
            classes = params['sample_model']['filter_class']
        else:
            classes = [params['sample_model']['filter_class']]
        if len(classes) > 1:
            params['sample_model']['filter_class'] = classes[0]         
    else:
        params['sample_model']['filter_class'] = None
        classes = []
        namedet_samp = get_namedet(params)

    all_sis = params['feature_model']['spec_indices']
    idx_bases = list(set([si.split('-')[0] for si in all_sis]))
    logger.info(f'idx_bases: {idx_bases}')
    smoothed_sis = list(set([s for s in all_sis if 'raw' not in s]))
    raw_sis = list(set([s for s in all_sis if 'raw' in s]))
            
    for idx in idx_bases:
        logger.info(f'working on {idx}...')
        idx_sm = [ix for ix in smoothed_sis if ix.split('-')[0] == idx]
        idx_raw = [ix for ix in raw_sis if ix.split('-')[0] == idx]
        
    if len(idx_sm) > 0:
        ## prep the first index to get the plot dimensions
        ts0 = prep_ts_for_plotting(idx_sm[0], params)
        numcols, numrows = get_grid_layout(params,ts0)    
        fig, axs = plt.subplots(nrows=numrows, ncols=numcols, squeeze=False, figsize=(15, 5*numrows), dpi=params['plot']['dpi'], layout="constrained")

        if params['plot']['multi_idx']:
            plot_multiidx = True
            break
        
        if params['plot']['multi_coords']:
            plot_multicoords = True
            break
        
        #if not plot_multiidx and not plot_multicoords:    
        if len(idx_sm) > 1:
            ## making plot comparing multiple smoothed indices (e.g. NDVI_smwh, NDVI_smsg)
            logger.info(f'making plot comparing different smoothing methods for {idx}: {[sm[-2:] for sm in idx_sm]}')
            if params['sample_model']['filter_class']:
                title = "Smoothing method comparison of {idx} for {params['sample_model']['filter_class']}"
            else:
                title = "Smoothing method comparison of {idx}"
            out_fig = Path(figdir) / f"Smoothing_comparison_{idx}_{namedet_main}_{startyr}-{endyr}.png"
            n_colors = len(idx_sm)
            cmap = plt.get_cmap(params['plot']['color'], n_colors)
            colors = [cmap(i) for i in range(n_colors)]
            for i, idxs in enumerate(idx_sm):
                if i == 0:
                    ## first smoothed ts is already calculated above
                    ts = ts0
                else:
                    ts = prep_ts_for_plotting(idxs, params)
                for p, pt in enumerate(ts.columns.to_list()):
                    row, col = get_row_col(p, numcols)
                    axs[row, col].plot(ts.index, ts[pt], color=colors[i], label=idxs)
                    axs[row, col].set_title(f'pt{pt}', y=1.0, pad=-14)
            lines, labels = axs[0,0].get_legend_handles_labels()
            params['feature_model']['spec_indices'] = all_sis
            fig_final = format_figure(params,fig,title,out_fig,lines,labels)
                
        elif len(idx_sm) > 0 and len(idx_raw) > 0: 
            ## making plot comparing smoothed to raw indices (e.g. NDVI, NDVI-raw)
            logger.info(f'making smooth vs raw plot for {idx}')
            if len(classes) >0:
                if len(classes) == 2:
                    title = f"{idx} signatures for {classes[0]} vs {classes[1]}"   
                elif len(classes) > 2:
                    title = f"{idx} signatures"
                else:
                    title = f"Smoothed vs raw {idx} for {classes[0]}"
            else:
                title = f"Smoothed vs raw {idx}"
            out_fig = Path(figdir) / f"Smooth_Vs_Raw_{idx}_{namedet_main}_{startyr}-{endyr}.png"
            ## starting with first smoothed ts, which is ts0 from above:
            if len(classes) > 1:
                n_colors = len(classes)
                cmap = plt.get_cmap(params['plot']['color'], n_colors)
                # Access individual colors via index (0 to n_colors-1)
                colors = [cmap(i) for i in range(n_colors)]
            for c, lcclass in enumerate(classes):
                if c==0:
                    tss = ts0
                else:
                    params['sample_model']['filter_class'] = lcclass         
                    tss = prep_ts_for_plotting(idx, params)
                for p, pt in enumerate(ts0.columns.to_list()):
                    row, col = get_row_col(p, numcols)
                    if len(classes) > 1:
                        color = colors[p]
                        label = classes[p]
                    else:
                        color = 'k'
                        label = 'smoothed'
                    axs[row, col].plot(tss.index, tss[pt], color=color, label=label)
                if len(classes) > 1 and f'{idx}-raw' in all_sis:
                    tsr = prep_ts_for_plotting(f'{idx}-raw', params)
                    for p, pt in enumerate(tsr.columns.to_list()):
                        row, col = get_row_col(p, numcols)
                        axs[row, col].scatter(tsr.index, tsr[pt], color=colors[0], edgecolor='white', lw=0.1, s=10, label=lcclass)
                else:
                    for sen in sensors:    
                        params['image_type'] = sen
                        tsr = prep_ts_for_plotting(f'{idx}-raw', params)                
                        for p, pt in enumerate(tsr.columns.to_list()):
                            row, col = get_row_col(p, numcols)
                            axs[row, col].scatter(tsr.index, tsr[pt], color=SENSORS[sen]['color'], 
                                    edgecolor='white', lw=0.1, s=10, label=f"raw   { SENSORS[sen]['name']}")
            for p, pt in enumerate(tss.columns.to_list()):
                row, col = get_row_col(p, numcols)
                if params['plot']['coords']:
                    pt = f"{float(params['plot']['coords'][p*2]):.04f}_{float(params['plot']['coords'][(p*2)-1]):.04f}"
                axs[row, col].set_title(pt, y=1.0, pad=-14)
                axs[row,col] = format_subplot(axs[row,col], params)
            lines, labels = axs[0,0].get_legend_handles_labels()
            params['feature_model']['spec_indices'] = all_sis
            fig_final = format_figure(params,fig,title,out_fig,lines,labels)
        
        elif len(smoothed_sis) > 0:
            if isinstance(classes, list) and len(classes) > 1:
                logger.info(f'plotting multiple classes on single plot...')
                if len(classes) == 2:
                    title = f"{idx} signatures for {classes[0]} vs {classes[1]}"   
                else:
                    title = f"{idx} signatures"
                out_fig = Path(figdir) / f"Smoothed_{idx}_{namedet_main}_{startyr}-{endyr}.png"
                n_colors = len(classes)
                cmap = plt.get_cmap(params['plot']['color'], n_colors)
                # Access individual colors via index (0 to n_colors-1)
                colors = [cmap(i) for i in range(n_colors)]
                    
                for p, pt in enumerate(ts0.columns.to_list()):
                    row, col = get_row_col(p, numcols)
                    axs[row, col].plot(ts0.index, ts0[pt], color=colors[0], label=classes[0])
                for c, lcclass in enumerate(classes):
                    if c==0:
                        pass
                    else:
                        params['sample_model']['filter_class'] = lcclass         
                        tss = prep_ts_for_plotting(idx, params)
                        for p, pt in enumerate(tss.columns.to_list()):
                            row, col = get_row_col(p, numcols)
                            axs[row, col].plot(tss.index, tss[pt], color=colors[c], label=lcclass)
                for row in range(0,numrows):
                    for col in range(0,numcols):
                        axs[row,col] = format_subplot(axs[row,col], params)
                lines, labels = axs[0,0].get_legend_handles_labels()
                params['feature_model']['spec_indices'] = all_sis
                fig_final = format_figure(params,fig,title,out_fig,lines,labels)
                
            else:
                ## multiple indices on one plot; Need to extract from index loop first
                plot_multiidx = True
                break
            
    else:
        ## No smoothed indices. Making simple raw index plot
        logger.info(f'making plot of raw {idx}...')
        params['image_type'] = sensors[0]
        tsr0 = prep_ts_for_plotting(idx_raw[0], params)
        numcols, numrows = get_grid_layout(params,tsr0)    
        fig, axs = plt.subplots(nrows=numrows, ncols=numcols, squeeze=False, figsize=(15, 5*numrows), dpi=params['plot']['dpi'], layout="constrained")
                
        if len(classes) > 1:
            logger.info(f'plotting multiple classes on single plot...')
            if len(classes) == 2:
                title = f"{idx} signatures for {classes[0]} vs {classes[1]}"   
            else:
                title = f"{idx} signatures"
            out_fig = Path(figdir) / f"Raw_{idx}_{namedet_main}_{startyr}-{endyr}.png"
            n_colors = len(classes)
            cmap = plt.get_cmap(params['plot']['color'], n_colors)
            # Access individual colors via index (0 to n_colors-1)
            colors = [cmap(i) for i in range(n_colors)]
            for c, lcclass in enumerate(classes):
                if c == 0:
                    tsr = tsr0
                else:
                    params['sample_model']['filter_class'] = lcclass         
                    tsr = prep_ts_for_plotting(f'{idx}-raw', params)
                    for p, pt in enumerate(tsr.columns.to_list()):
                        row, col = get_row_col(p, numcols)
                        axs[row, col].scatter(tsr.index, tsr[pt], edgecolor='white', lw=0.1, s=10, color=colors[c], label=lcclass)
            out_fig = Path(figdir) / f"{idx}_simple_raw_signature_{namedet_main}.png"
            title = ''
            for s, sen in enumerate(sensors):
                if s == 0:
                    tsr = tsr0
                else:
                    logger.info(f'getting info for sensor: {sen}')
                    params['image_type'] = sen
                    tsr = prep_ts_for_plotting(idx_raw[0], params) 
                for p, pt in enumerate(tsr.columns.to_list()):
                    row, col = get_row_col(p, numcols)
                    axs[row, col].scatter(tsr.index, tsr[pt],color=SENSORS[sen]['color'], 
                            edgecolor='white', lw=0.1, s=10, label=f"raw  {SENSORS[sen]['name']}")
                            
        for p, pt in enumerate(tsr.columns.to_list()):   
            row, col = get_row_col(p, numcols)
            if params['plot']['coords']:
                pt = f"{float(params['plot']['coords'][((p+1)*2)-1]):.04f}_{float(params['plot']['coords'][((p+1)*2)-2]):.04f}"
            if title:
                axs[row, col].set_title(pt,y=1.0, pad=-14)
            else:
                axs[row, col].set_title(pt)
            axs[row,col] = format_subplot(axs[row,col], params)
        lines, labels = axs[0,0].get_legend_handles_labels()
        params['feature_model']['spec_indices'] = all_sis
        fig_final = format_figure(params,fig,title,out_fig,lines,labels)
                    
    if plot_multiidx:
        ## Making plot of multiple smoothed indices (e.g. NDVI, NBR, WI, etc.)
        logger.info(f'Making plot comparing smoothed indices:{smoothed_sis}')
        title = f"index comparison for {params['sample_model']['filter_class']}"
        out_fig = Path(figdir) / f"MultiIndex_comparison_{namedet_main}_{startyr}-{endyr}.png"
        n_colors = len(smoothed_sis)
        cmap = plt.get_cmap(params['plot']['color'], n_colors)
        colors = [cmap(i) for i in range(n_colors)]
        for i, idxs in enumerate(smoothed_sis):
            if i == 0:
                ## first smoothed ts is already calculated above
                ts = ts0
            else:
                ts = prep_ts_for_plotting(idxs, params)
            
            for p, pt in enumerate(ts.columns.to_list()):
                row, col = get_row_col(p, numcols)
                axs[row, col].plot(ts.index, ts[pt], color=colors[i], label=idxs)
            if f"{idxs.split('-')[0]}-raw" in raw_sis:
                ts = prep_ts_for_plotting(f"{idxs.split('-')[0]}-raw", params)
                for p, pt in enumerate(ts.columns.to_list()):
                    row, col = get_row_col(p, numcols)
                    axs[row, col].scatter(ts.index, ts[pt], color=colors[i])
            
        for p, pt in enumerate(ts.columns.to_list()):
            row, col = get_row_col(p, numcols)
            if params['plot']['coords']:
                pt = f"{float(params['plot']['coords'][((p+1)*2)-1]):.04f}_{float(params['plot']['coords'][((p+1)*2)-2]):.04f}"
            axs[row, col].set_title(pt,y=1.0, pad=-14)
            axs[row, col] = format_subplot(axs[row, col], params)
        lines, labels = axs[0,0].get_legend_handles_labels()
        params['feature_model']['spec_indices'] = all_sis
        fig_final = format_figure(params,fig,title,out_fig,lines, labels)

    elif plot_multicoords:
        ## Making spectral signature plot with multiple points on sampe plot (note: will work with just one point, too)
        logger.info(f'making spectral signature plot of multiple points')
        if len(classes) == 1:
            title = f"{params['sample_model']['filter_class']} signatures"
        else:
            title = "Sample signatures"
        out_fig = Path(figdir) / f"Signatures_{namedet_main}.png"
        if len(classes) > 1:
            n_colors = len(classes)
        else:
            n_colors = ts0.shape[1]
        cmap = plt.get_cmap(params['plot']['color'], n_colors)
        # Access individual colors via index (0 to n_colors-1)
        colors = [cmap(i) for i in range(n_colors)]

        if not ts0:
            ts0 = tsr0
            
        for i, idx in enumerate(idx_bases):
            logger.info(f'working on {idx}...')
            row,col = get_row_col(i, numcols)
            for c, lcclass in enumerate(classes):
                tss = None
                if i == 0 and c == 0:
                    tss = ts0
                else:
                    params['sample_model']['filter_class'] = lcclass
                    if idx in all_sis:
                        tss = prep_ts_for_plotting(idx, params)
                if tss is not None:
                    if params['plot']['multi_coords'] == 'median':
                        if len(classes) <= 1:
                            axs[row,col].errorbar(tss.index, tss['ALL'], yerr=tss['stdv'], fmt='o', color='k')
                        else:
                            axs[row,col].errorbar(tss.index, tss['ALL'], yerr=tss['stdv'], fmt='o', color=colors[c])
                    else: 
                        for p, pt in enumerate(tss.columns.to_list()):
                            if (pt != 'ALL') and (pt !='std'):  ## these will only exist if <plot:multi_coords> = individual_w_med
                                if len(classes) <= 1:
                                    if params['plot']['color'] == 'mono':
                                        axs[row,col].plot(tss.index, tss[pt], color='dark grey')
                                    else:
                                        if ts0.shape[1] < 10:   ## only label pts if there is a manageable amount!
                                             axs[row,col].plot(tss.index, tss[pt], color=colors[p],label=pt)
                                        else:
                                            axs[row,col].plot(tss.index, tss[pt], color=colors[p])
                                else:
                                    axs[row,col].plot(tss.index, tss[pt], color=colors[c])
                            elif (pt == 'ALL'):
                                if len(classes) <= 1:
                                    axs[row,col].plot(tss.index, tss[pt], color='k', linewidth=4, label='med')
                                else:
                                    axs[row,col].plot(tss.index, tss[pt], color=colors[c], linewidth=4, label=lcclass)

                if f'{idx}-raw' in all_sis:
                    params['feature_model']['spec_indices'] = f'{idx}-raw'
                    tsr = prep_ts_for_plotting(f'{idx}-raw', params)
                    for p, pt in enumerate(tss.columns.to_list()):
                        if len(classes) <= 1:
                            axs[row,col].scatter(tsr.index, tsr[pt], edgecolor='white', lw=0.1, s=10, color=colors[p])
                        else:
                            axs[row,col].scatter(tsr.index, tsr[pt], edgecolor='white', lw=0.1, s=10, color=colors[c])
                                      
            axs[row,col].set_title(f'{idx} signatures')        
            axs[row,col] = format_subplot(axs[row,col], params)
        params['feature_model']['spec_indices'] = all_sis
        lines, labels = axs[0,0].get_legend_handles_labels()
        fig_final = format_figure(params,fig,title,out_fig,lines, labels)

        
    params['image_type'] = sensors 
    params['sample_model']['train_yrs'] = orig_train_yrs
    
    return fig_final
    

def pre_post_df(params):
    '''
    Returns None. Prints a dataset of pre/post spectral index values for an event (e.g. burning, harvest). 
             Includes "postdelta" and "predelta" column with the number of days between the observation date "eventobs" and the closest clear RS viewing.
    Inputs: -- <obs_db> = Dataframe (or .csv file) with cols <sample_model:obscol> <sample_model:preobscol> and "OID_" (id column for linking). 
                  The Post-event value will be from the first observation after the data of the event, recorded in <obscol>. 
                  Pre-event values will be from the closest observation prior to or on the pre-event observation <preobs_col>
            -- timeseries database of the spectral index, named <si>_<filter_col>-<filter_class>_<img_type>_<yrs[0]}-{yrs[1]>.csv', column names = "OID_" 
    '''
    
    ## Args
    obs_db = params['sample_model']['point_file'] ## str
    obscol = params['sample_model']['obs_col'] ## str  
    preobscol = params['sample_model']['preobs_col'] ## str
    filter_col = (params['sample_model']['filter_col']).replace('_','-') ## str -- just for file naming/searching here
    filter_class = (params['sample_model']['filter_class']).replace('_','-') ## str -- just for file naming/searching here
    sis = params['feature_model']['spec_indices'] ## str or list
    yrs = params['sample_model']['train_yrs'] ## list [XXXX,XXXX]
    img_type = params['image_type'] ## str 'LS2'| 'S2' | 'L'| 'LT05'| 'LE07'| 'LC08'| 'LC09'| 'PS'| 'CHIRPS_'
    #<spectsdir> holds the input timeseries data
    spectsdf_dir = params['explore']['spectsdf_dir'] ## str
    ## <spectsdf_dir> holds sample data for preliminary exploration
    if not spectsdf_dir:
        ppaths=ProjectPaths(params)
        spectsdf_dir = ppaths.tssigs
    #<opt_dir> holds outputs of exploratory and optimization outputs
    opt_dir = params['iter_models']['opt_dir'] ## str
    if not opt_dir:
        ppaths=ProjectPaths(params)
        opt_dir = ppaths.optimization
    def get_closest_post_date(x, obscol, date_cols):
        future_dates = [d for d in date_cols if d >= x[obscol]]
        done=False
        if not future_dates:                
            done=True
            return 999
        for fd in future_dates:
            if not done:
                if (pd.notna(x[fd])) and (x[fd]>0):
                    done=True
                    return fd

    def get_closest_pre_date(x, obscol, date_cols):
        prior_dates = [d for d in date_cols if d < x[obscol]]
        done=False
        if not prior_dates:                
            done=True
            return -999
        else:
            prior_dates.sort(reverse=True)
            for prd in prior_dates:
                if not done:
                    if (pd.notna(x[prd])) and (x[prd]>0):
                        done=True
                        return prd
                        
    if isinstance(sis,str):
        sis = [sis]
        params['feature_model']['spec_indices'] = sis
    for si in sis:
        logger.info(f'...for {si}... \n')
        if '-' in si:
            sidx = si.split('-')[0]
        else:
            sidx = si
        if 'raw' in si:
            ts_type = 'raw'
        else:
            ts_type = 'smooth'

        sig_prefix = params['explore']['sig_prefix']
        procprefix = get_tsdir_name(params)
        if procprefix == '' or procprefix == 'ms':
            siall = si
        else:
            siall = f'{procprefix}_{si}'
        if params['project_ver'] == 'Py0' or params['project_ver'] == 'Biltong0':
            spectsdf_base = f'{sig_prefix}_{si}_{filter_col}-{filter_class}_{img_type}_{yrs[0]}-{yrs[1]}.csv'
        else:
            spectsdf_base = f'{sig_prefix}_{siall}_{filter_col}-{filter_class}_{img_type}_{yrs[0]}-{yrs[1]}.csv'
        ts = pd.read_csv(Path(spectsdf_dir)/spectsdf_base, index_col=0)
    
        ## convert date-like values to actual dates for use later
        ts.index = pd.to_datetime(ts.index)
        ## remove nonnumeric columns like 'ALL' and 'std" for later join to work
        ts_num = [c for c in ts.columns if c.isdigit() or '.' in str(c)]
        ts = ts[ts_num]
        ## transpose to join pt-specific info
        tsT=ts.T
        ## reset imdex to integer -- some methods are outputting double for some reason TODO: fix this at source
        tsT_rs = tsT.reset_index()
        tsT_rs['OID_'] = pd.to_numeric(tsT_rs['index'], errors='coerce')
        tsT_rs['OID_']= tsT_rs['OID_'].astype('int64')
        tsT = tsT_rs.set_index('OID_')
    
        events = pd.read_csv(obs_db)

        if preobscol:
            pf = events[['OID_',obscol,preobscol]].set_index('OID_')
        else:
            pf = events[['OID_',obscol]].set_index('OID_')
        df = tsT.join(pf, how='left')
        df= df.dropna(subset=[obscol])

        df[obscol] = pd.to_datetime(df[obscol])
        if preobscol:
            df[preobscol] = pd.to_datetime(df[preobscol])
           
        date_cols = [d for d in df.columns if isinstance(d, pd.Timestamp)]
        #df.dropna(axis=1, how='all',inplace=True)
        #df.dropna(axis=0, how='all',inplace=True)
        df['post1_obs'] = df.apply(lambda x: get_closest_post_date(x, obscol, date_cols), axis=1)
        df['event_idx'] = df.apply(lambda x: df.columns.get_loc(x['post1_obs']), axis=1)
        df['postdelta'] = df['post1_obs'] - df[obscol]

        ##do steps above for closest pre-event observation:
        if not preobscol:
            preobscol = obscol
        df['pre1_obs'] = df.apply(lambda x: get_closest_pre_date(x, preobscol, date_cols), axis=1)
        df['predelta'] = df['pre1_obs'] - df[obscol]
    
        df['post1_obs'] = df['post1_obs'].dt.strftime('%Y-%m-%d')
        df['pre1_obs'] = df['pre1_obs'].dt.strftime('%Y-%m-%d')
    
        ts.index = ts.index.strftime('%Y-%m-%d')
        prepost_dict = {}
        for i,r in df.iterrows():
            prepost_dict[i]={}
            prepost_dict[i]['eventobs'] = df.at[i,obscol]
            prepost_dict[i][f'post_{si}'] = 0 if pd.isna(r['post1_obs']) else ts.at[r['post1_obs'], str(i)]
            prepost_dict[i][f'pre_{si}'] = 0 if pd.isna(r['pre1_obs']) else ts.at[r['pre1_obs'],str(i)]
            prepost_dict[i]['postdelta']= r['postdelta']
            prepost_dict[i]['predelta']= r['predelta']

        prepost_ts = pd.DataFrame.from_dict(prepost_dict, orient='index')
        if filter_class:
            out_file = Path(opt_dir)/f"prepost_{obscol.replace('_','-')}_{filter_class.replace('_','-')}_{img_type}_{siall}.csv"
        else:
            out_file = Path(opt_dir)/f"prepost_{obscol.replace('_','-')}_{img_type}_{siall}.csv"
        prepost_ts.to_csv(out_file) 

        logger.info(f'saving to {out_file}')

def pre_post_separability(params, printdf=True):
    '''
    returns summary statistics for each index of pre- and post-event observations including average deltas and delta quantiles 
        in three dataframes: all observations, wet season obseravations, and dry season observations. Option to print output to optimization folder.
    input is a pre-post observation dataframe formated with pre_post_df() method above. Wet and dry season months are defined with <calendar> parameters.
    '''
        
    def make_prepost_dict(dct,df,vi):
        
        sep = get_separability(df,f'pre_{vi}',f'post_{vi}')
        
        dct[vi] = {}
        dct[vi]['n'] = df.shape[0]
        dct[vi]['delta_avg'] = int((df[f'pre_{vi}'] - fulldfa[f'post_{vi}']).mean())
        dct[vi]['delta_std'] = int((df[f'pre_{vi}'] - fulldfa[f'post_{vi}']).std())
        dct[vi]['delta_20'] = int((df[f'pre_{vi}'] - fulldfa[f'post_{vi}']).quantile(0.2))
        dct[vi]['delta_10'] = int((df[f'pre_{vi}'] - fulldfa[f'post_{vi}']).quantile(0.1))
        dct[vi]['delta_5'] = int((df[f'pre_{vi}'] - fulldfa[f'post_{vi}']).quantile(0.05))
        dct[vi]['delta_80'] = int((df[f'pre_{vi}'] - fulldfa[f'post_{vi}']).quantile(0.8))
        dct[vi]['delta_90'] = int((df[f'pre_{vi}'] - fulldfa[f'post_{vi}']).quantile(0.9))
        dct[vi]['delta_95'] = int((df[f'pre_{vi}'] - fulldfa[f'post_{vi}']).quantile(0.95))
        dct[vi]['pre_avg'] = int(df[f'pre_{vi}'].mean())
        dct[vi]['pre_std'] = int(df[f'pre_{vi}'].std())
        dct[vi]['post_avg'] = int(df[f'post_{vi}'].mean())
        dct[vi]['post_std'] = int(df[f'post_{vi}'].std())
        dct[vi]['sep_M'] = sep[0]
        dct[vi]['sep_F1'] = sep[1]
        dct[vi]['sep_bhatt'] = sep[2]
        dct[vi]['sep_jm'] = sep[3]
        dct[vi]['sep_jen'] = sep[4]
        
        return dct

    sis = params['feature_model']['spec_indices']
    obscol = (params['sample_model']['obs_col']).replace('_','-') ## str --for file naming/searching only here 
    filter_col = (params['sample_model']['filter_col']).replace('_','-') ## str --for file naming/searching only here
    filter_class = (params['sample_model']['filter_class']).replace('_','-') ## str --for file naming/searching only here
    img_type = params['image_type'] ## str 'LS2'| 'S2' | 'L'| 'LT05'| 'LE07'| 'LC08'| 'LC09'| 'PS'| 'CHIRPS_'
    maxpost = params['explore']['maxobsgap_post']
    maxpre = params['explore']['maxobsgap_pre']
    
    yrs = params['sample_model']['train_yrs']
    allyr_str = get_train_yrs_str(yrs)
    if isinstance(yrs,str):
        yrs = [yrs]
    elif len(yrs) == 2:
        yrs = [yr for yr in range(yrs[0],yrs[1]+1)]
        logger.info(f'yrs = {yrs}')
   
    opt_dir = params['iter_models']['opt_dir'] ## str
    if not opt_dir:
        ppaths=ProjectPaths(params)
        opt_dir = ppaths.optimization

    ## get seasons to divide datasets by season
    dry_start_mo = datetime.datetime.strptime(f"2024 {params['calendar']['start_dry']}", '%Y %j').month
    dry_end_mo = datetime.datetime.strptime(f"2024 {params['calendar']['end_dry']}", '%Y %j').month
    if dry_start_mo > dry_end_mo: 
        dry_months = [f"{mo:02d}" for mo in range(dry_start_mo,13)] + [f"{mo:02d}" for mo in range(1,(dry_end_mo+1))]
    else:
        dry_months = [f"{mo:02d}" for mo in range(dry_start_mo,dry_end_mo+1)]
    logger.info(f" getting dry season stats using months:{dry_months}")
    wet_start_mo = datetime.datetime.strptime(f"2024 {params['calendar']['start_wet']}", '%Y %j').month
    wet_end_mo = datetime.datetime.strptime(f"2024 {params['calendar']['end_wet']}", '%Y %j').month
    if wet_start_mo > wet_end_mo: 
        wet_months = [f"{mo:02d}" for mo in range(wet_start_mo,13)] + [f"{mo:02d}" for mo in range(1,(wet_end_mo+1))]
    else:
        wet_months = [f"{mo:02d}" for mo in range(wet_start_mo,wet_end_mo+1)]
    logger.info(f" getting wet season stats using months:{wet_months}") 

    all_dict = {}
    dry_dict = {}
    wet_dict = {}
    
    for si in (sis):
        logger.info(f'working on {si}...')
        dfs = []
        
        procprefix = get_tsdir_name(params)
        if procprefix == '' or procprefix == 'ms':
            siall = si
        else:
            siall = f'{procprefix}_{si}'

        for yr in yrs:
            yrstr = str(yr)[2:]
            logger.debug(f'working on yr {yr} with yrstr{yrstr}...')
            
            if len(yrs) > 1:
                try:
                    int(obscol.split('-')[-1])
                    obscol = obscol.replace(obscol.split('-')[-1],yrstr)
                    logger.info(f'obscol = {obscol}')
                except:
                    logger.warning('obscol does not end with year. need to incorporate new method to find each yr file')
                    
            if filter_class:
                in_file = Path(opt_dir)/f"prepost_{obscol}_{filter_class}_{img_type}_{siall}.csv"
            else:
                in_file = Path(opt_dir)/f"prepost_{obscol}_{img_type}_{siall}.csv"
            
            df = pd.read_csv(in_file, index_col=0)
            logger.info(f'there are {len(df)} pre-post observations for year {yr}') 
            dfs.append(df)

        if len(dfs) > 1:
            logger.debug(f'merging data from {len(dfs)} years...')
            fulldf = pd.concat(dfs)
        else:
            fulldf = dfs
        logger.debug(f'There are {fulldf.shape[0]} total prepost observations')
        fulldf.dropna(inplace=True)
        logger.debug(f'There are {fulldf.shape[0]} prepost observations after dropping nas')
        ## delta outputs responses as 'x days' -- need to extract the integer from this string:
        fulldf['postd'] = fulldf['postdelta'].str.split(' ').str.get(0).astype('int')
        fulldf['pred'] = fulldf['predelta'].str.split(' ').str.get(0).astype('int')
        ## use only observations where the obs to post-obs period is <= 10 days and the preobs to obs period is less than 30 days 
        fulldfa = fulldf[(fulldf['postd'] < maxpost) & (fulldf['pred']> (-1*maxpre)) & (fulldf[f'post_{si}'] > 0) & (fulldf[f'pre_{si}']>0)]
        logger.debug(f'There are {fulldfa.shape[0]} high quality prepost observations after dropping long observation gaps')

        all_dict = make_prepost_dict(all_dict,fulldfa,si)

        fulldfa['season'] = fulldfa['eventobs'].apply(lambda x: 'dry' if x.split('-')[1] in dry_months 
                                                 else 'wet' if x.split('-')[1] in wet_months else '')
    
        vidfdry = fulldfa[fulldfa['season']=='dry']
        dry_dict = make_prepost_dict(dry_dict,vidfdry,si)

        vidfwet = fulldfa[fulldfa['season']=='wet']
        wet_dict = make_prepost_dict(wet_dict,vidfwet,si)

    alldf = pd.DataFrame.from_dict(all_dict, orient='index')
    alldf.reset_index(inplace=True)
    drydf = pd.DataFrame.from_dict(dry_dict, orient='index')
    drydf.reset_index(inplace=True)
    wetdf = pd.DataFrame.from_dict(wet_dict, orient='index')
    wetdf.reset_index(inplace=True)
    if printdf:
        obs_out = obscol = obscol.replace(obscol.split('-')[-1],allyr_str)
        alldf.to_csv( Path(opt_dir) / f"prepost-summary_{obs_out}_ALL.csv")
        wetdf.to_csv( Path(opt_dir) / f"prepost-summary_{obs_out}_wet.csv")
        drydf.to_csv( Path(opt_dir) / f"prepost-summary_{obs_out}_dry.csv")
                     
    return alldf, wetdf, drydf
    
        
def get_ts_for_days_since_event(ts, num_days_pre, num_days_post1, num_days_post2, single_post_obs, 
                                print_df=True, out_dir='../../data', out_prefix=None):
    '''
    Converts full TS dataframes into days since event (or observation of event) for 0-<num_days_post>.
    Gets pre-event obs from first <num_days_pre> days before event to compare.
    Output dataframe is columns -1 (pre_event) and each day since up to <num_days_post> 
    with obs value for each (where available) for each field (row)
    If <single_post_obs> == True, post observation is single cloumn with first value since event
    '''
    if isinstance(ts, pd.DataFrame):
        obs_data = ts
    else:
        obs_data = pd.read_csv(ts, index_col=0)
    
    ## Narrow time series <num_days_pre> days pre-event and <numDaysPost2> days post event:
    obs_data_narrow = obs_data[(obs_data.index >= -1*num_days_pre) & (obs_data.index <num_days_post2)]
    ts = obs_data_narrow.transpose()
    
    ## Get pre-event value (if within <numDaysPre> days of event obs)
    pre_evt = ts[range(-1*num_days_pre,0,1)]
    post_evt = ts[range(num_days_post1,num_days_post2,1)]
    ## Fill NAs such that most recent pre-event obs is in -1 position
    pre_evt_f = pre_evt.ffill(axis=1)
    pre_evt_val = pre_evt_f[[-1]]
    
    if single_post_obs:
        post_evt_f = post_evt.bfill(axis=1)
        post_evt_val = post_evt_f[[num_days_post1]]
        tsdsb = pd.concat([pre_evt_val, post_evt_val], axis=1)
        tsdsb.rename({-1:'Pre', num_days_post1:'Post'}, axis=1, inplace=True)
        if print_df:
            pd.DataFrame.to_csv(tsdsb, Path(out_dir/ f'PrePost_{out_prefix}.csv'), sep=',', index=True)
    else:
        tsdsb = pd.concat([pre_evt_val, post_evt], axis=1)
        if print_df:
            pd.DataFrame.to_csv(tsdsb, Path(out_dir / f'DaysFromEvent_{out_prefix}.csv'), sep=',', index=True)
    
    return tsdsb

def ts_to_days_from_event(params):
    
    '''Standardizes a set of points with different dates for event occurance into a comparable time series from event date
    Returns time series (10 day intervals) with observation closest to event for each point as 0
    '''
    logger.info('fill in methods')