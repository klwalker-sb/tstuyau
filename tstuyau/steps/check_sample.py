import sys
from pathlib import Path
import json
import csv
import random
import numpy as np
import rasterio as rio
import pandas as pd
import geowombat as gw
import geopandas as gpd
from shapely.geometry import Point, box, shape
import shutil
import tempfile
from .project import ProjectPaths
#from .mod_utils import get_train_yrs_str, get_class_col
from ..handler import logger


def get_ran_pt_in_poly(polyg, seed):
    '''
    Returns a shapely Point object for a random point within a polygon <polyg>
    '''
    minx, miny, maxx, maxy = polyg.bounds
    while True:
        np.random.seed(seed)
        pp = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polyg.contains(pp):
            logger.debug(f"pp={pp}")
            return pp
        
def get_ran_pts_in_polys(polys, npts, seed=88):
    '''
    Returns a geodataframe with geometry column containing shapely Point objects
    With <npts> random samples from each polygon in <polys>
    (<polys> is path to json file with polygon geometries)
    '''
    with open(polys, 'r') as polysrc:
        polys2 = json.load(polysrc)
        logger.debug(f"poly type = {polys['type']}") #FeatureCollection
        poly_list = []
        pt_dict = {}
        for poly in polys2['features']:
            poly_list.append(poly['geometry'])
            polyobj = shape(poly['geometry'])
            for i in range(0,npts):
                pt_name = str(poly['properties']['id'])+'_'+str(i+1)
                pt_in_poly = get_ran_pt_in_poly(polyobj, seed)
                pt_dict[pt_name] = pt_in_poly

    ptsdb = pd.DataFrame.from_dict(pt_dict, orient='index')
    ptsgdb = gpd.GeoDataFrame(ptsdb, geometry=ptsdb[0])
    ptsgdb.drop(columns=[0], inplace=True)

    return ptsgdb


def get_variables_at_pts(in_dir, feature_model, feature_mod_dict, start_yr, polys, numpts, seed, load_samp=False, ptgdb=None, out_dir=None):
    '''
    Gets values for all sampled points <numpts> in all polygons <polys> for all images in <in_dir>
    OR gets values for points in a previously generated dataframe <ptgdb> using <load_samp>=True.
    output is a dataframe with a pt (named polygonID_pt#)
    on each row and an image index value(named YYYYDDD) in each column
    '''
    stack_path = Path(in_dir) / f"{feature_model}_{start_yr}_stack.tif"
    if not stack_path.is_file():
        if 'Poly' in feature_model and 'NoPoly' not in feature_model:
            nopoly_model = str(feature_model).replace('Poly','NoPoly')
            stack_path = Path(in_dir) / f"{nopoly_model}_{start_yr}_stack.tif"
    if not stack_path.is_file():       
        logger.warning(f"path {stack_path} does not exist. \n")
        logger.warning(f"need to create variable stack for {feature_model}_{start_yr} first. \n")
        ptsgdb = None
    
    else:
        #band_names = getset_feature_model(feature_mod_dict, feature_model)[7]
    
        if not load_samp:
            if polys:
                ptsgdb = get_ran_pts_in_polys (polys, numpts, seed)
            else:
                logger.info('There are no polygons or points to process in this cell \n')
                return None
        elif load_samp:
            ptsgdb = ptgdb

        xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
        coords = list(map(list, zip(*xy)))
    
        logger.info('Extracting variables from stack \n')
        
        with gw.open(stack_path) as src0:
            logger.debug(f"attributes = {src0.attrs}")
            band_names = src0.attrs['descriptions']
        
        with rio.open(stack_path ,'r') as comp:
            ## Open each band and get values
            for b, band in enumerate(band_names):
                logger.info(f'{b}:{band},')
                comp.np = comp.read(b+1)
                varn = (f'var_{band}')
                ptsgdb[varn] = [sample[b] for sample in comp.sample(coords)]
            if out_dir:
                pd.DataFrame.to_csv(ptsgdb,Path(out_dir)/'ptsgdb.csv', sep=',', index=True)
    
    return ptsgdb

def get_pts_in_grid (grid_file, grid_cell, ptfile):
    '''
    loads point file (from .csv with 'XCoord' and 'YCoord' columns) and returns points that overlap a gridcell
    as a geopandas GeoDataFrame. Use this if trying to match/append data to existing sample points
    rather than making a new random sample each time (e.g. if matching Planet and Sentinel points)
    Note that crs of point file is known ahead of time and hardcoded here to match specific grid file.
    '''
    out_path = Path(grid_file).parent

    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(grid_file).name
        shutil.copy(grid_file, temp_file)
        df = gpd.read_file(temp_file)
        crs_grid = df.crs
        logger.debug(f'grid is in:{crs_grid} \n')

    if isinstance(ptfile, gpd.GeoDataFrame):
        pts = ptfile
    elif isinstance(ptfile, pd.DataFrame):
        ptsdf = ptfile
        pts = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=crs_grid)
    elif (ptfile.endswith('.shp')) or (ptfile.endswith('.gpkg')):
        pts = gpd.read_file(ptfile)
    else:
        ptsdf = pd.read_csv(ptfile, index_col=0)
        pts = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=crs_grid)
        
    if df.shape[0] > 1:
        bb = df.query(f'UNQ == {grid_cell}').geometry.total_bounds
    else:
        bb = df.geometry.total_bounds
    logger.debug(f'bb = {bb} \n')
    
    grid_bbox = box(bb[0],bb[1],bb[2],bb[3])
    grid_bounds = gpd.GeoDataFrame(gpd.GeoSeries(grid_bbox), columns=['geometry'], crs=crs_grid)
    logger.debug(f'grid_bounds = {grid_bounds} \n ')

    pts_in_grid = gpd.sjoin(pts, grid_bounds, predicate='within')
    pts_in_grid = pts_in_grid.loc[:,['geometry']]

    logger.info(f'Of the {pts.shape[0]} ppts, {pts_in_grid.shape[0]} are in gridCell {grid_cell} \n')

    ## Write to geojson file
    if pts_in_grid.shape[0] > 0:
        pt_clip = Path(out_path)/ f"ptsGrid_{str(grid_cell)}.json"
        pts_in_grid.to_file(pt_clip, driver="GeoJSON")
        logger.debug(f'pts in grid: {pts_in_grid.head(n=5)}')
        
        return pts_in_grid
        

def get_polygons_in_grid (grid_file, grid_cell, poly_path, oldest=None, newest=None, obs_col=None, out='json'):
    '''
    Filters polygon layer to contain only those overlapping selected grid cell (allows for iteration through grid)
    Allows filtering by optional columns: 'FirstYrObs','ObsYr','Year'or'Acquired' 
    to remove polygons obseserved during years outside the period of interest
    Outputs new polygon set to a .json file stored in the <grid_cell> directory
    '''
    polys = gpd.read_file(poly_path)
    out_path = Path(grid_file).parent

    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(grid_file).name
        shutil.copy(grid_file, temp_file)
        df = gpd.read_file(temp_file)
        crs_grid = df.crs

    bb = df.query(f'UNQ == {grid_cell}').geometry.total_bounds

    grid_bbox = box(bb[0],bb[1],bb[2],bb[3])
    grid_bounds = gpd.GeoDataFrame(gpd.GeoSeries(grid_bbox), columns=['geometry'], crs=crs_grid)
    polys_in_grid = gpd.overlay(grid_bounds, polys, how='intersection')

    logger.info(f"Of the {polys.shape[0]} polygons, {polys_in_grid.shape[0]} are in grid_cell {grid_cell} \n")

    ## Filter out polygons that were observed before year set as 'oldest' or after year set as 'newest'
    if obs_col in polys_in_grid.columns:
        yr_filter = polys_in_grid[obs_col]
    elif 'Year' in polys_in_grid.columns:
        yr_filter = polys_in_grid['Year']
        
    if oldest:
        polys_in_grid = polys_in_grid[int(yr_filter) >= oldest]
    if newest:
        polys_in_grid = polys_in_grid[int(yr_filter) <= newest]
        logger.info(f"{len(polys_in_grid)} polygons first observed between {oldest} and {newest} in AOI \n")

    if out=='json':
        ## Write to geojson file
            if polys_in_grid.shape[0] > 0:
                poly_clip = Path({out_path}/f'polysGrid_{str(grid_cell)}.json')
                polys_in_grid.to_file(poly_clip, driver="GeoJSON")

            return poly_clip
    
    else:
        return polys_in_grid
    
    
def make_var_dataframe(params):
    '''
    Produces a dataframe with all sample points as rows and all model features as columns
    some additional information (such as coordinates and labels) is also retained in columns. 
    Model feature columns are given prefix "var_" for easy subsetting during modelling
    Note this method assumes all variables are already calculated and in a stack called 
    {feat_mod}_{start_yr}_stack.tif> in the comp folder for each cell.
    
    It is most efficient to pull all data for all possible sample points, then adjust the dataframe 
       to subsamples during model optimization using the "append_vars_to_dataframe" method below. 
       This output is saved as ptsfeats_model_name in the vector/training directory
       it can be used as the final pixeldf used to build a model, but that is usually a subset
       created from this output with format_ptfeat_set().
    '''
    
    all_pts = pd.DataFrame()
    if isinstance(params['grids'], list):
        cells = params['grids']
    elif str(params['grids']).endswith('.csv'): 
        cells = []
        with open(params['grids'], newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])
    else:
        logger.warning('cell_list needs to be a list or path to .csv file with list \n')

    for cell in cells:
        ppaths = ProjectPaths(params, grid=int(cell))
        if params['classify']['comp_dir'] == 'in_dir':
            in_dir = ppaths.ms.parent / 'comp'
        elif params['classify']['comp_dir'] == 'tmp':
            in_dir = ppaths.scratch / 'comp'
        else:
            in_dir = ppaths.comp
        logger.info(f'working on cell {cell} \n')

        ftset_dir = ['classify']['ptsfeat_dir']
        if not ftset_dir:
            ftset_dir = ppaths.trainfeatsets

        if params['sample_model']['load_samp']:
            logger.info(f'loading sample from points for cell {cell} \n')
            points = get_pts_in_grid (params['grid_file'], cell, params['sample_model']['point_file'])
            polys = None
        else:
            logger.info(f'loading sample from polygons for cell {cell} \n')
            polys = get_polygons_in_grid (params['grid_file'], cell, params['sample_model']['poly_file'], 
                                          oldest=params['sample_model']['oldest'], newest=params['sample_model']['newest'],
                                          obs_col=params['sample_model']['obs_col'])
            points = None

        feat_mod = params['feature_model']['name']
        start_yr = params['feature_model']['start_yr']
        logger.info(f'looking for {feat_mod}_{start_yr}_stack.tif in {ppaths.comp} to extract variables... \n')
        
        if polys or isinstance(points, gpd.GeoDataFrame):
            
            feature_mod_dict = params['feature_model']['feature_mod_dict']
            if not feature_mod_dict:
                feature_mod_dict = str(ppaths.fmoddict)

            if params['sample_model']['load_samp']:
                polys=None
                pts = get_variables_at_pts(in_dir, params['feature_model']['name'], feature_mod_dict, 
                        params['feature_model']['start_yr'], polys, params['sample_model']['npts'], 
                        params['sample_model']['poly_samp_seed'], load_samp=True, ptgdb=points, out_dir=None)
            else:
                pts = get_variables_at_pts(in_dir, params['feature_model']['name'], feature_mod_dict,
                        params['feature_model']['start_yr'], polys, params['sample_model']['npts'], 
                        params['sample_model']['poly_samp_seed'], load_samp=False, ptgdb=None, out_dir=None)
            if pts:
                pts.drop(columns=['geometry'], inplace=True)
                all_pts = pd.concat([all_pts, pts])
          
        else:
            logger.info('skipping this cell \n')
            pass
    
    pts_in = pd.read_csv(params['sample_model']['point_file'], index_col=0)
    if params['log_level'] == 'DEBUG':
        rfdf = all_pts.merge(pts_in, left_index=True, right_index=True)
        pd.DataFrame.to_csv(rfdf,Path(f"{ftset_dir}/forchecking_ptsfeats_{params['feature_model']['name']}_{params['feature_model']['start_yr']}.csv"), sep=',', index=True)
    
    pd.DataFrame.to_csv(all_pts,Path(ftset_dir) / f'ptsfeats_{feat_mod}_{start_yr}.csv', sep=',', index=True)
