from pathlib import Path
import shutil
import csv
import yaml
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, Polygon
from osgeo import gdal, ogr, gdal_array
from ..handler import logger
from .project import ProjectPaths
from .date_utils import get_date_range
#from .image_utils import img_to_bbox_offsets
#######################################################################################################################################
### Culltionet prep
    
def prep_user_train(training_digitizations, user_train_dir, end_yr,chip_dim):
    '''
    Makes training chips as .gpkg files in 'user_train' folder in segmentation dir
    '''
    class_col = "class"
    polys = gpd.read_file(training_digitizations, mode="r")
    chips = gpd.read_file(training_digitizations.replace("_Polys", "_Chips"),  mode="r")
    proj_crs = polys.crs
    polys = polys[polys.columns.drop(list(polys.filter(regex='Notes')))]
    chips = chips[chips.columns.drop(list(chips.filter(regex='Notes')))]
    adjusted_chips = []
    skipped_chips = []
    #chips = sorted(chips)
    ## CHIPS
    for i, chp in chips.iterrows():
        process_polys = False
        chip_region = chp.region
        out_chip_name = Path(user_train_dir)/f"{chip_region}_grid_{str(end_yr)}.gpkg"
        ## check dimensions
        xdim = chips.bounds.iloc[i]['maxx']-chips.bounds.iloc[i]['minx']
        ydim = chips.bounds.iloc[i]['maxy']-chips.bounds.iloc[i]['miny']
        
        if xdim != chip_dim or ydim != chip_dim:
            process_polys = True
            if abs(chip_dim-xdim) < 1 and abs(chip_dim-ydim) < 1:
                adjusted_chips.append(chip_region)
                logger.info(f'dims for chip {chip_region} not quite 1000; adjusting now...')
                logger.info(f'old bounds: {chips.bounds.iloc[i]}')
                new_box = [box(chips.bounds.iloc[i]['minx'],chips.bounds.iloc[i]['miny'],
                          chips.bounds.iloc[i]['minx']+chip_dim,chips.bounds.iloc[i]['miny']+chip_dim)]
                df = chp.rename(None).to_frame().T
                logger.debug(f'df={df}')
                chip_gdf = gpd.GeoDataFrame(df, crs=proj_crs, geometry=new_box)
                logger.info(f'new bounds: {chip_gdf.bounds.iloc[0]}')
                chip_gdf.to_file(out_chip_name, crs=proj_crs, driver="GPKG") ### output chip
            else:
                logger.warning(f'ERROR: cannot process chip: {chip_region} because dims not close to 1000: x-dim = {xdim}, y-dim = {ydim} \n')
                skipped_chips.append(chip_region)
                process_polys = False
                
        elif not out_chip_name.exists():
            process_polys = True
            df = chp.rename(None).to_frame().T ## format chip dataframe
            chip_gdf = gpd.GeoDataFrame(df, crs=proj_crs, geometry=df.geometry)  ## create chip's GeoDataFrame
            chip_gdf.to_file(out_chip_name, crs=proj_crs, driver="GPKG") ### output chip
        
        if process_polys == True:
            chip_polys = polys.sjoin(chip_gdf, how="inner") ## select digitizations that intersect chip
            chip_polys = chip_polys[chip_polys.columns.drop(list(chip_polys.filter(regex='right')))] ## drop duplicate columns
            chip_polys = chip_polys[chip_polys.columns.drop(list(chip_polys.filter(regex='left')))] ## drop duplicate columns
            chip_polys['Name'] = f"{str(chip_region)}_poly_{str(end_yr)}" ## Name column for cultionet 
            chip_polys['region'] = str(chip_region) ## region column for cultionet 
            chip_polys['class'] = chip_polys[class_col]
            if len(chip_polys['class'].unique()) == 1 and 0 in chip_polys['class'].unique(): ## if there are only non-crop digitizations (based on recoded 'class')
                chip_polys = chip_polys.drop(chip_polys.index[1:]) ## delete all rows after the first 
                new_geom = chip_gdf.geometry.buffer(50) ## copy the chip's geometry and buffer by 50m               
                chip_polys['geometry'] = new_geom.iloc[0] ## assign new buffered geometry ### MAYBE RM ILOC[0]??

            else:
                geom_tmp = chip_polys.geometry.buffer(-0.000001) ## using buffered 'geom' removes the Z dimension 
                geom = geom_tmp.buffer(+0.000001) ## using buffered 'geom' removes the Z dimension 
                chip_polys = gpd.GeoDataFrame(chip_polys, crs=proj_crs, geometry=geom) ## create field digitization's GeoDataFrame
            chip_polys = chip_polys[['class', 'region', 'Name', 'geometry']]
            logger.info(f'chip_polys: {chip_polys}')
            out_polys_name = Path(user_train_dir)/f"{str(chip_region)}_poly_{str(end_yr)}.gpkg"
            chip_polys.to_file(out_polys_name,  crs=proj_crs, driver="GPKG", mode="w") ## , layer=str(chip_region) ## export digitization polys             

        else:
            logger.info(f'user_train already made for {str(out_chip_name)}')

    logger.info(f'skipped chips:{skipped_chips} \n')
    logger.info(f'adjusted chips:{adjusted_chips} \n')
    
def clip_to_chips(ras_list, grid_num, spec_index, version_dir, grid_file, end_yr, mmdd):

    seg_dir = Path(version_dir).parent
    ## get list of training chips for grid number
    suffix = f"_poly_{end_yr}.gpkg"
    names = [
        path.name.replace(suffix, "") 
        for path in (Path(seg_dir) / 'user_train').glob(f"{grid_num}*")
        if "_grid_" not in path.name
    ]
    logger.info(f'training chips for this grid: {names}')
    ## get chips that match current year
    chip_list = [f"{str(i)}_grid_{str(end_yr)}.gpkg" for i in names]
    for chip in chip_list:
        chip_num = int(chip.split("_")[0])
        chip_clip_shape = gpd.read_file(Path(seg_dir)/"user_train"/chip)
        bounds = tuple(chip_clip_shape.total_bounds)
        
        for rast in sorted(ras_list):
            rast_base = str(rast).split(f'{grid_num:06d}')[0]
            logger.debug(f'rast={rast}, rast_base={rast_base}')
            out_dir_f =  Path(str(rast).replace(rast_base, f'{version_dir}/time_series_vars/'))
            out_dir_f.mkdir(parents=True,exist_ok=True)
            out_rast = Path(out_dir_f)/Path(rast).name   
            logger.debug(f'raster_out: {out_rast}')
            if not out_rast.exists():
                with rio.open(rast_path, 'r') as src:
                    window = from_bounds(*bounds, transform=src.transform)
                if int(window.height) == 100 and int(window.width) == 100:
                    with rio.open(rast_path, 'r') as src:
                        clipped_rast = src.read(1, window=window)
                        new_gt = src.window_transform(window)
                        out_meta = {'driver': 'GTiff','width': 100,'height': 100,'count': 1,
                                         'dtype': np.int16,'crs': src.crs,'transform': new_gt}
                        with rio.open(out_rast, "w",  **out_meta) as dst:
                            dst.write(clipped_rast, 1)
                else:
                    # load grid shape to find grids that intersect with chip shape 
                    grids = gpd.read_file(grid_file)

                    ## mosaicking intersecting chips -- don't know if this is necessary
                    chip_within_grids = gpd.sjoin(grids, chip_clip_shape, op='intersects') 
                    both_grids = chip_within_grids.UNQ.to_list()
                    logger.info(f'both grids: {both_grids}')
                    grid_folder1 = Path(rast.replace(f'{grid_num:06d}',f'{both_grids[0]:06d}')).parent
                    raster1 = Path(grid_folder1)/rast.name                           
                    if len(both_grids) == 2:
                        grid_folder2 = Path(rast.replace(f'{grid_num:06d}',f'{both_grids[1]:06d}')).parent 
                        raster2 = Path(grid_folder2)/rast.name  
                        mosaic_list = [raster1, raster2]
                    else:
                        mosaic_list = [raster1]

                    grid_mosaic=Path(out_dir_f)/f"tmp_mos_{rast.stem}.vrt"
                    gdal.BuildVRT(grid_mosaic, mosaic_list)
                    # read in window of chip bounds 
                    with rio.open(grid_mosaic) as src2:
                        window = from_bounds(*bounds, transform=src2.transform)
                        new_gt2 = src2.window_transform(window)
                        clipped_rast2 = src2.read(1, window=window)
                        h, w = int(window.height), int(window.width)
                        out_meta = {'driver': 'GTiff', 'width': w, 'height': h, 'count': 1, 
                                        'dtype': np.int16, 'crs': src2.crs, 'transform': new_gt2}
                        with rio.open(out_rast, "w", **out_meta) as dst:
                            dst.write(clipped_rast2, 1)     
                    # delete tmp mosaic 
                    grid_mosaic.unlink()
            else:
                logger.info(f'mosaic already made: {str(out_dir_f)}')

def update_cultionet_config(seg_dir, yr, params):

    def null_tag_constructor(loader, node):
        ## This is a fix to kee yaml safe loader from reading '!!null' as Null
        ## Register the rule under PyYAML's safe scanner
        return "!!null"
        
    yaml.SafeLoader.add_constructor('tag:yaml.org,2002:null', null_tag_constructor)
    yaml.SafeLoader.add_constructor('!null', null_tag_constructor)

    region_id_file = str(Path(seg_dir) / "cnet_training_regions.txt")
    null="!!null"
    sis = params['segment']['spec_indices']
    yml_params={"image_vis":sis, "regions":null, "region_id_file": region_id_file,
            "years":[int(yr)],  "start_year":int(yr)-1,  "predict_year":int(yr)}
    logger.info(f'params = {yml_params}')

    ## if runnning local python script in same folder as config file (e.g. bash):
    #local_path = str(Path(__file__).parent)
    #bash_dir ="/".join(local_path.split("/")[1:-2])

    ## if running from installed tstuyau package:
    #config_file = (Path(__file__).resolve().parents[1] / "config" / "config_cultionet.yaml").resolve()

    ## but this makes no sense if cultionet will be run from a different environment (very likely). 
    ##   just need to provide path to cultionet config explicitly, so the same path can be called by cultionet:
    
    config_file = (Path(seg_dir) / "config_cultionet.yml")
    if not config_file.exists():
        logger.info(f'writing new cultionet config file at: {config_file}')
        #config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            yaml.dump(yml_params, f, sort_keys=False)
    else:
        logger.info(f'modifying existing new cultionet config file at: {config_file}')
        with open(config_file, 'r') as file:
            value = yaml.safe_load(file)
        if value is None:
            value = {}
        value.update(yml_params)
        #for key in yml_params:
        #    value[key] = yml_params[key]
        with open(config_file, 'w') as file:
            yaml.dump(value, file, sort_keys=False)
        
    #shutil.copy(config_file, Path(seg_dir)/'config_cultionet.yml')
    #logger.info(f" config file saved to: {str(Path(seg_dir)/'config_cultionet.yml')}")

    
def prep_training_ts_for_segmentation(params):
    
    '''look in user_train directory, for user input grid cell, grab all regions within that grid cell and clip chip for each region 
     (e.g. rgn1-evi2, rgn2-evi2, rgn1-gcvi, rgn2-gcvi, rgn1-wi, rgn2-wi)
     '''
    
    ppaths=ProjectPaths(params)
    if params['segment']['temp_inputs']:
        seg_dir = ppaths.segdir_temp
    else:
        seg_dir = params['segment']['seg_dir_main']
        if not seg_dir:
            seg_dir = ppaths.segmentation

    user_train_dir =  Path(seg_dir)/'user_train'
    user_train_dir.mkdir(parents=True, exist_ok=True)
    
    if params['segment']['seg_dir_mod']:
         mod_versiondir = Path(seg_dir) / params['segment']['seg_dir_mod']
    else:
         mod_versiondir = seg_dir
        
    grid_file = params['grid_file']
    
    yr = params['sample_model']['train_yrs']
    if isinstance(yr,list):
        yr = yr[0]
    ## can only predict for one year at a time because all cultionet components need to have the same name
    
    sis = params['segment']['spec_indices']
    if isinstance(sis,str):
        sis = [sis]
    
    if params['segment']['step'] == 'train':
        ## preps vector training chips from 'ready regions' file and <seg_train_polys>. Saves to 'user_train'
        if (params['segment']['update_polys']) or (not any(user_train_dir.iterdir())):
            training_polys = params['segment']['seg_train_polys']
            chip_dim = params['segment']['train_chip_size'] * params['res']
            regions_fi = prep_user_train(training_polys, user_train_dir, yr, chip_dim)
        ## need to copy training files for all versions so all components are in the same folder for cultionet.
        elif (params['segment']['seg_dir_mod']) and (not any((Path(mod_versiondir)/'user_train').iterdir())):
            shutil.copytree(seg_dir, mod_versiondir)

    tsvar_dir = Path(mod_versiondir)/'time_series_vars'
    tsvar_dir.mkdir(parents=True, exist_ok=True)
    data_dir =  Path(mod_versiondir)/'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    ## make classes.info file in data folder -- required by cultionet   TODO: make these parameters
    class_dict_file = (Path(data_dir) / "classes.info")
    if not class_dict_file.exists():
        class_dict = {
            "max_crop_class": 1, 
            "edge_class": 2
        }    
        
        with open(class_dict_file, "w", encoding='utf-8') as f:
            json.dump(class_dict, f, indent=4)
    
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
        logger.info(f'working on cell {cell}... \n')
        ppaths = ProjectPaths(params, grid=cell)
        for si in sis:
            spec_index = si
            logger.info(f'working on {spec_index}... \n')
            ts_dir = ppaths.ts / si
            if not ts_dir.exists():
                logger.warning(f'there is no directory {ts_dir}')
            else:
                dr = get_date_range(yr-1,'yr',params,return_type='doy',padded=False)
                logger.debug(f'getting images from {dr[0]} to {dr[1]}')
                all_images = sorted([i for i in list(ts_dir.glob('*.tif')) if int(i.stem) >= int(dr[0])-3 and int(i.stem) <= int(dr[1])+3])
                logger.debug(f"found images: { [i.stem for i in all_images]} \n")
                ## take every third image [start:stop:step] in time series
                copy_images = all_images[::3]
                logger.debug(f"keeping images: {[i.stem for i in copy_images]} \n")
                if len(copy_images) != 13:
                    logger.warning(f'Expecting 13 images but found {len(copy_images)} between {dr[0]} and {dr[1]}  in {ts_dir}')
                else:
                    if params['segment']['step'].startswith('train') and (params['segment']['clip_imagery']):
                        mmdd = f"{params['calendar']['first_mo']:02d}-01"
                        clip_to_chips(copy_images,grid_num=cell, spec_index=si, version_dir=mod_versiondir, 
                                grid_file=grid_file, end_yr=yr, mmdd=mmdd)
                    else:
                        pre_ts_base = str(ppaths.ts/si).split(f'{cell:06d}')[0]
                        outdir =  Path(str(ppaths.ts/si).replace(pre_ts_base, f'{tsvar_dir}/'))
                        outdir.mkdir(parents=True, exist_ok=True)
                        logger.info(f'copying images from {dr[0]} and {dr[1]} into {str(outdir)}')
                        for fi in copy_images:
                            out_fi = Path(outdir)/fi.name
                            #if not out_fi.exists():
                            shutil.copyfile(fi, out_fi)   

    if (params['segment']['step'] == 'train') and (params['segment']['get_chip_list']):
        logger.info('getting list of chips with complete imagery...')
        user_train_regions = [r for r in tsvar_dir.iterdir() if r.is_dir()]
        logger.info(f'user_train_regions = {user_train_regions}')
        not_ready = []
        for rgn in user_train_regions:
            #region_folder = Path(tsvar_dir)/f"{str(rgn)}"
            if params['project_ver'] == 'Py_0': 
                rgn_ts_dir = Path(rgn)/'brdf_ts/ms'
            else:
                rgn_ts_dir = Path(rgn)/'brdf_ts'
            sis_found = [s.name for s in rgn_ts_dir.iterdir() if s.is_dir()]
            logger.info(f'sis_found = {sis_found}')
            for si in sis:
                if si not in sis_found:
                    not_ready.append(f"{str(rgn)}_{str(si)}")
                    logger.warning(f"{str(rgn)} missing for {str(si)}")
                elif si in sis_found:
                    num_tifs = list((Path(rgn_ts_dir)/si).glob("*.tif"))
                    if len(num_tifs) != 13:
                        logger.warning(f"{rgn.stem} does not have 13 images of {str(si)}. there are only {len(num_tifs)}")
                        not_ready.append(f"{rgn.stem}_{str(si)}")
                        #logger.info(f'region={rgn}')
        #logger.info(f'not ready = {not_ready}')
        regions_not_ready = list(set([i.split("_")[0] for i in not_ready]))
        ready = sorted([i.name for i in user_train_regions if i not in regions_not_ready])
        #config_file = Path(seg_dir)/"config_cultionet.yml"
    
        ## save file for user_train_regions that are ready (for the config file)
        if len(ready) > 0:
            chip_file = Path(seg_dir)/"cnet_training_regions.txt"
            txt = open(chip_file, 'w')                           
            txt.write('id \n')     
            for rdy in ready:
                if not str(rdy).startswith("."):
                    txt.write(f'{rdy} \n')       
            txt.close()
        else:
            logger.warning('oops -- there are no training chips ready')
                
        ## create holdout list for accuracy assessment from chips that weren't used in model training bcuz they had incomplete TS
        ## Note: for this to work, need to run this first for non-holdout cells, then later for holdout cells (with <get_chip_list>=False) 
        txt_holdout = open(str(chip_file).replace(".txt", "_holdout.txt"), 'w')
        txt_holdout.write('id \n')     
        for incomplete_ts in regions_not_ready:
            if not incomplete_ts.startswith("."):
                txt_holdout.write(f'{incomplete_ts} \n')       
        txt_holdout.close()
