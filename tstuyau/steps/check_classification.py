from pathlib import Path
from datetime import datetime
from contextlib import ExitStack

from . import lookup
from ..handler import logger
from .project import ProjectPaths
from .check_model_prep import make_variable_stack,  make_and_score_model
from .mod_utils import getset_feature_model, get_train_yrs_str, get_class_col, prep_test_train
from .lookup import SCHEMATIC_MODS, LC_CATS
from . import date_utils, image_utils, prechecks
from .. import errors

import geowombat as gw
import rastercrf as rcrf
import csv
import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio as rio
import dask
from tqdm import tqdm
import json
import joblib
import time
from fiona.drvsupport import supported_drivers
supported_drivers['LIBKML'] = 'rw'


def get_predictions_gw(saved_stack, model_bands, mod_path, class_img_out):
    '''
    apply random forest model to full raster using xarray with geowombat wrapper for named bands and windowed operations
    
    Parameters
    ----------
    saved_stack: path to multiband geotiff containing a band for each model variable
        The bands need to have names that match the model variables
        The stack can have extra bands; only those used in the model will be used. Likewise, the order of the 
        bands in the file does not matter, as it will be rearanged to match the model here.
    model_bands: The ordered bands used in the model. 
        This can be retrieved from 'band_names' in the feature model dictionary
    mod_path: The path to the .joblib file with the model information
    class_img_out:  The path for the classified output image
    '''
    logger.info('getting predictions...\n')
    mod = joblib.load(mod_path) #this load is from joblib -- careful if there are other packages with 'Load' module

    chunks=256
    with rio.open(saved_stack) as src0:
        profile = dict(blockxsize=chunks,
            blockysize=chunks,
            crs=src0.crs,
            transform=src0.transform,
            driver='GTiff',
            height=src0.height,
            width=src0.width,
            nodata=0,
            count=1,
            dtype='uint8',
            compress='lzw',
            tiled=True)
        
    ## reduce stack bands to match model variables, ensuring same order as model df 
    with gw.open(saved_stack) as src0:
        logger.debug(f'attrs= {src0.attrs} \n')
        stack_bands = src0.attrs['descriptions']
        logger.info(f'bands in stack: {stack_bands} \n')
    bands_out = []
    band_names = []
    for b in model_bands:
        logger.debug(f'looking for {b}:')
        found_band = False
        for i, v in enumerate(stack_bands):
            logger.debug(f'v={v}')
            if v == b:
                bands_out.append(i+1)
                band_names.append(v)
                logger.debug(' -- found! \n')
                found_band = True
                break
            elif b.startswith('sing') and ((v == b.split('_')[1]) or (v == b.split('_',1)[1])):
                bands_out.append(i+1)
                band_names.append(f'sing_{v}')
                logger.debug(' -- found! \n')
                found_band = True
                break
            elif b.startswith('poly') and v == b.split('_',1)[1]:
                bands_out.append(i+1)
                band_names.append(f'poly_{v}')
                logger.debug(' -- found! \n')
                found_band = True
                break
            else:
                logger.debug('...')
                pass
        if not found_band:
            logger.warning(f'ERROR: band {b} not found in stack \n')
            return False
            
    logger.info(f'bands used for model: {bands_out}')
    
    new_stack = src0.sel(band=bands_out)
    new_stack.attrs['descriptions'] = band_names
    logger.info(f"band names: {new_stack.attrs['descriptions']}")

    #new_stack = new_stack.chunk({"x": len(new_stack.x), "y": len(new_stack.y)})
    with gw.open(saved_stack) as src:
        windows = list(src.gw.windows(row_chunks=chunks, col_chunks=chunks))
    
    for w in tqdm(windows, total=len(windows)):
        #with ExitStack() as stack:
        stackblock = new_stack[:, w.row_off:w.row_off+w.height, w.col_off:w.col_off+w.width]
        
        X = stackblock.stack(s=('y', 'x'))\
            .transpose()\
            .astype('int16')\
            .fillna(0)\
            .data\
            .rechunk((stackblock.gw.row_chunks * stackblock.gw.col_chunks, 1))

        feature_band_count = X.shape[1]
        logger.info(f'num features in df = {feature_band_count} \n')
            
        X = dask.compute(X, num_workers=4)[0]

        class_prediction = mod.predict(X)
        logger.info(f'class_prediction out = {class_prediction} \n')
        class_out = np.uint8(np.squeeze(class_prediction))
        class_out = class_out.reshape(w.height, w.width)
        #logger.info(f'class_out = {class_out} \n')
    
        if not class_img_out.is_file():
            with rio.open(class_img_out, mode='w', **profile) as dst:
                pass
        with rio.open(class_img_out, mode='r+') as dst:
            #dst.write(class_out, window=w)
            dst.write(class_out, indexes=1, window=w)
         
    return class_prediction


def classify_timestep(params):

    getset_feature_model(params)
    
    ## derive model name from components
    class_col = get_class_col(params['schematic_model']['lc_mod'],params['schematic_model']['lut'])[0]
    trainyrs =  get_train_yrs_str(params['sample_model']['train_yrs'])

    feat_mod_name = params['feature_model']['name']
    samp_mod_name = params['sample_model']['name']
    
    if isinstance(params['classify']['out_yrs'], int):
        params['classify']['out_yrs'] = [params['classify']['out_yrs']]
    out_yrs = params['classify']['out_yrs'][0]
    
    mod_type = params['classify']['mod_type']
    
    base_model_name = f'{feat_mod_name}_{samp_mod_name}_{class_col}_{trainyrs}'
    model_name_train = f'{feat_mod_name}_{samp_mod_name}_{class_col}_{trainyrs}_{mod_type}'
    model_name_class = f'{feat_mod_name}_{samp_mod_name}_{class_col}_{trainyrs}_{mod_type}_{out_yrs}'
    
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
        ppaths = ProjectPaths(params, grid=cell)
        logger.info(f'working on cell {cell}... \n')
        if params['classify']['comp_dir'] == 'backup':
            comp_dir = ppaths.bk /'comp'  
        elif params['classify']['comp_dir'] == 'input_dir':
            comp_dir = ppaths.ms.parent / 'comp'
        elif params['classify']['comp_dir'] == 'tmp':
            comp_dir = Path(params['scratch_dir']) / params['project_name'] / 'comp'

        comp_dir.mkdir(parents=True, exist_ok=True)
        mod_dir_base = params['classify']['mod_dir']
        if not mod_dir_base:
            mod_dir_base = ppaths.classification
        mod_dir = Path(mod_dir_base) / f'{mod_type}'
        vdf_dir = params['classify']['vardf_dir']
        if not vdf_dir:
            vdf_dir = ppaths.fulltrainsets

        class_img_out = Path(comp_dir) / f"{int(cell):06d}_{model_name_class}.tif"
        if params['classify']['overwrite_image']:
            if class_img_out.is_file():
                class_img_out.unlink()
        if class_img_out.is_file():
            logger.warning(f'{class_img_out} already exists. set classify:overwrite_image param to True to overwrite.')
            pass
        else:    
            stack_path = ppaths.ms.parent / 'comp' / f'{feat_mod_name}_{out_yrs}_stack.tif'
            logger.info(f'looking for stack: {stack_path}... \n')
            if stack_path.is_file():
                logger.info(f'stack file already exists for model {feat_mod_name} \n')
                var_stack = stack_path
            ## Specific poly/no poly method no longer needed because we can just use general subset parameters
            #elif 'NoPoly' in str(feat_mod_name):
            #    ## Can make a noPoly model with a Poly stack
            #    poly_model = str(feat_mod_name).replace('NoPoly','Poly')
            #    alt_path =  ppaths.ms.parent / 'comp' / f'{poly_model}_{out_yrs}_stack.tif'
            elif (params['feature_model']['subset_features']==True) or (params['feature_model']['subset_features']=='True'):
                alt_path = ppaths.ms.parent / 'comp' / f"{params['feature_model']['full_feature_mod']}_{out_yrs}_stack.tif"
                logger.info(f'looking for alternative stack at {alt_path}')
                if alt_path.is_file():
                    logger.info(f'using larger stack file that already exists at {alt_path} \n')
                    var_stack = alt_path
            else:
                ## make variable stack if it does not exist (for example for cells without sample pts)
                ## -- will not be remade if a file named {feature_model}_{start_year}_stack.tif already exists in ts_dir/comp
                ##      or is overwrite param is set to false.
                logger.info(f'writing variable stack for model {feat_mod_name} ...\n') 
                var_stack = make_variable_stack(params)
        
            ## Try to use existing ml model. By default, will use model with default name:
            ##  f'{feature_model}_{samp_mod_name}_{class_col}_{trainyrs}_{mod_type)mod.joblib'
            ##    but can force use of a different existing model with <alt_mod> param
            if params['classify']['existing_mod']:
                existing_mod = params['classify']['existing_mod']
                if Path(f'{mod_dir}/{existing_mod}').is_file():
                    ml_mod = Path(mod_dir) / existing_mod
                    logger.info(f'using existing ml model at:{ml_mod} \n')
                elif Path(existing_mod).is_file():
                    ml_mod = existing_mod
                    logger.info(f'using existing ml model at:{ml_mod} \n')
                else:
                    logger.info(f'cannot find existing model {existing_mod} specified with existing_mod parameter. Check name or set alt_mod to None to create new model \n')
            else:
                default_model = Path(f'{mod_dir}/{model_name_train}mod.joblib')
                if default_model.is_file():
                    ml_mod = default_model
                    logger.info(f'using existing ml model at {ml_mod} \n')
                else:
                    logger.info(f'Could not find existing model at{default_model}. Creating new ml model... \n')
                    df_in = Path(vdf_dir) / f'pixdf_{base_model_name}.csv'
                    ml_mod = make_and_score_model(params, df=df_in)
        
            feature_mod_dict = params['feature_model']['feature_mod_dict']
            if not feature_mod_dict:
                feature_mod_dict = str(ppaths.fmoddict)
            with open(Path(feature_mod_dict), 'r+') as fmd:
                dic = json.load(fmd)
                model_bands = dic[params['feature_model']['name']]['band_names']
                logger.info(f'model bands from dict: {model_bands}: \n')

            class_prediction = get_predictions_gw(var_stack, model_bands, ml_mod, class_img_out)
            
            if class_prediction is not None:
                logger.info(f'Image saved to: {class_img_out} \n')    
            else:
                logger.warning('got an error \n')
    
    return None

def classify_CRF(params):

    """
    Classifies land cover time-series by looking at change / no-change samples across years

    Args:
        params (dict)

    Returns:
        None
    """

    prechecks.precheck_classify(params)

    # Combined samples for all grids
    train_samples = Path(params['classify_crf']['train_samples'])

    train_grid_path = train_samples.parent / 'grids'

    train_grid_path.mkdir(parents=True, exist_ok=True)

    if params['classify_crf']['method'] != 'predict':
        if params['classify_crf']['update_samples']:
            if train_samples.is_file():
                while True:
                    try:
                        train_samples.unlink()
                    # if classifying multiple grids at once (multiple processes are trying to write to same file):
                    except FileNotFoundError:
                        time.sleep(1)
                        continue
                    else:
                        break

    model_file = Path(params['classify_crf']['model_file'])

    if params['classify_crf']['method'] in ['fit', 'fit_predict']:
        if params['classify_crf']['overwrite_model']:
            if model_file.is_file():
                model_file.unlink()

    ###############################
    # Extract samples for each grid
    ###############################

    if not train_samples.is_file():

        # Read the KML file with all samples
        df = gpd.read_file(params['classify_crf']['lc_vector'], mode='r')

        df[['lc', 'acquired']] = df.Name.str.split('-', expand=True)

        idx = []
        for i, date in enumerate(df.acquired.tolist()):

            try:
                datetime.strptime(date, '%Y%m%d')
                idx.append(i)
            except:
                pass

        # Get samples with normal date tags
        df = df.iloc[idx].reset_index(drop=True)

        # Set the dates
        df['datetimes'] = [datetime.strptime(date, '%Y%m%d') for date in df.acquired.tolist()]
        df['lc'] = df['lc'].astype(int)

        # TODO: different levels for orchards and plantations?
        df = df.replace({78: lookup.LABELS_DICT_r[b'plt'],     # orchards -> trees
                         121: lookup.LABELS_DICT_r[b'dev'],    # developed
                         174: lookup.LABELS_DICT_r[b'grs']})   # grassland

        for grid in params['grids']:

            # Output samples for the current grid
            train_samples_grid = train_grid_path / (train_samples.stem + f'_{grid:06d}_train_grid' + train_samples.suffix)
            test_samples_grid = train_grid_path / (train_samples.stem + f'_{grid:06d}_test_grid' + train_samples.suffix)

            if params['classify_crf']['overwrite_samples']:

                if train_samples_grid.is_file():
                    train_samples_grid.unlink()

                if test_samples_grid.is_file():
                    test_samples_grid.unlink()

            if train_samples_grid.is_file():
                logger.warning(f'  Training samples for grid {grid} already exist.')
                continue

            ppaths = ProjectPaths(params, grid=grid)

            vrt_files, image_names, time_names = image_utils.open_images(ppaths, params, 'classify')

            grid_years, dft, df_clip = date_utils.get_sample_years(df, image_names[0], time_names)

            # Split the polygons into train/test
            train_df_clip, test_df_clip = prep_test_train(df_clip, df_clip.parent, 'lc', 'crf_mod', train_frac=0.7, min_count=5)
            ## TODO: need to figure out mod_name and out_dir and follow up with read on these, as they will be returned as 
            ##   paths rather than open dataframes. Also, min_count is not being used currently.

            #####################################
            # Extract samples for each time slice
            #####################################

            logger.info(f'  Extracting data for grid {grid} at land cover samples ...')

            train_grid_year_df_list = []
            test_grid_year_df_list = []

            # The years come in pairs (e.g., 2019/2020), so don't process the end year
            for year in tqdm(grid_years[:-1], total=len(grid_years[:-1])):

                # Get the samples that intersect the grid
                if train_df_clip.empty:
                    train_df_clip_year = gpd.GeoDataFrame(data=[])
                else:
                    train_df_clip_year = date_utils.query_frame_date(train_df_clip, year, year+1, params, 'classify')

                if test_df_clip.empty:
                    test_df_clip_year = gpd.GeoDataFrame(data=[])
                else:
                    test_df_clip_year = date_utils.query_frame_date(test_df_clip, year, year+1, params, 'classify')

                if train_df_clip_year.empty and test_df_clip_year.empty:
                    continue

                # Get the the time slice indices needed to slice the timeframe.
                image_index = date_utils.year_to_index(dft, year, year+1, params, 'classify')

                train_df_extract = None
                test_df_extract = None

                with ExitStack() as stack:

                    # Open a stack of image variables at the time slice ``image_index``.
                    src = image_utils.scale_stack(stack,
                                                  vrt_files,
                                                  params,
                                                  'classify',
                                                  'image_bands',
                                                  'image_bands_pred',
                                                  image_index)

                    if not train_df_clip_year.empty:

                        train_df_extract = gw.extract(src,
                                                      train_df_clip_year,
                                                      band_names=src.band.values.tolist(),
                                                      id_column='lc',
                                                      frac=params['classify_crf']['poly_frac'],
                                                      min_frac_area=params['classify_crf']['min_frac_area'],
                                                      n_jobs=params['num_workers'],
                                                      num_workers=params['num_workers'])

                    if not test_df_clip_year.empty:

                        test_df_extract = gw.extract(src,
                                                     test_df_clip_year,
                                                     band_names=src.band.values.tolist(),
                                                     id_column='lc',
                                                     frac=params['classify_crf']['poly_frac'],
                                                     min_frac_area=params['classify_crf']['min_frac_area'],
                                                     n_jobs=params['num_workers'],
                                                     num_workers=params['num_workers'])

                # Store each annual DataFrame in the list
                if isinstance(train_df_extract, gpd.GeoDataFrame):
                    if not train_df_extract.empty:
                        train_grid_year_df_list.append(train_df_extract)

                if isinstance(test_df_extract, gpd.GeoDataFrame):
                    if not test_df_extract.empty:
                        test_grid_year_df_list.append(test_df_extract)

            # Save the samples to file
            if train_grid_year_df_list:
                pd.concat(train_grid_year_df_list, axis=0).to_file(train_samples_grid, driver='GPKG')

            if test_grid_year_df_list:
                pd.concat(test_grid_year_df_list, axis=0).to_file(test_samples_grid, driver='GPKG')

            image_utils.clean_vrt_files(vrt_files)

            logger.info(f'  Finished sampling grid {grid}.')

    if params['classify_crf']['method'] == 'sample':
        return

    ###################
    # Load grid samples
    ###################

    if model_file.is_file() and (params['classify_crf']['method'] == 'predict'):

        clf = rcrf.CRFClassifier()
        clf.from_file(model_file)

    else:

        if params['classify_crf']['method'] == 'predict':
            logger.exception('  A model should exist with method (classify_crf:method:predict).')
            return

        logger.info('  Loading all grids ...')

        if train_samples.is_file():
            df_samples = gpd.read_file(train_samples).fillna(value=0)
        else:

            # Combine all samples from each grid
            sample_list = []
            for train_samples_grid in train_grid_path.glob(f'{train_samples.stem}*train*{train_samples.suffix}'):
                sample_list.append(gpd.read_file(train_samples_grid))

            if not sample_list:
                logger.warning('  Re-sample the grids with (classify_crf:method:sample).')
                raise errors.TrainingGridsError(train_grid_path)

            df_samples = pd.concat(sample_list, axis=0).fillna(value=0)

            logger.info('  Writing the grids to file ...')

            while True:
                try:
                    df_samples.sample(frac=params['classify_crf']['sample_save_frac']).to_file(train_samples, driver='GPKG')
                except RuntimeError:
                    continue
                else:
                    break
                
        #######################
        # Temporal augmentation
        #######################

        # TODO: check column names
        feature_cols = [col for col in df_samples.columns.tolist() if col not in
                        ['Name', 'Description', 'acquired', 'datetimes', 'lc', 'index', 'poly', 'point', 'lc', 'geometry', 'x', 'y']]

        # TODO: add parameter to control series length
        feature_cols = feature_cols[:params['classify_crf']['series_length']]

        pred_cols = list(map(str, range(1, len(feature_cols) + 1)))

        logger.info('  Generating temporally-augmented samples ...')

        df_samples = df_samples.rename(columns=dict(zip(feature_cols, pred_cols)))

        keep_features = []
        for i in range(0, len(params['classify_crf']['keep_features']), 2):
            keep_features.append(tuple(map(bool, params['classify_crf']['keep_features'][i:i+2])))

        # Transform the time series into extra features
        scaled_features = image_utils.calc_features(df_samples[pred_cols].values,
                                                    keep_features=keep_features,
                                                    nbands=len(params['classify_crf']['image_bands_pred']))

        pred_cols = list(map(str, range(1, scaled_features.shape[1] + 1)))
        df_samples[pred_cols] = scaled_features

        X, y = [], []

        def _sampler(Xlist, ylist, dataframe_list, fractions, shuffle=False):

            return rcrf.sample_dataframe(Xlist,
                                         ylist,
                                         dataframe_list,
                                         pred_cols,
                                         lookup.LABELS_DICT,
                                         id_column='lc',
                                         shuffle=shuffle,
                                         fractions=fractions,
                                         max_workers=params['num_workers'],
                                         **params['classify_crf']['augment'])

        for label in map(int, sorted(df_samples.lc.unique().tolist())):

            dfa = df_samples.query(f"lc == {label}")

            if dfa.empty:
                continue

            df_list = [dfa]
            fractions = [1.0]

            X, y = _sampler(X, y, df_list, fractions)

        # Cropland <--> grassland
        dfa = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'crp']}")
        dfb = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'grs']}")

        if dfa.empty or dfb.empty:
            pass
        else:

            df_list = [dfa, dfb]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)
            df_list = [dfb, dfa]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)

        # Cropland <--> barren
        dfa = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'crp']}")
        dfb = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'bar']}")

        if dfa.empty or dfb.empty:
            pass
        else:

            df_list = [dfa, dfb]
            fractions = [0.8, 0.2]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.2, 0.8]
            X, y = _sampler(X, y, df_list, fractions)
            df_list = [dfb, dfa]
            fractions = [0.8, 0.2]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.2, 0.8]
            X, y = _sampler(X, y, df_list, fractions)

        # Trees <--> shrubs
        dfa = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'trs']}")
        dfb = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'shr']}")

        if dfa.empty or dfb.empty:
            pass
        else:

            df_list = [dfa, dfb]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)
            df_list = [dfb, dfa]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)

        # Trees <--> grassland
        dfa = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'trs']}")
        dfb = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'grs']}")

        if dfa.empty or dfb.empty:
            pass
        else:

            df_list = [dfa, dfb]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)
            df_list = [dfb, dfa]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)

        # Trees <--> cropland
        dfa = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'trs']}")
        dfb = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'crp']}")

        if dfa.empty or dfb.empty:
            pass
        else:

            df_list = [dfa, dfb]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)
            df_list = [dfb, dfa]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)

        # Shrubs <--> grassland
        dfa = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'shr']}")
        dfb = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'grs']}")

        if dfa.empty or dfb.empty:
            pass
        else:

            df_list = [dfa, dfb]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)
            df_list = [dfb, dfa]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)

        # Shrubs <--> cropland
        dfa = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'shr']}")
        dfb = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'crp']}")

        if dfa.empty or dfb.empty:
            pass
        else:

            df_list = [dfa, dfb]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)
            df_list = [dfb, dfa]
            fractions = [0.7, 0.3]
            X, y = _sampler(X, y, df_list, fractions)
            fractions = [0.3, 0.7]
            X, y = _sampler(X, y, df_list, fractions)

        # Wetlands <--> grassland <--> cropland
        dfa = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'wtl']}")
        dfb = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'grs']}")
        dfc = df_samples.query(f"lc == {lookup.LABELS_DICT_r[b'crp']}")

        if dfa.empty or dfb.empty:
            pass
        else:

            df_list = [dfa, dfb, dfc]
            fractions = [1/3.0, 1/3.0, 1/3.0]
            X, y = _sampler(X, y, df_list, fractions, shuffle=True)

        X_filter = []
        y_filter = []

        for Xlist, ylist in zip(X, y):
            if list(set(ylist))[0] != 'null':
                X_filter.append([xfeas for xfeas in Xlist])
                y_filter.append([ylab for ylab in ylist])

        X = X_filter
        y = y_filter

        # logger.info('  Tuning CRF parameters ...')
        #
        # clf = rcrf.CRFClassifier()
        # clf.tune(X, y, test_size=0.6, num_cpus=params['num_workers'])
        #
        # import ipdb;ipdb.set_trace()

        logger.info('  Fitting a CRF classifier ...')

        # Gradient descent using the limited-memory BFGS method (with L1 and L2 regularization)
        clf = rcrf.CRFClassifier(algorithm='lbfgs',
                                 c1=0.01,
                                 c2=0.1,
                                 max_iterations=None,
                                 num_memories=40,
                                 epsilon=0.05,
                                 delta=0.05,
                                 period=30,
                                 linesearch='MoreThuente',
                                 max_linesearch=40,
                                 all_possible_states=False,
                                 all_possible_transitions=False,
                                 verbose=False)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
        # y_test = [[ylab.decode() if isinstance(ylab, bytes) else ylab for ylab in ylist] for ylist in y_test]
        # clf.fit(X_train, y_train)
        # y_pred = clf.model.predict(X_test)
        # logger.info(metrics.flat_classification_report(y_test, y_pred, labels=list(lookup.LABELS_DICT_str.values())))

        clf.fit(X, y, columns=pred_cols, func=image_utils.calc_features)

        clf.to_file(model_file, overwrite=True)

        logger.info('  Finished fitting the land cover model.')

    ######################
    # Make the predictions
    ######################

    if params['classify_crf']['method'] in ['predict', 'fit_predict']:

        for grid in params['grids']:

            logger.info(f'  Predicting land cover for grid {grid} ...')

            ppaths = ProjectPaths(params, grid=grid)

            vrt_files, image_names, time_names = image_utils.open_images(ppaths, params, 'classify')

            cls_image = ppaths.cls / f'{grid:06d}.tif'

            if params['classify_crf']['overwrite_image']:

                if cls_image.is_file():
                    cls_image.unlink()

            grid_years = np.sort(np.unique(np.array([dt.year for dt in time_names])))

            with gw.open(image_names[0]) as src:

                profile = dict(blockxsize=src.gw.col_chunks,
                               blockysize=src.gw.row_chunks,
                               crs=src.crs,
                               transform=src.transform,
                               driver='GTiff',
                               count=grid_years.shape[0]-1,
                               height=src.gw.nrows,
                               width=src.gw.ncols,
                               nodata=params['classify_crf']['nodata'],
                               dtype='uint8',
                               compress='lzw',
                               tiled=True)

            dft = pd.DataFrame(data=range(0, len(time_names)), columns=['image_index'], index=time_names)

            # Get windows from one image
            with gw.open(image_names[0]) as src:

                windows = list(src.gw.windows(row_chunks=params['classify_crf']['chunks'],
                                              col_chunks=params['classify_crf']['chunks']))

            for w in tqdm(windows, total=len(windows)):

                # If the file exists, check if the block has been mapped
                if cls_image.is_file():

                    with rio.open(cls_image, mode='r') as src:
                        block = src.read(1, window=w)

                    if block.mean() != params['classify_crf']['nodata']:
                        continue

                X_list = []
                processed_years = []
                feature_band_count = []

                for year in grid_years[:-1]:

                    image_index = date_utils.year_to_index(dft, year, year+1, params, 'classify')

                    with ExitStack() as stack:

                        src = image_utils.scale_stack(stack, vrt_files, params, 'classify', 'image_bands', 'image_bands_pred', image_index)

                        # Reshape to [samples x features]

                        if len(src.shape) == 4:

                            src = src[:, :, w.row_off:w.row_off+w.height, w.col_off:w.col_off+w.width]

                            X = src.stack(s=('y', 'x'))\
                                    .stack(X=('time', 'band'))\
                                    .astype('float64')\
                                    .fillna(0)\
                                    .data\
                                    .rechunk((src.gw.row_chunks * src.gw.col_chunks, 1))

                        else:

                            src = src[:, w.row_off:w.row_off+w.height, w.col_off:w.col_off+w.width]

                            X = src.stack(s=('y', 'x'))\
                                    .transpose()\
                                    .astype('float64')\
                                    .fillna(0)\
                                    .data\
                                    .rechunk((src.gw.row_chunks * src.gw.col_chunks, 1))

                        # if feature_band_count:
                        #
                        #     if X.shape[1] != int(stats.mode(feature_band_count).mode[0]):
                        #         logger.warning(f'Year {year} has {X.shape[1]} bands, but there are {int(stats.mode(feature_band_count).mode[0])} predictive bands. Excluding it from the stack.')
                        #         continue

                        # feature_band_count.append(X.shape[1])
                        X_list.append(X[:, :60])
                        processed_years.append(year)

                # feature_mode = int(stats.mode(feature_band_count).mode[0])
                #
                # # Check the end years
                # if X_list[0].shape[1] != feature_mode:
                #     X_list = X_list[1:]
                #     processed_years = processed_years[1:]
                #
                # if X_list[-1].shape[1] != feature_mode:
                #     X_list = X_list[:-1]
                #     processed_years = processed_years[:-1]

                X_list = dask.compute(X_list, num_workers=params['num_workers'])[0]

                keep_features = []
                for i in range(0, len(params['classify_crf']['keep_features']), 2):
                    keep_features.append(tuple(map(bool, params['classify_crf']['keep_features'][i:i+2])))

                probas = clf.predict_probas(X_list,
                                            w.height,
                                            w.width,
                                            y_names=sorted(list(lookup.LABELS_DICT_str.values())),
                                            keep_features=keep_features,
                                            nbands=len(params['classify_crf']['image_bands_pred']))

                argmax = np.uint8(np.squeeze(probas.argmax(axis=1)))

                # Recode the positional indices to classes
                for i, k in enumerate(sorted(lookup.LABELS_DICT.values())):
                    argmax = np.where(argmax == i, lookup.LABELS_DICT_r[k], argmax)

                profile['count'] = probas.shape[0]

                if not cls_image.is_file():

                    with rio.open(cls_image, mode='w', **profile) as dst:
                        pass

                lookup.CLS_METADATA['coverage_dimensions'] = f'layers:{probas.shape[0]}, rows:1000, columns:1000'
                lookup.CLS_METADATA['title'] = f'Annual land cover ({processed_years[0]}-{processed_years[-1]})'
                lookup.CLS_METADATA['coverage_timeframe'] = f"{params['classify_crf']['start']} year 1 to {params['classify_crf']['end']} year 2"

                for b in range(0, probas.shape[0]):
                    lookup.CLS_METADATA[f'band_{b+1:02d}'] = f'{processed_years[b]}/{processed_years[b]+1}'

                # Write the predictions to file
                with rio.open(cls_image, mode='r+') as dst:

                    dst.update_tags(**lookup.CLS_METADATA)

                    dst.write(argmax,
                              indexes=1 if probas.shape[0] == 1 else list(range(1, probas.shape[0]+1)),
                              window=w)

            image_utils.clean_vrt_files(vrt_files)

            logger.info(f'Finished predicting land cover for grid {grid}.')




