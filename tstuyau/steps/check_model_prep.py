
import sys
from pathlib import Path
import json
import joblib
import csv
import random
from collections.abc import Iterable
import numpy as np
import rasterio as rio
from rasterio.mask import mask
import pandas as pd
import geowombat as gw
import geopandas as gpd
from shapely.geometry import Polygon
import shutil
import tempfile
from .project import ProjectPaths
from .aggregate import make_ts_composite
from .mod_utils import getset_feature_model, get_train_yrs_str, get_class_col,  multiclass_mod, get_confusion_matrix
from .mod_utils import get_holdout_scores, get_binary_holdout_score, prep_test_train, log_acc_results
from .lookup import CROP_CATS_Py0, CROP_CATS, MIXED_CROPS_Py0, MIXED_CROPS, MIXED_NONCROPS_Py0, MIXED_NONCROPS, LC_FOCUS_DICT, LC_VALS_DICT
from ..handler import logger


def prioritize_row(row, lccol, project_v=None, focus='All'):
    '''
    These are very specific quality flags to maximize/minimize the sample for certain over/under represented groups.
    '''

    if project_v == 'Py0':
        if (row[f'{lccol}_name'] == 'NoVeg_Built') & (row['sampgroup'] != 'rd_samp') & (row['source'] == 'GE'):
            priority = 1
        elif (row[f'{lccol}_name'] == 'NoVeg_Bare') & (row['sampgroup'] != 'rd_samp') & (row['source'] == 'GE'):
            priority = 1
        elif focus=='All' and row['estrat'] != 'BH':
            ## For full maps, want to prioritize Chaco data because there is less compared to E Py. but not ideal for E Py based optimization
            priority = 1
        elif (row['source'] == 'GE') & (row['entry_lev'] > 1):
            priority = 2
    for doubt_col in ['doubt_CNC', 'doubt_LC', 'doubt_LC5']:
        if (doubt_col in row) and (row[doubt_col] == 1):
            priority = 4
    else:
        priority = 3
    
    
    return priority

def balance_training_data(params, pixdf=None, stats_only=False, print_file=False, out_path=None):
    '''
    balances class samples based on map proportion, relative to sample size for class with max map proportion
       this estimated map proportion is a columnin the LUT <bal_col>. bal_col should start with 'per', followed by LCXX 
       (LCXX should be an actual column name and LCXX_name should also exist. The CELPy crop maps used a column named "perLC32E")
    allows a minimum threshold to be set <params['sampe_model']['minsamp']> so that sample sizes are not reduced below the minimum
    allows a factor to be set for mixed (heterogeneous) classes to sample them more heavily than main classes
    (the maximum value will depend on the available samples for these classes.)
    Samples can be given higher priority based on characteristics. This is highly specific to a given project, so <project_ver> is needed
        if <project_v> is None, all samples given same priority unless they are flagged as having doubts.  
        when provided along with <project_v>, <focus> allows different prioritization of sample data depending on region
       (can select higher % of sample in regions where there is less data sample data (for example, if sample == 'All' in <project_v> = 'Py0' 
       samples from the Chaco are given higher priority because they are less represented.) 
       
    <reduce> allows dataset to be reduced by that percentage (0 to 1) for rebustness checks (0 = don't reduce)
    '''

    minsamp = params['sampe_model']['minsamp']
    minbal = params['sampe_model']['minbal']
    lut = pd.read_csv(params['schematic_model']['lut'])
    trainyrs = params['sample_model']['train_yrs']
    allyrstr = get_train_yrs_str(trainyrs)
    focus_geo = params['sampe_model']['focus_area']
    project_v = params['project_ver']
    reduce = params['sample_model']['reduce']

    if reduce==0:
        sampmod0 = f'min{minsamp}upsamp{minbal}'
    else:
        sampmod0 = f'red{int(100*reduce)}min{minsamp}upsamp{minbal}'
    
    if not focus_geo or focus_geo.startswith('All'):
        sampmod = f'{sampmod0}_{allyrstr}'
    else:
        sampmod = f'{sampmod0}_{allyrstr}-{focus_geo}'

    if pixdf:
        if isinstance(pixdf, pd.DataFrame):
            pixdf = pixdf
        else:
            pixdf = pd.read_csv(pixdf)
    else:
        pixdf = pd.read_csv(params['sample_model']['point_file'])

    if project_v == 'Py0':
        bal_col = 'perLC32E'
        lccol = 'LC32'
    else:
        bal_col = ['sample_model']['balance_col']
    if bal_col not in pixdf:
        lccol = bal_col.split('per')[1]
        #if lccol not in pixdf:
        #    logger.warning(f"ERROR: cannot find {lccol} in pixdf")
        #else:
        pixdf = pixdf.merge(lut[['LC_UNQ',bal_col,lccol,f'{lccol}_name']], left_on='LC_UNQ', right_on='LC_UNQ', how='left')
    
    ## add internal pixel homogeneity: this is already done in CollectCube prior 
    #pixdf['HOMOINT'] = pixdf.apply(lambda row : get_internal_homogeneity(row,'LC32_name',True), axis=1) 
    ##TODO: Add optional filters here if desired -- but for training probably best to include impure data
    
    ## Allow only one pixel per neighborhood per class:
    if 'HOMOINT' in pixdf:
        pixdf = pixdf.sort_values('HOMOINT', ascending=False).drop_duplicates(subset=['PID0', f'{lccol}_name'])
        ##TODO: consider alternatives to taking only pixel with highest internal purity
        ## Separate highest quality sample to select from first 
    pixdf['priority'] = pixdf.apply(lambda row : prioritize_row(row, lccol, project_v= project_v, focus=focus_geo), axis=1) 

    ## get number of samples for each class and priority level
    counts = pixdf[f'{lccol}_name'].value_counts().reset_index()
    counts.columns = [f'{lccol}_name', 'counts']
    logger.info(f'counts={counts}')
    counts.set_index(f'{lccol}_name',inplace=True)
    logger.info(f'Total sample size before balancing is: {sum(counts["counts"])}')
    
    p1_pixdf = pixdf[pixdf['priority'] == 1]
    p1_counts = p1_pixdf[f'{lccol}_name'].value_counts().reset_index()
    p1_counts.columns = [f'{lccol}_name', 'p1_counts']
    p1_counts.set_index(f'{lccol}_name',inplace=True)
    p2_pixdf = pixdf[pixdf['priority'] == 2]
    p2_counts = p2_pixdf[f'{lccol}_name'].value_counts().reset_index()
    p2_counts.columns = pixdf[f'{lccol}_name', 'p2_counts']
    p2_counts.set_index(f'{lccol}_name',inplace=True)
    p3_pixdf = pixdf[pixdf['priority'] == 3]
    p3_counts = p3_pixdf[f'{lccol}_name'].value_counts().reset_index()
    p3_counts.columns = pixdf[f'{lccol}_name', 'p3_counts']
    p3_counts.set_index(f'{lccol}_name',inplace=True)

    all_counts = pd.concat([counts,p1_counts,p2_counts,p3_counts],axis=1)
    all_counts['p1_counts'].fillna(0,inplace=True)
    all_counts['p2_counts'].fillna(0,inplace=True)
    all_counts['p3_counts'].fillna(0,inplace=True)
    
    ## to balance, get estimated class percents (this is estimated from other maps and in <sample_model:balance_col> column in LUT)
    classprev = lut.sort_values(bal_col)[[bal_col, f'{lccol}_name']]
    classprev= classprev.dropna()
    classprev = classprev.drop_duplicates(subset = bal_col) 
    tot = classprev[bal_col].sum()
    logger.info(f"checking total percent for balance_col:...{tot}...(should be 1)")
    ## get highest class percent
    mmax = classprev[bal_col].max()
    ## convert all class percents to ratio of highest
          ## (keeping all samples from mmax class (n), n = mmax * TotalSamp  -> TotalSamp = n/mmax )
    classprev[bal_col] = (1-reduce) * classprev[bal_col]/mmax
    ratiodf = classprev.merge(all_counts, left_on=f"{lccol}_name", right_on=bal_col, how='left')
    maxsamp = ratiodf.at[ratiodf[bal_col].idxmax(), 'counts']
    logger.info(f'samp size for class with max proportion is {maxsamp}')
    ##  Get resample ratio based on class proportion. if existing samples are < minsamp, keep all samples. 
    ##    Otherwise reduce according to proportion, but do not allow to go below minsamp
    ##    Allow for separate treatment of mixed classes based on minbal
    
    if params['iter_models']['optimize_on'] == 'smCrop':
        if project_v == 'Py0':
            CROP_CATS = CROP_CATS_Py0
            MIXED_CROPS = MIXED_CROPS_Py0
            MIXED_NONCROPS = MIXED_NONCROPS_Py0
    
        mixed_classes = MIXED_CROPS + MIXED_NONCROPS
        ## For crop-type map, "Crops-Mandioca", "Crops-Horticulture","Crops-Sesame" are removed from this list so that they can use the
        ##  full sample (up to the minsamp), whereas for mixed-ratio progression analyses they are included so not to exceed the intended
        ##  ratio. TODO: make a parameter to indicate which way they will be treated. 
        ratiodf['ratios'] = np.where(ratiodf[f'{lccol}_name'].isin(mixed_classes),(minbal * ratiodf[f"{lccol}_name"] * maxsamp / ratiodf["counts"]),
                           np.where(ratiodf["counts"] < minsamp, 1, 
                              np.where(ratiodf[bal_col] * maxsamp < ratiodf["counts"], 
                                 np.maximum((minsamp/ratiodf["counts"]), (ratiodf[bal_col] * maxsamp / ratiodf["counts"])),   
                            1))).round(3)
    
        ratiodf['p1_draw'] =  np.where( ratiodf['p1_counts'] == 0, 0, 
                                   np.where(ratiodf['p1_counts'] <= ratiodf['counts'] * ratiodf['ratios'], 1,  
                                            (ratiodf['ratios']*ratiodf['counts'])/ratiodf['p1_counts'])).round(3)
        ratiodf['p2_draw'] =  np.where( ratiodf['p2_counts'] == 0, 0,
                                   np.where(ratiodf['p1_counts'] + ratiodf['p2_counts'] < ratiodf['counts'] * ratiodf['ratios'], 1,
                                            (ratiodf['ratios']*ratiodf['counts'] - ratiodf['p1_counts']) / ratiodf['p2_counts'])).round(3)
        ratiodf['p3_draw'] =  np.where(ratiodf['p1_counts'] + ratiodf['p2_counts'] >  ratiodf['counts'] * ratiodf['ratios'], 0,
                                   (ratiodf['ratios']*ratiodf['counts'] - (ratiodf['p1_counts']+ratiodf['p2_counts']))/ (ratiodf['counts'] - (ratiodf['p1_counts']+ratiodf['p2_counts']))).round(3)
                                                      
        ## Use random column to select samples (already in df here, for easy reproduction, but could make new ran column from 0-1)
        if 'p1_draw' in pixdf.columns:  ## drop old values if already exist
            pixdf.drop(['ratios','p1_draw','p2_draw','p3_draw'], axis=1, inplace=True)
        pixdf_ratios = pixdf.merge(ratiodf[[f'{lccol}_name','ratios','p1_draw','p2_draw','p3_draw']],
                               left_on=f"{lccol}_name", right_on=f"{lccol}_name", how='left')
    
    if reduce > 0: # make new rand col because original has reduced range (but keep original for replication purposes)
        pixdf_ratios['rand3'] = np.random.uniform(0, 1, len(pixdf_ratios))
        rand_col = 'rand3'
    else:
        rand_col = 'rand'
    p1_samp = pixdf_ratios[(pixdf_ratios['priority'] == 1) & (pixdf_ratios[rand_col] < pixdf_ratios['p1_draw'])]
    p2_samp = pixdf_ratios[(pixdf_ratios['priority'] == 2) & (pixdf_ratios[rand_col] < pixdf_ratios['p2_draw'])]
    p3_samp = pixdf_ratios[(pixdf_ratios['priority'] == 3) & (pixdf_ratios[rand_col] < pixdf_ratios['p3_draw'])]
    full_samp = pd.concat([p1_samp,p2_samp,p3_samp],axis=0)
    counts=full_samp[f'{lccol}_name'].value_counts()
    logger.info(f'counts={counts}')
    totsamp = sum(full_samp[f'{lccol}_name'].value_counts())
    logger.info(f'Total sample size after balancing is: {totsamp}')

    samp_entry={}
    samp_name = f'red{int(100*reduce)}min{minsamp}upsample{minbal}'
    samp_entry['totsamp'] = int(totsamp)
    samp_entry['maxcat'] = int(maxsamp*(1-reduce))

    if params['iter_models']['optimize_on'] == 'smCrop':
        mixed_crops = full_samp[full_samp[f'{lccol}_name'].isin(MIXED_CROPS)]
        mixed_noncrops = full_samp[full_samp[f'{lccol}_name'].isin(MIXED_NONCROPS)]
        samp_entry['totmix_crop'] =  int(sum(mixed_crops[f'{lccol}_name'].value_counts()))
        samp_entry['totmix_nocrop'] = int(sum(mixed_noncrops[f'{lccol}_name'].value_counts()))
    
    logger.info(f'samp_entry:{samp_entry}')

    ## save sample model dictionary to optimization directory
    samp_stat_dir = params['iter_models']['opt_dir']
    if not samp_stat_dir:
        ppaths=ProjectPaths(params)
        samp_stat_dir = ppaths.optimization
    samp_stat_dir.mkdir(parents=True, exist_ok=True)
    samp_stat_dict = Path(samp_stat_dir) / 'sample_breakdown_dict.json'
    try:
        with open(samp_stat_dict, 'r+') as stat_dict:
            dic = json.load(stat_dict)
        dic.update({samp_name : samp_entry})
          
    except IOError:
        logger.warning('File not found, will create a new one.')
        with open(samp_stat_dict, 'w') as new_dict:
            dic = {samp_name : samp_entry}
        
    with open(samp_stat_dict, "w") as new_dict:
        json.dump(dic, new_dict)
    
    if not stats_only:
        if not print_file: 
            return full_samp
        else:
            if not out_path:
                ptsamp_dir = params['sample_model']['point_samp_dir']
                if not ptsamp_dir:
                    ppaths=ProjectPaths(params)
                    ptsamp_dir = ppaths.trainptsets
                out_path = Path(ptsamp_dir) /f'{sampmod}.csv'
            pd.DataFrame.to_csv(full_samp, out_path)
            return out_path
        
    else:
        return samp_stat_dict

def get_stable_holdout(params, df_in=None, overwrite=False):
    '''
    Separates stable holdout from point file (with features already applied) to use during model optimization / testing
        creates a ...TRAINING_all... file in the <ptsfeat_dir> directory and a HOLDOUT_all file in the <fixed_ho_dir> directory
    By default removes from training set all points with value in 'rand2' column <sample_model:test_thresh>/100 
        (note: rand2 is a decimal and test_thresh is an integer 0-100 (thus gets divided by 100 here)
    Option to run subsamples (if <num_subsamples> > 0), in which case that number of training and holdout sets
        are generated in dirictories in the scratch directory. 

    Note: This holdout is used for optimization. If samples are to be used for validation, need to separate before running this.

    It is possible that the holdout was already removed when the sample was created (e.g with CollectCube). If so, the holdout should
         be named GENERAL_HOLDOUT_all_YY and there should be no rand2 values <.2 in the training set. We check for this here,
         but cannot properly run subsamples if this is the case.
    '''
    
    ho_dir = params['sample_model']['fixed_ho_dir']
    if not ho_dir:
        ppaths=ProjectPaths(params)
        ho_dir = ppaths.hos
        ho_dir.mkdir(parents=True, exist_ok=True)
    ftset_dir = ['classify']['ptsfeat_dir']
    if not ftset_dir:
        ppaths=ProjectPaths(params)
        ftset_dir = ppaths.trainfeatsets
        ftset_dir.mkdir(parents=True, exist_ok=True)
      
    feat_mod_name = params['feature_model']['name']
    thresh = params['sample_model']['test_thresh'] / 100
    num_subsamples =  params['sample_model']['num_subsamples']  ## 0 if not running subsamples
    subsample_seed = params['sample_model']['subsamp_seed']    ## only used if subsample >0    
    scratch_dir = params['scratch_dir'] ## only used if subsample > 0   
    balanced_ho = params['sample_model']['use_balanced_ho']
    subsample = params['sample_model']['subsample']
    
    trainyrs = params['sample_model']['train_yrs']
    if isinstance(trainyrs, int):
        yrlist = [trainyrs]
    elif (len(trainyrs) == 2) and (trainyrs[0]<trainyrs[1]):
        yrlist = list(range(trainyrs))
    else:
        yrlist = trainyrs

    for y in yrlist:
        yrst = get_train_yrs_str(y)

        ## first check if fixed holdout already exists
        if not num_subsamples:
            ho_out_path = Path(ho_dir) / f"{feat_mod_name}_HOLDOUT_all_{yrst}.csv"
            ho_bal_path = Path(ho_dir) / f"{feat_mod_name}_HOLDOUT_balno_{yrst}.csv"
        else:
            ho_out_path = Path(params['scratch_dir']) / 'hos' / f"{feat_mod_name}_HOLDOUT_all_{yrst}_ss{subsample}.csv"
            ho_bal_path = Path(params['scratch_dir'] / 'hos') / f"{feat_mod_name}_HOLDOUT_balno_{yrst}_ss{subsample}.csv"
        if not (balanced_ho or overwrite) and ho_out_path.is_file():
            logger.info(f'holdout file {ho_out_path} already exists. Change name or set overwrite=True to save new holdout')
           
            if isinstance(trainyrs,int) or len(trainyrs) == 1:
                return pd.read_csv(ho_out_path)
        
        elif balanced_ho and overwrite and ho_bal_path.is_file():
            logger.info(f'holdout file {ho_bal_path} already exists. Change name or set overwrite=True to save new holdout')
            
            if isinstance(trainyrs,int) or len(trainyrs) == 1:
                return pd.read_csv(ho_bal_path)

        else:  ## make new holdout sets

            if isinstance(df_in, pd.DataFrame):
                ## for use in hands-on tinkering -- make sure params['sample_model']['train_yrs'] is a single year
                df_in = df_in
            elif df_in:
                df_in = pd.read_csv(df_in)
            else: ## for normal pipeline 
                df_path = Path(ftset_dir) / f'ptsfeats_{feat_mod_name}_{y}.csv'    
                if df_path.exists():
                    df_in = pd.read_csv(df_path)
                    logger.info(f'removing fixed holdout from {df_path} \n')
                else:
                    logger.warning(f'ERROR: Cannot find the input point-feature set at {df_path} \n')
        
            ## Check if fixed holdout has already been removed from point set:
            if len(df_in[df_in['rand2']<.1]) == 0:
                logger.info('fixed holdout has alreday been removed from training set')
                if not num_subsamples:
                    logger.warning('ERROR: training set is already reduced. Rurun with full set')
                else:  ## legacy
                    gen_ho = Path(ho_dir) / f"GENERAL_HOLDOUT_all_{y}.csv"
                    if gen_ho.is_file():
                        logger.info(f'getting features for existing holdout at {gen_ho}')
                        params['sample_model']['point_file'] = gen_ho
                        orig_vardf = params['classify']['vardf_dir']
                        orig_ptsamp_dir = params['sample_model']['point_samp_dir']
                        params['classify']['vardf_dir'] = ho_dir
                        params['sample_model']['point_samp_dir']=None
                        ho = format_ptfeat_set(params)
                        params['classify']['vardf_dir'] = orig_vardf
                        params['sample_model']['point_samp_dir'] = orig_ptsamp_dir
                        
                        if isinstance(trainyrs,int) or len(trainyrs) == 1:
                            return ho
                    else:
                        logger.warning(f'ERROR: cannot find holdout at {ho_out_path} or {gen_ho}. Check parameters')
                        return None
            
            else: ## make a new holdout set
                
                ## the following are only needed if <sample_model:balanced_ho> = True
                focus = params['iter_models']['optimize_on'] 
                reduce_orig = params['sampe_model']['reduce']
                min_orig = params['sampe_model']['minsamp']
                bal_orig=params['sampe_model']['minbal']
                ##  reset these temporarily for holdout balancing
                params['sampe_model']['reduce'] = 0
                params['sampe_model']['minsamp'] = 100
                params['sampe_model']['minbal'] = 1
                
                ## without subsamples -- this is the most common scenario
                if not num_subsamples:
                    ho = df_in[df_in['rand2']<=thresh]
                    non_hos = df_in[df_in['rand2'] > thresh] 
                    logger.info(f'there are {len(non_hos)} pixels remaining after removing fixed holdout \n')
                    train_out_path = Path(ftset_dir) / f"{feat_mod_name}_TRAINING_all_{yrst}.csv"
                    non_hos.to_csv(train_out_path)
                    ho.to_csv(ho_out_path)
                    if balanced_ho:
                        if focus in LC_FOCUS_DICT.keys():
                            for cat in LC_FOCUS_DICT[focus]['cats']:
                                if cat.startswith('no'):  ## currently only balancing the negative category
                                    ho_no = ho.loc[ho['LC_UNQ'].isin(LC_VALS_DICT[cat])]
                                    logger.info(f'balancing {cat}. Originally has {len(ho_no)} records')
                                    hono_bal = balance_training_data(params, ho_no, stats_only=False, print_file=False, out_path=None)
                                    logger.info(f"there are {len(hono_bal)} pixels in the {cat} holdout \n")
                            ho_other = ho.loc[~ho['LC_UNQ'].isin(LC_VALS_DICT[cat])]
                            ho_bal = pd.concat([ho_no,ho_other],axis=1)
                            ho_bal.to_csv(ho_bal_path)
                
                else:  ## with subsamples
                    logger.info(f'generating {num_subsamples} pt-feature sets for {feat_mod_name} in {ftset_dir}')
                    ## seed the generator sequence for exact reproducibility 
                    rng = np.random.default_rng(seed=subsample_seed)
                    for n in range(num_subsamples):
                        ## pull a random number for each point, following the generator sequence for iterations
                        df_in['iran'] = rng.random(size=len(df_in))
                        ho_out_path = Path(ho_dir)/ f"{feat_mod_name}_HOLDOUT_all_{yrst}_ss{n}.csv"
                        ho = df_in[df_in['iran']<=thresh]
                        non_hos = df_in[df_in['iran'] > thresh] 
                        logger.info(f'there are {len(non_hos)} pixels remaining after removing fixed holdout \n')
                        train_out_path = Path(ftset_dir)/ f"{feat_mod_name}_TRAINING_all_{yrst}_ss{n}.csv"
                        non_hos.to_csv(train_out_path)
                        if balanced_ho:
                            if focus in LC_FOCUS_DICT.keys():
                                for cat in LC_FOCUS_DICT[focus]['cats']:
                                    if cat.startswith('no'):  ## currently only balancing the negative category
                                        ho_no = ho.loc[ho['LC_UNQ'].isin(LC_VALS_DICT[cat])]
                                        logger.info(f'balancing {cat}. Originally has {len(ho_no)} records')
                                        hono_bal = balance_training_data(params, ho_no, stats_only=False, print_file=False, out_path=None)
                                        logger.info(f"there are {len(hono_bal)} pixels in the {cat} holdout \n")
                            ho_other = ho.loc[~ho['LC_UNQ'].isin(LC_VALS_DICT[cat])]
                            ho_bal = pd.concat([ho_no,ho_other],axis=1)
                            ho_out_path_bal = ho_dir/ f"{feat_mod_name}_HOLDOUT_balno_{yrst}_ss{n}.csv"
                            ho_bal.to_csv(ho_out_path_bal)                  
                        else:
                            ho.to_csv(ho_out_path)
                
                # reset params to original setting
                params['sampe_model']['reduce'] = reduce_orig
                params['sampe_model']['minsamp'] = min_orig
                params['sampe_model']['minbal'] = bal_orig             
                       
            logger.info(f'holdout sets are in {ho_dir} \n')

        '''
        ## legacy code. All hos should be combined into one file now.
        if (params['iter_models']['optimize_on'] == 'smCrops') & (
            Path(params['sample_model']['fixed_ho_dir']) / f'{feat_mod_name}_HOLDOUT_smallCrop_{trainyrs}.csv').is_file():
            ho_smallCrop = Path(params['sample_model']['fixed_ho_dir']) / f'{feat_mod_name}_HOLDOUT_smallCrop_{trainyrs}.csv'
            ho_bigCrop = Path(params['sample_model']['fixed_ho_dir']) / f'{feat_mod_name}_HOLDOUT_bigCrop_{trainyrs}.csv'
            ho_noCrop = Path(params['sample_model']['fixed_ho_dir']) / f'{feat_mod_name}_HOLDOUT_noCrop_{trainyrs}.csv'
        '''

def separate_field_level_holdout(training_pix_path, holdout_field_pix_path, out_dir):
    '''
    USE THIS WHEN WE AUGMENT BY POLYGON
    Generates separate pixel databases for training data and 20% field-level holdout
    Use this instead of generate_holdout() to fit a model to an exsisting holdout set
       to avoid having points from the same polygon in both the training and holdout sets (as this would inflate accuracy)
    '''    
    holdout_set = pd.read_csv(holdout_field_pix_path)
    pixels = pd.read_csv(training_pix_path)
    
    pixels_holdouts = pixels[pixels.field_id.isin(holdout_set['unique_id'])]
    pixels_holdouts['set']='HOLDOUT'
    pixels_training = pixels[~pixels.field_id.isin(holdout_set['unique_id'])]
    pixels_training['set']='TRAINING'

    logger.info(f"original training set had {len(pixels)} rows. \n") 
    logger.info(f"Current training set has {len(pixels_training)} rows and holdout has {len(pixels_holdouts)} rows. \n")
    
    ##TODO: add print option 

    return training_pix_path, holdout_field_pix_path
   
def format_ptfeat_set(params):
    '''
    Filters the full set of points and features to fit a smaller point and feature sample.
    Formats the pt-feature set for final modeling, includind making sure all bands are in the same 
    order as they appear in the model (as read from the band_names attribute of the model in the <feature_mod_dict>).
    If multiple training years set in <'sample_model':'train_yrs'>, will make a multi-year df as well as the individuals.
    
    This assumes that all features in the feature model have already been calculated for all sample points
    and exist in a dataframe at <classify:ptsfeat_dir>. A subset can be extracted from a larger feature set using the 
    feature_model param <subset_features> = True and providing the name of the larger model as the <full_feature_mod>.
    Will separate a testing holdout if <'sample_model':'fixed_ho'> is set. This is used for optimization, so another
       holdout should already be separated from the data if it is needed for final validation.

    If variables have not yet been calculated, run "make_var_dataframe" in check_sample first.
    '''
    ftset_dir = params['classify']['ptsfeat_dir']
    ## <ftset_dir> holds the df with the features extracted for all possible points
    if not ftset_dir:
        ppaths=ProjectPaths(params)
        ftset_dir = ppaths.trainfeatsets
    ## <ptsamp_dir> holds the point sets for each sample model without features appended
    ptsamp_dir = params['sample_model']['point_samp_dir']
    if not ptsamp_dir:
        ppaths=ProjectPaths(params)
        ptsamp_dir = ppaths.trainptsets
    ## <vardf_dir> holds the model-ready sets. Default <backup_path>.parents[1]/classification/inputs/pixdf_<model_name>.csv'
    ##   but may be in <temp_dir>/'optimization'/'vardfs' if part of optimization run
    vardf_dir = params['classify']['vardf_dir']
    if not vardf_dir:
        ppaths=ProjectPaths(params)
        vardf_dir = ppaths.fulltrainsets
        vardf_dir.mkdir(parents=True, exist_ok=True)
    
    feat_mod = params['feature_model']['name']
    ## read model from dict to make sure that band order is correct
    feature_mod_dict = params['feature_model']['feature_mod_dict']
    if not feature_mod_dict:
        ppaths=ProjectPaths(params)
        feature_mod_dict = str(ppaths.fmoddict)
    with open(feature_mod_dict, 'r+') as fmd:
        dic = json.load(fmd)
    model_bands = dic[feat_mod]['band_names']
    keep_vars = [f"var_{v.split('_', 1)[1]}" if v.startswith('sing') else f"var_{v}" for v in model_bands]
    logger.info(f'model bands from dict: {keep_vars}: \n')

    ## these parameters are all used her to get the model name
    ##     and later in the function to make a new sample model if it does not already exist
    class_mod_name = get_class_col(params['schematic_model']['lc_mod'], params['schematic_model']['lut'])[0]
    focus_geo = params['sample_model']['focus_area']  ## use if a geographical subset of the model is being built
    trainyrs = params['sample_model']['train_yrs']
    allyrstr = get_train_yrs_str(trainyrs)
    minsamp = params['sampe_model']['minsamp']
    minbal = params['sampe_model']['minbal']
    reduce = params['sample_model']['reduce']
    pt_key = params['sample_model']['point_file']
    fixed_ho = params['sample_model']['fixed_ho']  ## if True, will append vars to the subsetted _TRAINING Set
    lut=pd.read_csv(params['schematic_model']['lut'])
    
    if reduce==0:
        sampmod0 = f'min{minsamp}upsamp{minbal}'
    else:
        sampmod0 = f'red{int(100*reduce)}min{minsamp}upsamp{minbal}'
    
    if not focus_geo or focus_geo.startswith('All'):
        sampmod = f'{sampmod0}_YYYYY'   ### YYYY gets replaced as needed
    else:
        sampmod = f'{sampmod0}_YYYYY-{focus_geo}' ### YYYY gets replaced as needed

    multiyr_model_name = f"{feat_mod}_{sampmod.replace('YYYY',allyrstr)}_{class_mod_name}"
    if not params['sample_model']['subsample']:
        multiyr_vardf_path = Path(vardf_dir)/f"pixdf_{multiyr_model_name}.csv"
    else:
        multiyr_vardf_path = Path(vardf_dir)/f"pixdf_{multiyr_model_name}_ss{params['sample_model']['subsample']}.csv" 

    if multiyr_vardf_path.is_file():
        multiyr_vardf = pd.read_csv(multiyr_vardf_path)
        keep_other = [v for v in multiyr_vardf.columns.tolist() if not v.startswith('var_')]
        cols = keep_other + keep_vars
        if len(cols) != len(multiyr_vardf.columns.tolist()):
            multiyr_vardf = multiyr_vardf[cols]
            logger.debug(f'new order is: {multiyr_vardf.columns.tolist()}')
            pd.to_csv(multiyr_vardf,multiyr_vardf_path)
            logger.info('reorganized existing vardf \n')
        else:
            logger.info('vardf already exists \n')
        
        return multiyr_vardf

    elif params['feature_model']['subset_features']:
        fullfeat_mod =  params['feature_model']['full_feature_mod']
        fullmultiyr_vardf_path = multiyr_vardf_path.replace(feat_mod,fullfeat_mod)
        if fullmultiyr_vardf_path.is_file():
            fullmultiyr_vardf = pd.read_csv(multiyr_vardf_path)
            keep_other = [v for v in fullmultiyr_vardf.columns.tolist() if not v.startswith('var_')]
            cols = keep_other + keep_vars
            multiyr_vardf = fullmultiyr_vardf[cols]
            logger.debug(f'new order is: {multiyr_vardf.columns.tolist()}')
            logger.info(f'made reduced vardf \n')
            pd.to_csv(multiyr_vardf,multiyr_vardf_path)

        return multiyr_vardf
    
    elif list(Path(vardf_dir).glob(f"{feat_mod}_{sampmod.replace('YYYY',allyrstr)}*")) > 0:
        altdf_path= list(Path(vardf_dir).glob(f"{feat_mod}_{sampmod.replace('YYYY',allyrstr)}*"))[0]
        logger.info(f'found an alternate vardf: {altdf_path}. Joining new LC col... \n')
        altdf = pd.read_csv(altdf_path)
        if class_mod_name not in list(altdf.columns):
            new_vardf = altdf.merge(lut[['LC_UNQ',class_mod_name,f'{class_mod_name}_name']],on='LC_UNQ',how='left')
        else:
            new_vardf = altdf
        pd.to_csv(new_vardf,multiyr_vardf_path)
        
        return new_vardf

    else:  ## vardf does not already exist. get maximum training set and filter to the points included in the sample model
        if params['feature_model']['subset_features']:
            feat_mod = params['feature_model']['full_feature_mod']
        else:
            feat_mod = params['feature_model']['name']

        if isinstance(trainyrs, int):
            yrlist = [trainyrs]
        elif (len(trainyrs) == 2) and (trainyrs[0]<trainyrs[1]):
            yrlist = list(range(trainyrs))
        else:
            yrlist = trainyrs
    
        allyrs_vardfs=[]
        for y in yrlist:
            yrst = get_train_yrs_str(y)
            if fixed_ho:
                fullfeats_path = Path(ftset_dir)/f'{feat_mod}_TRAINING_all_{yrst}.csv'
                if not fullfeats_path.is_file():
                    get_stable_holdout(params, df_in=None, overwrite=False)
            else:
                fullfeats_path = Path(ftset_dir)/f'{feat_mod}_{y}.csv'
                if not fullfeats_path.is_file():
                    fullfeats_path = Path(ftset_dir)/f'ptsfeats_{fullfeat_mod}_{y}.csv'

            fullfeats = pd.read_csv(fullfeats_path)
            logger.debug(f'full features: {fullfeats.head()}')
        
            keep_other = [v for v in fullfeats.columns.tolist() if not v.startswith('var_')]
            cols = keep_other + keep_vars 
            mod_feats = fullfeats[cols]
            logger.info(f'new order is: {mod_feats.columns.tolist()}')
    
            if 'var_poly_area' in list(mod_feats.columns):
            ## hacky fix for issue of numbers over signed 16-bit max being converted to negative in var dataframe 
                mod_feats['var_poly_area'] = np.where(mod_feats['var_poly_area']<0,32767,mod_feats['var_poly_area'])
            #if 'OID_' in mod_feats.columns.tolist():
            #    mod_feats = pd.merge(mod_feats,pt_key[['OID_','PID']],left_on='OID_',right_on='OID_', how='left')
            #else:
            #    mod_feats = pd.merge(mod_feats,pt_key[['OID_','PID']],left_index=True,right_on='OID_', how='left')

            ## reduce the full feature set by the sample model
            pt_set = Path(ptsamp_dir) / f'{sampmod}.csv'
            if pt_set.is_file():
                samp = pt_set
            else:
                logger.info(f'cannot find {pt_set} \n')
                logger.info(f'making new balanced training set from {pt_key}  \n')
                params['sample_model']['train_yrs'] = y
                samp = balance_training_data(params, pixdf=pd.read_csv(pt_key), stats_only=False, print_file=True, out_path=pt_set)
                params['sample_model']['train_yrs'] = trainyrs

            samp_pts = pd.read_csv(samp)
            logger.debug(f'samp_pts: {samp_pts.head()} \n')
            logger.info(f'There are {samp_pts.shape[0]} sample points \n')

            ## formatting to deal with issues caused by prior methods. TODO: check what is still necessary and edit.
            if ('OID_' not in list(samp_pts.columns)) & ('Unnamed: 0' in list(samp_pts.columns)):
                samp_pts.rename(columns = {"Unnamed: 0": 'OID_'}, inplace = True)
            if 'LC_UNQ_y' in list(samp_pts.columns):
                samp_pts.drop(['LC_UNQ_y'], axis=1, inplace=True)
            ## dropping LC2 from sample points bc used different classification system in previous versions 
            if 'LC2' in list(samp_pts.columns):
                samp_pts.drop(['LC2'], axis=1, inplace=True)
            if 'LC3' in list(samp_pts.columns):
                samp_pts.drop(['LC3'], axis=1, inplace=True)
            if 'LC3_name' in list(samp_pts.columns):
                samp_pts.drop(['LC3_name'], axis=1, inplace=True)
    
            sample_feats = pd.merge(mod_feats,samp,on='PID',how='inner')  ## maybe merge on OID_  ??
            
            sample_feats['year'] = y
            ## more formatting corrections that might not be needed anymore
            if 'LC_UNQ_x' in list(sample_feats.columns):
                if 'LC_UNQ' not in list(sample_feats.columns):
                    sample_feats.rename(columns={'LC_UNQ_x': 'LC_UNQ'},inplace=True)
                else:
                    sample_feats.drop(['LC_UNQ_x'], axis=1, inplace=True)
            if ('USE_NAME' in list(sample_feats.columns)) and ('Class' not in list(sample_feats.columns)):
                sample_feats.rename(columns={'USE_NAME':'Class'},inplace=True)    
            if (params['sample_model']['optimize_on'] == 'smCrop') & ('smlhld_1ha' not in list(sample_feats.columns)):
                sample_feats = apply_smalls(sample_feats,lut,project_v='Py0')
            if 'OID__x' in list(sample_feats.columns):
                sample_feats.rename(columns={'OID__x': 'OID_'},inplace=True)
                sample_feats.drop(['OID__y'], axis=1, inplace=True)
            
            if class_mod_name not in list(sample_feats.columns):  
                sample_feats = sample_feats.merge(lut[['LC_UNQ',class_mod_name,f'{class_mod_name}_name']],on='LC_UNQ',how='left')

            logger.info(f"sample breakdown by {class_mod_name} class: ")
            logger.info(f" \n ...{ sample_feats[f'{class_mod_name}_name'].value_counts()}")
            logger.debug(f'sample_feats: {sample_feats.head()}')

            singlyr_model_name = f"{feat_mod}_{sampmod.replace('YYYY',yrst)}_{class_mod_name}"
            if not params['sample_model']['subsample']:
                vardf_out_path = Path(vardf_dir)/f"pixdf_{singlyr_model_name}.csv"
            else:
                vardf_out_path = Path(vardf_dir)/f"pixdf_{singlyr_model_name}_ss{params['sample_model']['subsample']}.csv"
    
        allyrs_vardfs.append(sample_feats)
        pd.DataFrame.to_csv(sample_feats, vardf_out_path, index=False)

        allyr_vardf = pd.concat(allyrs_vardfs, axis=0, ingnore_index=True)
        if len(allyrs_vardfs > 1):     
            ## print final multiyr var_df
            pd.DataFrame.to_csv(allyr_vardf, multiyr_vardf_path, index=False)
    
        return allyr_vardf


def apply_smalls(pixdf,lut,outpath=None, project_v='Py0'):
    '''
    Adds smallholder flag columns ('smlhld_1ha','smlhd_halfha') to pixdf based on polygon area (if exists) and crop classification 
    If no polygon area feature, applies if classified as mixed crop)
    '''
    if project_v == 'Py0':
        lccol2 = 'LC2'
        lccolmulti = 'LC32'
        lowvegmax = 40
        croppos = 30
        cropmix = 35
    else: 
        lccol2 = 'LCcrop2' 
        lccolmulti = 'LC25'
        lowvegmax = 147
        croppos = 100
        cropmix = 137
        
    if (lccolmulti not in list(pixdf.columns)) or (lccol2 not in list(pixdf.columns)):
        pixdf = pixdf.merge(lut[['LC_UNQ',lccolmulti,lccol2]],on='LC_UNQ',how='left')
    logger.info(f'df columns: {pixdf.columns.values.tolist()}')
    ### <=1 hectare
    if 'var_poly_area' in pixdf.columns.values.tolist():
        pixdf['smlhld_1ha'] = pixdf.apply(lambda x: 1 if ((
            (x['var_poly_area'] < 100) and (x[lccol2] == 30)) and (x[lccolmulti] < lowvegmax)) or (
            (x['Width'] <= 100) and (x[lccol2 == croppos])) or (x[lccolmulti] == cropmix) else 0, axis=1)
    else:
        pixdf['smlhld_1ha'] = pixdf.apply(lambda x: 1 if ((
            (x['Width'] <= 100) and (x[lccol2] == croppos)) or (x[lccolmulti] == cropmix)) else 0, axis=1)
    num_smlhld_1ha = pixdf['smlhld_1ha'].sum()
    logger.info(f'{num_smlhld_1ha} of the sample points are small fields < 100 m across')
    ### <= .5 hectare
    if 'var_poly_area' in pixdf.columns.values.tolist():
        pixdf['smlhd_halfha'] = pixdf.apply(lambda x: 1 if (
        ((x['var_poly_area'] < 50) and (x[lccol2] == croppos) and (x[lccolmulti] < lowvegmax)) or (
        (x['Width'] <= 50) and (x[lccol2] == croppos)) or (x[lccolmulti] == 35)) else 0, axis=1)
    else:
        pixdf['smlhd_halfha'] = pixdf.apply(lambda x: 1 if ((
        (x['Width'] <= 50) and (x[lccol2] == croppos)) or (x[lccolmulti] == cropmix)) else 0, axis=1)
    num_smlhld_halfha = pixdf['smlhd_halfha'].sum()
    logger.info(f'{num_smlhld_halfha} of the sample points are very small fields < 50 m across')
    if outpath:
        pd.DataFrame.to_csv(pixdf, outpath)
    
    return pixdf


def make_multiyr_pixdf(params,yrs,feature_model=None,sample_model=None,class_mod=None):
    '''
    TODO: fix or remove -- this is already implemented within format_ptfeat_set()
    '''
    df_list = []
    if feature_model:
        params['feature_model']['name'] = feature_model
    else:
        feature_model = params['feature_model']['name']
        
    if sample_model:
        params['sample_model']['name'] = sample_model
    else:
        sample_model = params['sample_model']['name']

    if class_mod:
        params['schematic_model']['lc_mod'] = class_mod
    else:
        class_mod = params['schematic_model']['lc_mod']

    ptsamp_dir = params['sample_model']['point_samp_dir']
    if not ptsamp_dir:
        ppaths=ProjectPaths(params)
        ptsamp_dir = ppaths.trainptsets
    fset_dir = params['sample_model']['ptsfeat_dir']
    if not fset_dir:
        ppaths=ProjectPaths(params)
        fset_dir = ppaths.trainptsets
    vardf_dir = params['sample_model']['vardf_dir']
    if not vardf_dir:
        ppaths=ProjectPaths(params)
        vardf_dir = ppaths.fulltrainsets

    if isinstance(yrs, int):
        yrlist = [yrs]
    elif (len(yrs) == 2) and (yrs[0]<yrs[1]):
        yrlist = list(range(yrs))
    else:
        yrlist = yrs

    for y in yrlist:
        yrst = get_train_yrs_str(y)
        logger.info(f'finding pixdf for year {y}')
        model_name = f"{feature_model}_{sample_model}_{y}_{class_mod}"
        pixdf_path = Path(vardf_dir) / f"pixdf_{model_name}.csv"
        if not pixdf_path.is_file():
            (params)
        pixdf = pd.read_csv(pixdf_path)
        pixdf['year']=y
        vardf = pixdf.filter(regex='var_')
        nancols = vardf.columns[vardf.isna().any()].tolist()
        if len(nancols) > 0:
            logger.warning('ERROR -- NaNs in:', nancols)
        df_list.append(pixdf)
    
    allpix = pd.concat(df_list)
    model_name_all = model_name.replace(str(yrs[1])[-2:],str(yrs[0])[-2:]+str(yrs[-1])[-2:])
    allpix_path = Path(vardf_dir) /f"pixdf_{model_name_all}.csv"
    pd.DataFrame.to_csv(allpix, allpix_path)
    logger.info(f'new vardf has {allpix.shape[0]} samples')

def make_multiyr_ho(ho_dir,feature_model,years):
    full_fixed_ho = Path(ho_dir) / f"{feature_model}_HOLDOUT_all_{str(years[0])[-2:]}{str(years[-1])[-2:]}.csv"
    if not full_fixed_ho.is_file():
        hos=[]
        for y in range(years[0],years[1]+1):
            ho = pd.read_csv(Path(ho_dir) / f'{feature_model}_HOLDOUT_all_{str(y)[-2:]}.csv')
            ho['year']=y
            hos.append(ho)
        allhos = pd.concat(hos)
        pd.DataFrame.to_csv(allhos, full_fixed_ho)

def make_variable_stack(params):
    '''
    Creates stack of all features variables for each cell in cell list.
    This creates a lot of extra temp files, which will be sent to scratch_dir if set.

    can clip large ancillary maps ('ancillary' features) to grid cell to include in stack
    as well as calculate new data
    '''
    
    # get model paramaters if model already exists in dict. Else create new dict entry for this model
    getset_feature_model(params)
  
    cells = []
    if isinstance(params['grids'], list):
        cells = params['grids']
    elif str(params['grids']).endswith('.csv'): 
        with open(params['grids'], newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])
    elif isinstance(params['grids'], int) or isinstance(params['grids'], str): # if runing individual cells as array via bash script
        cells.append(params['grids']) 
    else:
        logger.warning(f"ERR: Problem parsing input as cell list. Needs to be list, .csv, or single int or string")
        
    for cell in cells:
        ppaths = ProjectPaths(params, grid=int(cell))
        logger.info(f"working on cell: {cell}.... \n")
        
        # set the path for the temporary output files prior to final stacking
        if params['scratch_dir']:
            temp_dir = Path(params['scratch_dir']) / f'{cell:06d}/comp'
        else:
            temp_dir = ppaths.ms.parent.joinpath('comp')
        if params['feature_model']['treat_out'] == 'archive':
            out_dir = ppaths.ms.parent.joinpath('comp')
        elif params['feature_model']['treat_out'] == 'tmp':
            out_dir = temp_dir

        logger.debug(f'making dirs {temp_dir} and {out_dir} \n')
        out_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        mod_name = params['feature_model']['name']
        mod_start_yr = params['feature_model']['start_yr']
        
        stack_exists = False
        stack_path = Path(out_dir) / f'{mod_name}_{mod_start_yr}_stack.tif'
        if stack_path.is_file():
            stack_exists = True
            logger.debug(f"stack file already exists at {stack_path} \n")
        ## Can build noPoly models from existing stacks containing polys
        ##    but can just use subset option now instead
        #elif 'NoPoly' not in mod_name:
        #    no_poly_model = str(mod_name).replace('Poly','NoPoly')
        #    alt_path = Path(out_dir) / f'{no_poly_model}_{mod_start_yr}_stack.tif'
        elif params['feature_model']['subset_features']==True:
            alt_path = Path(out_dir) / f"{params['feature_model']['full_feature_mod']}_{mod_start_yr}_stack.tif"
            logger.info(f'looking for alternative stack at {alt_path}')
            if alt_path.is_file():
                stack_exists = True
                logger.debug(f"larger stack file already exists that contains the features we need at {alt_path} \n")
        keep_running = True
        if stack_exists:
            logger.info('stack file already exists. \n')
            if not params['feature_model']['overwrite']:
                keep_running = False
        if keep_running:
            stack_paths = []
            band_names = []
            logger.info(f"making variable stack for cell {cell} \n")

            logger.info(f'prechecking ts directories...')
            for si in params['feature_model']['spec_indices']:
                ## check if all spec indices exist before going on:
                img_dir = ppaths.ts / si
                if not img_dir.is_dir():
                    logger.warning(f"ERROR: missing spec index: {si} at {img_dir}\n")
                    return False
                else:
                    logger.info(f"{si}...exists")

            all_indices = params['feature_model']['spec_indices']
            for si in all_indices:
                logger.info(f"working with {si}...")
                params['feature_model']['spec_indices'] = [si]
                img_dir = ppaths.ts/si
                params['feature_model']['use_pheno'] = False
                new_vars, new_bands = make_ts_composite(params)
                try:
                    with rio.open(new_vars) as src:
                        num_bands = src.count
                except:    
                    logger.warning(f"ERROR: there is a problem with the time series for {si} \n")
                    return False

                if num_bands < len(params['feature_model']['si_vars']):
                    logger.warning(f"ERROR: not all variables could be calculated for {si} \n")
                    return False
                    
                else:
                    stack_paths.append(new_vars)
                    for b in new_bands:
                        new_band_name = f'{si}_{b}'
                        band_names.append(new_band_name)
                    logger.info(f'Added {si} with {num_bands} bands \n')
                       
            if len(stack_paths) < len(params['feature_model']['spec_indices']):
                logger.warning('ERROR: did not find ts data for all the requested spec_indices \n')
            else:
                if params['feature_model']:
                    logger.info('getting pheno variables... \n')
                    for psi in params['feature_model']['spec_indices_pheno']:
                        #try:
                        img_dir = ppaths.ts/psi 
                        params['feature_model']['use_pheno'] = True
                        new_pheno_vars, pheno_bands = make_ts_composite(params)
                        stack_paths.append(new_pheno_vars)
                        for pb in pheno_bands:
                            new_band_name = f'{psi}_{pb}'
                            band_names.append(new_band_name)
                        #except Exception as e:
                        #    logger.warning(f'ERROR: {e} \n')
                if params['feature_model']['ancillary_vars']:
                    ## Clips portion of ancillary raster corresponding to gridcell 
                    ## and saves with stack files (if doesn't already exist there)

                    ancillary_feat_dict = params['feature_model']['ancillary_var_dict']
                    if not ancillary_feat_dict:
                        ancillary_feat_dict = str(ppaths.singfeatdict)
                    for sf in params['feature_model']['ancillary_vars']:
                        if sf != '':
                            with open(ancillary_feat_dict, 'r+') as sfd:
                                dic = json.load(sfd)
                                if sf in dic: 
                                    sf_path = dic[sf]['path']
                                    sf_col = dic[sf]['col']
                                    logger.info(f'getting {sf} from {sf_path} \n')    
                                else:
                                    logger.warning(f'ERROR: do not know path for {sf}. Add to ancillary_var_dict and rerun \n')
                                    sys.exit()

                            ancillary_clipped = ppaths.bk/'comp'/f'{sf}.tif'
                            if ancillary_clipped.is_file():
                                stack_paths.append(ancillary_clipped)
                                band_names.append(sf)
                            else:
                                comp_dir = ppaths.bk/'comp'
                                comp_dir.mkdir(parents=True, exist_ok=True)
                                ## clip large ancillary raster to extent of other rasters in stack for grid cell
                                small_ras = stack_paths[0]
                                logger.debug(f"stack_paths: {stack_paths}")
                                ## with gdal:
                                #src_small = gdal.Open(small_ras)
                                #ulx, xres, xskew, uly, yskew, yres  = src_small.GetGeoTransform()
                                #lrx = ulx + (src_small.RasterXSize * xres)
                                #lry = uly + (src_small.RasterYSize * yres)
                                #geometry = [[ulx,lry], [ulx,uly], [lrx,uly], [lrx,lry]]
                                ## with geowombat.
                                with gw.open(small_ras) as src:
                                    logger.info(f'src: {src}')
                                minx = src.x.min().item()
                                maxx = src.x.max().item()
                                miny = src.y.min().item()
                                maxy = src.y.max().item()
                                    #bounds = src.bounds
                                #ulx = bounds[0]
                                #urx = bounds[2]
                                #lry = bounds[1]
                                #uly = bounds[3]
                                geometry = [[maxx,miny], [maxx,maxy], [minx,maxy], [minx,miny]]
                                roi = [Polygon(geometry)]
                                with rio.open(small_ras) as src0:
                                    out_meta = src0.meta.copy()
                                    out_meta.update({"count":1})
                                with rio.open(sf_path) as src:
                                    out_image, transformed = mask(src, roi, crop = True)
                                with rio.open(ancillary_clipped, 'w', **out_meta) as dst:
                                    dst.write(out_image)
                                stack_paths.append(ancillary_clipped)
                                band_names.append(f'sing_{sf}')

                poly_vars = params['feature_model']['poly_vars']
                if poly_vars:
                    logger.info('getting poly variables... \n')
                    for pv in poly_vars:
                        if str(pv).startswith('poly'):
                            stored_name = str(pv).replace('poly','pred')
                            poly_path = Path(params['feature_model']['poly_var_path']) / f'{stored_name}_{cell}.tif'
                        else:
                            poly_path = Path(params['feature_model']['poly_var_path'])/ f'{pv}_{cell}.tif'
                        poly_comp_path = Path(params['feature_model']['poly_var_path']) / f'pred_PY_{cell}.tif'
                        if poly_path.is_file():
                            #if pv in ['pred_APR','poly_APR']:
                            #    poly_path = Path(poly_var_path) / f'pred_APRef_{cell}.tif'
                            #    stack_paths.append(poly_path)
                            #    band_names.append(pv)
                            #else:
                            logger.info(f'adding {poly_path} to stack for var {pv} \n')
                            stack_paths.append(poly_path)
                            band_names.append(pv)
                        elif pv in ['NovDecGCVI_Std','poly_NovDecStd','NovDecStd']:
                            alt_poly_path = Path(params['feature_model']['poly_var_path']) / f'AvgNovDec_FieldStd_{cell}.tif'
                            if alt_poly_path.is_file():
                                stack_paths.append(alt_poly_path)
                                band_names.append(pv)
                        elif poly_comp_path.is_file():
                            if pv in ['pred_ext','poly_ext']:
                                with rio.open(poly_comp_path) as src:
                                    vals = src.read([3])
                                    profile = src.profile
                                    profile.update(count = 1)
                                    new_file = Path(temp_dir) / 'pred_ext.tif'
                                    with rio.open(new_file, mode="w",**profile) as new_b:
                                        new_b.write(vals)
                                stack_paths.append(new_file)
                                band_names.append(pv)           
                            elif pv in ['pred_dst','poly_dst']:
                                with rio.open(poly_comp_path) as src:
                                    vals = src.read([1])
                                    profile = src.profile
                                    profile.update(count = 1)
                                    new_file = Path(temp_dir) / 'pred_dst.tif'
                                    with rio.open(new_file, mode="w",**profile) as new_b:
                                        new_b.write(vals)
                                stack_paths.append(new_file)
                                band_names.append(pv)    
                            elif pv in ['pred_cropbnds','poly_cropbnds']:
                                with rio.open(poly_comp_path) as src:
                                    vals = src.read([2])
                                    profile = src.profile
                                    profile.update(count = 1)
                                    new_file = Path(temp_dir) / 'pred_bnds.tif'
                                    with rio.open(new_file, mode="w",**profile) as new_b:
                                        new_b.write(vals)
                                stack_paths.append(new_file)
                                band_names.append(pv)       
                        else:   
                            logger.warning(f'variable {pv} does not exist for cell {cell} \n')
                            ## Write stack without poly variables, but change name to specify
                            if 'Poly' in params['feature_model']['name'] and 'NoPoly' not in params['feature_model']['name']:
                                nop_mod = str(params['feature_model']['name']).replace('Poly', 'NoPoly')
                                stack_path = Path(out_dir) / f'{nop_mod}_{mod_start_yr}_stack.tif'
                                poly_vars = None
                                      
                logger.info(f'Final stack will have {band_names} bands \n')
                logger.info('making variable stack... \n')
                logger.debug(f'All paths are {stack_paths} \n')

                output_count = 0
                indexes = []
                for path in stack_paths:
                    if ('ModVars' in Path(path).name) or ('Phen' in Path(path).name):
                        with rio.open(path, 'r') as src:
                            src_indexes = src.indexes
                            #logger.info(f'got indices {src.indexes} for path {path} \n')
                            indexes.append(src_indexes)
                            output_count += len(src_indexes)
                    else:
                        indexes.append(1)
                        output_count += 1       
                #logger.info(f'final indexes: {indexes} \n')

                with rio.open(stack_paths[0],'r') as src0:
                    profile = src0.profile
                    profile.update(count = output_count) 
                
                dst_idx = 1
                with rio.open(stack_path,'w',**profile) as dst:
                    for path, index in zip(stack_paths, indexes):
                        with rio.open(path) as src:
                            logger.info(f'inserting {path} at index {dst_idx} \n')
                            if isinstance(index, int):
                                data = src.read(index)
                                data_profile = src.profile
                                logger.debug(str(data_profile))
                                dst.write(data, dst_idx)
                                dst_idx += 1
                            elif isinstance(index, Iterable):
                                logger.debug(f'inserting {path} at index {dst_idx} \n')
                                data = src.read(index)
                                dst.write(data, range(dst_idx, dst_idx + len(index)))
                                dst_idx += len(index)

                    logger.debug(f'new stack has {dst_idx - 1} bands \n')
                    logger.debug(f'we have {len(band_names)} band names \n')
                    dst.descriptions = tuple(band_names)
                logger.info(f'done writing {stack_path.name} with {len(band_names)} bands for cell {cell} \n') 


def make_and_score_model(params, df=None, out_dir=None):
    '''
    Current mod_types are 'RF' for random forest and 'GB' for gradient boosting
    if running subsample iterations or sample model tests, the launch point is "iterate_sample_model()".
    if testing different feature models, class models, training years, etc. the launch point is "iterate_all_model_components()".
    '''
    project_v = params['project_ver']
    ## legacy code for original CELPy maps:
    if project_v == 'Py0':
        CROP_CATS = CROP_CATS_Py0
    
    ppaths=ProjectPaths(params)
    if not out_dir():
        mod_dir = params['classify']['mod_dir']
        if not mod_dir():
            mod_dir = ppaths.classification
        out_dir = mod_dir

    score_dict = params['iter_models']['model_score_dict']  ## path/str for score .csv file. optional
    lut = params['schematic_model']['lut']  ## path or str for .csv file
    lc_mod = params['schematic_model']['lc_mod']
    class_col,lut = get_class_col(lc_mod,lut)
    allyrs = params['sample_model']['train_yrs']
    allyrs_str =  get_train_yrs_str(allyrs)
    feat_mod_name = params['feature_model']['name']
    samp_mod_name = params['sample_model']['name']
    focus_geo = params['sample_model']['focus_area']
    focus = params['iter_models']['optimize_on']
    mod_type = params['classify']['mod_type']
    nest = params['classify']['n_est']
    runnum = params['iter_models']['iter']   ## set by "iterate_sample_model" if not None
    subsample = params['sample_model']['subsample']  ## set by "iterate_sample_model" if not None
    fixed_ho = ['sample_model']['fixed_ho'] ## True/False
    balanced_ho = params['sample_model']['use_balanced_ho']
    if fixed_ho:
        thresh = 1
    else:
        thresh = params['sample_model']['test_thresh']  ## int from 0-100

    vardf_dir = params['classify']['vardf_dir']
    if not vardf_dir:
        vardf_dir = ppaths.fulltrainsets
    ho_dir = params['sample_model']['fixed_ho_dir']
    if not ho_dir:
        ho_dir = ppaths.hos

    if not focus_geo or focus_geo.startswith('All'):
        model_base_name = f'{feat_mod_name}_{samp_mod_name}_{allyrs_str}_{class_col}'   

    else:
        model_base_name = f'{feat_mod_name}_{samp_mod_name}_{allyrs_str}-{focus_geo}_{class_col}'  

    if not subsample:
        vardf_name = f"pixdf_{model_base_name}.csv"
        if not runnum:
            full_model_name = f'{model_base_name}_{mod_type}{nest}'
        else:
            full_model_name = f'{model_base_name}_{mod_type}{nest}_run{runnum}'
    else:
        vardf_name = f"pixdf_{model_base_name}_ss{subsample}.csv"
        if not runnum:
            full_model_name = f'{model_base_name}_{mod_type}{nest}_ss{subsample}'
        else:
            full_model_name = f"{model_base_name}_{mod_type}{nest}_ss{subsample}-run{runnum}"

    logger.info(f'model name = {full_model_name}')
    
    if df:
        df_in = df
    else:
        df_in = Path(vardf_dir) / vardf_name

    if not df_in.exists():
        logger.warning('cannot find vardf {df_in} to train the model')
    else:
        if isinstance(df_in, pd.DataFrame):
            df = df_in.set_index('OID_')
            logger.info('reading in df as database')
        else:
            df = pd.read_csv(df_in)
            logger.info(f'reading in df file: {df_in} \n')
 
        logger.debug(f'class_col = {class_col} \n')
        if f'{class_col}_name' in df.columns:
            df2 = df
        else:
            lutdf = pd.read_csv(lut)
            lut_cols = ['USE_NAME',f'{class_col}',f'{class_col}_name']
            filtered_lut = lutdf.filter(lut_cols)
            df2 = df.merge(filtered_lut, left_on='Class',right_on='USE_NAME', how='left')

        ## update model dictionary if necessary
        feature_mod_dict = params['feature_model']['feature_mod_dict']
        if not feature_mod_dict:
            feature_mod_dict = str(ppaths.fmoddict)
        if params['feature_model']['update_model_dict']:
            ## add columns back into feature dict to make sure they are in the right order:
            ordered_vars = [v[4:] for v in df2.columns.to_list() if v.startswith('var')]
            logger.info(f'there are {len(ordered_vars)} variables in the model \n')
            logger.info(f'model bands are: {ordered_vars} \n')
            with open(feature_mod_dict, 'r+') as fmd:
                dic = json.load(fmd)
                dic[params['feature_model']].update({'band_names':ordered_vars})
            with open(feature_mod_dict, 'w') as new_feature_model_dict:
                json.dump(dic, new_feature_model_dict)

    ## use existing model if one exists and overwrite == False (or if specified as model with <classify:existing_mod>)
    exiting_mod_name = full_model_name.replace(f'{mod_type}{nest}',f'{mod_type}{nest}mod')
    if (Path(out_dir) / f"{exiting_mod_name}.joblib".isfile()) and (not params['classify']['overwrite_model']):
        params['classify']['existing_mod'] = Path(out_dir) / f"{exiting_mod_name}.joblib"
    if params['classify']['existing_mod']:
        mod = joblib.load(params['classify']['existing_mod']) #this load is from joblib -- careful if there are other packages with 'Load' module
        logger.info(f"loading existing mod from: {params['classify']['existing_mod']}")
    else: ## make new model
        train, ho = prep_test_train(df2, out_dir, class_col, full_model_name, thresh=thresh, stable=True)
        mod = multiclass_mod(train, full_model_name, out_dir, params, runnum=runnum, subsample=subsample)

    if isinstance(allyrs, int):
        yrlist = [allyrs]
    elif (len(allyrs) == 2) and (allyrs[0]<allyrs[1]):
        yrlist = list(range(allyrs))
    else:
        yrlist = allyrs

    ## Get holdout scores.
    if not fixed_ho:  
        ## scoring the model on the fly, based on the test set (% set with <sample_model:test_thresh>)
        logger.info('scoring model on the fly...\n')
        #score = get_holdout_scores(ho, mod[0], class_col, out_dir) ## this should run below
        yrlist = [yrlist[0]] ## if multiyr, holdout is for all yrs. This is not used except as dict entry
    else:
        logger.info('scoring model based on fixed houldouts... \n')  
        ##    created with "get_stable_holdout", but should already exist because df is made from the training split of this set
        # score by year, to be able to evaluate whther some years are better then others 
                
    for y in yrlist:

        if (fixed_ho is True) or (fixed_ho == 'True'):      
            yrst = get_train_yrs_str(y)

            if not subsample:
                if balanced_ho:
                    ho_name = f"{feat_mod_name}_HOLDOUT_balno_{yrst}.csv"
                else:
                    ho_name = f"{feat_mod_name}_HOLDOUT_all_{yrst}.csv"
            else:
                if balanced_ho:
                    ho_name = f"{feat_mod_name}_HOLDOUT_balno_{yrst}_ssm{subsample}.csv"
                else:
                    ho_name = f"{feat_mod_name}_HOLDOUT_all_{yrst}_ss{subsample}.csv"

            ho_path = Path(ho_dir) / ho_name
            ho = pd.read_csv(ho_path)
            score = {}

            training_path = ppaths.trainfeatsets.join(ho_name.replace('HOLDOUT','TRAINING'))
            training_data = pd.read_csv(training_path)
            score["Sn"] = training_data.shape[0]  ## training sample size
            score["Smax"] = training_data[class_col].value_counts().max()  ## MaxCat sample size
            
        score["I"] = params['image_type']
        score["R"] = params['res']
        score["P"] = params['procseq']
        score["F"] = full_model_name.split('_')[0]
        score["S"] = full_model_name.split('_')[1]
        score["C"] = class_col
        score["A"] = f"{params['classify']['mod_type']}-{params['classify']['n_est']}"
        score["Ytr"] = allyrs_str
        score["G"] = focus_geo
        score["Ytst"] = y
    
        ## if optimizing for smallholder crops, add the smallholder indication variables to the output df
        if params['iter_models']['optimize_on'] == 'smCrops':
            score["smalls_1ha"] = df["smlhld_1ha"]
            score["smalls_halfha"] = df["smlhd_halfha"]

        if focus in LC_FOCUS_DICT.keys():
            acccat = LC_FOCUS_DICT[focus][class_col]
            s_hos={}
            for cat in LC_FOCUS_DICT[focus]['cats']:
                ho_cat = ho.loc[ho['LC_UNQ'].isin(LC_VALS_DICT[cat])]
                if focus == 'smCrops':  ## removing crop_edge from crop class, as this is ambiguous
                    if params['project_ver'] == 'Py0':
                        ho['bigCrop'] = ho.loc[(ho['LC2'] == 30) & (ho['LC_UNQ'].isin(CROP_CATS['bigcrops']))]
                        ho['noCrop'] = ho.loc[(ho['LC2'] == 98) & (ho['LC_UNQ'] != 19)]
                    else:
                        ho['bigCrop'] = ho.loc[(ho['LCcrop2'] == 100) & (ho['LC_UNQ'].isin(CROP_CATS['bigcrops']))]
                        ho['noCrop'] = ho.loc[(ho['LCcrop2'] == 98) & (ho['LC_UNQ'] != 93)]  
                score[f'recall_{cat}'] = get_binary_holdout_score(ho_cat, mod[0],out_dir,lut,project_v)
                s_ho = get_holdout_scores(ho_cat,mod[0], acccat, out_dir, cat)[["pred","label","OID"]]
                s_hos.append(s_ho)
            ho_scores = pd.concat(s_hos)
            if params['log_level'] == 'DEBUG':
                ho_score_path = Path(f'{out_dir}/full_ho_check.csv')
                ho_scores.to_csv(ho_score_path)
                logger.debug(f'ho looks like: \n {ho_scores.head()}.  Sent full ho file to {ho_score_path}')

        cm = get_confusion_matrix(ho['pred'], ho['label'], lut, params['schematic_model']['lc_mod'],
                        acccat, print_cm=False, out_dir=None, model_name=None)
        logger.info(f'confusion matrix : \n {cm}')

        cats = pd.read_csv(lut)[acccat].unique()
        if len(cats) == 2:
            neg_lab = [c for c in cats if c.startswith('no')][0]
            pos_lab = cats.remove(neg_lab)[0]
            score["Kappa_bi"] = cm.at[pos_lab,'Kappa']
            score["F1_bi"] = cm.at[pos_lab,'F1']
            score["F_5_bi"] = cm.at[pos_lab,'F_5']
            score["F_25_bi"] = cm.at[pos_lab,'F_25']        
            score["OA_bi"] = cm.at['All','UA']
        else:
            logger.warning('FINISH for multicat models')

        if score_dict:
            model_name_full = f'{model_base_name}_{mod_type}{nest}'
            log_acc_results(score_dict, model_name_full,score,subsample=subsample,runnum=runnum)

    if params['log_level'] == 'DEBUG':
        pd.DataFrame.to_csv(score, Path(out_dir) / f'{full_model_name}_HO_SCORES', sep=',', na_rep='NaN', index=True)     
    
    logger.info(f' score: \n {score}')
    return mod, score