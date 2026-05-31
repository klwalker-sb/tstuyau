
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import shutil
from .project import ProjectPaths
from .mod_utils import get_train_yrs_str, get_class_col, getset_feature_model
from .check_model_prep import format_ptfeat_set, get_stable_holdout, make_and_score_model
from .check_classification import classify_timestep
from ..handler import logger
from .lookup import LC_FOCUS_DICT

def aggregate_run_scores(score_dict, agg_score_tab):
    with open(score_dict, 'r+') as full_dict:
        dic = json.load(full_dict)
    new_scores = pd.DataFrame.from_dict(dic).T
    
    if 'recall_smallCrop' in new_scores:
        agg_scores = new_scores.groupby(['I','R','P','F','S','C','A','Ytr','G','Ytst']).agg(avgF1=('F1_bi', 'mean'),stdF1=('F1_bi', 'std'),avgOA=('OA_bi','mean'),stdOA=('OA_bi','std'),
                                                        recallsc=('recall_smallCrop','mean'),stdsc=('recall_smallCrop','std'),
                                                        recallbc=('recall_bigCrop','mean'),stdbc=('recall_bigCrop','std'),
                                                        recallnc=('recall_noCrop','mean'),stdnc=('recall_noCrop','std'))
    elif 'recall_crop' in new_scores:
        agg_scores = new_scores.groupby(['I','R','P','F','S','C','A','Ytr','G','Ytst']).agg(avgF1=('F1_bi', 'mean'),stdF1=('F1_bi', 'std'),avgOA=('OA_bi','mean'),stdOA=('OA_bi','std'),
                                                        recallc=('recall_crop','mean'),stdc=('recall_crop','std'),
                                                        recallnc=('recall_noCrop','mean'),stdnc=('recall_noCrop','std'))
    elif 'recall_mgmtBurn' in new_scores:
        agg_scores = new_scores.groupby(['I','R','P','F','S','C','A','Ytr','G','Ytst']).agg(avgF1=('F1_bi', 'mean'),stdF1=('F1_bi', 'std'),avgOA=('OA_bi','mean'),stdOA=('OA_bi','std'),
                                                        recallmb=('recall_mgmtBurn','mean'),stdmb=('recall_mgmtBurn','std'),
                                                        recallb=('recall_burn','mean'),stdb=('recall_burn','std'),
                                                        recallnb=('recall_noBurn','mean'),stdnb=('recall_noBurn','std'))
    elif 'recall_wetBurn' in new_scores:
        agg_scores = new_scores.groupby(['I','R','P','F','S','C','A','Ytr','G','Ytst']).agg(avgF1=('F1_bi', 'mean'),stdF1=('F1_bi', 'std'),avgOA=('OA_bi','mean'),stdOA=('OA_bi','std'),
                                                        recallwb=('recall_wetBurn','mean'),stdwb=('recall_wetBurn','std'),
                                                        recallb=('recall_burn','mean'),stdb=('recall_burn','std'),
                                                        recallnb=('recall_noBurn','mean'),stdnb=('recall_noBurn','std'))
    elif 'recall_highBurn' in new_scores:
        agg_scores = new_scores.groupby(['I','R','P','F','S','C','A','Ytr','G','Ytst']).agg(avgF1=('F1_bi', 'mean'),stdF1=('F1_bi', 'std'),avgOA=('OA_bi','mean'),stdOA=('OA_bi','std'),
                                                        recallhb=('recall_wetBurn','mean'),stdhb=('recall_wetBurn','std'),
                                                        recallb=('recall_burn','mean'),stdb=('recall_burn','std'),
                                                        recallnb=('recall_noBurn','mean'),stdnb=('recall_noBurn','std'))
    elif 'recall_burn' in new_scores:
        agg_scores = new_scores.groupby(['I','R','P','F','S','C','A','Ytr','G','Ytst']).agg(avgF1=('F1_bi', 'mean'),stdF1=('F1_bi', 'std'),avgOA=('OA_bi','mean'),stdOA=('OA_bi','std'),
                                                        recallb=('recall_burn','mean'),stdb=('recall_burn','std'),
                                                        recallnb=('recall_noBurn','mean'),stdnb=('recall_noBurn','std'))
    elif 'recall_clearGrass' in new_scores:
        agg_scores = new_scores.groupby(['I','R','P','F','S','C','A','Ytr','G','Ytrt']).agg(avgF1=('F1_bi', 'mean'),stdF1=('F1_bi', 'std'),avgOA=('OA_bi','mean'),stdOA=('OA_bi','std'),
                                                        recallcg=('recall_clearGrass','mean'),stdcg=('recall_clearGrass','std'),
                                                        recallg=('recall_allGrass','mean'),stdg=('recall_allGrass','std'),
                                                        recallng=('recall_noGrass','mean'),stdng=('recall_noGrass','std'))
    elif 'recall_allGrass' in new_scores:
        agg_scores = new_scores.groupby(['I','R','P','F','S','C','A','Ytr','G','Ytst']).agg(avgF1=('F1_bi', 'mean'),stdF1=('F1_bi', 'std'),avgOA=('OA_bi','mean'),stdOA=('OA_bi','std'),
                                                        recallg=('recall_allGrass','mean'),stdg=('recall_allGrass','std'),
                                                        recallng=('recall_noGrass','mean'),stdng=('recall_noGrass','std'))
    agg_scores.to_csv(agg_score_tab)

    return agg_scores


def get_best_models(score_dict, final_models_tab, final_dict):
    with open(score_dict, 'r+') as full_dict:
        dic = json.load(full_dict)
    new_scores = pd.DataFrame.from_dict(dic).T
    new_scores['F1_bi'] = new_scores['F1_bi'].astype('float64')
    best_models = new_scores.groupby(['I','R','P','F','S','C','A','Ytr','G','Ytst'])['F1_bi'].idxmax()
    keep_models = new_scores.loc[best_models]
    keep_models.to_csv(final_models_tab)
    keep_dict = keep_models.to_dict(orient='index')
    ## remove iteration number from key name for final dictionary
    key_parts = [k.split('_') for k in keep_dict.keys()]
    new_keys = ['_'.join(kps[:-1]) for kps in key_parts]
    keep_clean = dict(zip(new_keys, list(keep_dict.values())))
    if final_dict.is_file():
        with open(final_dict, 'r+') as in_dict:
            fin_dict = json.load(in_dict)
        for k,v in keep_clean.items():
            if k not in fin_dict:
                fin_dict[k] = v
    else:
        fin_dict = keep_clean
    with open(final_dict, 'w+') as out_dict:
        json.dump(fin_dict, out_dict)
        
    #logger.info(f'final models: {fin_dict}')
    return keep_models


def optimize_feature_model(params):
    '''
    systematically removes sets of variables from dataframe to assess importance and 
    cull feature load prior to fine-tune optimization
    '''
    feature_mod_dict = params['feature_model']['feature_mod_dict']
    if not feature_mod_dict:
        ppaths=ProjectPaths(params)
        feature_mod_dict = str(ppaths.fmoddict)
    vardf_dir = params['classify']['vardf_dir']
    if not vardf_dir:
        ppaths=ProjectPaths(params)
        vardf_dir = ppaths.fulltrainsets
    
    feat_mod = params['feature_model']['name']
    samp_mod = params['sample_model']['name']
    yr_str = get_train_yrs_str(params['sample_model']['train_yrs'])
    class_col = get_class_col(params['schematic_model']['lc_mod'], params['schematic_model']['lut'])[0]
    dropf_indices = params['iter_models']['dropf_indices']
    dropf_vars = params['iter_models']['dropf_vars']
    dropf_combo = params['iter_models']['dropf_combo']
    dropf_list = params['iter_models']['dropf_list']
    dropf_method = params['iter_models']['dropf_method']
    new_fmodname = params['iter_models']['new_fmodname']
    original_dict = params['iter_models']['model_score_dict']
    original_importance = params['classify']['importance_method']
    params['iter_models']['model_score_dict'] = params['iter_models']['fmodel_score_dict']
    
    mod_name = f'{feat_mod}_{samp_mod}_{class_col}_{yr_str}'
    fulldf = Path(vardf_dir) / f'pixdf_{mod_name}.csv'
    
    if not fulldf.is_file():
        format_ptfeat_set(params)

    ptsdf = pd.read_csv(fulldf, index_col=0)
    cols = list(ptsdf.columns)

    if dropf_method == 'top':
        params['classify']['importance_method'] = 'Impurity'
        make_and_score_model(params, df=ptsdf, out_dir=None)
        numkeep = params['iter_models']['dropf_keepnum']
        params['iter_models']['new_fmodname'] = f'{feat_mod}_keep{numkeep}'
        ## TODO: read in variable importance file, sort, and drop features below top <numkeep> 
        ## drop_cols = 
    elif dropf_method == 'thresh':
        params['classify']['importance_method'] = 'Permutation'
        make_and_score_model(params, df=ptsdf, out_dir=None)
        drop_thresh = params['iter_models']['dropf_thresh']
        params['iter_models']['new_fmodname'] = f'{feat_mod}_keep{numkeep}'
        ## TODO: read in variable importance file and drop all features below <drop_thresh> (usually 0)
        ## drop_cols = 
    elif dropf_method == 'top_w_thresh':
        numkeep = params['iter_models']['dropf_keepnum']
        drop_thresh = params['iter_models']['dropf_thresh']
        ## drop_cols = 
    elif dropf_list:
        logger.info(f'dropping features from list {dropf_list}')
        ## drop_cols = 
    else:
        drop_cols = set()
        if dropf_indices: 
            for i in dropf_indices:
                drop = [c for c in cols if c.split('_')[1] == i]
                drop_cols.update(drop)
        if dropf_vars: 
            for v in dropf_vars:
                drop = [c for c in cols if c.split('_')[2] == v]
                drop_cols.update(drop)
        if dropf_combo: 
            for cb in dropf_combo:
                drop = [c for c in cols if c==f'var_{cb}']
                drop_cols.update(drop)
    ptsdf.drop(list(drop_cols), axis=1, inplace=True)
    logger.info(f'dropping {drop_cols} from model \n')
    final_cols = [c for c in cols if c not in list(drop_cols)]
    logger.info(f'new model has bands: {final_cols} \n')
    new_model = getset_feature_model(feature_mod_dict,new_fmodname,spec_indices=None,si_vars=None,
                                     spec_indices_pheno=None,pheno_vars=None, ancillary_vars=None,poly_vars=None, combo_bands=final_cols)
    newname_full = f'{new_fmodname}_{samp_mod}_{class_col}_{yr_str}'
    df_out = pd.DataFrame.to_csv(ptsdf,Path(vardf_dir)/f"pixdf_{newname_full}.csv", sep=',', index=True)
    logger.debug(f'created new model {new_fmodname} in {vardf_dir}')
     
    params['feature_model']['name'] = params['iter_models']['new_fmodname']
    make_and_score_model(params, df=ptsdf, out_dir=None)
  
    scores_csv = ['iter_models']['model_score_dict'].replace('.json','.csv')
    with open(params['iter_models']['model_score_dict'], 'r+') as fmod_score_dict:
        fsdict = json.load(fmod_score_dict)
    pd.DataFrame.to_csv(fsdict, scores_csv, sep=',', na_rep='NaN', index=True)
    logger.info(f'printed feature model scores to {scores_csv}')
    
    ## reset params to original state
    params['iter_models']['model_score_dict'] = original_dict
    params['classify']['importance_method'] = original_importance
    params['feature_model']['name'] = feat_mod 

    return df_out
    
def iterate_sample_model(params):
    '''
    Iterates over multiple versions of the sample model to find the optimal sample set
    Allows for variation in:
        the number of estimators being used to build the model <iter_mods:range_est> (e.g. [100,300]
        the minimum number of samples allowed for any class <iter_mods:range_minsamp> (e.g. [100,300]
        and the balance factor for the minority class of interest <iter_mods:range_minbal> (e.g. [0,10]
    Assumes that the full sample set with appended features exists as <feature_model:name>_YYYY.csv",
        where YYYY (or YY) is a string representing the yera-span of the training sample, derived from <sample_model:train_yrs>
    '''
    ppaths=ProjectPaths(params)
    score_dict = params['iter_models']['model_score_dict']
    lut = params['schematic_model']['lut']
    class_mod = params['schematic_model']['lc_mod']
    feat_mod = params['feature_model']['name']
    focus_geo = params['sample_model']['focus_area'] # string for focus area, if subset of full (naming purposes only)
    #reduce = params['sample_model']['reduce'] # 0-1, the percentage reduction of training dataset (for robustness checks) - 0 = no reduction
    range_reduce = params['iter_mods']['range_reduce']
    inc_reduce = params['iter_mods']['inc_reduce']
    range_minsamp = params['iter_mods']['range_minsamp']  # e.g. [100,300]. minsamp is the minimum number of samples allowed for any class
    inc_minsamp = params['iter_mods']['inc_minsamp']  # step number for minsamp increments (e.g. 100)
    range_minbal = params['iter_mods']['range_minbal'] # e.g. [0,7]. minbal is the representation of the minority class of interest (e.g. mixed)
    inc_minbal = params['iter_mods']['inc_minbal']
    numruns = params['iter_mods']['iterations'] # e.g. 10  # The number of models (e.g. RFs) run with the same parameters and holdout
    num_subsamples = ['sample_model']['num_subsamples']  # The number of subsamples run on the original point sample
    mod_type = params['classify']['mod_type'] # 'RF' | 'GB'
    range_est = params['iter_mods']['range_est']  # e.g. [100,300]. range for nunber of estimators to use.
    inc_est = params['iter_mods']['inc_est'] # step number for estimator increments (e.g. 100)
    
    ftset_dir = params['classify']['ptsfeat_dir'] ## input directory containing FULL pt sets with features appended
    if not ftset_dir:
        ftset_dir = ppaths.self.trainfeatsets
        ftset_dir.mkdir(parents=True, exist_ok=True)

    vardf_dir = params['classify']['vardf_dir'] ## output directory containing pt-feature sets for each model ready for model input
    if not vardf_dir:
        vardf_dir = ppaths.fulltrainsets
        vardf_dir.mkdir(parents=True, exist_ok=True)

    class_mod_name = get_class_col(class_mod,lut)[0]
    logger.info(f' working on model using class column: {class_mod}.\n')
    
    ## make each parameter a two number list to use for range. If int X or single number list [X], reformat as [X,X+1] 
    if isinstance(range_est,int): 
        range_est = [range_est,range_est]
        inc_est = 1
    elif len(range_est) == 1:
        range_est.append(range_est[0])
        inc_est = 1
    if isinstance(range_minsamp,int):
        range_minsamp = [range_minsamp,range_minsamp]
        inc_minsamp = 1
    elif len(range_minsamp) == 1:
        range_minsamp.append(range_minsamp[0])
        inc_minsamp = 1
    if isinstance(range_minbal,int):
        range_minbal = [range_minbal,range_minbal]
        inc_minbal = 1
    elif len(range_minbal) == 1:
        range_minbal.append(range_minbal[0])
        inc_minbal = 1
    if isinstance(range_reduce,int):
        range_reduce = [range_reduce,range_reduce]
        inc_reduce = 1
    elif len(range_reduce) == 1:
        range_reduce.append(range_reduce[0])
        inc_reduce = 1 

    trainyrs = params['sample_model']['train_yrs']
    trainyrstr = get_train_yrs_str(trainyrs)

    for n in range(range_est[0],range_est[1]+inc_est, inc_est):
        for m in range(range_minsamp[0],range_minsamp[1]+inc_minsamp, inc_minsamp):
            for b in range(range_minbal[0],range_minbal[1]+inc_minbal, inc_minbal):
                for r in range(range_reduce[0],range_reduce[1]+inc_reduce, inc_reduce):
                    if r==0:
                        samp_mod = f'min{m}upsamp{b}'
                    else:
                        samp_mod = f'red{int(100*r)}min{m}upsamp{b}'

                    if not focus_geo or focus_geo.startswith('All'):
                        multiyrmod_name = f'{feat_mod}_{samp_mod}_{trainyrstr}_{class_mod_name}' 
                    else:
                        multiyrmod_name = f'{feat_mod}_{samp_mod}_{trainyrstr}-{focus_geo}_{class_mod_name}'   

                    params['sampe_model']['reduce'] = r
                    params['sampe_model']['minsamp'] = m
                    params['sampe_model']['minbal'] = b
                    params['classify']['n_est'] = n
                    params['sample_model']['name'] = samp_mod
                    model_name_full = f'{multiyrmod_name}_{mod_type}{n}'
                    logger.info(f'building {model_name_full}...')

                    if not num_subsamples:
                        sampledraws = 1
                    else:
                        sampledraws = num_subsamples
                        
                    for ss in range(sampledraws):
                        logger.info(f'subsample {ss}...\n')
                        if not num_subsamples:
                            vardf_path = Path(vardf_dir) / f'pixdf_{multiyrmod_name}.csv'
                            #ssn = None
                        else:
                            #ssn = ss
                            params['sample_model']['subsample'] = ss
                            vardf_path = Path(vardf_dir) / f'pixdf_{multiyrmod_name}_ss{ss}.csv'
                            if not vardf_path.is_file():
                                format_ptfeat_set(params)
                                    
                        for rn in range(numruns):
                            logger.info(f'iteration {rn}...\n')
                            params['iter_models']['iter'] = rn
                            mod0 = make_and_score_model(params)
                            ## logging now occurs during make_and_score_model to handle multiyr holdouts
                            #sdict = log_acc_results(score_dict, model_name_full, mod0[1],subsample=ssn,runnum=rn)

                        
    ## TODO: copy winning models and vardfs from tmp to main paths
    ## TODO:  then delete temp directories

    scores_out = ppaths.optimization/'MODEL_SCORES.csv'
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    with open(score_dict, 'r+') as model_score_dict:
        sdict = json.load(model_score_dict)
    pd.DataFrame.to_csv(sdict, scores_out, sep=',', na_rep='NaN', index=True)
    logger.info(f'finished making and scoring the models. Saved scores to: {scores_out}')
    
    return sdict    


def iterate_all_model_components(params):
    '''
    iterates through model options and saves score results to dictionary to select optimum model
    scores each year seperately for model with multiple training years (this is an optimization task; scoring is based on training holdouts) 
    allows for different feature models, class models and sample models...
    '''

    ftset_dir = params['classify']['ptsfeat_dir']
    if not ftset_dir:
        ppaths=ProjectPaths(params)
        ftset_dir = ppaths.trainfeatsets

    opt_dir = params['iter_models']['opt_dir']
    if not opt_dir:
        ppaths=ProjectPaths(params)
        opt_dir = ppaths.optimization

    ## Feature models: 
    ##    feature models should already be defined in feature_model_dict and have a full dataset for all points from make_variable_dataframe
    for fm in params['iter_mods']['feat_models']:
        logger.info(f'Working on feature model {fm}....\n')
        params['feature_model']['name'] = fm
        get_stable_holdout(params, df_in=None, overwrite=False)
        
        ## Class_models:
        for lcmod in params['iter_mods']['class_models']:
            logger.info(f'Working on class model {lcmod}...\n')
            params['schematic_model']['lc_mod'] = lcmod
            
            ## Training_year_depth:
            for yrset in params['iter_models']['trainyr_sets']:
                logger.info(f'Working on training year set {yrset}...\n')
                params['sample_model']['train_yrs'] = yrset

                ## Sample models (adding in mixed)
                sdict = iterate_sample_model(params)

    with open(sdict, 'r+') as full_dict:
        dic = json.load(full_dict)
    scores = pd.DataFrame.from_dict(dic).T
    pd.to_csv(scores, opt_dir/'All_scores')
    agg_score_tab = opt_dir /'Aggregated_scores.csv'
    best_scores_tab = opt_dir /'Best_scores.csv'
    best_scores_dict = opt_dir /'Best_scores.json'
    agg_scores = aggregate_run_scores(sdict, agg_score_tab)
    best_scores = get_best_models(sdict, best_scores_tab, best_scores_dict)

    logger.info(f"DONE! Full scores dictionary is at: {params['iter_models']['model_score_dict']}.") 
    logger.info(f"here is a preview: {scores.head()}")
    
    return scores, agg_scores, best_scores