from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import joblib
from ..handler import logger
from .project import ProjectPaths
from .lookup import SCHEMATIC_MODS
import shutil
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
# from sklearn_crfsuite import metrics
    
def getset_feature_model(params):
    '''
    This writes the specs of a feature model to a feature_mod_dict so that they can be kept track of and easily used to build a replicate model
    '''
    modname = params['feature_model']['name']
    feature_mod_dict = params['feature_model']['feature_mod_dict']
    if not feature_mod_dict:
        ppaths=ProjectPaths(params)
        feature_mod_dict = str(ppaths.fmoddict)
        
    with open(feature_mod_dict, 'r+') as fmd:
        dic = json.load(fmd)
        if modname in dic:
            params['image_type'] = dic[modname]['imod1it']
            params['procseq'] = dic[modname]['imod3pseq']
            params['res'] = dic[modname]['imod2res']
            params['feature_model']['spec_indices'] = dic[modname]['spec_indices'] 
            params['feature_model']['si_vars'] = dic[modname]['si_vars']
            params['feature_model']['spec_indices_pheno'] = dic[modname]['spec_indices_pheno']
            params['feature_model']['pheno_vars'] = dic[modname]['pheno_vars']
            params['feature_model']['ancillary_vars'] = dic[modname]['ancillary_vars']
            params['feature_model']['poly_vars'] = dic[modname]['poly_vars']
            params['feature_model']['combo_bands'] = dic[modname]['combo_bands']
            params['feature_model']['band_names'] = dic[modname]['band_names']
            logger.debug(f"using existing model: {modname} \n")
            logger.debug(f"spec_indices = {params['feature_model']['spec_indices']} \n si_vars = {params['feature_model']['si_vars']} \n")
            logger.debug(f"pheno_vars = {params['feature_model']['poly_vars']} on {params['feature_model']['spec_indices_pheno']} \n")
            logger.debug(f"ancillary_vars={ params['feature_model']['ancillary_vars']} \n poly_vars = {params['feature_model']['poly_vars']} \n ")
        else:
            dic[modname] = {}
            dic[modname]['imod1it'] = params['image_type']
            dic[modname]['imod2res'] = params['res']
            dic[modname]['imod3pseq'] = params['procseq']
            dic[modname]['spec_indices'] = params['feature_model']['spec_indices']
            dic[modname]['si_vars'] = params['feature_model']['si_vars']
            dic[modname]['ancillary_vars'] = params['feature_model']['ancillary_vars']
            dic[modname]['poly_vars'] = params['feature_model']['poly_vars']
            dic[modname]['spec_indices_pheno'] = params['feature_model']['spec_indices_pheno']
            dic[modname]['pheno_vars'] = params['feature_model']['pheno_vars']
            dic[modname]['combo_bands'] = params['feature_model']['combo_bands']
            
            band_names = []
            if params['feature_model']['spec_indices']:
                for si in params['feature_model']['spec_indices']:
                    for sv in params['feature_model']['si_vars']:
                        band_names.append(f'{si}_{sv}')
            if params['feature_model']['combo_bands']:
                for cb in params['feature_model']['combo_bands']:
                    band_names.append(cb)
            if params['feature_model']['spec_indices_pheno']:
                for sip in params['feature_model']['spec_indices_pheno']:
                    for pv in params['feature_model']['pheno_vars']:
                        band_names.append(f'{sip}_{pv}')
            if params['feature_model']['ancillary_vars']:
                for sin in params['feature_model']['ancillary_vars']:
                    band_names.append(f'sing_{sin}')
            if params['feature_model']['poly_vars']:       
                for pv in params['feature_model']['poly_vars']:
                    band_names.append(f'poly_{pv}')

            dic[modname]['band_names'] = band_names
            with open(feature_mod_dict, 'w') as new_feature_model_dict:
                json.dump(dic, new_feature_model_dict)
            logger.debug(f"created new model: {modname} \n spec_indices={params['feature_model']['spec_indices']} \n")
            logger.debug(f"si_vars={params['feature_model']['si_vars']} \n pheno_vars={params['feature_model']['pheno_vars']}")
            logger.debug(f"on {params['feature_model']['spec_indices_pheno']} \n ancillary_vars={params['feature_model']['ancillary_vars']} \n")
            logger.debug(f"poly_vars={params['feature_model']['poly_vars']} \n combo_bands={params['feature_model']['combo_bands']} \n")
        
    return None

def get_train_yrs_str(train_yrs):
    '''
    receives year info as int or list and returns:
          -- two-character string for single year (e.g. 2024 --> 24)
          -- YYtoYY for year range (two year list with oldest year first) (e.g. [2017,2024] --> 17to24)
          -- concatonated two-character strings for multiple years, sorted with newest first (e.g. [2017,2022,2025] --> 252217)
    This is for labelling purposes in output file names
    '''
    if isinstance(train_yrs, int):
        trainyrs = str(train_yrs)[-2:]
    elif len(train_yrs) == 1:
        trainyrs = str(train_yrs[0])[-2:]
    elif (len(train_yrs)==2) and (train_yrs[0]<train_yrs[1]):
        trainyrs = f"{str(train_yrs[0])[-2:]}to{str(train_yrs[-1])[-2:]}"
    else:
        ystrs = [str(y)[-2:] for y in sorted(train_yrs, reverse=True)]
        trainyrs = "".join(ystrs)
    
    return trainyrs

def get_class_col(lc_mod,lut):
    '''
    gets the column name of the column to use in the LUT based on the SCHEMATIC_MOD name (<lc_mod>)
    if <lc_mod> is SINGLE_<X> and <X> is a unique name in the 'USE_NAME' column of the lut, will create a binary model for that class
    This is used to switch between schematic models for accuracy assessments, classification counts, etc.
    '''
    if lc_mod.startswith('LC'):
        class_col = lc_mod  
    elif lc_mod in SCHEMATIC_MODS.keys():
        class_col = SCHEMATIC_MODS[lc_mod]
    elif lc_mod.startswith('single'):
        lc_base =  lc_mod.split('_')[1].lower()
        target_class = lut.index[lut['USE_NAME'].map(lambda s: lc_base in s.lower())].to_list()
        if len(target_class) == 0:
            logger.warning(f'no match found for {lc_base} in lut \n')
        elif len(target_class) > 1:
            logger.warning(f'there are more than one entries with {lc_base} in USE_NAME column of lut \n')
        else:
            class_col = 'LC1'
            if class_col not in lut.columns:
                logger.info(f'making new virtual {lc_base} column in lut \n')
                lut['LC1'] = 0
                lut['LC1_name'] = f'no_{lc_base}'
                lut.at[target_class[0],'LC1_name'] = lc_base
    else:
        logger.warning(f'current options for lc_mod are: {SCHEMATIC_MODS.keys()} and single_X with X as any category. You put {lc_mod} \n')
        
    return class_col,lut
        
def get_holdout_scores(holdoutpix, ml_model, class_col, out_dir,class_type=None, project_v=None):
    '''
    gets predictions for a holdout sample in .csv file <holdout_pix>. Expects columns "OID_" and <class_col> in holdout_pix
       as well as all "var_" columns that match model features
    '''
    ## Save info for extra columns and drop (model is expecting only variable input columns)
    
    if isinstance(holdoutpix, pd.DataFrame):
        holdout_pix = holdoutpix
        holdout_pix.reset_index(drop=True, inplace=True)
    else:
        holdout_pix = pd.read_csv(holdoutpix)
    
    if 'entry_lev' in list(holdout_pix.columns):
        ## filter to remove less confident entries
        #holdout_pix = holdout_pix[(holdout_pix['entry_lev'] == 4) | (holdout_pix['source'].isin(['ground','GE']))]
        holdout_pix = holdout_pix[(holdout_pix['entry_lev'] > 1)]
        if project_v == 'Py0':
            ## mixed fields are removed from from no-crop test set in CELPy, as this is ambiguous. TODO: expand for all
            if class_type=='noCrop':
                holdout_pix = holdout_pix[(holdout_pix['LC'] != 19) & (holdout_pix['smlhld_1ha'] == 0)]
        holdout_pix.reset_index(drop=True, inplace=True)

    ## legacy code to handle the fact that 'LCcrop2' was originally just 'LC2':
    if class_col == 'LCcrop2':
        if 'LCcrop2' in holdout_pix.columns.values.tolist():
            class_col = class_col
        elif project_v == 'Py0':
            class_col = 'LC2'
        else:
            logger.warning('WARNING: cannot find LCcrop2 column (maybe change project_ver param to Py0 if using original CELPy dfs)')
            
    holdout_labels = holdout_pix[class_col]
    h_IDs = holdout_pix['OID_']
    logger.debug(f'number of holdout pixels = {len(holdout_pix)} \n')
    
    ## Get list of variables to include in model:
    vars = [col for col in holdout_pix if col.startswith('var_')]
    holdout_fields = holdout_pix[vars]

    ## Calculate scores
    #holdout_fields_predicted = ml_model.predict_proba(holdout_fields)
    holdout_fields_predicted = ml_model.predict(holdout_fields)
    
    #holdout_fields_predicted.to_csv('~/data/ho_p_check.csv')
    
    ## Add extra columns back in
    holdout_fields = pd.concat([holdout_fields,pd.Series(holdout_fields_predicted),holdout_labels,h_IDs],axis=1)
    new_cols = [-3,-2,-1]
    new_names = ["pred","label","OID"]
    old_names = holdout_fields.columns[new_cols]
    holdout_fields.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    
    if class_type=='noCrop': ## remove crop edges from no-crop test set, as this is ambiguous
        if project_v == 'Py0':
            holdout_fields = holdout_fields[(holdout_fields['label'] != 19)]
        else:
            holdout_fields = holdout_fields[(holdout_fields['label'] != 93)]

    ## Print to file
    if class_type:
        out_file = f'Holdout_predictions_{class_type}.csv'
    else:
        out_file = 'Holdout_predictions.csv'
        
    pd.DataFrame.to_csv(holdout_fields, Path(f'{out_dir}/{out_file}'), sep=',', na_rep='NaN', index=True)
   
    return holdout_fields

def get_binary_holdout_score(ho_path, ml_model, out_dir, lut, class_type, project_v=None):
    '''
    Returns percent correct for binary model.
    Current models are crops: ['crop', 'smallCrop', 'bigCrop', 'noCrop'] or burn: ['burn', 'noburn', 'wet_burn', 'dry_burn', mgmt_burn']
    other models can be added by adding column to LUT to match <class_type> 
          (negative class needs to start with 'no' and value for that class should be 255 or added to neg_classes)
    '''

    neg_classes = [98,255]  #(255 = noevent, 98 = nocrop)

    if class_type.lower().endswith('crop'):
        if project_v == 'Py0':
            lutcol = 'LC2'
            #posval = 30
        else:
            lutcol = 'LCcrop2'
            #posval = 100
    elif class_type.lower().endswith('burn'):
        lutcol = 'LCburn2'
        #posval = 95
    elif class_type.lower().endswith('grass'):
        lutcol = 'LCgrass2'
    else:
        lutcol = class_type
        
    mcho_score = get_holdout_scores(ho_path, ml_model, lutcol, out_dir, class_type)
    ## Need to rejoin with LUT and get L2 class if using any other classification system
    lut2 = pd.read_csv(lut)
    accdf = mcho_score.merge(lut2[['LC_UNQ',lutcol]], left_on='pred', right_on='LC_UNQ',how='left')

    ## convert negative cases to 0: 
    accdf[lutcol] = np.where(accdf[lutcol].isin(neg_classes), 0, accdf[lutcol])
    ## get value for posivive case (should be only remaining value in binary model)
    posvals = [v for v in accdf[lutcol].unique() if v > 0]
    if len(posvals) > 0:
        logger.warning(f'OOPS -- there is more than one possible positive value in the binary model: {posvals}')
    posval = posvals[0]

    if class_type.lower().startswith('no'):
        num_correct = len(accdf) - (accdf[lutcol].sum() / posval)
    else:
        num_correct = accdf[lutcol].sum() / posval
    
    per_correct = (num_correct / len(accdf)).round(3)
    
    return per_correct

def get_confusion_matrix(pred_col, obs_col, class_lut, lc_mod_map, lc_mod_acc, print_cm=False, out_dir=None, model_name=None):
    '''
    returns confusion matrix with optional regrouping of classes based on LUT 
    classification schema and class columns defined in get_class_col
    '''
    if isinstance(class_lut, pd.DataFrame):
        lut = class_lut
    else:
        lut = pd.read_csv(class_lut)
    
    cmdf = pd.DataFrame()
    cmdf['pred'] = pred_col
    cmdf['obs'] = obs_col
    
    if lc_mod_map.startswith('single'):
        cm=pd.crosstab(cmdf['pred'],cmdf['obs'],margins=True)
    else: 
        map_cat = get_class_col(lc_mod_map,lut)[0]
        acc_cat = get_class_col(lc_mod_acc,lut)[0]
        logger.info(f"getting confusion matrix based on {acc_cat}...\n")
        cats = lut[acc_cat].unique()
        logger.info(f" with categories: {cats}")
        cmdf2 = cmdf.merge(lut[['LC_UNQ',f'{acc_cat}_name']], left_on='obs', right_on='LC_UNQ',how='left')
        cmdf2.rename(columns={f'{acc_cat}_name':'obs_reclass'}, inplace=True)
        cmdf2.drop(['LC_UNQ'],axis=1,inplace=True)
        cmdf3 = cmdf2.merge(lut[['LC_UNQ', f'{acc_cat}_name']], left_on='pred', right_on='LC_UNQ',how='left')
        cmdf3.rename(columns={f'{acc_cat}_name':'pred_reclass'}, inplace=True)
        cmdf3.drop(['LC_UNQ'],axis=1,inplace=True)
        cm=pd.crosstab(cmdf3['pred_reclass'],cmdf3['obs_reclass'],margins=True)
    cm['correct'] = cm.apply(lambda x: x[x.name] if x.name in cm.columns else 0, axis=1)
    cm['sumcol'] = cm.apply(lambda x: cm.loc['All', x.name] if x.name in cm.columns else 0)
    cm['UA'] = (cm['correct']/cm['All']).round(3)
    cm['PA'] = (cm['correct']/cm['sumcol']).round(3)
    cm['F1'] = ((2 * cm['UA'] * cm['PA'])/(cm['UA'] + cm['PA'])).round(3)
    cm['F_5'] = ((1.5 * cm['UA'] * cm['PA'])/(.5 * cm['UA'] + cm['PA'])).round(3)
    cm['F_25'] = ((1.25 * cm['UA'] * cm['PA'])/(.25 * cm['UA'] + cm['PA'])).round(3)
    total = cm.at['All','correct']
    cm.at['All','UA'] = ((cm['correct'].sum() - total) / total).round(3)
    cm.at['All','PA'] = ((cm['correct'].sum() - total) / total).round(3)
    if len(cats) == 2:
        neg_cat = [c for c in cats if c.startswith('no')][0]
        pos_cat = cats.remove(neg_cat)[0]
        cm.at['All','F1']=cm.at[pos_cat,'F1']
        TP = cm.at[pos_cat, pos_cat]
        FP = cm.at[pos_cat, neg_cat]
        FN = cm.at[neg_cat, pos_cat]
        TN = cm.at[neg_cat,neg_cat]
        #All = TP + FP + FN + TN
        cm['Kappa'] = (2*(TP*TN - FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))).round(3)
        
    logger.info(f'Confusion Matrix: {cm}')
    if print_cm:
        mod_path = Path(out_dir / f'{model_name}_{lc_mod_acc}.csv')
        pd.DataFrame.to_csv(cm, mod_path, sep=',', index=True)
    
    return cm

def quick_accuracy(X_test, y_test, ml_model, lc_mod, out_dir,model_name,lut,mod_type):
    
    predicted = ml_model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    if mod_type == 'RF':
        logger.info(f'Out-of-bag score estimate: {ml_model["forest"].oob_score_:.3} \n')
    logger.info(f'Mean accuracy score: {accuracy:.3} \n')

    
    cm = get_confusion_matrix(predicted, y_test,lut, lc_mod, lc_mod, out_dir,model_name,lut)                    
    
    return accuracy, cm

def prep_test_train(df_in, out_dir, class_col, mod_name, thresh=20, stable=True):
    '''
    This is used instead of internal sklearn train_test_split or slt method below when a stable holdout set is desired 
       to compare models. (e.g. from make_and_score_mod) 
    '''
    logger.info(f'df_in = {df_in}\n')
    if isinstance(df_in, pd.DataFrame):
        df_in = df_in
    else:
        df_in = pd.read_csv(df_in, index_col=0)
    logger.info(f'there are {df_in.shape[0]} pts in the full data set \n')
   
    ## remove unknown and other entries where class is not specified a given level (i.e. crop_low if crop type is desired)
    if class_col in ['LC2','CropNoCrop']:
        df_in = df_in[(df_in[class_col] <= 100) & (df_in[class_col] > 0)]
    else:
        df_in = df_in[(df_in[class_col] > 0) & (df_in[class_col] < 256) & (df_in[class_col] != 98)]
    logger.info(f'there are {df_in.shape[0]} sample points after removing those without clear class. \n')
    
    ## Separate training and holdout datasets to avoid confusion with numbering
    training_pix_path = Path(out_dir) / f'{mod_name}_TRAINING.csv'
    holdout_pix_path = Path(out_dir) / f'{mod_name}_HOLDOUT.csv'
    if stable and ((thresh is not None) and (int(thresh) == 0)):
        df_train = df_in  
        df_test = None
    elif not stable:
        value_counts = df_in.value_counts(class_col)
        # Get the count of the smallest class
        class_count_min = value_counts.min()
        # Split the sample into train/test
        n = int(class_count_min * float(1 - thresh)/100)
        # Take the sample count if it is < n
        df_train = df_in.groupby(class_col)\
                    .apply(lambda x: x.sample(n if (int(value_counts[int(x[class_col].max())]) > n) and
                                                   (int(value_counts[int(x[class_col].max())]) > class_count_min) else
                                                    int(value_counts[int(x[class_col].max())]), replace=False))

        if df_train.empty or (df_train.shape[0] == 1):
            return df_in, gpd.GeoDataFrame(data=[])

        try:
            df_train.index = df_train.index.droplevel(level=0)
        except:
            logger.warning(df_train.shape)
            return df_in, gpd.GeoDataFrame(data=[])

        # Use the remaining sample for testing
        df_test = df_in[~df_in.index.isin(df_train.index)]

    logger.info(f'saving training datastes at:{training_pix_path}')
    pd.DataFrame.to_csv(df_train, training_pix_path, sep=',', na_rep='NaN', index=False)
    
    if df_test:
        pd.DataFrame.to_csv(df_test, holdout_pix_path, sep=',', na_rep='NaN', index=False)
        logger.info(f'saving test datastes at:{holdout_pix_path}')  
    return(training_pix_path, holdout_pix_path)

def multiclass_mod(trainfeatures, mod_name, out_dir, params, runnum=None, subsample=None):
    '''
    Current mod_types are 'RF' for random forest and 'GB' for gradient boosting
    '''
    
    df_train = pd.read_csv(trainfeatures)
    logger.info(f'There are {df_train.shape[0]} training samples \n')

    class_col = get_class_col(params['schematic_model']['lc_mod'],params['schematic_model']['lut'])[0]
    y = df_train[class_col]
           
    vars_mod = [col for col in df_train if col.startswith('var_')]
    
    logger.info(f'There are {len(vars_mod)} training features \n')
    X = df_train[vars_mod]
    nan_cols = [i for i in X.columns if X[i].isnull().any()]
    if len(nan_cols)>0:
        logger.warning(f'OOPS -- THere are NANs in the following model variables: {nan_cols}')
    
    logger.debug(pd.Series(y).value_counts())
    X = X.values
    y = y.values
    
    if params['sample_model']['fixed_ho']:
        test_size = .01
    else:
        test_size = params['test_thresh'] / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=params['sample_model']['ran_hold_seed'])
    ## TODO: Don't really need Scikit method for this. Modify prep_train_data so that it returns same outputs

    ## wrap sklearn models in Pipeline to ensure they create the same predictions when loaded cold
    ##    was getting very different results without this whenever models were reloaded!
    if params['classify']['mod_type'] == 'RF':
        stable_model = Pipeline([('forest', RandomForestClassifier(n_estimators=params['classify']['n_est'], oob_score=True))])
    elif params['classify']['mod_type'] == 'GB':
        stable_model = Pipeline([('boost', GradientBoostingClassifier(n_estimators=params['classify']['n_est']))])
    stable_model.fit(X_train,y_train)
    
    if not runnum:
        if not subsample:
            modname = f'{mod_name}mod.joblib'
        else:
            modname = f'{mod_name}mod_ss{subsample}.joblib'
    else:
        if not subsample:
            modname = f'{mod_name}mod_run{runnum}.joblib'
        else:
            modname = f'{mod_name}mod_ss{subsample}-run{runnum}.joblib'
        
    joblib.dump(stable_model, Path(f'{out_dir}/{modname}'))

    cm = quick_accuracy (X_test, y_test, stable_model, params['schematic_model']['lc_mod'], out_dir,mod_name,params['schematic_model']['lut'],params['classify']['mod_type'])

    if (params['classify']['mod_type'] == 'RF'):
        if params['classify']['importance_method'] == "Impurity":
            var_importances = pd.Series(stable_model["forest"].feature_importances_, index=vars_mod)
            pd.Series.to_csv(var_importances,Path(f'{out_dir}/VarImportance_{mod_name}.csv'),sep=',',index=True)
        elif params['classify']['importance_method'] == "Permutation":
            result = permutation_importance(stable_model, X_test, y_test, n_repeats=10,random_state=params['classify']['perm_seed'], n_jobs=2)
            var_importances = pd.Series(result.importances_mean, index=vars_mod)
            pd.Series.to_csv(var_importances,Path(f'{out_dir}/VarImportance_{mod_name}.csv'),sep=',', index=True)

    return stable_model, cm

def log_acc_results(scores_dict, model_name, these_scores, subsample=None, runnum=None):
    try:
        with open(scores_dict, 'r+') as full_dict:
            dic = json.load(full_dict)

        if runnum:
            if subsample:
                model_name = f'{model_name}_ss{subsample}-run{runnum}'
            else:
                model_name = f'{model_name}_run{runnum}'
        else:
            if subsample:
                model_name = f'{model_name}_ss{subsample}'
            else:
                model_name = model_name 
        
        dic.update({model_name : these_scores})

    except IOError:
        logger.info('File not found, will create a new one.')
        dic = {model_name : these_scores}

    with open(scores_dict, 'w') as new_dict:
        json.dump(dic, new_dict)

    return scores_dict
    
def save_best_models(keep_models, temp_mod_dir, main_mod_dir=None, params=None):
    '''
    moves keep models into main model dir
    use this to copy model from scratch dir to final storage dir for replicaiton
    '''
    if main_mod_dir:
        out_dir = main_mod_dir
    else: 
        main_mod_dir = params['classify']['mod_dir']
        if not main_mod_dir:
            ppaths=ProjectPaths(params)
            main_mod_dir = ppaths.classification
            
    keepers = list(keep_models.index.values)
    logger.debug(f'keepers = {keepers}')

    for k in keepers:
        logger.info(f'copying model for {k} to permamant directory')
        if len(k.split('_'))==5:
            outname = k
        elif len(k.split('_'))==6:  ## the last part of the string should be the runnum
            ## remove the runnum from the model name
            outname = k.rsplit('_',1)[0]
        else:
            logger.warning(f"ERROR -- this string has {len(k.split('_'))} parts -- it should have 5 or 6 parts")
            
        current_file = Path(temp_mod_dir) / f'{k}.joblib'
        final_file = Path(main_mod_dir) / f'{outname}.joblib'
        shutil.copy(current_file, final_file)
