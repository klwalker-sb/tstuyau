from pathlib import Path

import numpy as np
import pandas as pd

from ..handler import logger
from .constants import FILENAME_DATE_INDEX, FILENAME_DATE_INDEX_GEE
from .project import ProjectPaths


def match_brdf_files_to_record(brdf_dir, processing_db, exclude=None, method='stac'):
    """
    Maps each BRDF file to its processing.info scene id, via 'brdf_id' column -- 
        This column matches the beginning of the actual filename, 
        but the filename can have extra text appended before the suffix (e.g '_coreg').
    Suffix is usually .nc (based on default method 'stac', but can be .tif (e.g. with method ='GEE')
    Option to skip files with text that matches <exclude> (usually 'X') just before the suffix
    """

    file_to_scene = {}
    unmatched = []

    if method == 'stac':
        date_pos=FILENAME_DATE_INDEX
        suffix='.nc'
    else:
        date_pos=FILENAME_DATE_INDEX_GEE
        suffix='.tif'
        
    match_str = f'*{suffix}' if exclude is None else f'*[!{exclude}]{suffix}'
    candidates = list(Path(brdf_dir).glob(match_str))
    logger.info(f'there are {len(candidates)} images to process that match the criteria')
    for f in candidates:
        if 'coreg' in str(f):
            fbase = f.stem.split('_coreg')[0]
        elif f'_{suffix}' in str(f):
            fbase = f.name.split(f'_{suffix}')[0]
        else:
            fbase = f.stem
        rec_match = processing_db.index[processing_db['brdf_id'].str.startswith(fbase, na = False)].tolist()
        
        if len(rec_match)==0:
            unmatched.append(f.name)
            continue
        file_to_scene[f] = rec_match[0]

    if unmatched:
        preview = unmatched[:5]
        logger.warning(
            f"{len(unmatched)} brdf files had no matching brdf_id: "
            f"{preview}{'...' if len(unmatched) > 5 else ''}"
        )

    return file_to_scene
    

def reconstruct_db(gridcell, params):
    '''
    This checks for an existing processing.info database and creates one if needed from download and brdf folders.
    This is only for cases of corruption or accidental deletion. -- 
       processing.info is normally created as files are downloaded -- 
    Note: It is best to use original database whenever possible, as this will not recreate error notes,
       nor populate the numpix or coreg shift_x and shift_y columns that are in the original db
    '''
    ppaths = ProjectPaths(params, grid=gridcell)
    brdf_path = ppaths.ms
    landsat_path = ppaths.ms.parent / 'landsat'
    sentinel2_path = ppaths.ms.parent / 'sentinel2'
    processing_info_path = ppaths.ms.parent / 'processing.info'
    
    modified = False
    
    if brdf_path.is_dir():
        brdf_files = [fi.name for fi in brdf_path.glob('*.nc')]
    else:
        brdf_files = []
    if landsat_path.is_dir():
        brdf_files = [fi.name for fi in landsat_path.glob('*.tif')]
    else:
        landsat_files = []
    if sentinel2_path.is_dir():
        sentinel2_files = [fi.name for fi in sentinel2_path.glob('*.tif')]
    else:
        sentinel2_files = []
        
    if len(landsat_files) + len(sentinel2_files) + len(brdf_files) == 0:
        logger.info(f'WARNING: no images have been downloaded for cell {gridcell}')
    else:
        ## Make new processing db if it does not already exist:
        if not processing_info_path.is_file():
            processing_dict = {}
            # First check for for files in the brdf folder (these are at the most processed stage)
            if len(brdf_files) > 0:
                for b in brdf_files:
                    ## get corresponding dl id:
                    if b.split("_")[1].startswith('L'):
                        dlid = '{}_{}_{}_{}_{}_{}'.format(b.split("_")[1],
                                                         'L2SP',
                                                          b.split("_")[2][4:10],
                                                          b.split("_")[3],
                                                          b.split("_")[2][10:12],
                                                          b.split("_")[2][12:14])
                    else:
                        dlid = '{}_{}_{}_{}_{}'.format(b.split("_")[1],
                                                       b.split("_")[2][4:9],
                                                       b.split("_")[3],
                                                       b.split("_")[2][9:10],
                                                       b.split("_")[2][10:13])
                    bp = 'True' if b.split('_')[0] == 'L3B' else ('False' if b.split('_')[0] == 'L3A' else np.nan)
                    processing_dict[dlid] = {'dl':f'{dlid}',
                                           'beforeDB':True,
                                           'redownload':False,
                                           'brdf_id':f"{b.split('_coreg')[0].split('X')[0]}.tif",
                                           'brdf':'True',
                                           'brdf_error':np.nan,
                                           'bandpass':bp}
            ## If no files in brdf folder, reconstruct db from download folders
            else:
                for f in landsat_files:
                    processing_dict[Path(f).stem] = {'dl': f'{landsat_path}/{f}',
                                                     'beforeDB': True, 
                                                     'redownload': False
                                                    }
                for f in sentinel2_files:
                    processing_dict[Path(f).stem] = {'dl': f'{sentinel2_path}/{f}',
                                                     'beforeDB': True, 
                                                     'redownload': False
                                                    }

            new_processing_info = pd.DataFrame.from_dict(processing_dict,orient='index')
            new_processing_info.rename_axis('id', axis=1, inplace=True)
            pd.to_pickle(new_processing_info, processing_info_path)
            logger.info(f'{len(new_processing_info)} images downloaded and added to database.')
            
        # read in existing db (can be the one that was just created or pre-existing):
        processing_db = pd.read_pickle(processing_info_path)
        
        ## to fix issues from older version of db already created for some cells:
        if 'id' not in processing_db:
            processing_db.rename_axis('id', axis=1, inplace=True)
        
        logger.info(f'{len(processing_db)} records in db. {len(landsat_files)} landsat and {len(sentinel2_files)} sentinel images in downloads.')

        if len(processing_db) >= len(landsat_files) + len(sentinel2_files):
            logger.info('all downloaded images have probably been added to db already')
        else:
            logger.info('adding images to db...')
            new_dls = {}
            for f in landsat_files:
                if Path(f).stem in processing_db.values:
                    continue
                else:
                    new_dls[Path(f).stem]={'dl':f'{landsat_path}/{f}','beforeDB':True}
            for s in sentinel2_files:
                if Path(s).stem in processing_db.values:
                    continue
                else:
                    new_dls[Path(s).stem]={'dl':f'{sentinel2_path}/{s}','beforeDB':True}
        
            if len(new_dls)>0:
                new_dl_db = pd.DataFrame.from_dict(new_dls,orient='index')
                new_dl_db.rename_axis('id', axis=1, inplace=True)
                processing_db.append(new_dl_db)
                modified = True
            
        if brdf_path.is_dir():
            if 'brdf' in processing_db:
                logger.info('brdf data already in database')
            
            else: 
                logger.info('adding brdf info to db...')
                processing_db['brdf_id'] = np.nan
                processing_db['brdf_error'] = np.nan
                processing_db['brdf'] = np.nan
                processing_db['bandpass'] = np.nan
                for idx, row in processing_db.iterrows():
                    match=None
                    for fi in Path(brdf_path).iterdir():
                        if fi.endswith('.nc') and (
                            (idx.startswith('S') and (idx.split('_')[1] in fi.split('_')[2]) and (idx.split('_')[2] == fi.split('_')[3])) or (
                                idx.startswith('L') and (idx.split('_')[0] == fi.split('_')[1]) and (
                                idx.split('_')[2] in fi.split('_')[2]) and (idx.split('_')[3] == fi.split('_')[3]))):
                                match = fi
                    processing_db.at[idx,'brdf_id']=match
                    if match is not None:
                        if match.split('_')[0] == 'L3B':
                            processing_db.at[idx,'bandpass']=True
                        elif match.split('_')[0] == 'L3A':
                            processing_db.at[idx,'bandpass']=False
                
                modified = True
            
            num_coreged_files = len([fi.name for fi in brdf_path.glob('*coreg.nc')])
            logger.info(f'{num_coreged_files} images have been coreged')
            if num_coreged_files == 0:
                logger.info('coregistration has not yet occured. Processing database is up to date')
            else:
                if 'shift_x' in processing_db:
                    logger.info('coreg data has already been added to database')
                else:
                    logger.info('adding coreg info to db...')
                    processing_db['coreg'] = np.nan
                    processing_db['shift_x'] = np.nan
                    processing_db['shift_y'] = np.nan
                    processing_db['coreg_error'] = np.nan
                    for idx, row in processing_db.iterrows():
                        match=None
                        for fi in  Path(brdf_path).iterdir():
                            if fi.endswith('.nc'):
                                if idx.startswith('S'):
                                    if (idx.split('_')[1] in fi.split('_')[2]) and (idx.split('_')[2] == fi.split('_')[3]):
                                        match = fi 
                                elif idx.startswith('L') and (
                                    idx.split('_')[0] == fi.split('_')[1]) and (idx.split('_')[2] in fi.split('_')[2]) and (idx.split('_')[3] == fi.split('_')[3]):
                                    match = fi
                        if match is not None:
                            if 'coreg' in match:
                                processing_db.at[idx,'coreg']=True
                            elif match.endswith('X.nc'):
                                processing_db.at[idx,'coreg']=False
                                processing_db.at[idx,'coreg_error']='unknown'
                            else:
                                processing_db.at[idx,'coreg']='NaN'                           
                    modified = True                        
        else:
            logger.info('brdfs have not yet been created. Processing database is up to date')

        if modified == True:
            pd.to_pickle(processing_db, processing_info_path)
            logger.info('saving new database')
        
        return processing_db