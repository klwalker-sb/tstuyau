from pathlib import Path
import yaml
import datetime
import shutil
import ast
import sys
import csv
import geopandas as gpd
import geowombat as gw
import rasterio as rio
import pandas as pd
import matplotlib.pyplot as plt

from ..db import TuyauDataBase
from .project import ProjectPaths
from ..handler import logger

pd.options.mode.chained_assignment = None


def read_db(db_path,db_version):
    pyver = float((sys.version)[:3])
    if pyver < 3.8:
        logger.info(f"python version is {pyver}")
        ### note this may not work depending on version of Pandas. If it doesn't, maybe upgrade Pandas
        import pickle5 as pickle
        with open(db_path, 'rb') as file1:         
            df = pickle.load(file1)
    else:
        if db_version == 'current':
            df = pd.read_pickle(db_path)
            df['sensor']=df.index.map(lambda x:(x[:4].lower()))
            # get date -- in current version, this is in different part of id for Landsat and Sentinel2
            df_lan = df[df['sensor'].str.startswith('l')]
            df_lan['date'] = df_lan.index.map(lambda x: int(x.split('_')[3][:8]))
            df_sen = df[df['sensor'].str.startswith('s')]
            df_sen['date'] = df_sen.index.map(lambda x: int(x.split('_')[2][:8]))
            df = pd.concat([df_lan,df_sen],axis=0)
            df['yr']=df.date.map(lambda x:(int(str(x)[:4])))
        else:
            df = pd.read_pickle(db_path).set_index('date')
            df.index.rename(None, inplace=True)
            df = df.assign(date=df.index)
            df = df.sort_index()
    
    return df

###########################################################################################################################
####  DOWNLOAD CHECK ####
###########################################################################################################################
def find_gaps(ranges, start_date, stop_date):
    '''
    find gaps log files
    '''
    if len(ranges) <= 1:
        return []
    gaps = []
    # check for gap at the beginning of range:
    startd = datetime.datetime.strptime(start_date,'%Y-%m-%d').date()
    if ranges[0][0] > startd:
        gaps.append([startd,ranges[0][0]])
    # Set marker at the end of the first range
    current = ranges[0][1]
    # Iterate through ranges, ignoring the first range
    for pair in ranges:
        # if next start time is before current end time, keep going until we find a gap
        # if next start time is after current end time, found the first gap
        if pair[0] > current:
            # ignore gaps between 31-Dec and 1-Jan:
            if pair[0].day==1 and pair[0].month==1 and current.day==31 and current.month==12:
                pass
            else:
                gaps.append([current,pair[0]])
        # advance "current" if the next end time is past the current end time
        current = max(pair[1], current)
    # check for gap at the end of range:
    logger.debug(f'stop_date={stop_date}')
    stopd = datetime.datetime.strptime(stop_date,'%Y-%m-%d').date()
    if ranges[-1][1] < stopd:
        gaps.append([ranges[-1][1],stopd])
    return gaps

def check_logfile_dl(logfile, cell_dict,stop_date='2025-10-01', start_date='2000-01-01', ignore_dates=None):
    '''
    read in logfiles from run folder and check for errors
    '''
    cell_id=None
    core_requested=None
    runtime=None
    periods = []
    errors=[]
    if ignore_dates:
        ignore = [d for d in ignore_dates.split('--')]
        ignore_dt = [datetime.datetime.strptime(d,'%Y-%m-%d').date() for d in ignore]
    with open(logfile) as f:
        for line in f:
            if 'cell_id' in line:
                cell_id = int(line.split(' ')[2])
            if 'Working on' in line:
                period=[line.split(' ')[3][:10],line.split(' ')[5][:10]]
                periods.append(period)
            if 'TimeoutError' in line or 'pystac_client.exceptions.APIError' in line or 'urllib.error.HTTPError' in line:
                errors.append(period)
            if 'full process took:' in line:
                runtime = int(line.split(' ')[3])
            if 'core used' in line:
                core_requested = int(line.split(' ')[2])
    ## core used was not always in download log. Use 4 if value is not known:
    if not core_requested or core_requested >90:
        core_requested = 4
    ## also, cell id in logfile title is only 3 digits (because it is array id).
    ## Started printing id within file, but need this for old files that didn't have that printed.
    logger.info(f'working on {cell_id} from logfile: {logfile}')
    if cell_id is None:
        logger.info(f'ERROR: Cannot find cell_id in log for {logfile}')
        cell_id3 = int(logfile.split('.')[2])
        if cell_id3 < 101:
            cell_id = cell_id3 + 4000
        else:
            cell_id = cell_id3 + 3000
    if runtime is None:
        runtime = 0
    
    logger.info(f'cores used:{core_requested}')

    if len(periods)==0:
        logger.info('this log file contains no info')
    else:
        dates = [[datetime.date(int(x[:4]),int(x[5:7]),int(x[8:10])) for x in p] for p in periods]
        ranges = sorted(dates)
        date_range = [ranges[0][0], ranges[-1][1]]
        logger.info(f'downloaded from {date_range}')

        if cell_id in cell_dict:
            logger.info('updating cell info...')
            ## note: if dict has been saved as dataframe and reconstructed as dict, entries will be strings
            ## update start and end value
            if isinstance(cell_dict[cell_id]['dllog_start'],str):
                old_start = datetime.datetime.strptime(cell_dict[cell_id]['dllog_start'],'%Y-%m-%d').date()
            else:
                old_start = cell_dict[cell_id]['dllog_start']
            if ranges[0][0] < old_start:
                cell_dict[cell_id]['dllog_start']=ranges[0][0]
            if isinstance(cell_dict[cell_id]['dllog_end'],str):
                old_end = datetime.datetime.strptime(cell_dict[cell_id]['dllog_end'],'%Y-%m-%d').date()
            else:
                old_end = cell_dict[cell_id]['dllog_end']
            if ranges[-1][1] > old_end:
                cell_dict[cell_id]['dllog_end']=ranges[-1][1]
            ## add former errors to current dllog_errors if time period not recorded in new ranges (without error)
            new_ranges = [[x.strftime('%Y-%m-%d') for x in r] for r in ranges]
            if isinstance(cell_dict[cell_id]['dllog_errors'],str):
                old_errors =  ast.literal_eval(cell_dict[cell_id]['dllog_errors'])
            else:
                old_errors = cell_dict[cell_id]['dllog_errors']
            unresolved_errors = [e for e in old_errors if e in errors or e not in new_ranges]
            new_errors = [e for e in errors if datetime.datetime.strptime(e[0],'%Y-%m-%d').date() < old_start 
                          or datetime.datetime.strptime(e[1],'%Y-%m-%d').date() > old_end]
            if len(unresolved_errors) > 0:
                new_errors.extend(unresolved_errors)
            cell_dict[cell_id]['dllog_errors']=new_errors
            ## error in most recent period (specific with ignore param) probably cant be fixed and should be ignored for now:
            errors_to_fix = new_errors
            if ignore_dates:
                if ignore in new_errors:
                    errors_to_fix = new_errors.remove(ignore)
                elif ignore_dt in errors:
                    errors_to_fix = new_errors.remove(ignore_dt)
            cell_dict[cell_id]['dl_fix_now']=errors_to_fix
            ## get new gap sequence:
            orig_range = [old_start,old_end]
            ranges.append(orig_range)
            date_gaps=find_gaps(ranges, start_date, stop_date)
            ## add runtime:
            cell_dict[cell_id]['dltime']=int(cell_dict[cell_id]['dltime'])+runtime
            cell_dict[cell_id]['dlcoremin']=int(cell_dict[cell_id]['dlcoremin'])+runtime*core_requested
        else:
            logger.info('adding cell to processing db...')
            errors_to_fix = errors
            if ignore_dates:
                if ignore in errors:
                    errors_to_fix = errors.remove(ignore)
                if ignore_dt in errors:
                    errors_to_fix = errors.remove(ignore_dt)
            date_gaps = find_gaps(ranges, start_date, stop_date)
            new_dict_entry={cell_id:{'dllog_start':ranges[0][0],
                                     'dllog_end':ranges[-1][1],
                                     'dllog_gaps':date_gaps,
                                     'dllog_errors':errors,
                                     'dltime':runtime,
                                     'dlcoremin':runtime*core_requested,
                                     'dl_fix_now':errors_to_fix}}
            cell_dict.update(new_dict_entry)

        if len(date_gaps) == 0:
            logger.info('found no gaps')
        else:
            logger.info(f'found gaps: {date_gaps}')

    return cell_id, cell_dict

def archive_logfile(logfile,cell_dict,archive_path):
    if isinstance(archive_path, str):
        archive_path = Path(archive_path)
    archive_path.mkdir(parents=True, exist_ok=True)
    cell_id = None
    with open(logfile) as f:
        for line in f:
            if 'cell_id' in line:
                cell_id = int(line.split(' ')[2])
                break
    if cell_id is None:
        cell_id3 = int(logfile.split('.')[2])
        if cell_id3 < 101:
            cell_id = cell_id3 + 4000
        else:
            cell_id = cell_id3 + 3000
    if isinstance(cell_dict[cell_id]['dl_fix_now'],str):
        if cell_dict[cell_id]['dl_fix_now'] == '[]':
            shutil.move(str(logfile), archive_path)
    elif cell_dict[cell_id]['dl_fix_now'] is None:
        shutil.move(str(logfile), archive_path)
    elif len(cell_dict[cell_id]['dl_fix_now']) == 0:
        shutil.move(str(logfile), archive_path)

def check_dl_logs(params):
    '''
    checks log files for errors (.err files produced from download process).
    (download process must be run with specific SLURM script that prints info)
    logs errors and gaps in time sequence to master database
    updates errors and gaps with subsequent logs for same cell
    archives all log files once there are no errors in 'dl_to_fix'
    deletes all .out files (they contain no information)
    
    note: there still may be gaps and unfixable errors that can be seen in database
    
    .out file from this script contains info on this current batch of cells for
    quick reference.
    '''

    ## get the existing cell database as a dictionary
    dldb_path = Path(params['status']['download_db_path'])
    if not dldb_path:
        ppaths=ProjectPaths(params)
        dldb_path_path = ppaths.dldb

    if Path(dldb_path).is_file():
        #TODO: make a temp copy so the original does not get corrupted
        cell_dict = pd.read_csv(str(dldb_path),index_col=[0]).to_dict(orient='index')
        logger.info('using existing download database')
    else:
        cell_dict = {}
        logger.info('creating new download database')
    
    ## update records based on curent logfiles
    logpath = Path(params['status']['log_path'])
    logger.info(f"    Looking for log files in {logpath}")
    logger.info(f"      processing images between: {params['status']['period'][0]}, {params['status']['period'][1]}")
    cell_batch = set([])
    for logfile in logpath.glob(f"{params['status']['log_prefix']}*.err"):
        processed = check_logfile_dl(logfile, cell_dict, params['status']['period'][0], params['status']['period'][0], params['status']['ignore_dates'])
        cell_batch.add(processed[0])
    ## These output files contain no information; can just remove
    for outfile in logpath.glob(f"{params['status']['log_prefix']}*.out"):
        outfile.unlink()
    ## Archive logfiles after running all because errors in some may be removed by subsequent files
    archive_path=params['status']['archive_path']
    if not archive_path:
        ppaths=ProjectPaths(params)
        archive_path = ppaths.logfiles
        archive_path.mkdir(parents=True, exist_ok=True)
    for logfile in logpath.glob(f"{params['status']['log_prefix']}*.err"):
        archive_logfile(logfile,cell_dict,archive_path)
    ## Save full updated database
    new_processing_info = pd.DataFrame.from_dict(cell_dict,orient='index')
    new_processing_info.rename_axis('cell_id', axis=1, inplace=True)
    pd.DataFrame.to_csv(new_processing_info, dldb_path, index='cell_id')
    
    ## Print just the current batch of cells to logfile for easy error checking
    logger.info(f'all cells prcessed in this batch:{cell_batch}')
    batch_status = new_processing_info[new_processing_info.index.isin(cell_batch)].sort_index()
    pd.set_option("display.max_columns", None)
    pd.set_option('display.max_rows', 500)
    logger.info(batch_status)

###########################################################################################################################
####  TS INDEX CHECK ####
###########################################################################################################################
def check_ts_windows(cell_list, processed_dir, spec_indices, start_check, end_check):
    '''
    Checks whether files exist over the expected duration (['status']['period'][0] to ['status']['period'][1]) in [YYYYdoy, YYYYdoy]
    as well as whether there is data in all windows of the images.
    '''
    
    cells = []
    if isinstance(cell_list, list):
        cells = cell_list
    elif isinstance(cell_list, str) and cell_list.endswith('.csv'): 
        with open(cell_list, newline='') as cell_file:
            for row in csv.reader(cell_file):
                cells.append (row[0])
    elif isinstance(cell_list, int) or isinstance(cell_list, str): # if runing individual cells as array via bash script
        cells.append(cell_list) 

    data_status = {}
    if isinstance(spec_indices, str):
        spec_indices = [spec_indices]
    for ix in spec_indices:
        data_status[ix] = {}
        logger.info(f"WORKING ON {ix}..... \n")
        for cell in cells:
            ts_dir = Path(processed_dir) / f"{int(cell):06d}/brdf_ts/ms/{ix}"
            if not ts_dir.is_dir():
                logger.debug(f'    ERROR: no {ix} created for cell {cell} \n')
                data_status[ix][cell] =  'no ts'
            else: 
                ts_imgs = sorted(list(ts_dir.glob('*.tif')))
                if len(ts_imgs) == 0:
                    logger.debug(f'    ERROR: there are no images in the {ix} folder for cell {cell} \n')
                    data_status[ix][cell] =  'no ts'
                else:
                    first_img_date = int(ts_imgs[0].stem)
                    last_img_date = int(ts_imgs[-1].stem)
                
                    if int(first_img_date) > start_check:
                        logger.debug(f'    ERROR: cell {cell} index {ix} starts at {first_img_date} \n')
                        data_status[ix][cell] = 'missing images'            
                    elif int(last_img_date) < end_check:
                        logger.debug(f'    ERROR: cell {cell} index {ix} ends at {last_img_date} \n')
                        data_status[ix][cell] = 'missing images'
                    else:
                        with gw.open(ts_imgs[-1]) as src:
                            windows = list(src.gw.windows(row_chunks=256, col_chunks=256))
                            bad_wins = 0
                            for w in windows:
                                with rio.open(ts_imgs[-1], mode='r') as src:
                                    block = src.read(1, window=w)
                                    if block.mean() != 0:
                                        continue
                                    else:
                                        bad_wins = 1
                                        logger.debug(f'  ERROR: Cell {cell}, index {ix} missing data for w {w} \n')
                        if bad_wins == 1:
                            data_status[ix][cell] = 'missing windows'
                        else:
                            data_status[ix][cell] = 'clean'
    
    logger.info(f"data_status: {data_status}")
    return data_status

###########################################################################################################################
####  GENERAL ####
###########################################################################################################################
    
def plot_status(out_fig, dataframe, zoom, other_layers, plt_title=None, cent_dist=50000):

    logger.info('      plotting grid status...')
    with plt.style.context('seaborn-dark'):
        fig, ax = plt.subplots(constrained_layout=True)
        legend_kwds = {'ncol': 1,
                       'fontsize': 4,
                       'loc': 'lower center',
                       'markerscale': 0.5,
                       'facecolor': 'white',
                       'frameon': True,
                       'framealpha': 0.8}
        legend_kwds = legend_kwds if zoom else None
        dataframe.plot(column='status', legend=True, legend_kwds=legend_kwds, ax=ax)
        dataframe.plot(color='none', edgecolor='w', lw=0.4, legend=False, ax=ax)

        if other_layers:
            if isinstance(other_layers, str):
                other_layers = [other_layers]
            lw = 0.2
            for layer in other_layers:
                anc_df = gpd.read_file(layer)
                anc_df.to_crs(dataframe.crs).plot(color='none', edgecolor='k', lw=lw, ax=ax)
                lw += 0.1

        if zoom:
            if not dataframe.query("status != 'n'").empty:
                dataframe.query("status == 'n'").apply(lambda x: ax.annotate(str(x.UNQ),
                                                                             x.geometry.centroid.coords[0],
                                                                             fontsize=3,
                                                                             ha='center',
                                                                             va='center',
                                                                             color='white'), axis=1)

            if not dataframe.query("status != 'n'").empty:
                dataframe.query("status != 'n'").apply(lambda x: ax.annotate(str(x.UNQ),
                                                                             x.geometry.centroid.coords[0],
                                                                             fontsize=5,
                                                                             ha='center',
                                                                             va='center',
                                                                             color='black'), axis=1)

            if isinstance(zoom, str) and zoom.startswith('completed'):
                if "_" in zoom:
                    left, bottom, right, top = dataframe.query(f"status == {zoom.split('_')[1]}").total_bounds
                else:    
                    left, bottom, right, top = dataframe.query("status != 'n'").total_bounds
            else:
                left, bottom, right, top = dataframe.query(f"UNQ =={zoom}").total_bounds
            offset = cent_dist
            ax.set_xlim(left-offset, right+offset)
            ax.set_ylim(bottom-offset, top+offset)
            ax.set_axis_off()

        plt.suptitle(plt_title, fontsize=6) 
        #plt.tight_layout(pad=0.1)
        plt.savefig(out_fig, dpi=300)

def status(params):

    '''
    Checks processing status of all cells for process steps (preproceesing, time series) for time series, can specify sis to check for as list.
    Produces gridded image for visualization as well as .csv file for quick lookup. Also saves results as .geojson so that the image can easily
       be rebuilt to focus on different areas. The ['status']['zoom'] parameter enables refocuing of the image by area with a specific task
       completed 'compleded_<task>, or area surrounding a chosen grid cell, with ['status']['offset'] determining the zoom extent around the area.
       ['status']['other_layers'] allows polygon files (e.g. of the AOI) to be overlayed for context.

    When checking time series, checks not only whether files exist over the expected duration (['status']['period'][0] to ['status']['period'][1]), 
    but also whether there is data in all windows of the images.
    '''
    
    if params['log_level'] !='INFO':
        logger.logging.basicConfig(level=f"logging.{params['log_level']}")
     
    out_path = Path(params['status']['out_path'])
    out_path.mkdir(parents=True, exist_ok=True)

    if params['status']['use_existing'] and (out_path / "grid_status.geojson").is_file():
        df = gpd.read_file(out_path / "grid_status.geojson")
        title_preface = "spectral index (kndvi,gcvi,nbr,ndmi)"
        out_fig = str(out_path / f"grid_status_ts_{params['status']['period'][0]}-{params['status']['period'][1]}.png")

    else:
        grid_file = params['grid_file']
        df = gpd.read_file(grid_file)
        logger.debug(df.columns)
        if params['status']['filter']:
            logger.info(f"      Filtering gridfile by {params['status']['filter']} in column: {params['status']['filter_column']}")
            df = df[df[params['status']['filter_column']]==params['status']['filter']]
        df['status'] = 'n'

        for row in df.itertuples():
            grid = f'{row.UNQ:06d}'
            ## do not use ppaths here -- it will make new folders for ALL cells in grid!
            grid_path = Path(f"{params['main_path']}/{grid}")
            if grid_path.is_dir():
                logger.info(f'getting info for cell {grid}')
            
                ## mark with p if any processing has happened. 
                ##   If ppaths has ever been called on this cell, empty folders and db may have been created, so need to check whether empty
                ##   Also, sometimes the db file gets corrupted, so not perfect check by itself.
                db = TuyauDataBase(str(grid_path / f'{grid}_tuyau.db'))
                brdf_path = grid_path /'brdf'
                if (db.is_file() and db.has_data()) or (brdf_path.is_dir() and any(brdf_path.iterdir())):
                    df.loc[row.Index, 'status'] = 'p'
                    try:
                        # Check pre-preprocessing
                        if db.is_complete(int(row.UNQ), 'preprocess'):
                            df.loc[row.Index, 'status'] = 'p_preprocess'
                    except:
                        # raise LookupError(f'The database for grid {row.UNQ} could not be opened.')
                        continue
                
                    ## mark with p_sis if any indices have been processed
                    if params['feature_model']['spec_indices']:
                        sis = params['feature_model']['spec_indices']
                        title_preface = f"vi ({','.join(sis)})"
                        out_fig = str(out_path / f"grid_status_ts_{params['status']['period'][0]}-{params['status']['period'][1]}.png")
                    
                        completed_sis = []
                        #ts_dir = Path(f"{params['backup_path']}/{grid}/brdf_ts}")
                        for si in sis:
                            '''
                            si_window_file = ts_dir / si / f'{grid}.window'
                            if si_window_file.is_file():
                                with open(si_window_file, mode='r') as pf:
                                    window_tracker = yaml.load(pf, Loader=yaml.FullLoader)
                                if int(window_tracker[params['reconstruct']['chunks']]['latest']) == 1e9:
                                    completed_sis.append(si)
                                else:
                                    completed_sis.append(f'{si} incomplete')
    
                            if completed_sis:
                                df.loc[row.Index, 'status'] = ','.join(completed_sis)
                            '''
                            si_check = check_ts_windows(grid,params['backup_path'], si, params['status']['period'][0], params['status']['period'][1])
                            logger.debug(f'vi_check2: {si_check}')
                            if si_check[si][grid] == 'clean':
                                completed_sis.append(si)
                        
                        if len(completed_sis) > 0:
                            df.loc[row.Index, 'status'] = f"p_sis_{','.join(completed_sis)}"

                    else:
                        title_preface = 'preprocessing'
                        out_fig = str(out_path / f"grid_status_preprocessing_{params['status']['period'][0]}-{params['status']['period'][1]}.png")

    plot_status(out_fig, 
                df, 
                params['status']['zoom'], 
                params['status']['other_layers'],
                plt_title= f"{title_preface} status for {params['status']['period'][0]} to {params['status']['period'][1]}",
                cent_dist = params['status']['offset'])

    df.to_csv(str(out_path / "grid_status.csv")) 
    df.to_file(str(out_path / "grid_status.geojson"), driver='GeoJSON')
    logger.info(f"   done! status file saved at: {out_fig}")
    return df
