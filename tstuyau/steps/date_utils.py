from pathlib import Path
from datetime import datetime, timedelta
from calendar import monthrange, month_abbr
from collections import namedtuple
import geowombat as gw
import numpy as np
import pandas as pd
import geopandas as gpd
from ..handler import logger


def get_date_range(year,period,params,return_type='ymd',padded=False):

    '''
    returns start and stop dates for mapping period 
    
    The year is the starting year if period spans more than one year
    <period> can be 'yr', 'wet' or 'dry' based on seasonal <calendar> parameters (which are in julien doy)
        or a quarter ('Q1','Q2','Q3','Q4') or a single month ('Jan','Feb','Mar',etc) or multiple months ('NovDec','JanMar')
              (note: months must be consecutive and inclusive. 'JanMar' will behave the same as 'JanFebMar')
    
    if return_type is 'doy', returns int in format YYYYdoy
    otherwise returns strings in format YYYY-MM-DD 
    '''

    ## start of the year is the first day of the starting calendar month
    start_of_yr = datetime(year, params['calendar']['first_mo'], 1)
    ## end of the year is 364 days from the start of the year
    end_of_yr = datetime(year, params['calendar']['first_mo'], 1) + timedelta(364)
    
    if period == 'yr':
        start_date = start_of_yr
        end_date = end_of_yr

    elif any(month in period for month in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']):
        ## match month to month in calendar package to get date range info
        month1_num = list(month_abbr).index(period[:3])   ## gives index location (should = normal month num since sequence in calendar starts with a blank)
        if month1_num >= params['calendar']['first_mo']:
            target_yr = year
        else:
            target_yr = year + 1
        start_date = datetime(target_yr, month1_num, 1)
        ## if single month, end month is same as start month. But period can be range (e.g. 'JanMar') 
        end_month = month1_num if len(period) == 3 else list(month_abbr).index(period[-3:])
        last_day = monthrange(year, end_month)[1]
        end_date = datetime(target_yr, end_month, last_day)
    elif any (q in period for q in ['Q1','Q2','Q3','Q4']):
        quarter = int(period.split('Q')[1][0])
        qend = quarter * 91
        qstart = qend - 91
        start_date = datetime(year, params['calendar']['first_mo'], 1) + timedelta(qstart)
        end_date = datetime(year, params['calendar']['first_mo'], 1) + timedelta(qend)
    else:
        if period == 'wet':
            doys = [int(params['calendar']['start_wet']), int(params['calendar']['end_wet'])]

        elif period == 'dry':
            doys = [int(params['calendar']['start_dry']), int(params['calendar']['end_dry'])]
        
        if padded:
            doys = [doys[0] - params['feature_model']['pheno_pad_days'][0], doys[1] + params['feature_model']['pheno_pad_days'][1]]

        if datetime((int(year) + 1), 1, 1) + timedelta(days=doys[0] - 1) > end_of_yr:
            start_date = datetime((int(year)), 1, 1) + timedelta(days=(doys[0] - 1))
        else:
            start_date = datetime((int(year) + 1), 1, 1) + timedelta(days=(doys[0]  - 1))
            
        if datetime((int(year) + 1), 1, 1) + timedelta(days=doys[1] - 1) > end_of_yr:
            end_date = datetime((int(year)), 1, 1) + timedelta(days=(doys[1] - 1))
        else:
            end_date = datetime((int(year) + 1), 1, 1) + timedelta(days=(doys[1] - 1))

    if return_type == 'doy':
        return int(start_date.strftime("%Y%j")), int(end_date.strftime("%Y%j"))
    else:
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            
def get_img_date(img, ts_type, img_type, data_source=None):
    
    if '.' in img:
        img_base = Path(img).stem
    else:
        img_base = img
    
    if 'sm' in ts_type.lower():  #Expects images to be smoothed already and be named YYYYDDD
        YYYY = int(img_base[:4])
        doy = int(img_base[4:7])
        ydoy = img_base
    elif 'CHIRPS' in img_type: #Expects images to be named something like chirps-v3.0.YYYY.MM.3.tif
        chirps_intervals = {'pentad':5,'dekad':10,'daily':1}
        interval = img_type.split('-')[1]
        interval_int = chirps_intervals[interval]
        YYYY = int(img_base[2])
        MM = int(img_base[3])
        if interval in['pentad','dekad']:
            DD = int(img_base[4])*interval_int if ((MM !=2) | (int(img_base[4]) !=30/interval_int)) else 28
        else:
            DD = int(img_base[4])
        ymd = datetime.datetime(YYYY, MM, DD)
        ydoy = ymd.strftime("%Y%j")
        doy = int(ymd.strftime('%j'))
    elif img_type not in ['LS2','S2','L','LT05', 'LE07', 'LC08', 'LC09']:
        logger.info(f"Currently valid image types are 'LS2','S2','L' or specific landsat sensors (LT05, LE07, LC08, LC09)")
        logger.info(f"or CHIRPS(-pentad,-dekad,-daily). You put ts_type={ts_type},img_type={img_type}")
    else:
        if img_type == 'S2' and 'brdf' not in str(img_base):
            YYYYMMDD = img_base.split('_')[2][:8]
        else:
            YYYYMMDD = img_base.split('_')[3][:8]
        YYYY = int(YYYYMMDD[:4])
        MM = int(YYYYMMDD[4:6])
        DD = int(YYYYMMDD[6:8])
   
        ymd = datetime(YYYY, MM, DD)
        ydoy = ymd.strftime("%Y%j")
        doy = int(ymd.strftime('%j'))

        #return YYYY, doy
    return ydoy
        

def query_frame_date(df_clip, year1, year2, params, method):
    return df_clip.query(f"(datetimes >= '{year1}-{params[method]['start']}') & (datetimes <= '{year2}-{params[method]['end']}')")


def year_to_index(dft, year1, year2, params, method):

    """
    Gets indices for a year-to-year time slice
    """

    # Add 1 because the time indexes start with 1 and are sliced by DataArray.sel(time=<>)
    return dft[f"{year1}-{params[method]['start']}":f"{year2}-{params[method]['end']}"].image_index.values + 1


def get_grid_years(time_names):
    return np.unique(np.array([dt.year for dt in time_names]))


def time_names_to_frame(time_names):
    return pd.DataFrame(data=range(0, len(time_names)), columns=['image_index'], index=time_names)


def get_sample_years(df, image_name, time_names):

    grid_years = get_grid_years(time_names)

    dft = time_names_to_frame(time_names)

    # Clip the samples to the current grid
    with gw.open(image_name) as src:
        df_clip = gpd.clip(df.to_crs(src.crs), src.gw.geodataframe)

    df_clip.sort_values(by=['datetimes'], inplace=True)

    return grid_years, dft, df_clip


def filter_tile_groups(tile_list, start, end):

    """
    Filters a list of tile groups by date parameters

    Args:
        tile_list (list): A list of tile groups.
        start (str): The start date (yyyy-mm-dd).
        end (str): The end date (yyyy-mm-dd).

    Returns:
        filtered groups (list)
    """

    # Get the Julian days
    start_jd = datetime.strptime(start, '%Y-%m-%d')
    end_jd = datetime.strptime(end, '%Y-%m-%d')

    # Filter the list by the requested time slice
    return [fn_tuple[1] for fn_tuple in tile_list
            if (datetime.strptime(str(fn_tuple[0]), '%Y%j') >= start_jd)
            and (datetime.strptime(str(fn_tuple[0]), '%Y%j') <= end_jd)]


def date_attrs_to_datetime(year, month, day):
    return datetime.strptime(f"{year:d}-{month}-{day}", '%Y-%m-%d')


def julian_attrs_to_datetime(jd):
    return datetime.strptime(str(jd), '%Y%j')


def date_to_julian(date):

    """
    Converts a date to Day-of-year and Julian day

    Args:
        date (datetime | str): 'yyyymmdd'

    Returns:
        Day-of-year (int), Julian day (str)
    """

    if isinstance(date, datetime):
        dt = date
    else:
        dt = datetime.strptime(date, '%Y%m%d')

    doy = dt.timetuple().tm_yday

    jdr = f"{dt.timetuple().tm_year:d}{doy:03d}"

    return doy, jdr

def check_day_dist(dta, dtb, max_days):

    """
    Checks if two dates fall within a day range

    Args:
        dta (object): The first ``datetime.datetime`` object.
        dtb (object): The second ``datetime.datetime`` object.
        max_days (int): The maximum number of days.

    Returns:
        ``bool``
    """

    # Get the maximum number of days in the current month
    max_month_days = monthrange(dta.year, dtb.month)[1]
    month_day = min(dtb.day, max_month_days)

    dtc = datetime.strptime(f'{dta.year}-{dtb.month}-{month_day}', '%Y-%m-%d')

    if abs(dta - dtc).days <= max_days:
        return True

    # Get the maximum number of days in the current month
    max_month_days = monthrange(dta.year-1, dtb.month)[1]
    month_day = min(dtb.day, max_month_days)

    dtc = datetime.strptime(f'{dta.year-1}-{dtb.month}-{month_day}', '%Y-%m-%d')

    if abs(dta - dtc).days <= max_days:
        return True

    # Get the maximum number of days in the current month
    max_month_days = monthrange(dta.year+1, dtb.month)[1]
    month_day = min(dtb.day, max_month_days)

    dtc = datetime.strptime(f'{dta.year+1}-{dtb.month}-{month_day}', '%Y-%m-%d')

    if abs(dta - dtc).days <= max_days:
        return True

    return False


def prepare_x(X, start, end, skip):

    """
    Prepares the X data for modelling

    Args:
        X (list): A list of datetime objects.
        start (str): The desired times series start date.
        end (str): The desired times series end date.
        skip (int): The time series skip interval.

    Returns:
        X information (namedtuple)
    """

    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')

    xd = [1000]
    dist = 1000
    for i in range(1, len(X)):
        dist += (X[i] - X[i - 1]).days
        xd.append(dist)

    xd = np.ascontiguousarray(np.array(xd, dtype='float64'))

    indices_series = pd.Series(xd, index=pd.DatetimeIndex(X))

    # Remove duplicate rows
    indices_series = indices_series[~indices_series.index.duplicated()]

    dates = indices_series.index.to_pydatetime()

    # Resample the uneven dates to a daily series
    indices_series_daily = indices_series.resample('D').interpolate('linear')

    xd_smooth = np.ascontiguousarray(indices_series_daily.values, dtype=np.float64)
    dates_smooth = indices_series_daily.index.to_pydatetime()

    start_orig_idx = np.argmin(np.abs(dates - start_dt))
    end_orig_idx = np.argmin(np.abs(dates - end_dt))

    skip_orig_idx = np.array([i for i in range(0, indices_series.shape[0])
                              if start_dt <= indices_series.index[i].to_pydatetime() <= end_dt],
                             dtype='uint64')

    start_idx = np.argmin(np.abs(dates_smooth - start_dt))
    end_idx = np.argmin(np.abs(dates_smooth - end_dt))

    py_dates_slice = dates_smooth[start_idx:end_idx]

    ###########################################################################################
    interp_idx = np.array([dtidx for dtidx in range(0, dates_smooth.shape[0])
                           if (dates_smooth[dtidx].timetuple().tm_mday == 1) or
                           (dates_smooth[dtidx].timetuple().tm_mday % skip == 0)],
                          dtype='uint64')

    xd_interp = xd_smooth[interp_idx]

    interp_start_idx = np.argmin(np.abs(dates_smooth[interp_idx] - start_dt))
    interp_end_idx = np.argmin(np.abs(dates_smooth[interp_idx] - end_dt))

    skip_interp_idx = np.arange(interp_start_idx, interp_end_idx+1)

    skip_interp_slice = dates_smooth[interp_idx][skip_interp_idx]
    ###########################################################################################

    # Get the indices for the time series interval
    skip_idx = np.array([dtidx for dtidx in range(0, py_dates_slice.shape[0])
                         if (py_dates_slice[dtidx].timetuple().tm_mday == 1) or
                         (py_dates_slice[dtidx].timetuple().tm_mday % skip == 0)],
                        dtype='uint64')

    XInfo = namedtuple('XInfo', 'xd xd_smooth xd_interp skip_interp_idx skip_interp_slice dates dates_smooth start_orig_idx end_orig_idx skip_orig_idx start_idx end_idx skip_idx skip_orig_slice skip_slice')

    return XInfo(xd=xd,
                 xd_smooth=xd_smooth,
                 xd_interp=xd_interp,
                 skip_interp_idx=skip_interp_idx,
                 skip_interp_slice=skip_interp_slice,
                 dates=dates,
                 dates_smooth=dates_smooth,
                 start_orig_idx=start_orig_idx,
                 end_orig_idx=end_orig_idx,
                 skip_orig_idx=skip_orig_idx,
                 start_idx=start_idx,
                 end_idx=end_idx,
                 skip_idx=skip_idx,
                 skip_orig_slice=dates[skip_orig_idx],
                 skip_slice=py_dates_slice[skip_idx])
