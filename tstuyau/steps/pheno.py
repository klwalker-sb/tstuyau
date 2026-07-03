import sys
import re
from pathlib import Path
import datetime
import rasterio as rio
import numpy as np
import geowombat as gw
import xarray as xr
import pandas as pd
#import bottleneck
from ..handler import logger


def add_var_to_stack(arr, var, attrs, out_dir, comp_band_names, ras_list, **gw_args):
    logger.debug(f'gw_args are:{gw_args}')
    logger.debug(f'file attrs are:{attrs}')
    ras = Path(out_dir) / f'{var}.tif'
    if not ras.is_file():  
        arr.attrs = attrs
        logger.info(f'making {var} raster')
        if not ras.parent.is_dir():
            ras.parent.mkdir(exist_ok=True, parents=True)
        ras.touch(exist_ok=True)
        arr.gw.to_raster(ras,**gw_args)
    else:
        logger.info(f' using existing {var} band at {ras}')
    ras_list.append(ras)
    #band = var.rsplit('-', 1)[0]
    band=var
    comp_band_names.append(band)

##################################################################################################
### base functions
##################################################################################################

def norm_abs_energy(data, axis=1):
    """
    Calculates the normalized absolute energy
    """
    return (data**2).sum(axis=axis) / (data.max(axis=axis)**2 * data.shape[1])

def amp(data, axis=1):
    """
    Calculates the amplitude
    """
    return np.nanmax(data, axis=axis) - np.nanmin(data, axis=axis)

def cv(data, axis=1):
    """
    Calculates the coefficient of variation
    """
    return np.nanstd(data, axis=axis) / np.nanmean(data, axis=axis)

def cumsum_is_large(data, thresh=2, axis=1):
    """
    Calculates whether the cumulative sum is large (1) or small (0)
    """
    return np.where(data.cumsum(axis=axis)[:, -1] >= thresh, 1, 0)

def mean_abs_diff(data, axis=1):
    """
    Calculates the mean of the absolute differences
    """
    return np.nanmean(np.abs(np.diff(data, n=1, axis=axis)), axis=axis)

def _lstsq(data):

    n_samples, n_feas = data.shape

    x = np.arange(0, n_feas)

    # Fit a least squares solution to each sample
    return np.linalg.lstsq(np.c_[x, np.ones_like(x)], data.T, rcond=None)[0]

def abs_slope_q1(data, axis=1):
    """
    Calculates the absolute slope of the first quarter
    """
    n_samples, n_feas = data.shape
    b1 = _lstsq(data[:, :int(0.25*n_feas)])[0]
    return np.abs(b1)

def abs_slope_q2(data, axis=1):
    """
    Calculates the absolute slope of the second quarter
    """
    n_samples, n_feas = data.shape
    b1 = _lstsq(data[:, int(0.25*n_feas):int(0.5*n_feas)])[0]
    return np.abs(b1)

def abs_slope_q3(data, axis=1):
    """
    Calculates the absolute slope of the third quarter
    """
    n_samples, n_feas = data.shape
    b1 = _lstsq(data[:, int(0.5*n_feas):int(0.75*n_feas)])[0]
    return np.abs(b1)

def abs_slope_q4(data, axis=1):
    """
    Calculates the absolute slope of the fourth quarter
    """
    n_samples, n_feas = data.shape
    b1 = _lstsq(data[:, int(0.75*n_feas):])[0]
    return np.abs(b1)

def sum_rss(data, axis=1):
    """
    Calculates the coefficients of a linear least squares regression, and the residual sum of squares
    """
    n_samples, n_feas = data.shape
    x = np.arange(0, n_feas)
    b1, intercept_b0 = _lstsq(data)
    # Estimate
    yhat = intercept_b0[:, np.newaxis] + b1[:, np.newaxis] * np.dot(b1[:, np.newaxis], x[np.newaxis, :])
    # Calculate the normalized residual sum of squares
    return ((data - yhat)**2).sum(axis=1) / ((data.max(axis=axis) - data.min(axis=axis))**2 * data.shape[1])

def p5(data, axis=1):
    """
    Calculates the 5th percentile
    """
    return np.nanpercentile(data, 5, axis=axis)

def p95(data, axis=1):
    """
    Calculates the 95th percentile
    """
    return np.nanpercentile(data, 95, axis=axis)


##################################################################################################
### phenology functions
##################################################################################################
    
def get_sig_change(ts_stack, ds_stack, cng_thresh, base_thresh=None, image_buffer=0, cng_freq='doy', normalize=None):
    '''
    Captures moment of significant change (above <cng_thresh>) in index, either as a drop (negatve <cng_thresh>) or spike (positive <cng_thresh>) 
    Returns change events formated based on <cng_freq>: 'bi' = 0/1, 'count', 'doy' (day-of-year of 1st occurance), or 'mo' (month of first occurance)
       if doy or month, the day-of-year value for the last day of the mapping period will be used as nan value, 
          so mapping period should be designed such that the last day's doy is not on a day of year that the event is likely to happen.  
           
    Allows for <base_thresh> to filter out changes in spectral areas outside the targeted range. This value is above/below normal post-event values
        when the index decreases/increases with the event. If 0, this step will be skipped.
    <image_buffer> stes the number of images that are backfilled if a pixel has a nodata value. This is important in burn mapping because intense 
        fire events are often masked as cloud shadow. if not useful for mapping task, set <feature_model:pheno_imgbuf> to 0.
    '''

    logger.info(f'getting significant change raster for threshold {cng_thresh}...')
    logger.debug(f'cng_freq = {cng_freq}')

    ## TODO: pass these as variables
    lowest = 0
    highest = 10000
    
    with gw.open(ts_stack, time_names=ds_stack) as src:
        attrs = src.attrs.copy()

    ## set nodata. TODO: pass nodata variables
    valsin = src.where((src > lowest) & (src < highest))
    
    if normalize:
        allavg = valsin.mean(dim='time')
    if normalize == '0m': 
        valsin = valsin - allavg
    elif normalize == 'z':
        allstd = valsin.std(dim='time')
        valsin = (valsin - allavg) / allstd
    
    ## backfill n images based on <image_buffer>
    if image_buffer:
        valsin = valsin.bfill(dim='time', limit=int(image_buffer))

    ## If the change event results in a spike in values (<cng_thresh> is positive), values are inverted prior to inquiry:
    if int(cng_thresh) > 0:
        cng_thresh = -1 * cng_thresh
        valsin = lowest + highest - valsin
        if base_thresh:
            base_thresh = lowest + highest - base_thresh
        
    if base_thresh:
        ## set base_thresh <feature_model:pheno_basethresh> to restrict set of possible events to those with reasonable post-event value 
        plausvals = valsin.where(valsin <= base_thresh)
    else:
        plausvals = valsin

    logger.info(f'finding instances of values < {base_thresh} where values have dropped by at least {cng_thresh} since previous observation')
    pass1 = plausvals.where (valsin.fillna(highest) - valsin.shift(time=1).fillna(lowest) <= cng_thresh)  
    ## only count if the low is sustained for at least two more images
    pass2 = pass1.where (valsin.shift(time=-1).fillna(highest) - valsin.shift(time=1).fillna(lowest) <= (.7 * cng_thresh))
    pass3 = pass2.where (valsin.shift(time=-2).fillna(highest) - valsin.shift(time=1).fillna(lowest) <= (.7 * cng_thresh))
    ## remove cases where the previous value was not above the base thresh (often heavily shaded areas)
    if base_thresh:
        pass3 = pass3.where(valsin.shift(time=-1) > base_thresh)
    ##  only count if the previous two values are above the change threshold (not abnormal spike)
    pass4 = pass3.where((valsin.fillna(highest) - valsin.shift(time=2).fillna(lowest)) < (.8 * cng_thresh))
    pass5 = pass4.where((valsin.fillna(highest) - valsin.shift(time=3).fillna(lowest)) < (.8 * cng_thresh)).fillna(lowest)

    cng_v = pass5.max(dim="time").fillna(lowest).astype('int16')
    if cng_freq == 'bi': ## binary resolution: 1 if any significant change
       cgn_t = xr.where(pass5 > lowest, 1, lowest).max(dim="time").fillna(lowest)
    elif cng_freq == 'count': ## returns number of significant changes observed
       cgn_t = pass5.where(pass5 > lowest).count(dim="time").fillna(lowest)
    else:
        ## get day-of-year of significant change observations:
        ## need to fill nas with a valid date for min to work. passing a value in with timestamp or datetime.datetime does not work
        ##     using the last date in the array -- then replacing as 0 after conversion to doy. 
        logger.debug(f"last time vals: {valsin.time[-1].values}")                        
        change_t = pass5["time"].where(pass5 > lowest).fillna(valsin.time[-1].values).min(dim="time")
        logger.debug(f'change_t (before doy): \n {change_t}')
        
        ## convert values for the last day back to Nan (0 here)
        ##     Note: need to make sure last day of mapping time period does not fall on a day of year that the event is likely to happen!
        ## First get the doy for the last day. Dates are np.datetime64 objects. 
        last_doy = valsin.time[-1].values.astype('datetime64[us]').tolist()
        nan_doy = last_doy.timetuple().tm_yday
        logger.info(f"last doy: {nan_doy} being replaced as 0")

        cng_t0 = change_t.dt.dayofyear.fillna(0).astype('int16')
        if cng_freq == 'doy': ## daily resolution, output is day-of-year
            cng_t = cng_t0.where(cng_t0 != nan_doy, 0)
        elif cng_freq == 'mo':  ## monthly resolution
            ## integer division rounds down, so reverse so that first values will be 1, not 0
            #cgn_t =  -(-cng_t0 // 30).where(cng_t0 != nan_doy, 0)
            cng_t = change_t.dt.month.astype('uint8')
        
    return cng_t, cng_v 
    
def find_peak_simp(ts_stack,ds_stack):
    
    with gw.open(ts_stack, time_names = ds_stack) as src:
        highs = src.where(src >= 3000)
    #highs_masked = highs.where(src < 1000).all("time")
    peaksimp = highs.idxmax(dim='time',skipna=True)

    return peaksimp

def revise_peaks(t,vals,invert):
    i = -1 if invert else 1
    fill = 32767 if invert else 0
    f = 10000 if invert else 1
    
    if t == 1:
        peakcheck1 = vals.where(((i*(vals - vals.shift(time=-1).fillna(fill)) >= 0) | (vals==f)),fill).fillna(fill)
        peakcheck = peakcheck1.where(((i*(peakcheck1 - peakcheck1.shift(time=1).fillna(fill)) >= 0) | (peakcheck1==f)),fill).fillna(fill)
    else:
        peakcheck1 = vals.where(((i*(vals - vals.shift(time=-t).fillna(fill)) >= 0 ) | (vals.shift(time=-(t-1)).fillna(fill) == f) | (vals==f)),fill).fillna(fill)
        peakcheck = peakcheck1.where(((i*(peakcheck1 - peakcheck1.shift(time=t).fillna(fill)) >= 0 ) | (peakcheck1.shift(time=(t-1)).fillna(fill) == f) | (peakcheck1==f)),fill).fillna(fill)
    
    return peakcheck

def find_peaks_robust(ts_stack, ds_stack,peak_thresh,base_thresh,invert):
    with gw.open(ts_stack, time_names = ds_stack) as valsin:
        attrs = valsin.attrs.copy()
    ''' if invert == True, returns troughs
        returns number of peaks and day and value of first and last peak (/trough) for stack
    '''
    
    ## base the input data, shuch that all values <= base threshold are flagged as 1 (these separate potential peaks)
    #i = -1 if invert == True else 1
    if not invert:
        peakbase = valsin.where(valsin > base_thresh, 1)
        ## check whether each pixel is local max value by subtracting the next in sequence (both forward and backward) and retaining
        ##    only those with positive results. if the original pixel value was 1, it is retainied as 1 to flag troughs separating peaks
    
        peakcheck1 = peakbase.where(((peakbase - peakbase.shift(time=-1).fillna(0) >= 0) | (peakbase==1)),0).fillna(0)
        peakcheck = peakcheck1.where(((peakcheck1 - peakcheck1.shift(time=1).fillna(0) >= 0) | (peakcheck1==1)),0).fillna(0)
        ## repeat the process at increasing time steps. If the value before the next time step is 1, there is a trough separating
        ##    the two peaks and both should be retained.
        for t in range(2,10):
            peaktcheck = revise_peaks(t,peakcheck,invert=False)
         ## TODO: break ties
    
        true_peaks = peakcheck.where(peakcheck >= peak_thresh)
    
    else:
        peakbase = valsin.where(valsin < base_thresh, 10000)
        peakcheck1 = peakbase.where(((peakbase - peakbase.shift(time=-1).fillna(32767) <= 0) | (peakbase==10000)),32767).fillna(32767)
        peakcheck = peakcheck1.where(((peakcheck1 - peakcheck1.shift(time=1).fillna(32767) <= 0) | (peakcheck1==10000)),32767).fillna(32767)
        for t in range(2,10):
            peaktcheck = revise_peaks(t,peakcheck,invert=True)
        true_peaks = peakcheck.where(peakcheck <= peak_thresh)
         
    numpeaks = true_peaks.count(dim='band').sum(dim='time').fillna(0).astype('int16')
    ## get first peak (minimum time of valid peaks). First fill nas with last possible day because skipna doesn't work with datetime 
    peak0d = true_peaks["time"].where(~true_peaks.isnull()).fillna(true_peaks.time[-1].values).min(dim="time")
    peak0v = valsin.sel(time=peak0d, method='nearest').astype('int16')
    ## get last trough (maximum time of valid troughs). First fill nas with first possible day because skipna doesn't work with datetime
    peak9d = true_peaks["time"].where(~true_peaks.isnull()).fillna(true_peaks.time[1].values).max(dim="time")
    peak9v = valsin.sel(time=peak9d, method='nearest').astype('int16').squeeze()
    
    return numpeaks, peak0d, peak0v, peak9d, peak9v
        
def find_peaks_deriv(ts_stack,ds_stack,comp_band_names,peak_thresh,base_thresh,ras_list,out_dir, **gw_args):
    '''
    method to find the first peak of a season (in case more than one peaks occur)
    '''
    with gw.open(ts_stack, time_names = ds_stack) as src1:
        attrs = src1.attrs.copy()
    src_c = src1.chunk({"time": -1})
    
    deriv = src_c.differentiate("time")
    maxima = src1.where((deriv < 0) & (deriv.shift(time=1) > 0) & (src1 >= peak_thresh))
    minima = src1.where((deriv > 0) & (deriv.shift(time=1) < 0) & (src1 <= base_thresh))
    minmax_arr = xr.concat((minima,maxima),'band').mean(dim='band')
    minmaxf = minmax_arr.ffill(dim='time')
    minmaxff = minmaxf.bfill(dim='time')
    minmax_step = minmaxff - minmaxff.shift(time=1) 
    #falsepeaks = max_arr.where((minmax_step > 0) & (minmax_step < 2000))
    #falsepeak1 = falsepeaks.idxmax(dim="time",skipna=True)
    #minmax_arr.loc[dict(time=falsepeak1)] = 'nan' # this doesn't work because lost one timestep.
    true_peaks = maxima.where((np.abs(minmax_step) >= (peak_thresh - base_thresh)) | (minmax_step == 0))
    numpeaks = true_peaks.count(dim='band').sum(dim='time').fillna(0).astype('int16')
    ## get minimum time where peak condition is met 
    ##    (note need to fill nan with last time stamp because skipna doens't work with datetime64 -- will always give 1st date)
    peakds = true_peaks["time"].where(~true_peaks.isnull()).fillna(true_peaks.time[-1].values).min(dim="time")
    ## remask where no peaks were found
    mask = true_peaks.isnull().all("time")
    peak1d = peakds.where(~mask)
    ## get peak the simple way (max value above threshold) in case slopes don't create proper maxima/minima
    altpeakd = find_peak_simp(ts_stack,ds_stack)
    altpeaknum = altpeakd.count(dim='band').fillna(0).astype('int16')
    ## if a peak was found with the first method, give that, otherwise give simple peak
    peakout = peak1d.where(peak1d.notnull(), altpeakd)
    numpeaks_all = numpeaks.where(numpeaks > 0, altpeaknum).fillna(0).astype('int16')
    masked2 = src1.where(src1 < base_thresh).all("time")
    numpeaks_allm = numpeaks_all.where(~masked2, 0).fillna(0).astype('int16')

    return peakout, numpeaks_all
    
def get_greenup(ts_stack, ds_stack, peak_time, method='step'):
    with gw.open(ts_stack, time_names = ds_stack) as src:
        attrs = src.attrs.copy()
    prepeak1 = src.where(src['time'] < peak_time)
    if method == 'step':
        endslope = prepeak1.where(prepeak1 - prepeak1.shift(time=1) < 0)
        mask = endslope.isnull().all("time")
        sos0 = endslope['time'].where(~endslope.isnull()).fillna(prepeak1.time[-1].values).min(dim='time')
        sos=sos0.where(~mask, prepeak1.time[0].values)
    elif method == 'thresh':
        thresh = 2000
        endslope = prepeak1.where(prepeak1 < thresh)
        mask = endslope.isnull().all("time")
        sos0 = endslope['time'].where(~endslope.isnull()).fillna(prepeak1.time[0].values).max(dim='time')
        sos=sos0.where(~mask, prepeak1.time[0].values)
    elif method == 'deriv':
        prechunk = prepeak1.chunk({"time": -1})
        green_deriv = prechunk.differentiate("time")
        pos_green_deriv = green_deriv.where(green_deriv > 0)
        pos_greenup = prechunk.where(~np.isnan(pos_green_deriv))
        med_g = pos_greenup.median("time")
        dist = np.abs(pos_greenup - med_g)
        mask = dist.isnull().all("time")
        distfill = dist.fillna(dist.max() + 1)
        sos0 = distfill.idxmin(dim="time",skipna=True).where(~mask)
    sosv = prepeak1.sel(time=sos, method='nearest').astype('int16')
    #return ts_stack, peak_time, prepeak1, endslope, mask, sos0, sos, sosv
    return sos, sosv

def get_senescence(ts_stack, ds_stack, peak_time, method='step'):
    with gw.open(ts_stack, time_names = ds_stack) as src:
        attrs = src.attrs.copy()
    postpeak1 = src.where(src['time'] > peak_time)
    if method == 'step':
        endslope = postpeak1.where(postpeak1 - postpeak1.shift(time=1) > 0)
        #mask = endslope.isnull().all("time")
        eos = endslope['time'].where(~endslope.isnull()).fillna(postpeak1.time[-1].values).min(dim="time")
    elif method == 'thresh':
        thresh = 2000
        endslope = postpeak1.where(postpeak1 < thresh)
        #mask = endslope.isnull().all("time")
        eos = endslope['time'].where(~endslope.isnull()).fillna(postpeak1.time[-1].values).min(dim='time')
    elif method == 'deriv':
        postchunk = postpeak1.chunk({"time": -1})
        brown_deriv = postchunk.differentiate("time")
        neg_brown_deriv = brown_deriv.where(brown_deriv < 0)
        neg_brownup = postchunk.where(~np.isnan(neg_brown_deriv))
        med_b = neg_brownup.median("time")
        dist = np.abs(neg_brownup - med_b)
        mask = dist.isnull().all("time")
        distfill = dist.fillna(dist.max() + 1)
        eos0 = distfill.idxmin(dim="time",skipna=True).where(~mask)
    eosv = postpeak1.sel(time=eos, method='nearest').astype('int16')
    return eos, eosv

def prep_pheno_bands(pheno_vars,ts_stack,ds_stack,ts_stack_padded, ds_stack_padded, out_dir,start_yr, 
                     temp,start_doy,comp_band_names,ras_list,sigdif=None,sigbase=None,img_buf=None,params=None, **gw_args):

    logger.info('prepping pheno bands...')
    if isinstance(pheno_vars,str):
        pheno_vars = [pheno_vars]
    logger.info(f'...bands to prep: {pheno_vars}  for temp: {temp}')
    with gw.open(ts_stack, time_names = ds_stack) as src:
        attrs = src.attrs.copy()
        
    mmed = src.where(src > 0).median(dim='time',skipna=True).astype('int16')
    if f'med-{temp}' in pheno_vars:
        add_var_to_stack(mmed,f'med-{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'maxv-{temp}' in pheno_vars or f'amp-{temp}' in pheno_vars:
        #maxf = src.where(src < 10000,0)
        mmax = src.max(dim='time').astype('int16')
        add_var_to_stack(mmax,f'maxv-{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args) 
    if any(v.startswith(('minv','amp','burn')) and v.endswith(temp) for v in pheno_vars):
        #minf = src.where(src > 0,10000).astype('int16')
        mmin = src.where(src > 0,10000).min(dim='time').astype('int16')
        if f'minv-{temp}' in pheno_vars:
            add_var_to_stack(mmin,f'minv-{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args) 
    if f'amp-{temp}' in pheno_vars:
        aamp =  (mmax - mmin).astype('int16')
        add_var_to_stack(aamp,f'amp-{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)   

    delta_vars = [v for v in pheno_vars if v.startswith('sigcng') or v.startswith('burn')]
    if len(delta_vars) > 0:
        for dv in delta_vars:
            if '-' not in dv:  ## legacy code
                norm = None
                freq = 'doy'
                thresh = sigdif
            else:
                thresh = dv.split('.')[1]
                if thresh.startswith('n'):
                    thresh = -1 * int(thresh[1:])
                elif thresh.startswith('p'):
                    thresh = int(thresh[1:])
                else:
                    thresh = thresh
                if dv.split('.')[2] == 'v':
                    freq = 'doy'
                else:
                    freq = dv.split('.')[2].split('-')[0]
                ## Get temporal normalization -- note this is usually in the si_var component of the feature name
                ##    but z and 0m also need to be added to the 1st component of the feature_calc because only this part is passed here
                ##    these are easily calcualted on the fly, so it doesn't make sense to rerun and store the whole ts index anyway 
                if dv.split('.')[0].endswith('z'):
                    norm = 'z'
                elif dv.split('.')[0].endswith('0m'):
                    norm = '0m'
                else:
                    norm = None

            logger.info(f'calculating delta varaible {dv}...')
            cng_d0, cng_v0 = get_sig_change(ts_stack, ds_stack, thresh, sigbase, img_buf,cng_freq=freq,normalize=norm)
            cng_d = cng_d0.persist()
            if (dv.startswith('sigcngv')) or ((dv.startswith('sigcng')) and (dv.split('.')[2] == 'v')):
                cng_v1 = cng_v0.persist()
                cng_v = cng_v1.where(cng_d != 0, 0).astype('int16')
                add_var_to_stack(cng_v,dv,attrs,out_dir,comp_band_names,ras_list,**gw_args) 
            elif dv.startswith('burn'):
                medras = mmed.persist()
                minras = mmin.persist()
                ## remove very shady areas
                if params and params['feature_model']['pheno_shadethreshmed']: 
                    medthresh = params['feature_model']['pheno_shadethreshmed']
                else:
                    medthresh = 0
                if params and params['feature_model']['pheno_shadethreshmin']:
                    minthresh = params['feature_model']['pheno_shadethreshmin']
                else: 
                    minthresh = 0
                burn = cng_d.where(((medras > medthresh) & (minras > minthresh)),0).astype('int16')
                if '.' not in dv and 'burnmo' in dv:  ##legacy code
                    burn = -(-burn // 30).astype('uint8')
                add_var_to_stack(burn,dv,attrs,out_dir,comp_band_names,ras_list,**gw_args) 
            else:
                add_var_to_stack(cng_d,dv,attrs,out_dir,comp_band_names,ras_list,**gw_args) 
    
    if f'slp-{temp}' in pheno_vars:
        slp_path = Path(out_dir)/f'slp-{temp}.tif'
        if not slp_path.is_file():
            src.coords['ordinal_day'] = (('time', ), (src.time - src.time.min()).values.astype('timedelta64[D]').astype(int))
            vslope = src.swap_dims({'time': 'ordinal_day'}).polyfit('ordinal_day', 
                                                                deg=1,skipna=True).polyfit_coefficients[0].astype('int16')    
        else:
            vslope = slp_path
        add_var_to_stack(vslope,f'slp-{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)

    peak_prefixes = ['numrot','posd','posv','tosd','tosv','numlow','p1amp','sosd','sosv','rog','los','eosd','eosv','ros']
    peak_vars = [v for v in pheno_vars if v.startswith(tuple(peak_prefixes)) and v.endswith(temp)]
    if len(peak_vars) > 0:
        if '.' not in peak_vars[0]:
            sigdif = sigdif
            invert = False
        else:
            sigdif =  peak_vars[0].split('.')[1]
            if sigdif.startswith('n'):
                sigdif = int(sigdif[1:])
                invert = True
            elif sigdif.startswith('p'):
                sigdif = int(sigdif[1:])
                invert = False
            else:
                invert = False

        peaks = find_peaks_robust(ts_stack,ds_stack, mmed+int(sigdif), mmed, invert=invert)
        ## peaks returns number of peaks, date-of-1st-peak, val-of-first-peak, date-of-last-peak, val-of-last-peak

        if any(var.startswith('pos') for var in peak_vars):
            posd = [v for v in peak_vars if v.startswith('posd')]
            if len(posd) > 0:
                posd_path = Path(out_dir) / f'{posd[0]}.tif'
                posd2 = posd_path
            else:
                peak1s = peaks[1].where(peaks[0] > 0).squeeze().dt.dayofyear  # returns na if no peaks (to be filled with 0)
                ## add 365 to doy if it passed into the next year to avoid jump in values from Dec31 to Jan 1
                ## (keep everything >= start_doy (-pad) as is. If passes into next year, doy will be < start_doy, so add 365)
                posd2 = peak1s.where(peak1s >= start_doy, (peak1s + 365)).fillna(0).astype('int16')     
            if len(posd) > 0:   
                add_var_to_stack(posd2,posd[0],attrs,out_dir,comp_band_names,ras_list,**gw_args)

            for v in peak_vars:
                if v.startswith('posv'):
                    posv = peaks[2].where(peaks[0] > 0, mmed).astype('int16').squeeze()
                    add_var_to_stack(posv,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)
        
        for v in peak_vars:
            if v.startswith('numrot'):
                add_var_to_stack(peaks[0],v,attrs,out_dir,comp_band_names,ras_list,**gw_args)                
                        
        tos_vars = [var for var in peak_vars if var.startswith(tuple(['tos','numlow','p1amp']))]
        if len(tos_vars) > 0:
            if '.' in tos_vars[0]:
                sigt = f".{tos_vars[0].split('.')[1]}"
            else:
                sigt = ''
            tosd_path = Path(out_dir) / f'tosd{sigt}-{temp}.tif'
            tosv_path = Path(out_dir) / f'tosv{sigt}-{temp}.tif'
            numlow_path = Path(out_dir) / f'numlow{sigt}-{temp}.tif'
            if (not tosd_path.is_file()) or (not tosv_path.is_file) or (not numlow_path.is_file()):
                troughs = find_peaks_robust(ts_stack,ds_stack, mmed-int(sigdif), mmed, invert=True)
                tosd1 = troughs[3].where(troughs[0] > 0).squeeze().dt.dayofyear
                ## add 365 to doy if it passed into the next year to avoid jump in values from Dec31 to Jan 1
                ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
                tosd2 = tosd1.where(tosd1 >= start_doy, (tosd1 + 365)).fillna(0).astype('int16')
                tosv = peaks[4].where(peaks[0] > 0, mmed).astype('int16') 
                numlow = troughs[0]
            else: 
                tosd2 = tosd_path
                tosv = tosv_path
                numlow = numlow_path
            for v in peak_vars:
                if v.startswith('numlow'):
                    add_var_to_stack(numlow,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)
                if v.startswith('tosd'):
                    add_var_to_stack(tosd2,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)
                if v.startswith('tosv'):
                    add_var_to_stack(tosv,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)
                if v.startswith('p1amp'):                
                    p1amp00 = (posv - tosv).where(peaks[0] > 0, (mmed - tosv))  
                    p1amp0 = p1amp00.where(numlow > 0, (posv - mmed))
                    p1amp = p1amp0.where(((p1amp0 > 0) & ((peaks[0] > 0) | (numlow > 0))), 0).fillna(0).astype('int16').squeeze() 
                    add_var_to_stack(p1amp,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)

        sos_vars = [var for var in peak_vars if var.startswith(tuple(['sos','sov','rog','los']))]
        if len(sos_vars) > 0:
            if '.' in sos_vars[0]:
                sigt = f".{sos_vars[0].split('.')[1]}"
            else:
                sigt = ''
            sosv_path = Path(out_dir) / f'sosv{sigt}-{temp}_{start_yr}.tif'
            sosd_path = Path(out_dir)/f'sosd{sigt}-{temp}_{start_yr}.tif'
            if (not sosd_path.is_file) or (not sosv_path.is_file()):            
                sos, sosv = get_greenup(ts_stack_padded, ds_stack_padded, peaks[1], method='thresh')
                sosd = sos.dt.dayofyear
                ## add 365 to doy if it passes into the next year to avoid jump in values from Dec31 to Jan 1
                ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
                sosd1 = sosd.where(sosd >= start_doy -40, sosd + 365)
                sosd2 = sosd1.where(peaks[0] > 0, 0).astype('int16')   
            else:
                sosd2 = sosd_path
                sosv = sosv_path
            for v in peak_vars:
                if v.startswith('sosd'):
                    add_var_to_stack(sosd2,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)  
                if v.startswith('sosv'):
                    add_var_to_stack(sosv,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)

        eos_vars = [var for var in peak_vars if var.startswith(tuple(['eos','ros','los']))]
        if len(eos_vars) > 0:
            if '.' in eos_vars[0]:
                sigt = f".{eos_vars[0].split('.')[1]}"
            else:
                sigt = ''
            eosv_path = Path(out_dir)/f'eosv{sigt}-{temp}.tif'
            eosd_path = Path(out_dir)/f'eosd{sigt}-{temp}.tif'
            if (not eosd_path.is_file()) or (not eosv_path.is_file()):
                eos, eosv = get_senescence(ts_stack_padded, ds_stack_padded, peaks[1], method='thresh')
                eosd = eos.dt.dayofyear
                ## add 365 to doy if it passes into the next year to avoid jump in values from Dec31 to Jan 1
                ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
                eosd1 = eosd.where(eosd >= start_doy, eosd + 365)
                eosd2 = eosd1.where(peaks[0] > 0, 0).astype('int16')  
            else:
                eosd2 = eosd_path
                eosv = eosv_path
            for v in peak_vars:
                if v.startswith('eosd'):
                    add_var_to_stack(eosd2,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)
                if v.startswith('eosv'):
                    add_var_to_stack(eosv,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)

        rate_vars = [var for var in peak_vars if var.startswith(tuple(['rog','rol','los']))]
        if len(rate_vars) > 0:
            with gw.open(Path(out_dir) /f'posd-{temp}.tif') as posd2:
                pass
            for v in rate_vars:
                if v.startswith('rog') or v.startswith('los'):
                    with gw.open(Path(out_dir)/f'sosd-{temp}.tif') as sosd2:
                        with gw.open(Path(out_dir)/f'sosv-{temp}.tif') as sosv:
                            rog = sosd2.where(sosd2 == 0, (posv - sosv) / (posd2 - sosd2))
                            rog = rog.astype('int16')
                if v.startswith('rog'):
                    add_var_to_stack(rog,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)
                if v.startswith('ros') or v.startswith('los'):
                    with gw.open(Path(out_dir)/f'eosd-{temp}.tif') as eosd2:      
                        with gw.open(Path(out_dir)/f'eosv-{temp}.tif') as eosv:
                            ros = eosd2.where(eosd2 == 0, (posv - eosv) / (posd2 - eosd2)) 
                            ros = ros.astype('int16')
                if v.startswith('ros'):       
                    add_var_to_stack(ros,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)
                if v.startswith('los'):
                    with gw.open(Path(out_dir)/f'sosd-{temp}.tif') as sosd2:
                        with gw.open(Path(out_dir)/f'eosd-{temp}.tif') as eosd2:
                            los = eosd2 - sosd2
                    add_var_to_stack(los,v,attrs,out_dir,comp_band_names,ras_list,**gw_args)

    return comp_band_names,ras_list
    

def prep_ts_variable_bands(si_vars, ts_stack,ds_stack, out_dir,temp,start_doy,comp_band_names,ras_list,nodata_in,ppaths,**gw_args):
    
    ## sort images by date just in case, but if coming from smoothed time series, should be in order already
    sts_stack = [ts for ds, ts in sorted(zip(ds_stack, ts_stack))]
    sds_stack = [ds for ds, ts in sorted(zip(ds_stack, ts_stack))]
    
    with gw.open(sts_stack, time_names = sds_stack) as src0:
        attrs = src0.attrs.copy()

    #replace nodata_in with np.nan   
    logger.debug(f'replacing nodata values of {nodata_in} with Nan')
    src = src0.where((src0 != int(nodata_in)) & (src0 != 10000))

    ## remove glcm portion of variables in case it exists. For glcms, the underlying variable will be processed, then the glcm after.
    if isinstance(si_vars,str):
        si_vars = [si_vars]
    no_glcm = r'\.glcm.*'
    si_vars = [re.sub(no_glcm, '', siv) for siv in si_vars]
    logger.info(f'calculating bands: {si_vars}')

    ## if single date as si_var (e.g. for texture variables):
    singl_img_vars = [v for v in si_vars if isinstance(v.split('.')[0],int)]
    for v in singl_img_vars:
        singdate = v.split('.')[0]
        if len(str(singdate)) == 3:
            ## this is doy -- get full date
            fulldate = [d for d in sds_stack if ds.endswith(str(singdate))][0]
            logger.info(f' adding image from {fulldate} to stack')
        elif len(str(singdate)) == 7:
            fulldate = singdate
        else: 
            logger.warning(f'cannot parse {singdate} to get corresponing image -- should be doy or YYYdoy')   
        sing_img = ds.sel(time=v.split('.')[0])  ##TODO: verify that time is in YYYYdoy and not YYYY-mm-dd
        add_var_to_stack(sing_img,str(singdate),attrs,out_dir,comp_band_names,ras_list,**gw_args)
        
    if any(v in si_vars for v in [f'maxv-{temp}',f'amp-{temp}', f'maxd-{temp}', f'maxdc-{temp}']):
        mmax = src.max(dim='time').astype('int16')
        if f'maxv-{temp}' in si_vars:
            add_var_to_stack(mmax,f'maxv-{temp}', attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if any(v in si_vars for v in [f'minv-{temp}',f'amp-{temp}', f'mind-{temp}',f'mindc-{temp}']):
        mmin = src.min(dim='time').astype('int16')
        if f'minv-{temp}' in si_vars:
            add_var_to_stack(mmin,f'minv-{temp}', attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'amp-{temp}' in si_vars:
        aamp = (mmax - mmin).astype('int16')
        add_var_to_stack(aamp,f'amp-{temp}', attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'avg-{temp}' in si_vars or f'cv-{temp}' in si_vars:
        aavg = src.mean(dim='time').astype('int16')
        if f'avg-{temp}' in si_vars:
            add_var_to_stack(aavg,f'avg-{temp}', attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'med-{temp}' in si_vars:
        mmed = src.median(dim='time').astype('int16')
        add_var_to_stack(mmed,f'med-{temp}', attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'sd-{temp}' in si_vars or f'cv-{temp}' in si_vars:
        sstd = src.std(dim='time').astype('int16')
        if f'sd-{temp}' in si_vars:
            add_var_to_stack(sstd,f'sd-{temp}', attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'cv-{temp}' in si_vars:
        ccv = ((sstd * 1000) / aavg).astype('int16')
        add_var_to_stack(ccv,f'cv-{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if any(v in si_vars for v in [f'maxd-{temp}', f'maxdc-{temp}']):
        maxd = src.idxmax(dim='time',skipna=True)
        maxd1 = maxd.dt.dayofyear.astype('int16')
        ## add 365 to doy if it passed into the next year to avoid jump in values from Dec31 to Jan 1
        ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
        maxd2 = maxd1.where(maxd1 >= start_doy, maxd1 + 365)
        #max_date2 = max_date2.astype('int16')
        if f'maxd-{temp}' in si_vars:
            add_var_to_stack(maxd2,f'maxd-{temp}', attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'maxdc-{temp}' in si_vars:
        maxd_360 = 2 * np.pi * maxd1/365
        maxd_cos = 100 * (np.cos(maxd_360) + 1)
        maxd_cos = maxd_cos.astype('int16')
        add_var_to_stack(maxd_cos,f'maxdc-{temp}',attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'mind-{temp}' in si_vars or f'mindc-{temp}' in si_vars:
        mind = src.idxmin(dim='time',skipna=True)
        mind1 = mind.dt.dayofyear.astype('int16')
        ## add 365 to doy if it passed into the next year to avoid jump in values from Dec31 to Jan 1
        ## (keep everything >= start_doy as is. If passes into next year, doy will be < start_doy, so add 365)
        mind2 = mind1.where(mind1 >= start_doy, mind1 + 365)
        if f'mind-{temp}' in si_vars:
            add_var_to_stack(mind2,f'mind-{temp}', attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'mindc-{temp}' in si_vars:
        mind_360 = 2 * np.pi * mind1 / 365
        mind_cos = 100 * (np.cos(mind_360) + 1)
        mind_cos = mind_cos.astype('int16')
        add_var_to_stack(mind_cos,f'mindc-{temp}', attrs,out_dir,comp_band_names,ras_list,**gw_args)
    if f'deltaobs-{temp}' in si_vars:
        num_images = len(sts_stack)
        src_c = src.chunk({"time": -1})
        srcf = src_c.ffill(dim='time')
        srcff = srcf.bfill(dim='time')
        ## shift moves data one to the right 
        delta = srcff - srcff.shift(time=1)
        negdelta = xr.where(delta < 0, 1, 0)
        numnegs = negdelta.sum(dim='time').fillna(0).astype('int16')
        numnegs1 = (100 * numnegs / num_images).astype('int16')
        add_var_to_stack(numnegs1,f'numnegdelta-{temp}', attrs,out_dir,comp_band_names,ras_list,**gw_args)
    
    logger.debug(f"comp_band_names = {comp_band_names}, ras_list={ras_list}")
    return comp_band_names,ras_list

