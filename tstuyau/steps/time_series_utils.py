from __future__ import division

import os

from ..handler import logger
# from .date_utils import prepare_x

import satsmooth as sm
from satsmooth.utils import nd_to_columns, columns_to_nd, prepare_x
from satsmooth.anc import AncSmoothers
from satsmooth.anc._lowess_smooth import lowess_smooth
from satsmooth.utils.tfill import SFill

import numpy as np


def check_low_values(yd):

    """
    Checks for low outlier values in a flattened 1d array

    Args:
        yd (1d array)

    Returns:
        (1d array)
    """

    # Check low values.
    yd[yd == 0] = np.nan

    # Check for low values missed
    #   by outlier detection.
    y_perc_5 = np.nanpercentile(yd, 5, axis=1)[:, np.newaxis]
    perc_low = (y_perc_5 - yd) / ((y_perc_5 + yd) / 2.0)
    low_idx = np.where(perc_low > 0.25)

    if low_idx[0].shape[0] > 0:
        yd[low_idx] = np.nan

    # Recode any remaining nan values to 0.
    yd[np.isnan(yd)] = 0.0

    return yd


def smooth(X,
           y,
           start,
           end,
           rule='D',
           skip='WMS',
           write_skip=10,
           nodata=0,
           interp_step=True,
           max_window=61,
           min_window=21,
           mid_g=0.5,
           r_g=-10.0,
           mid_k=0.5,
           r_k=-10.0,
           mid_t=0.5,
           r_t=15.0,
           max_outlier_days1=120,
           max_outlier_days2=120,
           min_outlier_values=7,
           min_gap_length=-999,
           outlier_iters=1,
           dev_thresh1=0.2,
           dev_thresh2=0.2,
           remove_outliers=True,
           t_smoothing=True,
           spt_smoothing=True,
           k=3,
           t=3,
           sigma_time=0.1,
           sigma_color=0.1,
           sigma_space=0.1,
           n_iters=2,
           n_iters_spt=1,
           max_peak_weight=1.2,
           min_valley_weight=0.8,
           n_jobs=1,
           chunksize=10,
           extreme_rel_thresh=0.0,
           return_indexed=True,
           prefill_gaps=True,
           prefill_return=False,
           prefill_iters=5,
           prefill_wmax=25,
           prefill_wmin=9,
           prefill_min_count=20,
           prefill_dev_thresh=0.03,
           prefill_max_days=30,
           prefill_max_years=2,
           verbose=1,
           smooth_method='dts',
           cspline_s=0.5,
           lowess_w=15,
           whittaker_s=0.5,
           whittaker_order=2,
           sg_w=7,
           sg_p=3,
           fourier_period=365.25,
           fourier_poly_order=1,
           fourier_harmonic_order=1,
           dbl_max_iters=1000,
           dbl_reltol=1e-8):

    """
    Smooths a time series

    Args:
        X (list): A list of datetime objects.
        y (3d array): A stack of 2d images.
        start (str): The desired times series start date.
        end (str): The desired times series end date.
        rule (Optional[str]): The time series resampling rule.
        skip (Optional[str]): The time series skip procedure.
        write_skip (Optional[int]): The time series write skip interval.
        nodata (int | float): The 'no data' value.
        interp_step (Optional[bool])
        max_window (Optional[int]): The maximum AGS window size.
        min_window (Optional[int]): The minimum AGS window size.
        mid_g (Optional[float])
        r_g (Optional[float])
        mid_k (Optional[float])
        r_k (Optional[float])
        mid_t (Optional[float])
        r_t (Optional[float])
        max_outlier_days1 (Optional[int])
        max_outlier_days2 (Optional[int])
        min_outlier_values (Optional[int])
        min_gap_length (Optional[int])
        outlier_iters (Optional[int])
        dev_thresh1 (Optional[float])
        dev_thresh2 (Optional[float])
        remove_outliers (Optional[bool])
        t_smoothing (Optional[bool]): Whether to apply temporal smoothing.
        spt_smoothing (Optional[bool]): Whether to apply spatial-temporal smoothing.
        k (Optional[int]): The spatial window size for spatial-temporal smoothing.
        t (Optional[int]): The temporal window size for spatial-temporal smoothing.
        sigma_time (Optional[float]): The temporal sigma for spatial-temporal smoothing.
        sigma_color (Optional[float]): The color sigma for spatial-temporal smoothing.
        sigma_space (Optional[float]): The spatial sigma for spatial-temporal smoothing.
        n_iters (Optional[int]): The number of bilateral smoothing iterations to fit to the upper envelope.
        n_iters_spt (Optional[int]): The number of spatial-temporal smoothing iterations.
        max_peak_weight (Optional[float]): The peak stretch weight.
        min_valley_weight (Optional[float]): The valley stretch weight.
        n_jobs (Optional[int]): The number of parallel workers.
        extreme_rel_thresh (Optional[float]): The extreme relative deviation threshold.
        return_indexed (Optional[bool])
        prefill_gaps (Optional[bool]): Whether to pre-fill gaps.
        prefill_iters Optional[int])
        prefill_wmax Optional[int])
        prefill_wmin Optional[int])
        prefill_min_count Optional[int])
        prefill_dev_thresh Optional[float])
        prefill_max_days Optional[int])
        prefill_max_years Optional[int])
        verbose (Optional[int]): The verbosity level.
        smooth_method (Optional[str]): Choices are ['dts', 'csp', 'sg', 'dbl', 'gpr', 'harm', 'wh'].
        cspline_s (Optional[float]): The Cubic spline smoothing parameter.
        lowess_w (Optional[int]): The lowess window size.
        whittaker_s (Optional[float]): The Whittaker smoothing parameter.
        whittaker_order (Optional[int]): The Whittaker smoothing order.
        sg_w (Optional[float]): The S-G window size.
        sg_p (Optional[int]): The S-G order.
        fourier_period (Optional[float]): The Fourier period.
        fourier_poly_order (Optional[int]): The Fourier polynomial order.
        fourier_harmonic_order (Optional[int]): The Fourier harmonic order.

    Returns:
        Data as unsigned 16-bit type with 0-10,000 range (2d array)
    """
    xinfo = prepare_x(X, start, end, rule=rule, skip=skip, write_skip=write_skip)

    if y.max() == 0:
        return xinfo, np.array([], dtype='uint16')

    dims, nrows, ncols = y.shape

    y[y <= 0.005] = nodata

    if prefill_gaps:

        if verbose > 0:
            logger.info('  Filling gaps ...')

        sf = SFill(start=start,
                   end=end,
                   rule=rule,
                   skip=skip,
                   wmax=prefill_wmax,
                   wmin=prefill_wmin,
                   nodata=nodata,
                   n_iters=prefill_iters,
                   max_days=prefill_max_days,
                   max_years=prefill_max_years,
                   min_count=prefill_min_count,
                   dev_thresh=prefill_dev_thresh,
                   num_threads=n_jobs,
                   chunksize=chunksize)

        xdates = np.ascontiguousarray([dt.toordinal() for dt in X], dtype='float64')
        xdates = xdates - xdates[0] + 1

        yotl = sm.remove_outliers(xdates,
                                  np.ascontiguousarray(nd_to_columns(y.copy(), dims, nrows, ncols), dtype='float64'),
                                  no_data_value=0.0,
                                  max_outlier_days1=120,
                                  max_outlier_days2=120,
                                  outlier_iters=1,
                                  dev_thresh1=0.1,
                                  dev_thresh2=0.1,
                                  n_jobs=n_jobs,
                                  chunksize=chunksize)

        y = sf.impute(X, columns_to_nd(yotl, dims, nrows, ncols))

        if prefill_return:
            return y

    # Reshape to 2d for smoothing
    y = nd_to_columns(y, dims, nrows, ncols)

    if extreme_rel_thresh != 0:

        y[y == nodata] = np.nan

        # Median over the time series, ignoring missing values
        med = np.nanmedian(y, axis=1)[:, np.newaxis]

        # Deviation from the median
        med_diff = (y - med) / med
        med_diff[np.isnan(med_diff)] = nodata

        # Remove high outliers
        y[np.isnan(y) | (med_diff > extreme_rel_thresh)] = nodata

    if verbose > 0:
        logger.info(f'  Smoothing with {n_jobs} processes out of {os.cpu_count()} available ...')

    #import ipdb;ipdb.set_trace() 
    if t_smoothing:
        if verbose > 0:
            logger.info(f' smooth_method is {smooth_method}')

        if smooth_method == 'dts':

            # if smoothing_interval == 'daily':

            interpolator = sm.LinterpMulti(xinfo.xd, xinfo.xd_smooth)
            indices = np.ascontiguousarray(xinfo.skip_idx + xinfo.start_idx, dtype='uint64')

            # elif smoothing_interval == 'sparse':
            #
            #     interpolator = sm.LinterpMulti(xinfo.xd, xinfo.xd_interp)
            #     return_indexed = False
            #     indices = np.array([1], dtype='uint64')
            #     # indices = np.ascontiguousarray(xinfo.skip_interp_idx, dtype='uint64')
            #
            # else:
            #
            #     logger.exception('  The smoothing interval is not supported')
            #     raise NameError

            # Prophet harmonics
            # pdata = y.copy()
            # pdata[pdata == 0] = np.nan
            #
            # m = ProphetHarmonics(xinfo.dates,
            #                      crop_period=365.25*0.5,
            #                      yearly_seasonality=1,
            #                      uncertainty_samples=0,
            #                      fourier_order=1)
            #
            # pres = m.predict(pdata, max_workers=n_jobs, chunksize=n_jobs*2)
            #
            # pdata = None
            #
            # pres = interpolator.interpolate(pres,
            #                                 fill_no_data=True,
            #                                 no_data_value=0,
            #                                 n_jobs=n_jobs)

            xdates = np.ascontiguousarray([dt.toordinal() for dt in X], dtype='float64')
            xdates = xdates - xdates[0] + 1

           ## try commenting this out
            #y = sm.remove_outliers(xdates, np.ascontiguousarray(y, dtype='float64'), no_data_value=nodata, max_outlier_days1=max_outlier_days1, max_outlier_days2=max_outlier_days2, outlier_iters=1, dev_thresh1=dev_thresh1, dev_thresh2=dev_thresh2, n_jobs=n_jobs, chunksize=chunksize)

            w = 21
            wh = int(w / 2.0)

            #y = lowess_smooth(ordinals=np.ascontiguousarray(np.pad(xinfo.xd, (wh, wh), mode='linear_ramp'), dtype='int64'),y=np.pad(sm.interp2d(np.ascontiguousarray(y, dtype='float64'),no_data_value=nodata,n_jobs=n_jobs), ((0, 0), (wh, wh)), mode='reflect'),w=w,n_jobs=n_jobs,chunksize=chunksize)[:, wh:-wh]

            # 1) Interpolates between missing values
            # 2) Checks for and removes outliers
            # 3) Interpolates to a new, denser grid
            # 4) Smooths the data
            # 5) Indexes to weekly series
            y = interpolator.interpolate_smooth(np.ascontiguousarray(y, dtype='float64'),
                                                interp_step=interp_step,
                                                fill_no_data=True,
                                                no_data_value=nodata,
                                                remove_outliers=remove_outliers,
                                                max_outlier_days1=max_outlier_days1,
                                                max_outlier_days2=max_outlier_days2,
                                                min_outlier_values=min_outlier_values,
                                                min_gap_length=min_gap_length,
                                                outlier_iters=outlier_iters,
                                                dev_thresh1=dev_thresh1,
                                                dev_thresh2=dev_thresh2,
                                                return_indexed=return_indexed,
                                                indices=indices.copy(),
                                                max_window=max_window,
                                                min_window=min_window,
                                                mid_g=mid_g,
                                                r_g=r_g,
                                                mid_k=mid_k,
                                                r_k=r_k,
                                                mid_t=mid_t,
                                                r_t=r_t,
                                                sigma_color=sigma_color,
                                                n_iters=n_iters,
                                                n_jobs=n_jobs,
                                                chunksize=chunksize)

        elif smooth_method == 'csp':

            smt = AncSmoothers(xinfo,
                               columns_to_nd(np.ascontiguousarray(y, dtype='float64'),
                                             dims,
                                             nrows,
                                             ncols),
                               pad=max_window,
                               index_by_indices=True,
                               remove_outliers=remove_outliers,
                               max_outlier_days1=max_outlier_days1,
                               max_outlier_days2=max_outlier_days2,
                               min_outlier_values=min_outlier_values,
                               dev_thresh1=dev_thresh1,
                               dev_thresh2=dev_thresh2,
                               n_jobs=n_jobs)

            y = smt.csp(s=cspline_s, chunksize=chunksize)
            y = nd_to_columns(y, *y.shape)

        elif smooth_method == 'lw':

            smt = AncSmoothers(xinfo,
                               columns_to_nd(np.ascontiguousarray(y, dtype='float64'),
                                             dims,
                                             nrows,
                                             ncols),
                               pad=max_window,
                               index_by_indices=True,
                               remove_outliers=remove_outliers,
                               max_outlier_days1=max_outlier_days1,
                               max_outlier_days2=max_outlier_days2,
                               min_outlier_values=min_outlier_values,
                               dev_thresh1=dev_thresh1,
                               dev_thresh2=dev_thresh2,
                               n_jobs=n_jobs)

            y = smt.lw(w=lowess_w, chunksize=chunksize)

            y = nd_to_columns(y, *y.shape)

        elif smooth_method == 'wh':

            smt = AncSmoothers(xinfo,
                               columns_to_nd(np.ascontiguousarray(y, dtype='float64'),
                                             dims,
                                             nrows,
                                             ncols),
                               pad=max_window,
                               index_by_indices=True,
                               remove_outliers=remove_outliers,
                               max_outlier_days1=max_outlier_days1,
                               max_outlier_days2=max_outlier_days2,
                               min_outlier_values=min_outlier_values,
                               dev_thresh1=dev_thresh1,
                               dev_thresh2=dev_thresh2,
                               n_jobs=n_jobs)

            y = smt.wh(s=whittaker_s, order=whittaker_order, chunksize=chunksize)
            y = nd_to_columns(y, *y.shape)

        elif smooth_method == 'ac':

            smt = AncSmoothers(xinfo,
                               columns_to_nd(np.ascontiguousarray(y, dtype='float64'),
                                             dims,
                                             nrows,
                                             ncols),
                               pad=max_window,
                               index_by_indices=True,
                               remove_outliers=remove_outliers,
                               max_outlier_days1=max_outlier_days1,
                               max_outlier_days2=max_outlier_days2,
                               min_outlier_values=min_outlier_values,
                               dev_thresh1=dev_thresh1,
                               dev_thresh2=dev_thresh2,
                               n_jobs=n_jobs)

            y = smt.ac(pad=10, chunksize=chunksize)
            y = nd_to_columns(y, *y.shape)

        elif smooth_method == 'harm':

            smt = AncSmoothers(xinfo,
                               columns_to_nd(np.ascontiguousarray(y, dtype='float64'),
                                             dims,
                                             nrows,
                                             ncols),
                               pad=max_window,
                               index_by_indices=True,
                               remove_outliers=remove_outliers,
                               max_outlier_days1=max_outlier_days1,
                               max_outlier_days2=max_outlier_days2,
                               min_outlier_values=min_outlier_values,
                               dev_thresh1=dev_thresh1,
                               dev_thresh2=dev_thresh2,
                               n_jobs=n_jobs)

            y = smt.harm(period=fourier_period,
                         poly_order=fourier_poly_order,
                         harmonic_order=fourier_harmonic_order)

            y = nd_to_columns(y, *y.shape)

        elif smooth_method == 'gpr':

            smt = AncSmoothers(xinfo,
                               columns_to_nd(np.ascontiguousarray(y, dtype='float64'),
                                             dims,
                                             nrows,
                                             ncols),
                               pad=max_window,
                               index_by_indices=True,
                               remove_outliers=remove_outliers,
                               max_outlier_days1=max_outlier_days1,
                               max_outlier_days2=max_outlier_days2,
                               min_outlier_values=min_outlier_values,
                               dev_thresh1=dev_thresh1,
                               dev_thresh2=dev_thresh2,
                               n_jobs=n_jobs)

            y = smt.gpr()
            y = nd_to_columns(y, *y.shape)

        elif smooth_method == 'sg':

            smt = AncSmoothers(xinfo,
                               columns_to_nd(np.ascontiguousarray(y, dtype='float64'),
                                             dims,
                                             nrows,
                                             ncols),
                               pad=max_window,
                               index_by_indices=True,
                               remove_outliers=remove_outliers,
                               max_outlier_days1=max_outlier_days1,
                               max_outlier_days2=max_outlier_days2,
                               min_outlier_values=min_outlier_values,
                               dev_thresh1=dev_thresh1,
                               dev_thresh2=dev_thresh2,
                               n_jobs=n_jobs)

            y = smt.sg(w=sg_w, p=sg_p)
            y = nd_to_columns(y, *y.shape)

        elif smooth_method == 'dbl':

            smt = AncSmoothers(xinfo,
                               columns_to_nd(np.ascontiguousarray(y, dtype='float64'),
                                             dims,
                                             nrows,
                                             ncols),
                               pad=max_window,
                               index_by_indices=True,
                               remove_outliers=remove_outliers,
                               max_outlier_days1=max_outlier_days1,
                               max_outlier_days2=max_outlier_days2,
                               min_outlier_values=min_outlier_values,
                               dev_thresh1=dev_thresh1,
                               dev_thresh2=dev_thresh2,
                               n_jobs=n_jobs)

            y = smt.dbl(max_iters=dbl_max_iters,
                        reltol=dbl_reltol,
                        beta1=0.9,
                        beta2=0.99,
                        chunksize=chunksize)

            y = nd_to_columns(y, *y.shape)

    else:

        # Simple linear interpolation
        interpolator = sm.LinterpMulti(xinfo.xd, xinfo.xd)

        return_indices = xinfo.skip_orig_idx.copy()

        y = interpolator.interpolate(np.ascontiguousarray(y, dtype='float64'),
                                     interp_step=interp_step,
                                     fill_no_data=True,
                                     no_data_value=nodata,
                                     remove_outliers=remove_outliers,
                                     max_outlier_days1=max_outlier_days1,
                                     max_outlier_days2=max_outlier_days2,
                                     min_outlier_values=min_outlier_values,
                                     outlier_iters=outlier_iters,
                                     dev_thresh1=dev_thresh1,
                                     dev_thresh2=dev_thresh2,
                                     return_indexed=return_indexed,
                                     indices=return_indices,
                                     n_jobs=n_jobs,
                                     chunksize=chunksize)

    # if smoothing_interval == 'daily':
    #     xlen_smooth = xinfo.skip_slice.shape[0]
    # else:
    # y = y[:, :xinfo.skip_slice.shape[0]]
    xlen_smooth = y.shape[1]

    if spt_smoothing:

        if verbose > 0:
            logger.info('  Spatial-temporal smoothing ...')

        th = int(t / 2.0)

        # Spatial-temporal smoothing
        return xinfo, np.uint16(sm.spatial_temporal(np.pad(columns_to_nd(y[:, :xlen_smooth], xlen_smooth, nrows, ncols),
                                                           pad_width=((th, th), (0, 0), (0, 0)),
                                                           mode='edge'),
                                                    k=k,
                                                    t=t,
                                                    sigma_time=sigma_time,
                                                    sigma_color=sigma_color,
                                                    sigma_space=sigma_space,
                                                    n_jobs=n_jobs,
                                                    chunksize=chunksize,
                                                    n_iters=n_iters_spt,
                                                    max_peak_weight=max_peak_weight,
                                                    min_valley_weight=min_valley_weight)[th:-th] * 10000.0)

    else:

        return xinfo, np.uint16(columns_to_nd(y,
                                              xlen_smooth,
                                              nrows,
                                              ncols) * 10000.0)
