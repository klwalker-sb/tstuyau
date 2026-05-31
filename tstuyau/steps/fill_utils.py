from datetime import datetime
from collections import defaultdict

import geowombat as gw
import satsmooth as sm

import numpy as np
import xarray as xr
import numba as nb
from tqdm import trange


def cdf(histogram):

    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    return cdf / float(cdf.max())


@nb.jit(nogil=True)
def calc_lut(src_cdf, ref_cdf):

    lookup_table = np.zeros(256, dtype='uint8')
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


def match_histograms(src, tmp, dst):

    src = ((src * 0.0001) * 255).astype('uint8')
    tmp = ((tmp * 0.0001) * 255).astype('uint8')
    dst = ((dst * 0.0001) * 255).astype('uint8')

    src_hist, bins = np.histogram(src.flatten(), bins=254, range=[1, 255])
    tmp_hist, bins = np.histogram(tmp.flatten(), bins=254, range=[1, 255])

    # Compute the normalized cdf for the source and reference image
    src_cdf = cdf(src_hist)
    tmp_cdf = cdf(tmp_hist)

    # Make a separate lookup table for each color
    lut = calc_lut(src_cdf, tmp_cdf)

    # Use the lookup function to transform the colors of the original
    # source image
    dst_transformed = cv2.LUT(dst, lut).reshape(dst.shape)

    return ((dst_transformed / 255.0) * 10000.0).astype('float64')


def readjust(data, indices):

    """
    Iteratively re-adjusts an array with original indices
    """

    # CLAHE adjustment
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    #
    # for bidx in range(0, data.shape[0]):
    #     data[bidx] = ((clahe.apply(((data[bidx]*0.0001)*255.0).astype('uint8')) / 255.0) * 10000.0).astype('float64')

    for imidx in indices.keys():

        for bidx in range(0, data.shape[0]):

            data_band = data[bidx]

            idx = indices[imidx][bidx]

            mask = np.ones(data_band.shape, dtype=np.bool)
            mask = np.where(data_band == 0, False, mask)
            mask[idx] = False

            data_band[idx] = match_histograms(data_band[idx],
                                              data_band[mask],
                                              data_band[idx])

            data[bidx] = data_band

    return data


def fill_array(bap, imidx, full, new, indices):

    """
    Fills an array by histogram matching
    """

    indices_sub = defaultdict(list)

    for bidx in range(0, full.shape[0]):

        new_band = new[bidx]

        full[bidx] = np.where((bap.score[bidx] >= bap.max_score[bidx]) & (new_band != 0), new_band, full[bidx])

    #     # Histogram matching
    #     idx = np.where((full[bidx] != 0) & (new_band != 0))
    #
    #     indices_sub[bidx] = idx
    #
    #     new[bidx] = match_histograms(new_band[idx],
    #                                  full[bidx][idx],
    #                                  new_band)
    #
    #     full[bidx] = np.where(full[bidx] == 0, new[bidx], full[bidx])
    #
    # indices[imidx] = indices_sub

    return full, indices


def sort_range(dates, data, time_index, n_references, max_days):

    """
    Sorts images by nearest date to reference

    Args:
        dates (1d array-like): The dates.
        data (3d array): The data to slice and sort.
        time_index (int): The current time reference.
        n_references (int): The maximum number of references.
        max_days (int): The maximum number of days difference.

    Returns:
        3d array
    """

    dims = data.shape[0]

    if time_index - n_references < 0:
        dstart = 0
    else:

        if time_index + n_references + 1 >= dims:
            dstart = time_index
        else:
            dstart = time_index - n_references

    if dstart + n_references + 1 >= dims:
        dend = dims
    else:
        dend = dstart + n_references + 1

    ref_idx = list(range(dstart, dend))

    # filter references by date
    try:
        target_date = dates[time_index]
    except:
        return None, np.array([0])

    n_dates = len(dates)

    ref_idx = np.array([ref_idxer for ref_idxer in ref_idx if (ref_idxer < n_dates) and (abs((target_date - dates[ref_idxer]).days) <= max_days)], dtype='int64')

    if ref_idx.shape[0] == 0:
        return None, np.array([0])
    else:

        dates = dates[ref_idx[np.argsort(np.abs(ref_idx - time_index))]]

        days = [abs((dt - dates[0]).days) for dt in dates]

        # Sort by nearest to reference ``time_index``
        return days, data[ref_idx[np.argsort(np.abs(ref_idx - time_index))]]


def fill_gaps(ldate_dt, landsat_list, near_indices, ppaths, params):

    filled_data = []

    for band in trange(1, 7):

        fill_data = []
        dates = []

        # Attempt to fill the moderate-res
        for imidx, near_idx in enumerate(near_indices):

            mkdate_str = landsat_list[near_idx].name.split('_')[3][:8]
            mkdate_dt = datetime.strptime(mkdate_str, '%Y%m%d')

            if abs(mkdate_dt.timetuple().tm_yday - ldate_dt.timetuple().tm_yday) > params['fusion']['fill_max_days']:
                continue

            # Do not use the target date
            if ldate_dt == mkdate_dt:
                continue

            mkmask_image = ppaths.masks.joinpath(mkdate_str + '.tif')

            with gw.open(landsat_list[near_idx]) as mres_k_src, \
                    gw.open(mkmask_image) as mmask_k_src:

                attrs = mres_k_src.attrs.copy()

                mres_k_src = xr.where(mmask_k_src.sel(band=1) > params['masking']['min_mask'], params['nodata'], mres_k_src). \
                    transpose('band', 'y', 'x').assign_attrs(**attrs)

                mres_k_src = mres_k_src.gw.set_nodata(params['nodata'], 0, (0, 1), 'float64', scale_factor=0.0001)

                mres_0_data = mres_k_src.sel(band=band).data.compute(num_workers=params['num_workers'])
                mres_0_data[np.isnan(mres_0_data)] = 0

                fill_data.append(mres_0_data)

            dates.append(mkdate_dt)

        fill_data = np.array(fill_data, dtype='float64')
        dates = np.array(dates)

        ndims, nrows, ncols = fill_data.shape

        for didx in range(0, ndims):

            gap_days, gap_array = sort_range(dates, fill_data, didx, 5, params['fusion']['fill_max_days'])

            # The target date needs some data
            if gap_array[0].max() > 0:

                gdims = gap_array.shape[0]

                if len(fill_data.shape) > 3:
                    gap_array = np.float64(gap_array)
                else:
                    gap_array = np.float64(gap_array.reshape(gdims, 1, nrows, ncols))

                # Fill gaps at time ``didx``
                fill_data[didx] = np.squeeze(sm.fill_gaps(gap_array,
                                                          days=np.ascontiguousarray(gap_days, dtype='float64'),
                                                          wmax=params['fusion']['prefill_wmax'],
                                                          wmin=params['fusion']['prefill_wmin'],
                                                          nodata=0.0,
                                                          min_prop=params['fusion']['min_prefill_prop'],
                                                          min_rsquared=params['fusion']['min_prefill_rsquared'],
                                                          n_jobs=params['num_workers']))

        filled_data.append(fill_data)

    return dates, np.array(filled_data, dtype='float64')
