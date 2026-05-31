import numpy as np
import xarray as xr
import cv2
from numba import njit, prange, set_num_threads
from tqdm import tqdm


# @jit
# def calc_medoids(data):
#
#     medoid_results = np.zeros(data.shape[0], dtype=data.dtype)
#
#     for idx in range(0, data.shape[0]):
#
#         sample = data[idx]
#         diff = sample[:, None] - sample
#         ssum = np.einsum('ij,ij->ij', diff, diff)
#
#         if not np.isfinite(sample).any():
#
#             sample[np.isnan(sample)] = 0.0
#             medoid_results[idx] = sample[(ssum**0.5).sum(axis=1).argmin()]
#
#         else:
#
#             dist = np.nansum(ssum**0.5, axis=1)
#
#             dist[np.isnan(sample)] = np.nan
#             medoid_results[idx] = sample[np.nanargmin(dist)]
#
#     return medoid_results


@njit(nopython=True)
def argmin(data, sums, ip, ncols):

    arg_pos = 0
    min_val = 1e9

    for p in range(0, ncols):

        if not np.isnan(data[ip, p]):

            val = sums[p]

            if val < min_val:

                arg_pos = p
                min_val = val

    return data[ip, arg_pos]


@njit(nopython=True)
def calc_dists(samples, ip, ncols, dists_, out_sum_):

    for i in range(0, ncols):

        for j in range(0, ncols):

            dists_[i, j] = abs(samples[ip, i] - samples[ip, j])

            if not np.isnan(dists_[i, j]):
                out_sum_[j] += dists_[i, j]

    return out_sum_


@njit(nopython=True, parallel=False)
def compute_samples(samples, nrows, ncols, dists, out_sum, argmins, dists_copy, out_sum_copy):

    for s in range(0, nrows):

        dists[...] = dists_copy
        out_sum[...] = out_sum_copy

        out_sum = calc_dists(samples, s, ncols, dists, out_sum)

        argmins[s] = argmin(samples, out_sum, s, ncols)

    return argmins


def calc_medoids(samples, n_threads=1):

    set_num_threads(n_threads)

    nrows = samples.shape[0]
    ncols = samples.shape[1]

    dists = np.zeros((ncols, ncols), dtype='float64')
    dists_copy = dists.copy()
    out_sum = np.zeros(ncols, dtype='float64')
    out_sum_copy = out_sum.copy()
    argmins = np.zeros(nrows, dtype='float64')

    return compute_samples(samples, nrows, ncols, dists, out_sum, argmins, dists_copy, out_sum_copy)


def get_medoid(stack_list, bands, num_workers):

    res_ = xr.concat(stack_list, dim='time')\
                .transpose('time', 'band', 'y', 'x')

    medoids_ = np.zeros((len(bands), res_.gw.nrows, res_.gw.ncols), dtype='float64')

    windows = list(res_.gw.windows(row_chunks=512, col_chunks=512))
    niters = res_.gw.n_windows(row_chunks=512, col_chunks=512)

    # Warm-up numba
    __ = calc_medoids(np.random.rand(1, 10), n_threads=1)

    for bidx, band in enumerate(bands):

        res_band = res_.sel(band=band)

        for w in tqdm(windows, total=niters):

            slicer1 = (slice(0, None), slice(w.row_off, w.row_off+w.height), slice(w.col_off, w.col_off+w.width))
            slicer2 = (slice(bidx, bidx+1), slice(w.row_off, w.row_off+w.height), slice(w.col_off, w.col_off+w.width))

            X = res_band[slicer1]\
                    .transpose('time', 'y', 'x')\
                    .stack(z=('y', 'x'))\
                    .transpose()\
                    .data\
                    .compute(num_workers=num_workers)

            medoids_[slicer2] = calc_medoids(X, n_threads=num_workers)\
                                    .reshape(w.height, w.width)

    return medoids_


class BAP(object):

    """
    Best available pixel
    """

    def __init__(self, data_shape, max_cloud_dist=60, max_days=10, baseline_score=0.1):

        self.max_cloud_dist = max_cloud_dist
        self.max_days = max_days
        self.baseline_score = baseline_score

        # Best available pixel
        self.bap = np.zeros(data_shape, dtype='float64')

        # Maximum score
        self.max_score = np.zeros(data_shape, dtype='float64')

        # Total value
        # self.total_value = np.zeros(data_shape, dtype='float64')

        # Count of clear observations over all iterations
        # self.count = np.zeros(data_shape, dtype='float64')
        # self.spec_diff = np.zeros(data_shape, dtype='float64')
        # self.dates = np.zeros(data_shape, dtype='uint16')
        # self.sensor_code = np.zeros(data_shape, dtype='uint8')

        self.score = None

        self.weights = {'cloud': 1.0,
                        'doy': 1.0,
                        'year': 1.0,
                        'spec': 2.0,
                        'haze': 1.0,
                        'sza': 1.0}

    def finalize(self):
        return self.bap.copy()

    def calc_score(self,
                   trg_data=None,
                   ref_data=None,
                   sza=None,
                   dta=None,
                   dtb=None,
                   name=None,
                   mdata=None,
                   celly=None):

        """
        Args:
            trg_data (ndarray): The data to score.
            ref_data (ndarray): The reference data used for the spectral score.
            sza (ndarray): The solar zenith angle.
            dta (datetime): The datetime object used for the DoY and year scores.
            dtb (datetime): The datetime object used for the DoY and year scores.
            name (str): The data name with the sensor.
            mdata (Optional[ndarray]): The data mask.
            celly (Optional[float]): The cell size used for the cloud distance.
        """

        doy_diff = abs(dta.timetuple().tm_yday - dtb.timetuple().tm_yday)
        year_diff = abs(dta.year - dtb.year)

        if isinstance(mdata, np.ndarray):
            cloud_score_ = self.cloud_score(mdata, celly)
        else:
            cloud_score_ = 1.0

        if isinstance(sza, np.ndarray):
            sza_score_ = self.sza_score(sza)
        else:
            sza_score_ = 1.0

        if trg_data.shape[0] > 1:
            haze_score_ = self.haze_score(trg_data)
        else:
            haze_score_ = 1.0

        doy_score_ = self.doy_score(doy_diff)
        year_score_ = self.year_score(year_diff)
        spec_score_ = self.spec_score(ref_data, trg_data)
        # sensor_score_ = 1.0 if 'lc08' in name.lower() else 0.25

        self.score = cloud_score_ * self.weights['cloud'] + \
                     doy_score_ * self.weights['doy'] + \
                     year_score_ * self.weights['year'] + \
                     spec_score_ * self.weights['spec'] + \
                     haze_score_ * self.weights['haze'] + \
                     sza_score_ * self.weights['sza']
                     # sensor_score_ * self.weights['sensor']

        self.score /= sum(list(self.weights.values()))

        if isinstance(mdata, np.ndarray):
            self.score = np.where(mdata >= 2, 0, self.score)

        #############
        # self.count = np.where((self.score > self.max_score) & (mdata < 2), self.count + 1, self.count)
        # self.spec_diff = np.where((self.score > self.max_score) & (mdata < 2), spec_score_, self.spec_diff)
        # self.dates = np.where((self.score > self.max_score) & (mdata < 2), int(dtb.strftime('%Y%m%d')), self.dates)
        #############

        # Update the best available pixel
        if isinstance(mdata, np.ndarray):
            self.bap = np.where((self.score > self.max_score) & (mdata < 2), trg_data, self.bap)
        else:
            self.bap = np.where(self.score > self.max_score, trg_data, self.bap)

        self.bap[np.isnan(self.bap) | np.isinf(self.bap)] = 0.0

        self.max_score = np.maximum(self.max_score, self.score)

    @staticmethod
    def gaussian(x, tar, std):

        """
        Gaussian function for day of year score

        Args:
            x (float)
            tar (float): The target (mean).
            std (float): The standard deviation.
        """

        a = 1.0 / (std * np.sqrt(2.0 * np.pi))
        b = np.exp(-0.5 * ((x - tar) / std) ** 2)

        return a * b

    @staticmethod
    def logistic(x, x0, r):

        """
        Same as Eq. 8 in:

            Frantz et al. (2017) Phenology-adaptive pixel-based compositing using
                optical earth observation imagery, RSE.

                where,
                    1 / (1 + exp(-10 / d_req * (d - d_req / 2)))
        """

        return 1.0 / (1.0 + np.exp(-r * (x - x0)))

    @staticmethod
    def sigmoid(x, a, b):

        """
        Sigmoid function for distance to 'no data'

        Args:
            x (float)
            a (float): The center.
            b (float): The steepness.

        >>> dist_to_nodata(dists, 30, -0.2)
        """

        s = 1.0 / (1.0 + np.exp(b * (x - a)))

        return 1.0 * (s - s.min()) / (s.max() - s.min())

    def year_score(self, year_diff):

        if year_diff == 0:
            return 1.0
        elif year_diff == 1:
            return 0.68
        elif year_diff == 2:
            return 0.42
        else:
            return self.baseline_score

    def spec_score(self, ref_data, data):

        """
        sigmoid(|delta|)
        """

        # Score differences [0,1], where 1 is smaller spectral difference
        spec_score_ = self.logistic(np.abs(ref_data - data), 0.2, -20)

        return np.where(spec_score_ <= self.baseline_score, self.baseline_score, spec_score_)

    def cloud_score(self, mdata, celly):

        """
        Distance from cloud edge
        """

        binary = np.where((mdata < 2) | (mdata == 255), 1, 0)

        if binary.min() > 0:
            return np.ones(binary.shape, dtype='float64')

        # Get the pixel distance from cloud edges
        dists = np.float64(cv2.distanceTransform(np.uint8(binary), cv2.DIST_L2, 5))

        # Get the distance in meters
        dists *= celly

        # Score clouds [0,1], where 1 is farther from cloud edges
        cloud_score_ = self.logistic(dists/self.max_cloud_dist, 0.5, 10.0)

        return np.where(cloud_score_ <= self.baseline_score, self.baseline_score, cloud_score_)

    def doy_score(self, doy_diff):

        """
        Distance from target day of year
        """

        # Normalized gaussian
        doy_score_ = self.gaussian(float(doy_diff), 0, self.max_days*0.33) / self.gaussian(0, 0, self.max_days*0.33)

        return doy_score_ if doy_score_ > self.baseline_score else self.baseline_score

    def haze_score(self, data):

        """
        Args:
            data (ndarray): [blue, green, red, nir, swir1, swir2].
        """

        hot = data[0] - 0.5 * data[2] - 0.08
        hot = 1.0 - (1.0 / (1.0 + np.exp((1.0 / 0.05) * (hot + 0.075))))
        hot_score = self.logistic(hot, 0.5, -10.0)

        return np.where(hot_score <= self.baseline_score, self.baseline_score, hot_score)

    def sza_score(self, sza):

        theta_req = 1.0

        # Closer to nadir (0) will give a higher score
        sza_score = 1.0 / (1.0 + np.exp((10.0 / theta_req) * (np.deg2rad(sza) - theta_req / 2.0)))

        return np.where(sza_score <= self.baseline_score, self.baseline_score, sza_score)
