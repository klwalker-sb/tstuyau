from abc import ABC, abstractmethod
from datetime import datetime
import concurrent.futures

try:
    from prophet import Prophet
except:
    pass

import numpy as np
import pandas as pd
import numexpr as nx
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm import tqdm as tqdm_normal


class BaseAbstract(ABC):

    @abstractmethod
    def predict(self, yarray, max_workers=1, in_notebook=False, chunksize=10):
        raise NotImplementedError


class HarmonicsAbstract(ABC):

    @abstractmethod
    def fit_predict_array(self, y, max_workers=1):
        raise NotImplementedError

    @abstractmethod
    def fit_predict(self, y):
        raise NotImplementedError

    @abstractmethod
    def init_model(self):
        raise NotImplementedError


class Harmonics(BaseAbstract):

    def predict(self, yarray, max_workers=1, in_notebook=False, chunksize=10):

        # tqdm = tqdm_notebook if in_notebook else tqdm_normal

        futures = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:

            # Submit futures
            # futures = [executor.submit(self.fit_predict_array, yrow) for yrow in yarray]
            for res in executor.map(self.fit_predict_array, yarray, chunksize=chunksize):
                futures.append(res)

            # for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            #     futures.append(f.result())

        return np.array(futures, dtype='float64')


class DLMHarmonics(HarmonicsAbstract, Harmonics):

    def __init__(self):
        self.yhat = None
        self.m = None

    def fit_predict_array(self, y, max_workers=1):

        self.fit_predict(y)

        return self.yhat

    def fit_predict(self, y):
        self.yhat = None

    def init_model(self):
        pass


class ProphetProps(object):

    @property
    def as_datetime(self):
        return [datetime.strptime(dt, '%Y-%m-%d') for dt in self.ds]


class ProphetHarmonics(HarmonicsAbstract, Harmonics, ProphetProps):

    """
    Args:
        dt (list): A list of datetime objects.
        yearly_seasonality (Optional[int])
        period (Optional[float])
        fourier_order (Optional[int])
        uncertainty_samples (Optional[int])
        workers_per_series (Optional[int])

    Example:
        >>> from tstuyau.steps.prophet import ProphetModel
        >>>
        >>> m = ProphetModel(X, yearly_seasonality=1, uncertainty_samples=0, fourier_order=2)
        >>> res = m.fit_predict_array(ts_data, max_workers=8)
    """

    def __init__(self,
                 dt,
                 yearly_seasonality=1,
                 crop_period=365.25/2,
                 fourier_order=1,
                 seasonality_mode='additive',
                 uncertainty_samples=100,
                 workers_per_series=1):

        self.ds = [datetime.strftime(dt_date, '%Y-%m-%d') for dt_date in dt]
        self.yearly_seasonality = yearly_seasonality
        self.crop_period = crop_period
        self.fourier_order = fourier_order
        self.seasonality_mode = seasonality_mode
        self.uncertainty_samples = uncertainty_samples
        nx.set_num_threads(workers_per_series)

        self.m = None
        self.yhat = None

    def init_model(self):

        self.m = Prophet(yearly_seasonality=self.yearly_seasonality,
                         weekly_seasonality=False,
                         daily_seasonality=False,
                         seasonality_mode=self.seasonality_mode,
                         uncertainty_samples=self.uncertainty_samples)

        # Seasonality for double cropping
        self.m.add_seasonality(name='crop_season',
                               period=self.crop_period,
                               fourier_order=self.fourier_order)

    def fit_predict_array(self, y, adjust_lower=0.01):

        self.fit_predict(y)

        if self.uncertainty_samples == 0:
            return self.yhat
        else:
            return self.adjust_outliers(y)

    def fit_predict(self, y):

        self.init_model()
        df = self._create_data(y)
        est = self.m.fit(df).predict(df)

        if self.uncertainty_samples > 0:
            self.yhat_lower = est.yhat_lower.values
            self.yhat_upper = est.yhat_upper.values

        self.yhat = est.yhat.values

    def _create_data(self, y):

        dataframe = pd.DataFrame(data=self.ds, columns=['ds'])
        dataframe.loc[:, 'y'] = y
        return dataframe

    def adjust_outliers(self, y, adjust_lower=0, adjust_upper=0):

        y_ = y.copy()

        return np.where(y_ < self.yhat_lower+adjust_lower, self.yhat_lower+adjust_lower,
                        np.where(y_ > self.yhat_upper+adjust_upper, self.yhat_upper+adjust_upper, y_))
