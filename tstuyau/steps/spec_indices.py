from ..handler import logger
import numpy as np

SI_DICT = {'avi':{'band_names':['red', 'nir']},
        'evi2':{'band_names':['red', 'nir']},
        'gcvi':{'band_names':['green', 'nir']},
        'kndvi':{'band_names':['red', 'nir']},
        'kndviV0':{'band_names':['red', 'nir']},
        'nbr':{'band_names':['nir', 'swir2']},
        'nbrV0':{'band_names':['nir', 'swir2']},
        'nbr2':{'band_names':['swir1', 'swir2']},
        'wi':{'band_names':['red', 'swir1']},
        'ndvi':{'band_names':['red', 'nir']},
        'ndviV0':{'band_names':['red', 'nir']},
        'ndwi':{'band_names':['nir', 'green']},
        'ndmi':{'band_names':['swir1', 'nir']},
        'ndmiV0':{'band_names':['swir1', 'nir']},
        'savi':{'band_names':['red', 'nir']},
        'msavi':{'band_names':['red', 'nir']},
        'blue':{'band_names':['blue']}, 
        'green':{'band_names':['green']}, 
        'red':{'band_names':['red']}, 
        'nir':{'band_names':['nir']}, 
        'swir1':{'band_names':['swir1']},
        'swir2':{'band_names':['swir2']},
        'char':{'band_names':['blue','green','red']},
        'mirbi':{'band_names':['swir1','swir2']},
        'cai':{'band_names':['swir1','swir2']},
        'bai':{'band_names':['nir','red']},
        'baim':{'band_names':['nir','swir2']},
        'gemi':{'band_names':['nir','red']}
        }

def calculate_char_index(red_val, green_val, blue_val, scale_factor=10000):
    
    s = scale_factor  ## max value of valid data. Usually 10000 in saved data, but 1 in original data
    r, g, b = red_val/s, green_val/s, blue_val/s
    vsum = r + g + b
    bg = abs(b - g)
    br = abs(b - r)
    rg = abs(r - g)
    maxdiff = np.maximum(np.maximum(bg,br), rg)
    si = s * (vsum + (maxdiff * 15))
    ## set white values (where r, g, & b are all 1) to no data
    mask = np.abs(vsum - 3.0) > 1e-9
    index_val = si/3 * mask 
    return index_val

def calculate_raw_index(nir_val, b2_val, si, params=None):
    '''
    This is for on-the-fly calculations at a specific point. 
    For raster level calculations, xarray/geowombat methods in SpecIndices class are used
    this assumes that input nodata vals = 10000 (or 0) and output nodata=0. TODO: control nodata with parameter
    '''

    ## s is the scale factor, which is the maximum value of valid data. Usually 10000 in saved data, but 1 in original data
    s = 10000
    if params:
        if params['masking']['maxval']:
            s = params['masking']['maxval']
    
    spec_index = si.split('.')[0]
    
    if spec_index == 'evi2': ## b2=red
        index_val =  s * 2.5 * ((nir_val/s - b2_val/s) / (nir_val/s + 1.0 + 2.4 * b2_val/s))
    elif spec_index in ['gcvi','ndvi','nbr','nbr2']: ## gcvi: b2=green, ndvi: bs=red, nbr: b2=swir2, nbr2: b2=swir2 and nir_val is actually swir1
        orig_idx = (nir_val - b2_val) / ((nir_val + b2_val) + 1e-9)
        index_val = np.where(orig_idx == 0, 0, (orig_idx + 1) * s/2)
    elif spec_index == 'cai': #b2=swir2 and nir_val is actually swir1
        index_val = s *  b2_val / (nir_val + 1e-9)
    elif spec_index == 'savi':  ## b2=red  lfactor = 0-1, 0=very green, 1=very arid. .5 most common. Some use negative vals for arid env)
        if '.' in si:
            lfactor = si.split('.')[1]
            if lfactor.startswith('n'):
                lfactor = -1 * int(lfactor.split('n')[1])
        else:
            lfactor = .5  
        orig_idx = (1 + int(lfactor)) * ((nir_val/s - b2_val/s) / (nir_val/s + b2_val/s + int(lfactor)))
        if int(lfactor) > 0:
            index_val = np.where(orig_idx == 0, 0, s * (1.0 + orig_idx) / 2.0)
        else:
            logger.warning('OOPS  -- TODO: finish the negative case here')
    elif spec_index == 'msavi': ## b2=red
        index_val =  s/2 * (2 * nir_val/s + 1) - ((2 * nir_val/s + 1)**2 - 8*(nir_val/s - b2_val/s))**1/2
        index_val = np.where(nir_val < s, index_val, 0)
    elif spec_index == 'ndmi': ## b2=swir1
        orig_idx = s * (nir_val - b2_val) / ((nir_val + b2_val) + 1e-9)
        index_val = (orig_idx + 1) * s/2
    elif spec_index == 'kndvi':
        index_val = s * (np.tanh(((nir_val - b2_val) / ((nir_val + b2_val) + 1e-9))**2))
    elif spec_index == 'bai': ## b2=swir2
        index_v = s / ((0.06 - nir_val/s)**2 + (0.1 - b2_val/s)**2)
        index_val = np.where(b2_val + nir_val > 0, index_v, 0) 
    elif spec_index == 'baim': ## b2=swir2
        index_v = s / ((0.05 - nir_val/s)**2 + (0.2 - b2_val/s)**2)
        index_val = np.where(b2_val + nir_val > 0, index_v, 0) 
    elif spec_index == 'gemi':
        n = (2 * ((nir_val/s)**2 - (b2_val/s)**2) + 1.5*(nir_val/s) + 0.5*(b2_val/s)) / ((nir_val/s) + (b2_val/s) + .5)
        index_val = s * (n*(1 - .25*n) - ((b2_val/s - .125) / (1 - b2_val/s)))
    elif spec_index == 'ndwi': ## b2=green
        orig_idx = s * (b2_val - nir_val) / ((b2_val + nir_val) + 1e-9)
        index_val = (orig_idx + 1) * s/2
    elif spec_index == 'wi':  #note nir_val is actually swir1 here. b2=red
        tot = nir_val + b2_val
        tot = np.where((nir_val > 0) & (nir_val < 1), tot, 0)
        if tot > .5:
            index_val = .001 * s
        else:
            index_val = s * (1.0 - (tot / 0.5))
    elif spec_index == 'mirbi': #note nir_val is actually swir1 here. b2=swir2
        index_v = s/5 * ((10 * b2_val/s) - (9.8 * nir_val/s) + 2)
        index_val = np.where(b2_val + nir_val == 0, 0, index_v) 
    elif spec_index == 'nir':
        index_val = np.where(nir_val < s, nir_val, 0)
    elif spec_index in ['swir1','swir2','red','green']:
        index_val = np.where(b2_val < s, b2_val, 0)
   
    return index_val
    

def calc_si_gw(data_src, params):
    '''
    Calculates vegetation indices with methods in geowombat
    '''
    ## args:
    si = params['reconstruct']['si']
    nodata = 0
    scale_factor = 1
    
    if si not in ['avi','evi','evi2','kndvi','nbr','ndvi','wi']:
        logger.warning("  The calc_si_gw method only supports vegetation indices defined in geowombat ('avi','evi','evi2','kndvi','nbr','ndvi','wi')")
        raise LookupError

    if si == 'avi':
        si_data = data_src.gw.avi(nodata=nodata, scale_factor=scale_factor)
    elif si == 'evi':
        si_data = data_src.gw.evi(nodata=nodata, scale_factor=scale_factor)        
    elif si == 'evi2':
        si_data = data_src.gw.gcvi(nodata=nodata, scale_factor=scale_factor)
    elif si == 'nbr':
        si_data = data_src.gw.nbr(nodata=nodata, scale_factor=scale_factor)
    elif si == 'kndvi':
        si_data = data_src.gw.kndvi(nodata=nodata, scale_factor=scale_factor)
    elif si == 'wi':
        si_data = data_src.gw.wi(nodata=nodata, scale_factor=scale_factor)
    elif si == 'ndvi':
        si_data = data_src.gw.ndvi(nodata=nodata, scale_factor=scale_factor)
 
    return si_data


class SpecIndices(object):
    '''
    Calculates spectral indices with methods defined here. 
    
    New indices can be added by creating methods here and also adding the index to the SI_DICT above.
    Note: input and output data are in decimals. (if rasters are stored as integers, values are divided/multiplied by 10000 before/after processing).
    '''

    def __init__(self, si_name):
        self.si_name = si_name

    def __call__(self, data, extra_param):
        return np.nan_to_num(getattr(self, self.si_name)(data,extra_param), nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _normV0(b1, b2):
        '''
        not rescaled from [0,1] in previous versions. 
        Kept here for compatibility with CELPy0 maps,
        but negative numbers not treated correctly.
        '''
        return (b2 - b1) / ((b2 + b1) + 1e-9)
        
    @staticmethod
    def _norm(b1, b2):
        norm = (b2 - b1) / ((b2 + b1) + 1e-9)
        ## rescale from [-1,1] to [0,1]
        return np.where(norm == 0, 0, (1 + norm)/2)

    @staticmethod
    def _scale_min_max(xv, min_in, max_in, min_out, max_out):

        return ((((max_out - min_out) * (xv - min_in)) / (max_in - min_in)) + min_out)\
                    .clip(min_out, max_out)

    def gcvi(self, data,extra_param=None):
        res = (data[1] / (data[0] + 1e-9)) - 1.0
        return self._scale_min_max(res, 0, 10, 0, 1)
        
    @staticmethod

    def evi2(data,extra_param=None):
        if extra_param is None:
            lfactor = 1.0
        elif extra_param.startswith('n'):
            lfactor = -1*int(extra_param[1:])/100
        else:
            lfactor = int(extra_param)/100
            
        return 2.5 * ((data[1] - data[0]) / (data[1] + lfactor + 2.4 * data[0]))

    @staticmethod
    def savi(data,extra_param=None):
        '''
        Soil-adjusted ndvi (= ndvi if L-factor = 0)
        lfactor (passed in as value from 0 to 100) is usually 0-1, 0=very green, 1=very arid. .5 (default) is most common. 
        Some use negative vals for arid env (passed in as n0 to n100)
        '''
        if extra_param is None:
            lfactor = .5
        elif extra_param.startswith('n'):
            lfactor = -1*int(extra_param[1:])/100
        else:
            lfactor = int(extra_param)/100

        #logger.debug(f'calculating savi with l-factor of {lfactor}')
        nir = data[1].astype(np.float32)
        red = data[0].astype(np.float32)
        num = nir - red
        denom = nir + red + lfactor
        denom = np.where(denom == 0, 1e-10, denom)
        savi = (1.0 + lfactor) * (num / denom)

        if lfactor >= 0:
            savi_rescaled = np.where(savi == 0, 0, (1.0 + savi) / 2.0)
        else:
            #bound_edge = 1.0 / (1.0 + abs(lfactor))  
            #gmin = -1.0 * bound_edge
            #gmax = 1.0 * bound_edge
            gmin = -5.0
            gmax = 5.0
            denom_scale = gmax - gmin
            savi_clamped = np.clip(savi, gmin, gmax)
            savi_rescaled = np.where(savi == 0, 0, (savi_clamped - gmin) / denom_scale)
        return savi_rescaled
        

    @staticmethod
    def msavi(data,extra_param=None):
        '''
        version of Savi without the need to select an L-factor
        (The values in the msavi are all constants) 
        '''
        return 1/2 * (2 * data[1] + 1) - ((2 * data[1] + 1)**2 - 8*(data[1] - data[0]))**1/2

    ## Legacy indices -- no longer in use:
    def nbrV0(self, data,extra_param=None):
        return self._normV0(data[1], data[0])
    def ndmiV0(self, data,extra_param=None):
        res = (data[1] - data[0]) / (data[1] + data[0] + 1e-9)
        return np.where(res < 0, .01, res)
    def ndviV0(self, data,extra_param=None):
        res = (data[1] - data[0]) / (data[1] + data[0] + 1e-9)
        return np.where(res < .000001, 0, res)
        
    def kndvi(self, data,extra_param):
        ## note that kndvi is bounded 0-1 by design, so uses norm without rescale here
        return np.tanh(self._normV0(data[1], data[0])**2)

    def nbr(self, data,extra_param=None):
        return self._norm(data[1], data[0])

    def nbr2(self, data,extra_param=None):
        return self._norm(data[1], data[0])
        
    def ndvi(self, data,extra_param=None):
        return self._norm(data[0], data[1])
            
    def ndwi(self, data,extra_param=None):
        return self._norm(data[1], data[0])

    def ndmi(self, data,extra_param=None):
        return self._norm(data[1], data[0])

    @staticmethod
    def cai(data,extra_param=None):
        return (data[1]) / (data[0] + 1e-9)

    @staticmethod
    def mirbi(data,extra_param):
        idx = .2 * (10*data[1] -(9.8*data[0]) + 2)
        return np.where(data[1] + data[0] == 0, 0, idx) 

    @staticmethod
    def bai(data,extra_param):
        idx =  1/((0.06 - data[0])**2 + (0.1 - data[1])**2)
        return np.where(data[1] + data[0] == 0, 0, idx)

    @staticmethod
    def baim(data,extra_param):
        idx =  1/((0.05 - data[0])**2 + (0.2 - data[1])**2)
        return np.where(data[1] + data[0] == 0, 0, idx)

    @staticmethod
    def gemi(data,extra_param):
        n = (2 * (data[0]**2 - data[1]**2) + 1.5*data[0] + 0.5*data[1]) / (data[0] + data[1] + .5)
        return n * (1 - 0.25*n) - ((data[1] - 0.125) / (1 - data[1]))

    @staticmethod
    def red(data,extra_param=None):
        return data[0]
    @staticmethod
    def blue(data,extra_param=None):
        return data[0]
    @staticmethod
    def green(data,extra_param=None):
        return data[0]
    @staticmethod
    def nir(data,extra_param=None):
        return data[0]
    @staticmethod
    def swir1(data,extra_param=None):
        return data[0]
    @staticmethod
    def swir2(data,extra_param=None):
        return data[0]

    @staticmethod
    def wi(data,extra_param):
        res = data[1] + data[0]
        return np.where(res == 0, 0,
             (np.where(res > 0.5, .001, 1.0 - (res / 0.5))))
    @staticmethod
    def char(data,extra_param):
        vsum = data[0] + data[1] + data[2]
        bg = np.abs(data[0] - data[1])
        br = np.abs(data[0] - data[2])
        rg = np.abs(data[2] - data[1])
        maxdiff = np.maximum(np.maximum(bg, br),rg)
        
        return np.where(vsum==3, 0, (vsum + (maxdiff*15))/3)


        
        