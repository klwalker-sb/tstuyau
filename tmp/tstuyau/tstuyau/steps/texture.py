from pathlib import Path
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
from scipy.stats import entropy #note after scikit_image 0.25.0 entropy can be calculated directly with others
#from scipy.misc import imresize
from rasterio.fill import fillnodata
import rasterio as rio
from .image_utils import rescale_band
from .project import ProjectPaths
from ..handler import logger


def glcm_slidingwin(in_ras, size_win, texture, cv=256, th=0, normed=False):
    '''
    create textures using the GLCM function of Skimage
    run individual glcm filters using numpy sliding windows.
    This avoids multithreading cython issues that cause segmentation fault
    '''

    logger.info(f' making {texture} raster...\n')
    
    pad_amt = size_win // 2
    padded_ras = np.pad(in_ras, pad_amt, mode='reflect')

    windows = np.lib.stride_tricks.sliding_window_view(padded_ras, (size_win, size_win))

    h, w = in_ras.shape
    glcm_map = np.zeros((h, w), dtype=np.float32)

    for r in range(h):
        for c in range(w):
            win_2d = np.array(windows[r, c], dtype=np.uint8, copy=True)
                
            glcm = graycomatrix(win_2d, distances=[1], angles=[th], 
                levels=cv, symmetric=True, normed=normed)

            glcm_map[r, c] = graycoprops(glcm, texture)[0, 0]
        
    return np.atleast_3d(glcm_map)

'''
def glcm_cython(in_ras, size_win, texture):
    """
    create textures using the GLCM function of Skimage and ndimage.generic_filter for moving window
    this is throwing a Segmentation fault error. the above works fine.
    """
    def homogeneity_fun(window_flat):
        
        win_2d = window_flat.reshape((size_win, size_win)).copy(order='C')
        win_int = win_2d.astype(np.uint8)
        
        glcm = graycomatrix(in_ras,  distances=[1], angles=[0], levels=256, symmetric = True, normed = True)
        return graycoprops(glcm, 'homogeneity')[0,0]
        
    def correlation_fun(window_flat):

        win_2d = window_flat.reshape((size_win, size_win)).copy(order='C')
        win_int = win_2d.astype(np.uint8)
            
        glcm = graycomatrix(in_ras,  distances=[1], angles=[0], levels=256, symmetric = True, normed = True)
        return graycoprops(glcm, 'correlation')[0,0]
    
    def contrast_fun(window_flat):

        win_2d = window_flat.reshape((size_win, size_win)).copy(order='C')
        win_int = win_2d.astype(np.uint8)
            
        glcm = graycomatrix(in_ras,  distances=[1], angles=[0], levels=256, symmetric = True, normed = True)
        return graycoprops(glcm, 'contrast')[0,0]
     
    def  dissimilarity_fun(window_flat):

        win_2d = window_flat.reshape((size_win, size_win)).copy(order='C')
        win_int = win_2d.astype(np.uint8)
            
        glcm = graycomatrix(in_ras,  distances=[1], angles=[0], levels=256, symmetric = True, normed = True)
        return graycoprops(glcm, 'dissimilarity')[0,0]

    out_stack = []
    ## apply to moving window to glcm calcs
    # 'reflect' mode handles edges safely without shrinking the window size
    if texture == 'variance':
        logger.info('making variance raster...\n')
        glcm_prod = ndimage.generic_filter(in_ras, np.var, size=size_win)
    if texture == 'contrast':
        logger.info('making contrast raster...\n')
        glcm_prod = ndimage.generic_filter(in_ras, contrast_fun, size=size_win)
    if texture == 'dissimilarity':
        logger.info('making dissimilarity raster...\n')
        glcm_prod = ndimage.generic_filter(in_ras, dissimilarity_fun, size=size_win)
    if texture == 'correlation':
        logger.info('making correlation raster...\n')
        glcm_prod = ndimage.generic_filter(in_ras, correlation_fun, size=size_win)
    if texture == 'homogeneity':
        logger.info('making homogeneity raster...\n')
        glcm_prod = ndimage.generic_filter(in_ras, homogeneity_fun, size=size_win, mode='reflect')
    if texture == 'entropy':
        logger.info('making entropy raster...\n')
        glcm_prod = ndimage.generic_filter(in_ras, entropy, size=size_win)
    
    return glcm_prod
'''
##########################################################################################################
##alternative method
## framework from https://stackoverflow.com/questions/42459493/sliding-window-in-python-for-glcm-calculation

def offset(length, angle):
    """Return the offset in pixels for a given length and angle"""
    dv = length * np.sign(-np.sin(angle)).astype(np.int32)
    dh = length * np.sign(np.cos(angle)).astype(np.int32)
    return dv, dh

def crop(img, center, win):
    """Return a square crop of img centered at center (side = 2*win + 1)"""
    row, col = center
    side = 2*win + 1
    first_row = row - win
    first_col = col - win
    last_row = first_row + side    
    last_col = first_col + side
    return img[first_row: last_row, first_col: last_col]

def cooc_maps(img, center, win, d=[1], theta=[0], levels=256):
    """
    Return a set of co-occurrence maps for different d and theta in a square 
    crop centered at center (side = 2*w + 1)
    """
    shape = (2*win + 1, 2*win + 1, len(d), len(theta))
    cooc = np.zeros(shape=shape, dtype=np.int32)
    row, col = center
    Ii = crop(img, (row, col), win)
    for d_index, length in enumerate(d):
        for a_index, angle in enumerate(theta):
            dv, dh = offset(length, angle)
            Ij = crop(img, center=(row + dv, col + dh), win=win)
            cooc[:, :, d_index, a_index] = encode_cooccurrence(Ii, Ij, levels)
    return cooc

def encode_cooccurrence(x, y, levels=256):
    """Return the code corresponding to co-occurrence of intensities x and y"""
    return x*levels + y

def decode_cooccurrence(code, levels=256):
    """Return the intensities x, y corresponding to code"""
    return code//levels, np.mod(code, levels)    

def compute_glcms(cooccurrence_maps, levels=256):
    """Compute the cooccurrence frequencies of the cooccurrence maps"""
    Nr, Na = cooccurrence_maps.shape[2:]
    glcms = np.zeros(shape=(levels, levels, Nr, Na), dtype=np.float64)
    for r in range(Nr):
        for a in range(Na):
            values, counts = np.unique(cooccurrence_maps[:, :, r, a], return_counts=True)
            table = np.column_stack((values, counts))
            codes = table[:, 0]
            freqs = table[:, 1]/float(table[:, 1].sum())
            i, j = decode_cooccurrence(codes, levels=levels)
            glcms[i, j, r, a] = freqs
    return glcms

def compute_props(glcms, props=('homogeneity','entropy')):
    """Return a feature vector corresponding to a set of GLCM"""
    Nr, Na = glcms.shape[2:]
    features = np.zeros(shape=(Nr, Na, len(props)))
    for index, prop_name in enumerate(props):
        features[:, :, index] = graycoprops(glcms, prop_name)
    return features.ravel()

def haralick_features(img, win, props, d, theta=0, levels=256):
    """Return a map of Haralick features (one feature vector per pixel)"""
    rows, cols = img.shape
    margin = win + max(d)
    arr = np.pad(img, margin, mode='reflect')
    n_features = len(d) * len(theta) * len(props)
    print(f'n_features = {n_features}')
    feature_map = np.zeros(shape=(rows, cols, n_features), dtype=np.float64)
    for m in range(rows):
        for n in range(cols):
            coocs = cooc_maps(arr, (m + margin, n + margin), win, d, theta, levels)
            glcms = compute_glcms(coocs, levels)
            feature_map[m, n, :] = compute_props(glcms, props)
    return feature_map
    ## TODO: average theta outputs if len(theta) > 1 
#########################################################################################################################

def make_glcm(base_img, params=None, si_var=None, win=None, covals=None, th=None, print_out=True, out_path=None):
    '''
    Run the GLCM textures and print file for each in list (if <print_out>=True, or return single glcm
    The "ndimage.generic_filter" funtion perform the moving window of size <win>

    glcm parameters (type, window, #co-vals and angle) can be passed as lists in <si_var>, <win>, <covals> and <theta> 
    or by parsing the si_var (either from the params or entered directly as <si_var>
       <in_var>.glcm.<type>.<window>.<co-vals>.<theta>  e.g. med.glcm.variance.w5.c100.th0

    theta is the angle quadrant: th0=0, th1=pi/4, th2=pi/2, th3=3*pi/4
     '''
    
    if si_var and isinstance(si_var,list): 
        ## passing the values in directly here
        gt = si_var  ## direct string or list of types to run (e.g. ['variance','contrast','dissimilarity','homogeneity','entropy']
        size_win = win  ## list of window values to run (e.g. [5,11,25] -- or just single value
        covals = covals ## list of # of possible values to rescale data to (e.g. [32,64,100,255]  -- or just single value
        thetas = th  ## list of angles to run [th0,th1,th2,th3] or 0 if none
    else:
        ## otherwise type is parsed from si variable (e.g. med.glcm.variance.w5.c100.th0)
        if si_var:
            si_vars = si_var
        else:
            #si = params['feature_model']['spec_indices'][0]
            si_vars = params['feature_model']['si_vars'][0] 

        if si_vars.split('.')[1] != 'glcm':
            logger.warning(f'{si_vars} is not a glcm variable')
    
        glcm_type = si_vars.split('.')[2]    
        '''
        if glcm_type.startswith('var'):
            gt = 'variance'
        elif glcm_type.startswith('cont'):
            gt = 'contrast'
        elif glcm_type.startswith('dis'):
            gt = 'dissimilarity'
        elif glcm_type.startswith('corr'):
            gt = 'correlation'
        elif glcm_type.startswith('homo'):
            gt = 'homogeneity'
        elif glcm_type.startswith('ent'):
            gt = 'entropy'
        elif glcm_type == 'asm':
            gt = 'ASM'
        elif glcm_type == 'energy':
            gt = 'energy'
        else:
            gt = 'all'
        '''

        gt = [glcm_type]
        in_var = si_vars.split('.')[0]
        size_win = int(si_vars.split('.')[3].split('w')[1])
        covals= int(si_vars.split('.')[4].split('c')[1])
        thetas = si_vars.split('.')[5].split('-')[0]

    if (print_out) or (gt=='all') or (len(gt)>1):
        if out_path == None:
            cell = params['grids']
            ppaths = ProjectPaths(params, grid=cell) 
            out_dir =  ppaths.comp / si
        else:
            out_dir = out_path.parent

    if isinstance(covals, int):
        covals = [covals]
    if isinstance(thetas, str):
        thetas = [thetas]
    if isinstance(size_win, int):
        size_win = [size_win]
    numrasts = len(gt) * len(covals) * len(thetas) * len(size_win)
        
    ##TODO: derive rgb and band from variable if using a raw image
    with rio.open(base_img) as in_ras:
        num_bands = in_ras.count
        profile = in_ras.profile.copy()
    if num_bands > 2:
        if rgb == True:
            #transform multiband image to single intensity with rgb bands
            with rio.open(base_img) as in_ras:
                raster = in_ras.read()
            r = raster[0,:,:]
            g = raster[1,:,:]
            b = raster[2,:,:]
            # Transform RGB to intensity (or lightness) of the HSL color scales
            # Preserves distances and angles from the geometry of the RGB cube
            ing_in = imresize( (0.2989 * r) + (0.5870 * g) + (0.1140 * b), 100 )
        else:
            with rio.open(base_img) as in_ras:
                img_in = in_ras.read(band)
                profile = in_ras.profile.copy()
    else:
        with rio.open(base_img) as in_ras:
            img_in = in_ras.read(1)
            profile = in_ras.profile.copy()

    ## images must be dtype uint8 (0 to 255) for methods like ndimage to work. 
    ##    But the co-occurence matrix will be more stable with less possible combinations. We use <covals> to rescale the data 0-255, 0-100, 0-64, etc.
 
    for cv in covals:
        with rio.open(base_img) as in_ras:    
            scaled_ras = rescale_band(img_in, maxval=cv, profile=in_ras.profile, outpath=None).astype(np.uint8)

        #    temp_dir = out_path /'temp_rescaling'        
        #temp_ras = rescale_band(base_img, maxvalcv, profile=None, outpath=temp_dir)
        #with rio.open(temp_ras) as ras_src:
        #    scaled_ras = ras_src.read(1).astype(np.uint8)

        out_files = []
        for tex in ['homogeneity','dissimilarity','contrast','correlation','entropy','variance','energy','ASM']:
            if (gt == ['all']) or (gt.startswith(t[:3]) or (gt.upper() == t):  
                for win in size_win:
                    for th in thetas:
                        if th == 0:
                            th = [0]
                        elif th == 1: 
                            th = [np.pi/4]
                        elif th == 2:
                            th = [np.pi/2]
                        elif th == 3:
                            th = [np.3*pi/4]

                        if not params['feature_models']['stacked']: 
                            #glcm_ras = glcm_cython(scaled_ras, win, texture=t)  #getting segmentation fault 
                            glcm_ras = glcm_slidingwin(scaled_ras, win, tex, cv=cv, th=th, normed=False)
                            
                            ## change dim ordering from scipy to rio:
                            glcm_rio = np.moveaxis(glcm_ras, -1, 0) 
                            ## glcm outputs are floats. convert to integer
                            glcm_rio = np.round((glcm_rio+.00001) * 10000).astype(np.int16)

                            profile.update(
                                dtype='int16', 
                                    counts=1,           
                                    nodata=0
                                    )
                            
                        else:
                            ## more robust method - returns 4d array
                            glcm_ras = haralick_features(scaled_ras, win, tex, 1, th=0, levels=cv)
                        
                        if (print_out) or (gt == 'all') or (numrasts > 1):
                    
                            if params:
                                out_path = Path(out_dir)/f"{params['grids'][0]}_{params['feature_model']['spec_indices'][0]}_{t}_w{win}_c{cv}_th{th}.tif"
                            else:
                                out_path = Path(out_dir)/f"{t}_w{win}_c{cv}_th{th}.tif"
                            logger.info(f'output saved to {out_path}')

                            with rio.open(out_path, 'w', **profile) as dst:
                                dst.write(glcm_rio.astype(np.int16))

                        if numrasts > 1:
                            out_files.append(out_path)

    logger.info('all_done!')
    
    if numrasts == 1:
        if print_out == True:
            return out_path 
        else:
            return glcm_rio
    else:
        return out_files

