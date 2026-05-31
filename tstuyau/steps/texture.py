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


def glcm_fast(in_ras, texture, size_win, levels=8, theta=0):
    '''in_dev -- TODO finish
    note data should be scaled 0-255 in the rescale below (coval=255). Then <levels> does additional rescaling
    also note cannot stack outputs, so each theta must be passed in individually
    '''
    
    import glcm_fast
    ## needs angles in degrees, not radians like scimage:
    thdeg = np.rad2deg(theta)

    if texture == 'entropy':
        glcm_map = fast_glcm.fast_glcm_entropy(in_ras, size_win, levels, angle=thdeg)
    if texture == 'contrast':
         glcm_map = fast_glcm.fast_glcm_contrast(in_ras, size_win, levels, angle=thdeg)
    if texture == 'dissimilarity':
        glcm_map = fast_glcm.fast_glcm_dissimilarity(in_ras, size_win, levels, angle=thdeg)
    if texture == 'ASM':
        glcm_map = fast_glcm.fast_glcm_ASM(in_ras, size_win, levels, angle=thdeg)
    if texture == 'homogeneity':
         glcm_map = fast_glcm.fast_glcm_homogeneity(in_ras, size_win, levels, angle=thdeg)
    if texture == 'std':
         glcm_map = fast_glcm.fast_glcm_std(in_ras, size_win, levels, angle=thdeg)
    
    return glcm_map

def glcm_slidingwin(in_ras, size_win, texture, cv=256, th=0, l=1, normed=False):
    '''
    create textures using the GLCM function of Skimage
    run individual glcm filters using numpy sliding windows.
    Slower than other mothods, but avoids multithreading cython issues that cause segmentation fault

    <th> is the angle 
    <l> is the offset distance in pixels between 'neighbors' (normally 1, but may be higher if looking for evenly spaced rows, etc.)
    <cv> is the total possible values (maxval + 1)
    '''

    def get_glcm_entropy(glcm):
        '''entropy is in later versions of graycoprops but not earlier'''
        # Re-normalize to ensure the 2D matrix sums to 1
        glcm_norm = glcm / np.sum(glcm)
        ## reexpanding into 3 dimensions so that output is the same format as the others.
        return np.array([[[ -np.sum(glcm_norm * np.log2(glcm_norm + 1e-10)) ]]])
    
    # 2. Add a tiny epsilon value (e.g., 1e-10) to avoid executing np.log2(0)
    logger.info(f' making {texture} raster...\n')
    
    pad_amt = size_win // 2
    padded_ras = np.pad(in_ras, pad_amt, mode='reflect')

    windows = np.lib.stride_tricks.sliding_window_view(padded_ras, (size_win, size_win))

    h, w = in_ras.shape
    glcm_map = np.zeros((h, w), dtype=np.float32)

    for r in range(h):
        for c in range(w):
            win_2d = np.array(windows[r, c], dtype=np.uint8, copy=True)

            if texture == 'entropy':
                normed = True
                
            glcm = graycomatrix(win_2d, distances=[l], angles=th, 
                levels=cv, symmetric=True, normed=normed)

            if texture == 'entropy':
                ## best practice to average all angles first for this case
                glcm_avg = np.mean(glcm[:, :, 0, :], axis=2)
                glcm_map[r, c] = get_glcm_entropy(glcm_avg)[0, 0]

            else:
                glcm_map[r, c] = graycoprops(glcm, texture)[0, 0]

    return np.atleast_3d(glcm_map)


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

##########################################################################################################
##alternative method if want more room to tinker...
## old framework from https://stackoverflow.com/questions/42459493/sliding-window-in-python-for-glcm-calculation

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
    ii = crop(img, (row, col), win)
    for d_index, length in enumerate(d):
        for a_index, angle in enumerate(theta):
            dv, dh = offset(length, angle)
            ij = crop(img, center=(row + dv, col + dh), win=win)
            cooc[:, :, d_index, a_index] = encode_cooccurrence(ii, ij, levels)
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

def haralick_features(img, win, props, d=1, theta=0, levels=256):
    """Return a map of Haralick features (one feature vector per pixel)"""
    rows, cols = img.shape
    margin = win + max(d)
    arr = np.pad(img, margin, mode='reflect')
    n_features = len(d) * len(theta) * len(props)
    logger.info(f'n_features = {n_features}')
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

    theta is the angle quadrant: th0=0, th1=pi/4, th2=pi/2, th3=3*pi/4   thmix4 = mix of all 4
       if multiple thetas are given, the returned raster is the average of all thetas
     '''
    
    if si_var and isinstance(si_var,list): 
        ## The textures can be directly passed in with <si_var> or parsed from params 
        gt = si_var  ## direct string or list of types to run (e.g. ['variance','contrast','dissimilarity','homogeneity','entropy']
        size_win = win  ## list of window values to run (e.g. [5,11,25] -- or just single value
        covals = covals ## list of # of possible values to rescale data to (e.g. [32,64,100,255]  -- or just single value
        thetas = th  ## list of angles to run [th0,th1,th2,th3] or 0 if none
    else:
        ## otherwise these values are parsed from si variable (e.g. med.glcm.variance.w5.c100.th0)
        if si_var:
            si_vars = si_var
        ## if si var is not passed in directly, it is pulled from the params
        else:
            #si = params['feature_model']['spec_indices'][0]
            si_vars = params['feature_model']['si_vars'][0] 

        if si_vars.split('.')[1] != 'glcm':
            logger.warning(f'{si_vars} is not a glcm variable')
    
        glcm_type = si_vars.split('.')[2]    

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
        
    numrasts = len(gt) * len(covals) * len(size_win)

    th = []
    if 'th0' in thetas or 'thmix4' in thetas:
        th.append(0)  #horizontal
    if 'th1' in thetas or 'thmix4' in thetas: 
        th.append(np.pi/4)  #diag up-right
    if 'th2' in thetas or 'thmix4' in thetas:
        th.append(np.pi/2) #vertical
    if 'th3' in thetas or 'thmix4' in thetas:
        th.append(3*np.pi/4) #diag up-left
    logger.info(f'calculating glcms from the following angles: {th}')
                             
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

    out_files = []
    for cv in covals:
        logger.info(f' rescaling the data to have a max val of {cv}')
        with rio.open(base_img) as in_ras:    
            scaled_ras = rescale_band(img_in, maxval=cv, profile=in_ras.profile, outpath=None).astype(np.uint8)

        #    temp_dir = out_path /'temp_rescaling'        
        #temp_ras = rescale_band(base_img, maxvalcv, profile=None, outpath=temp_dir)
        #with rio.open(temp_ras) as ras_src:
        #    scaled_ras = ras_src.read(1).astype(np.uint8)
        
        for tex in ['homogeneity','dissimilarity','contrast','correlation','entropy','variance','energy','ASM','std']:
            if (gt == ['all']) or any(t[:3].upper() == tex[:3].upper() for t in gt):
                for win in size_win:
                    logger.info(f'calculating {tex} at window size {win}')
                    ## glcm_fast available for these and should be faster, but need to test  
                    '''
                    if tex in ['homogeneity','dissimilarity','contrast','entropy','ASM']:
                        glcm_stack = []
                        for t in th:
                            glcm_ras = glcm_fast(scaled_ras, texture, win, levels=8, theta=th[0])
                            glcm_stack.append(glcm_ras)
                        glcm_avg = sum(glcm_stack) / len(glcm_stack)
                        glcm_out = (glcm_avg * 10000).astype(np.int16)
                    else:
                    ''' 
                        #glcm_ras = glcm_cython(scaled_ras, win, texture=t)  #getting segmentation fault 
                        #glcm_ras = haralick_features(scaled_ras, win, tex, l=1, th=0, levels=cv+1)
                    glcm_ras = glcm_slidingwin(scaled_ras, win, tex, cv=cv+1, th=th, l=1, normed=False)
                    
                    ## change dim ordering from scipy to rio:
                    glcm_rio = np.moveaxis(glcm_ras, -1, 0) 

                    ## glcm outputs are floats. convert to integer
                    glcm_int = np.round((glcm_rio+.00001) * 10000).astype(np.int16)
                    
                    if len(th) > 1:
                        ## if multiple angles, output is average
                        logger.info('averaging thetas')
                        glcm_out = np.mean(glcm_int, axis=0).astype(np.int16)
                    else:
                        glcm_out =  np.squeeze(glcm_int)
                    
                    profile.update(
                        dtype='int16', 
                            count=1,           
                            nodata=0
                            )
                        
                if (print_out) or (gt == 'all') or (numrasts > 1):
                    
                    if params:
                        out_path = Path(out_dir)/f"{params['grids'][0]}_{params['feature_model']['spec_indices'][0]}_glcm-{tex}_w{win}_c{cv}_{thetas[0]}.tif"
                    else:
                        out_path = Path(out_dir)/f"glcm-{tex}_w{win}_c{cv}_{thetas[0]}.tif"
                    logger.info(f'output saved to {out_path}')

                    with rio.open(out_path, 'w', **profile) as dst:
                            dst.write(glcm_out,1)

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

