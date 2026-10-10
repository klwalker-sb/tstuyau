from datetime import datetime, timezone
import numpy as np

dt_today = datetime.now(tz=timezone.utc).date()

SENSORS = {'Sentinel2':{'unq':'S2','matchstr':['S2','S2A','S2B','S2C'], 'sensor':'sentinel-2','color':'magenta', 'name':'Sentinel-2', 'GEEunq':'L1C','GEE':'COPERNICUS/S2'}, 
              'S2':{'unq':'S2','matchstr':['S2','S2A','S2B','S2C'], 'sensor':'sentinel-2', 'color':'magenta','name':'Sentinel-2','GEEunq':'L1C','GEE':'COPERNICUS/S2'},
              'S2cp':{'unq':'S2cp','matchstr':['S2cp'], 'sensor':'sentinel-2', 'GEE':'COPERNICUS/S2_CLOUD_PROBABILITY'}, 
              'Landsat5':{'unq':'LT05','matchstr':['LT'], 'sensor':'landsat', 'name':'Landsat TM','color':'yellow','GEEunq':'LT05','GEE':'LANDSAT/LT05/C01/T1_SR'},
              'L5':{'unq':'LT05','matchstr':['LT','LT05'], 'sensor':'landsat', 'name':'Landsat TM', 'color':'yellow','GEEunq':'LT05','GEE':'LANDSAT/LT05/C01/T1_SR'}, 
              'LT05':{'unq':'LT05','matchstr':['LT','LT05'], 'sensor':'landsat', 'name':'Landsat TM','color':'yellow', 'GEEunq':'LT05','GEE':'LANDSAT/LT05/C01/T1_SR'},
              'Landsat7':{'unq':'LE07','matchstr':['LE','LE07'], 'sensor':'landsat', 'name':'Landsat ETM+','color':'orange','GEEunq':'LE07', 'GEE':'LANDSAT/LE07/C01/T1_SR'},
              'L7':{'unq':'LE07','matchstr':['LE','LE07'], 'sensor':'landsat', 'name':'Landsat ETM+','color':'orange','GEEunq':'LE07','GEE':'LANDSAT/LE07/C01/T1_SR'},
              'LE07':{'unq':'LE07','matchstr':['LE','LE07'], 'sensor':'landsat', 'name':'Landsat ETM+','color':'orange','GEEunq':'LE07','GEE':'LANDSAT/LE07/C01/T1_SR'},
              'Landsat8':{'unq':'LC08','matchstr':['LC08'], 'sensor':'landsat', 'name':'Landsat 8','color':'blue','GEEunq':'LC08','GEE':'LANDSAT/LC08/C01/T1_SR'},
              'L8':{'unq':'LC08','matchstr':['LC08'], 'sensor':'landsat', 'name':'Landsat 8','color':'blue','GEEunq':'LC08','GEE':'LANDSAT/LC08/C01/T1_SR'},
              'LC08':{'unq':'LC08','matchstr':['LC08'], 'sensor':'landsat', 'name':'Landsat 8','color':'blue','GEEunq':'LC08','GEE':'LANDSAT/LC08/C01/T1_SR'},
              'Landsat9':{'unq':'LC09','matchstr':['LC09'], 'sensor':'landsat', 'name':'Landsat 9','color':'cyan','GEEunq':'LC09','GEE':'LANDSAT/LC09/C01/T1_SR'},
              'L9':{'unq':'LC09','matchstr':['LC09'], 'sensor':'landsat', 'name':'Landsat 9','color':'cyan','GEEunq':'LC09','GEE':'LANDSAT/LC09/C01/T1_SR'},
              'LC09':{'unq':'LC09','matchstr':['LC09'], 'sensor':'landsat', 'name':'Landsat 9','color':'cyan','GEEunq':'LC09','GEE':'LANDSAT/LC09/C01/T1_SR'},
              'Landsat':{'unq':'L','matchstr':['LC','LT','LE','LT05','LE07','LC08','LC09'], 'sensor':'landsat','color':'darkorange'},
              'L':{'unq':'L','matchstr':['LC','LT','LE','LT05','LE07','LC08','LC09'], 'sensor':'landsat','color':'darkorange', 'name':'Landsat'},
              'LS2':{'unq':'LS2','color':'dodgerblue','matchstr':['S2','S2A','S2B','S2C','LC','LT','LE','LT05','LE07','LC08','LC09'], 'name':'Landsat+Sentinel-2'},
              'All':{'unq':'LS2','matchstr':['S2','S2A','S2B','S2C','LC','LT','LE','LT05','LE07','LC08','LC09'],'color':'dodgerblue', 'name':'Landsat+Sentinel-2'}, 
              'AllRaw':{'unq':'LS2','matchstr':['S2','S2A','S2B','S2C','LC','LT','LE','LT05','LE07','LC08','LC09'],'color':'dodgerblue', 'name':'Landsat+Sentinel-2'},
              'S2A':{'unq':'S2A','matchstr':['S2A'], 'sensor':'sentinel-2','name':'Sentinel-2A','color':'red'},
              'S2B':{'unq':'S2B','matchstr':['S2B'], 'sensor':'sentinel-2', 'name':'Sentinel-2B','color':'darkred'},
              'S2C':{'unq':'S2C','matchstr':['S2C'], 'sensor':'sentinel-2', 'name':'Sentinel-2C','color':'purple'},
              }

MASKS = {'terrain_shade': {'maskname':'shademask', 'mask_dir':'terrain/shade_masks', 'db_col':'shade_mask', 'mask_val':1},
        'cloud_CRF':{'maskname':'cloudCRF', 'mask_dir':'clouds/CRF_masks', 'db_col':'cloud_crf', 'mask_val':[]},
         's2cloudless': {'maskname':'s2cloudless', 'mask_dir':'clouds/s2cloudless', 'db_col':'s2cloudless', 'mask_val':1},
        }

CONT_STATS = {'avg': np.nanmean,
               'med': np.nanmedian,
               'std': np.nanstd,
               'q75': lambda d: np.nanquantile(d, 0.75),
               'q90': lambda d: np.nanquantile(d, 0.9),
               'q25': lambda d: np.nanquantile(d, 0.25), 
               'q10': lambda d: np.nanquantile(d, 0.1)}


## legacy for old pymaps:
SCHEMATIC_MODS_leg={'pyall':'LC32',
                'pymax':'LC36',
                'trans_cats':'LCTrans',
                'cropNoCrop':'LC2',
                'crop_nocrop_mixcrop':'LC3sm',
                'crop_nocrop_medcrop':'LC3',
                'crop_nocrop_medcrop_tree':'LC4',
                'veg':'LC5',
                'veg_with_crop':'LC8',
                'veg_with_cropType':'LC10',
                'cropType':'LC_crops'
               }
    
## Note: 'labels' need to have corresponding entry in LC_CATS and "col' needs to be columns in the LUT with the class values (
##        matching the items in the list for the LC_CAT keys involved. If binary recall scores are desired for a multi-class model, 
##        a column 'col2' can be supplied (but the function should now be able to work with the multi-cat column). The class label starting 
##        with  'No' will be assumed to be the negative class and all other classes will be assumed positive. If binary recall does not make
##        sense for the model, enter 'NA' for 'col2' so that the model does not complain or waste effort running nonsensical recalls. 
##        Use get_schematic_mod(focus) rather than indexing SCHEMATIC_MODS directly: it fills in a missing 'col2' ('NA')
##        and a missing 'short_names' (derived from 'labels').
SCHEMATIC_MODS={'pyall': {'col':'celPy1_LC32', 'col2':'NA'},
                'pymax': {'col':'celPy1_LC36', 'col2':'NA'},
                'trans_cats':{'col':'LCTrans', 'col2':'NA'},
                'cropNoCrop':{'col':'LCcrop2', 'col2':'LCcrop2', 'labels':['Crop', 'No crop'], 'short_names':['Crop', 'NoCrop']},
                'smCrop':{'col':'LCcrop3sm', 'col2':'LCcrop2', 'labels':['Homogeneous crop','No crop','Mixed crop'], 'short_names':['bigCrop','noCrop','smCrop']},
                'medCrop':{'col':'LCcrop3', 'col2':'LCcrop2', 'labels':['LowVeg_crop','No crop','Shrub or Tree crop'],
                           'short_names':['lowCrop','NoCrop','medCrop']},
                'crop_nocrop_medcrop_tree':{'col':'LCcrop4', 'col2':'LCcrop2','labels':['LowVeg_crop','No crop','Shrub or Tree crop','Trees'],
                                           'short_names':['lowCrop','NoCrop','medCrop','Trees']},   # FIX: was 'short_names' = [...]
                'veg':{'col':'LC5', 'labels':['NoVeg','LowVeg','MedVeg','HighVeg','Trees']},
                'veg_det':{'col':'LC15', 'col2':'NA'},
                'veg_with_crop':{'col':'LC5crop4', 'col2':'NA', 'labels':['NoVeg','LowVeg_noncrop','Homogeneous crop','Mixed crop',
                                                              'Shrub or Tree crop','MedVeg_noncrop','Trees_plantation','Trees_nat']},
                'veg_with_cropType':{'col':'LC5CropT', 'col2':'NA'},
                'cropType':{'col':'LC_crops', 'col2':'NA'},
                'burnType':{'col':'LCburn4', 'col2':'LCburn2', 'labels':['No-burn', 'Burned-wet', 'Burned-dry', 'Burned-woody']},
                'burnNoburn':{'col':'LCburn2', 'col2':'LCburn2', 'labels':['Burn', 'No-burn']},
                'mgmt_burn':{'col':'LCburn4', 'col2':'LCburn2', 'labels':['Burn','No-burn','mgmtBurn']},   # NOTE: 'mgmtBurn' has no LC_CATS entry yet
                'wet_burn':{'col':'LCburn4', 'col2':'LCburn2', 'labels':['Burn','No-burn','Burned-wet']},
                'dry_burn':{'col':'LCburn4', 'col2':'LCburn2', 'labels':['Burn','No-burn','Burned-dry']},
                'high_burn':{'col':'LCburn4', 'col2':'LCburn2', 'labels':['Burn','No-burn','Burned-woody']},
                'SAgrass_max':{'col':'LC25', 'col2':'NA'},
                'SAgrass_all':{'col':'LC20', 'col2':'NA'},
                'grassNoGrass':{'col':'LCgrass2', 'col2':'LCgrass2', 'labels':['Grass', 'No-Grass']},   # FIX: missing comma after col2
                'clear_grass':{'col':'LCgrassP', 'col2':'LCgrass2', 'labels':['Grass_all','No-grass','Grass_clear']}
               }

def get_schematic_mod(mod):
    '''
    Returns a copy of SCHEMATIC_MODS[focus] with defaults filled in
    '''
    spec = dict(SCHEMATIC_MODS[mod])
    spec.setdefault('col2', SCHEMATIC_MODS[mod]['col'])
    if 'labels' in spec:
        spec.setdefault('short_names', [l.replace(' ', '_') for l in spec['labels']])
    return spec

SMALLS_FLAGS = {'smalls_1ha': 'smlhld_1ha', 'smalls_halfha': 'smlhld_halfha'}
    
## Note: In the case of lists, the first value is the storae value for the category
LC_CATS_Py0 ={  'Mixed crop' : [35,23,24,25,26,32,34,36,39],
                'Mixed crop reduced':[35,26,32,36],
                'Homogeneous crop':[22,31,33,37,38,19],  ## note this should not include shrub or tree crops
                'Shrub or Tree crop':[40,41,42,43,45,46,47,54],
                'LowVeg_crop':[30, *range(22,40)],
                'Crop':[100, *range(22,48),19,54],
                'Mixed_Crop-edge':19,  ## this is double counted in (Homogeneous crop / Crop) and (No crop) and usually dealt with by methods 
                'sugar':38,
                'rice':37,
                'banana':43,
                'No crop':[98,*range(1,20),*range(48,54),*range(55,98)],
                'mixed_nonCrop':[9,18,19],
                'LowVeg_noncrop':[*range(10,21)],
                'LowVeg':[20,*range(10,20),*range(22,40)],
                'LowVeg_wet':[17],
                'wet':[7,17,57,77],
                'NoVeg':[*range(1,10)],
                'NoVeg_water':[7],
                'NoVeg_built':[3],
                'NoVeg_bare':[2],
                'first_veg':11,
                'first_highveg':50,
                'HighVeg':[54,56],
                'Mixed_Grass-edge':18,
                'Grass_all':[12,13,17,18],
                'grass_Py36':[12,13,17,18],
                'dry_grass':[12,13],
                'gtmix':[51],
                'MedVeg_noncrop':[52,51,53,54,55,56,57,58,59],
                'MedVeg_wet':[57],
                'first_medveg':40,
                'MedVeg':[50,*range(51,60),40,41,42,43,45,46,47],
                'TreePlant_young':[56],
                'TreePlant_start':11,
                'Trees_plantation':[60,66,56],
                'first_mature':60,
                'dense_for':[80],
                'open_for':[65],
                'shrub_for':[64],
                'palm_for':[68],
                'tree_water_mix':77,
                'Trees':[65,60,66,64,68,77,80],
                'forest_Py36':[64,65,68,80],
                'forest_open_stable':[64,65],
                'Trees_nat':[64,65,68,80],
                'maxcat':100
             }

    
## Note: In the case of lists, the first value is the storae value for the category
LC_CATS={       'Mixed crop':[137,118,133,129,131,147,138,132,97,177],
                'Mixed crop reduced':[137,129,131,138,132,177],
                'Homogeneous crop':[104,96,101,102,103,105,106,107,109,110,111,112,113,114,115,116,117,119,120,140,141,142,143,144,145,146], ## note this should not include shrub or tree crops
                'Shrub or Tree crop':[150,148,151,152,153,155,156,157,158,159,190,191,192,193,194,196,197,198,199,183],
                'LowVeg_crop':[*range(100,148)],
                'Crop':[*range(100,160),*range(191,199),96],
                'Mixed_Crop-edge':96,  ## this is double counted in (Homogeneous crop / Crop) and (No crop) and usually dealt with by methods 
                'sugar':143,
                'rice':114,
                'banana':148,
                'No crop':[-100,*range(1,100),*range(161,191),*range(199,255)],
                'mixed_nonCrop':[51,91,96,181,184,191,196,211],
                'LowVeg_noncrop':[*range(50,100)],
                'LowVeg':[60,*range(50,60),*range(61,148)],
                'LowVeg_wet': [74,64,61,65,63,68],   # FIX: missing trailing comma
                'NoVeg':[10,*range(1,10),*range(11,50)],
                'NoVeg_water':[40],
                'NoVeg_built':[30],
                'NoVeg_bare':20,
                'first_veg':50,    
                'Mixed_Grass-edge':86,
                'Grass_all':[75,51,55,58,71,72,73,74,76,77,79,80,81,82,83,84,85,86,87,88,89,91,108],
                'Grass_clear':[51,55,58,71,73,75,76,77,79,80,81,82,83,84,85,87,88,89,108],
                'grass_Py36':[74,75,80,91],
                'dry_grass':[80,75],
                'gtmix': [176],
                'Burn':[95,93,94,99,169],
                'Burned-woody':[169],
                'Burned-dry':[99],
                'Burned-wet':[94],
                'Burned-firebreak':[93],
                'No-burn':[255],
                'MedVeg_noncrop': [161,*range(162,170)],
                'MedVeg':[160,*range(161,170),148,150,151,152,153,155,156,157,158,159,173,178,179,181,184,186,187,182,185],
                'first_medveg':148,
                'MedVeg_wet':[164],
                'first_highveg':180,
                'HighVeg':[180,188,189,181,184,186,187,182,185,211,197],
                'Trees_plantation':[207,187,217,202,203,205,206,208,209,216],
                'TreePlant_young':[187,182,185],
                'TreePlant_start':98,
                'Trees': [200,*range(206,255),193,195],
                 'first_mature':200,
                'dense_for': [250],
                'open_for': [215],
                'shrub_for':[226],
                'palm_for':[221],
                'generic_for':[220,215],
                'tree_water_mix':184,
                'forest_Py36':[226,215,221,220],
                'forest_open_stable':[226,221],
                'Trees_nat':[220,212,213,214,215,218,219,*range(221,255)],
                'wet':[40,74,164,184],
                'maxcat':255
        }

def get_lc_cats(project_v=None):
    """ Returns the LC_CATS dict for the project version. 
    """
    return LC_CATS_Py0 if project_v == 'Py0' else LC_CATS


GEE_COLLECTIONS = ['COPERNICUS/S2',
                  'COPERNICUS/S2_CLOUD_PROBABILITY',# S2Cloudless 
                  'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED',  #alternative cloud masks to check out
                  'COPERNICUS/S2_SR_HARMONIZED',    # *See note below
                  'COPERNICUS/S1_GRD',              # Sentinel-1 SAR GRD: C-band Synthetic Aperture Radar Ground Range Detected, log scaling
                  'LANDSAT/LC08/C01/T1_SR',         # Tier 1 surface reflectance
                  'LANDSAT/LC09/C02/T1_L2',             
                  'LANDSAT/LE07/C01/T1_SR',
                  'LANDSAT/LT05/C01/T1_SR',
                  'MODIS/006/MCD43A4',              # Nadir BRDF-Adjusted Reflectance Daily 500m
                  'MODIS/006/MCD43A2',              # Nadir BRDF-Albedo Quality Daily 500m
                  'MODIS/006/MCD19A2_GRANULES',     # Land Aerosol Optical Depth Daily 1km
                  'MODIS/006/MOD09A1',              # MOD09A1.006 Terra Surface Reflectance 8-Day Global 500m
                  'MODIS/006/MYD09A1',              # MOD09A1.006 Aqua Surface Reflectance 8-Day Global 500m
                  'ASTER/AST_L1T_003',              # ASTER L1T Radiance
                  'USDA/NASS/CDL',                  # USDA NASS Cropland Data Layers
                  'AAFC/ACI',                       # Agriculture and Agri-Food Canada Agriculture Crop Survey
                  'UCSB-CHG/CHIRPS/PENTAD'          # CHIRPS Pentad: Climate Hazards Group InfraRed Precipitation with Station Data (version 2.0 final)
                  ]
## * Note on Sentinel-2: After 2022-01-25, Sentinel-2 scenes with PROCESSING_BASELINE '04.00' or above have their DN (value) range shifted by 1000. 
##         The HARMONIZED collection shifts data in newer scenes to be in the same range as in older scenes

HGT_CONTINENT_DICT = {'South America': 'SouthAmerica',
                      'North America': 'NorthAmerica',
                      'Africa': 'Africa',
                      'Australia': 'Australia',
                      'Europe': 'Eurasia',
                      'Asia': 'Eurasia'}

GEE_TRANSLATIONS = {'landsat': {'gcp': {'TM': 'l5',
                                    'ETM': 'l7th',
                                    'OLI_TIRS': 'l8'}},
                'extensions': {'geotiff': '.tif',
                               'netcdf': '.nc'}}

FILE_EXTENSIONS = {'netcdf': '.nc',
                   'geotiff': '.tif',
                   'sentinel-2_metadata': '.xml',
                   'landsat_metadata': '.txt'}