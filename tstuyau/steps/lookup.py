from datetime import date as dt_date

dt_today = dt_date.today()

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

SCHEMATIC_MODS={'pyall':'celPy1_LC32',
                'pymax':'celPy1_LC36',
                'trans_cats':'LCTrans',
                'cropNoCrop':'LCcrop2',
                'crop_nocrop_mixcrop':'LCcrop3sm',
                'crop_nocrop_medcrop':'LCcrop3',
                'crop_nocrop_medcrop_tree':'LCcrop4',
                'veg':'LC5',
                'veg_det':'LC15',
                'veg_with_crop':'LC5wCrop1',
                'veg_with_cropType':'LCwCrop2',
                'cropType':'LC_crops',
                'burnType':'LCburn4',
                'burnNoburn':'LCburn2',
                'SAgrass_max':'LC25',
                'SAgrass_all':'LC20',
                'grassNoGrass':'LCgrass2'
               }

LC_FOCUS_DICT = {'smCrop':{'cats':['smallCrop','bigCrop', 'noCrop'],'lutcol':'LCcrop2'},
                 'crop':{'cats':['crop', 'noCrop'],'lutcol':'LCcrop2'},
                 'burn':{'cats':['burn','noBurn'],'lutcol':'LCburn2'},
                 'mgmt_burn':{'cats':['burn','noBurn','mgmtBurn'],'lutcol':'LCburn2'},
                 'wet_burn':{'cats':['burn','noBurn','wetBurn'],'lutcol':'LCburn2'},
                 'dry_burn':{'cats':['burn','noBurn','dryBurn'],'lutcol':'LCburn2'},
                 'high_burn':{'cats':['burn','noBurn','highBurn'],'lutcol':'LCburn2'},
                 'grass':{'cats':['allGrass','noGrass'],'lutcol':'LCgrass2'},
                 'clear_grass':{'cats':['allGrass','noGrass','clearGrass'],'lutcol':'LCgrass2'}
                }
                            
LC_CATS_Py0 ={'smallcrop_main' : 35,
                'sugar' : 38,
                'rice' : 37,
                'banana':43,
                'smallcrops' : [23,24,25,26,32,34,35,36,39],
                'bigcrops' : [22,31,33,37,38],
                'med_crops' : [40,41,42,43,45,46,47,54],
                'low_crops':  [*range(22,40)],
                'all_crops' : [*range(22,48),54],
                'first_veg' : 11,
                'first_highveg' : 50,
                'mixed_edge' : 18,
                'crop_edge': 19,
                'shrub': [*range(50,60)],
                'shrub_main': 52,
                'tree_plant':[60,66,56],
                'young_treeplant':56,
                'baby_treeplant':11,
                'first_medveg':40,
                'first_mature':60,
                'dense_for': [80],
                'open_for': [65],
                'shrub_for': [64],
                'palm_for': [68],
                'gtmix': [51],
                'wet':[7,17,57,77],
                'tree_water_mix': 77,
                'wet_grass': 17,
                'wet_med':57,
                'built':[3],
                'wet_medveg': 57,
                'forest_Py36':[64,65,68,80],
                'forest_open_stable':[64,65],
                'forest_nat':[64,65,68,80],
                'allGrass':[12,13,17,18],
                'grass_Py36':[12,13,17,18],
                'water':[7]
             }

LC_CATS={'smallcrop_main' : 137,
                'sugar': 143,
                'rice': 114,
                'banana': 148,
                'smallcrops' : [117,129,130,134,137,138,147],
                'bigcrops' : [102,103,104,105,106,107,110,111,112,113,114,115,116,140,141,142,143,144,145,146],
                'med_crops' : [148,150,151,152,153,155,156,157,158,159,191,192,193,194,196,197,198],
                'low_crops': [*range(100,148)],
                'noCrop':[98],
                'all_crops': [*range(100,160),*range(191,199)],
                'first_veg' : 50,
                'first_highveg': 180,
                'mixed_edge': 86,
                'crop_edge': 93,
                'allGrass': [51,55,58,71,72,73,74,75,76,77,79,80,81,82,83,84,85,86,87,88,89,96,108],
                'clearGrass': [51,55,58,71,73,75,76,77,79,80,81,82,83,84,85,87,88,89,108],
                'burn':[91,94,99,95,169],
                'highBurn':[169],
                'dryBurn':[99],
                'wetBurn':[94],
                'mgmtBurn':[91],
                'noBurn':[255],
                'shrub': [*range(160,170)],
                'shrub_main':161,
                'tree_plant':[207,187,217,202,203,205,206,208,209,216],
                'young_treeplant':187,
                'baby_treeplant':97,
                'first_medveg':148,
                'first_mature':200,
                'dense_for': [220],
                'open_for': [215],
                'shrub_for':[226],
                'palm_for':[221],
                'generic_for':[220,215],
                'gtmix': [176],
                'wet':[40,74,164,184],
                'tree_water_mix': 184,
                'wet_grass': 74,
                'wet_med':164,
                'built':[3],
                'wet_medveg':164,
                'forest_Py36':[226,215,221,220],
                'forest_open_stable':[226,221],
                'forest_nat':[212,213,214,215,*range(218,224)],
                'grass_Py36':[12,13,17,18],
                'water':[40]
                 
        }

MIXED_CROPS_Py0 = ["Crops-mix", "Crops-Mandioca", "Crops-Horticulture","Crops-Sesame","Crops-Tobacco"]
MIXED_CROPS_Py1 = ["crop_mixed_small" ,"crop_cassava", "crop_horticulture", "crop_tobacco", "crop_sesame"]
MIXED_NONCROPS_Py0  = ["Mixed-VegEdge", "Mixed-path", "Mixed-GrassEdge", "Mixed-FieldEdge"]
MIXED_CROPS = ["crop_mixed_small" ,"crop_cassava", "crop_horticulture"]
MIXED_NONCROPS = ["mixed_path", "grass_edge", "crop_edge", "riparian_mixed", "mixed_highNo", "mixed_highNoLow", "rd_trees", "tree_path"]

mixed_classes = ["Mixed-VegEdge", "Mixed-path", "Crops-mix", "Mixed-GrassEdge", "Mixed-FieldEdge", 
                     "Crops-Mandioca", "Crops-Horticulture","Crops-Sesame","Crops-Tobacco"]

GEE_COLLECTIONS = ['COPERNICUS/S2',
                  'COPERNICUS/S2_CLOUD_PROBABILITY',# S2Cloudless 
                  'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'  #alternative cloud masks to check out
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