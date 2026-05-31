from pathlib import Path
import pandas as pd
import numpy as np
#from rasterio.features import shapes
import geopandas as gpd
#from rasterstats import zonal_stats, point_query
from ..handler import logger
from .project import ProjectPaths
#from . import date_utils


def chip_acc(predpoly_dir, pred_prefix, refshp, acc_id_file, out_acc_dir, class_col="class", target='crop', pos_classes=[1]):
    logger.info(f'predpoly_dir = {predpoly_dir}')
    
    refpolys = gpd.read_file(refshp) ## pred vector file 
    regions = sorted(refpolys['region'].to_list())
    chips_w_target=[]
    chips_no_target=[]
    for region in sorted(list(set(regions))):
        refpoly_gdf = refpolys[refpolys["region"]==int(region)] 
        refpoly_gdf = refpoly_gdf[refpoly_gdf[class_col].isin(pos_classes)]
        if len(refpoly_gdf) >= 1:
            chips_w_target.append(region)
        else:
            chips_no_target.append(region)

    chip_df = gpd.read_file(refshp.replace("_Polys_", "_Chips_"))
    chip_df['region'] = [int(i) for i in chip_df['region']]
    refpolys['region'] = [int(i) for i in refpolys['region']]
    refpolys['UNQ'] = [int(str(i)[:4]) for i in refpolys['region']]

    all_chips_w_target=[]
    for region in sorted(chips_w_target): ## subset for quick testing
        predvector = Path(predpoly_dir)/f"{pred_prefix}_{str(region)[:4]}_cut.gpkg"
        if predvector.exists():
            predpolys = gpd.read_file(predvector)
            chip_shape = chip_df[chip_df["region"]==int(region)] 
            refpoly_gdf = refpolys[refpolys["region"]==int(region)] 
            refpoly_gdf = refpoly_gdf[refpoly_gdf[class_col].isin(pos_classes)]
            refpoly_gdf = gpd.clip(refpoly_gdf, chip_shape)

            predpoly_gdf = predpolys.sjoin(chip_shape, how="inner")
            predpoly_gdf = gpd.clip(predpoly_gdf, chip_shape)

            if len(predpoly_gdf) == 0: ## underpredicting where pred = 0 and ref > 0
                all_chips_w_target.append([int(region), pred_prefix, len(refpoly_gdf)*-1, np.nanmean(refpoly_gdf.area)*-1, np.sum(refpoly_gdf.area)*-1 ])
            else: 
                if len(predpoly_gdf) > len(refpoly_gdf):
                    all_chips_w_target.append([region, pred_prefix, (len(predpoly_gdf)/len(refpoly_gdf))-1 ])
                else:
                    all_chips_w_target.append([region, pred_prefix, 1-(len(refpoly_gdf)/len(predpoly_gdf))])            
                if np.nanmean(predpoly_gdf.area) > np.nanmean(refpoly_gdf.area):
                    all_chips_w_target[-1].append((np.nanmean(predpoly_gdf.area)/np.nanmean(refpoly_gdf.area))-1 )
                else:
                    all_chips_w_target[-1].append(1-(np.nanmean(refpoly_gdf.area)/np.nanmean(predpoly_gdf.area)) )
                if np.sum(predpoly_gdf.area) > np.sum(refpoly_gdf.area):
                    all_chips_w_target[-1].append((np.sum(predpoly_gdf.area)/np.sum(refpoly_gdf.area))-1 )
                else:
                    all_chips_w_target[-1].append(1- (np.sum(refpoly_gdf.area)/np.sum(predpoly_gdf.area)) )
                    
            all_chips_w_target[-1] = [0 if i==-np.inf else i for i in all_chips_w_target[-1]]
            logger.info(f'all chips with {target}: {all_chips_w_target[-1]}')

    all_chips_no_target=[]
    for region in sorted(chips_no_target): ## subset for quick testing
        predvector = Path(predpoly_dir)/f"{pred_prefix}_{str(region)[:4]}_cut.gpkg"
        if predvector.exists():
            predpolys = gpd.read_file(predvector)
            chip_shape = chip_df[chip_df["region"]==int(region)] 
            refpoly_gdf = refpolys[refpolys["region"]==int(region)] 
            refpoly_gdf = refpoly_gdf[refpoly_gdf[class_col].isin(pos_classes)]
            refpoly_gdf = gpd.clip(refpoly_gdf, chip_shape)
            predpoly_gdf = predpolys.sjoin(chip_shape, how="inner")
            predpoly_gdf = gpd.clip(predpoly_gdf, chip_shape)

            if len(predpoly_gdf)==0:
                all_chips_no_target.append([str(region), pred_prefix, 0, 0, 0])
            else: ## if pred > 0 and ref = 0 (overpredict)
                all_chips_no_target.append([str(region), pred_prefix, len(predpoly_gdf), np.nanmean(predpoly_gdf.area), np.sum(predpoly_gdf.area)  ])        
                
            all_chips_no_target[-1] = [0 if i==-np.inf else i for i in all_chips_no_target[-1]]
            logger.info(f"all chips without target land cover: {all_chips_no_target[-1]}")

    w_target_df = pd.DataFrame(all_chips_w_target, columns=["region", "version", "numFields", "avgArea", f"total{target.capitalize()}Area"])
    no_target_df = pd.DataFrame(all_chips_no_target, columns=["region", "version", "numFields", "avgArea", f"total{target.capitalize()}Area"])
    all_chips_df = pd.concat([w_target_df, no_target_df])
    
    holdout_cells=pd.read_csv(acc_id_file)
    keep_grids = sorted(holdout_cells['id'].to_list())
    done_regions = [i for i in all_chips_df['region']]
    keep_regions = [i for i in done_regions if int(str(i)[:4]) in keep_grids]   
    w_target_df=w_target_df[w_target_df['region'].isin(keep_regions)]
    no_target_df=no_target_df[no_target_df['region'].isin(keep_regions)]    

    w_target_df['version'] = [f'{i}_w_{target}' for i in w_target_df['version']]
    no_target_df['version'] = [f'{i}_no_{target}' for i in no_target_df['version']]    
    
    all_chips_df = pd.concat([w_target_df, no_target_df])
    out_name = Path(out_acc_dir) / f"chip_acc_{pred_prefix}_allregions_{str(len(all_chips_df))}.csv"
    all_chips_df.to_csv(out_name)
    
    w_target_grouped = w_target_df.groupby(["version"])[["numFields", "avgArea", f"total{target.capitalize()}Area"]].mean()
    no_target_grouped = no_target_df.groupby(["version"])[["numFields", "avgArea", f"total{target.capitalize()}Area"]].mean()
    all_grouped = pd.concat([w_target_grouped, no_target_grouped])
    all_grouped.to_csv(out_name.replace(f"_allregions_{str(len(all_chips_df))}.csv", "_avg.csv"))
    
    return all_chips_df, all_grouped

#####################################
## field accuracy metrics (IoU)
### Note: these are not called at moment

def largest_overlap(ref_df, pred_df):

    ## find all Pred fields that intersect (spatial join), as a list (1:many)
    intersecting = pd.DataFrame(pred_df.sjoin(ref_df, how='inner')['Rindex']) #Find the polygons that intersect. Keep savedindex as a series
    pred_val_matches = intersecting.reset_index()
    pred_val_matches.columns = ["PredIndex", "RefIndex"]
    pred_ref_intersecting_index = pd.DataFrame(pred_val_matches.groupby(['RefIndex'])['PredIndex'].apply(list)).reset_index()

    ## find the polygon w/ largest overlap w/ each Ref field (1:1)
    overlap_areas_all_fields=[]
    for k,v in pred_ref_intersecting_index.iterrows():
        ref_index = v[0]
        pred_matches = v[1]
        rdf=ref_df[ref_df["Rindex"]==ref_index]
        overlap_areas_per_ref_field=[]
        pred_indices_per_ref_field=[]
        for pred_index in pred_matches:
            pdf = pred_df[pred_df["Pindex"]==pred_index]
            rdf['area'] = rdf.geometry.area
            pdf['area'] = pdf.geometry.area 
            intersect_df = gpd.overlay(rdf, pdf, how="intersection")
            if not len(intersect_df) == 0:
                interction_area = intersect_df['geometry'].area
                pred_indices_per_ref_field.append(pred_index)
                overlap_areas_per_ref_field.append(interction_area[0])
            else: ################################************************ append something else here to signify no match
                pred_indices_per_ref_field.append(0)
                overlap_areas_per_ref_field.append(0)                
        overlap_areas_all_fields.append(dict(zip(pred_indices_per_ref_field, overlap_areas_per_ref_field)))

    ## find largest overlap from list 
    largest_overlapping_pred = []
    for r in overlap_areas_all_fields:
        max_index = max(r, key=r.get)
        largest_overlapping_pred.append(max_index)
    rp_index_matches = list(zip(pred_ref_intersecting_index['RefIndex'].to_list(), largest_overlapping_pred))
    return rp_index_matches
 
def calc_metrics(predvector, ref_df, pred_df, rp_index_matches):
    ious=[]
    overseg_rates=[]
    underseg_rates=[]
    location_similarities=[]

    for i in rp_index_matches:
        ref_gdf = ref_df[ref_df['Rindex'] == i[0]]
        pred_gdf = pred_df[pred_df['Pindex'] == i[1]]
        intersect_df = gpd.overlay(ref_gdf, pred_gdf, how="intersection")
        ref_area = ref_gdf['geometry'].iloc[0].area
        pred_area = pred_gdf['geometry'].iloc[0].area    
        intersect_area = intersect_df['geometry'].loc[0].area
        union_area = ref_area+pred_area-intersect_area

        ## IoU
        ious.append(intersect_area/union_area)
        ## overseg rates 
        overseg_rates.append(1-(intersect_area/ref_area))

        ## underseg rates 
        underseg_rates.append(1-(intersect_area/pred_area))    

        ## location similarity 
        pred_centroid = pred_gdf.geometry.centroid.iloc[0]
        ref_centroid = ref_gdf.geometry.centroid.iloc[0]
        centr_dist=ref_centroid.distance(pred_centroid) 
        circradius=2*np.sqrt(union_area/np.pi)
        location_similarities.append(1-centr_dist/circradius)

    filename = Path(predvector).stem
    region = (list(set(ref_df['region'])))[0] ### UNQ or region
    logger.info(f'region = {region}')
    match_metrics = [filename, region, rp_index_matches,ious,overseg_rates,underseg_rates,location_similarities]
    match_metrics=pd.DataFrame(match_metrics).T
    
    ## show accuracy metrics by chip 
    match_metrics_per_grid = [filename, region, np.mean(ious), np.mean(overseg_rates), np.mean(underseg_rates), np.mean(location_similarities)]
    metrics_per_grid = pd.DataFrame(match_metrics_per_grid).T
    metrics_per_grid.columns=["version", "region", "IoU", "overseg", "underseg", "location_sim"]
    metrics_per_grid.fillna(0, inplace=True)
    
    return match_metrics_per_grid

def instance_field_accuracy(refshp, predvector, out_name, target='crop', pos_classes=[1]):
    ### Note: this is not called at moment

    refpolys = gpd.read_file(refshp)
    refchips=refshp.replace("_Polys", "_Chips")
    chip_df = gpd.read_file(refchips)
    pred_grid=int(predvector.stem.split("_")[2])

    if "PyCropSeg" in refshp:
        refpolys['region'] = refpolys['region_lef']
        refpolys['region'] = [int(i) for i in refpolys['region']]
        refpolys['Name'] =refpolys['Name_left']
        refpolys['UNQ'] = [int(str(i)[:4]) for i in refpolys['region']]
        class_col = "class"
    else:
        refpolys['region'] = [int(i) for i in refpolys['region']]
        chip_df['region'] = [int(i) for i in chip_df['region']]
        class_col = "code"  
        
    refpoly = refpolys[refpolys['UNQ'] == int(pred_grid)] 
    regions = [str(i) for i in list(set(refpoly['region'].to_list()))]
    
    stats_per_grid=[]
    exclude_in_mean=[]
    for region in sorted(regions):
        chip_shape = chip_df[chip_df["region"]==int(region)]
        
        pred_df=gpd.read_file(predvector)
        pred_df = pred_df.sjoin(chip_shape, how="inner")
        pred_df = gpd.clip(pred_df, chip_shape)
        pred_df['area'] = pred_df.area     
        
        ref_df = refpoly[refpoly["region"]==int(region)] 
        ref_df = ref_df[ref_df[class_col].isin(pos_classes)] 
        ref_df = gpd.clip(ref_df, chip_shape)
        
        if len(ref_df) > 0 and len(pred_df) > 0:
            ref_df['area'] = ref_df.area
            ref_df['Rindex']= ref_df.index 
            pred_df['Pindex']= pred_df.index
            if "PyCropSeg" in refshp: 
                pred_df=pred_df.drop(columns=["index_right", "UNQ8858", "Shape_Leng", "Shape_Area", "Name", "region"])
            else:
                pred_df=pred_df.drop(columns=[i for i in pred_df.columns.to_list() if "_left" in i or  "_right" in i])
            ref_df=ref_df.drop(columns=[i for i in ref_df.columns.to_list() if "_left" in i or  "_right" in i])
            ## FIND PRED MATCH FOR EACH REF VECTOR BASED ON LARGEST OVERLAP 
            rp_index_matches = largest_overlap(ref_df, pred_df)
            ## CALCULATE ACCURACY METRICS w/ MATCHES 
            chip_avg_metrics = calc_metrics(predvector, ref_df, pred_df, rp_index_matches)
            ## append to find average of all chips in the UNQ grid cell
            stats_per_grid.append(chip_avg_metrics)
        elif len(ref_df) == 0 and len(pred_df) == 0:
            stats_per_grid.append([predvector.stem, region, 1, 1, 1, 1])
            exclude_in_mean.append(region)
        else:
            logger.warning('fix this')
            logger.info(f'pred_df has {len(pred_df)} items')
            logger.info(f'ref_df has {len(ref_df)} items')
            
    logger.info(f'list of chip regions w/ 0 {target}: {exclude_in_mean}')

    stats_df=pd.DataFrame(stats_per_grid)
    stats_df.columns=["version", "region", "IoU", "overseg", "underseg", "location_sim"]
    stats_df = stats_df[~stats_df['region'].isin(exclude_in_mean)]
    logger.info(f'stats_df = {stats_df}')
    
    avg_per_grid = [str(region)[:4], predvector.stem, np.mean(stats_df['IoU']), np.mean(stats_df['overseg']), 
                    np.mean(stats_df['underseg']), np.mean(stats_df['location_sim']) ]
    out=pd.DataFrame(avg_per_grid).T
    out.columns=["UNQ", "version", "IoU", "overseg", "underseg", "location_sim"]
    out.to_csv(out_name)
    logger.info(f' stats out = {out}')

    return out 
    
###############################################################################################################

def segmentation_accuracy(params):

    seg_dir_main = params['segment']['seg_dir_main']
    if not seg_dir_main:
        ppaths=ProjectPaths(params)
        seg_dir_main = ppaths.segmentation
    seg_dir_main.mkdir(parents=True, exist_ok=True)
    version_dir = Path(seg_dir_main) / params['segment']['seg_dir_mod']
    
    accuracy_dir = Path(version_dir)/'accuracy'
    
    method = params['vectorize']['instance_method']
    if method == 'EO':
        eot = params['vectorize']['eo_thresh']
        threshs = eot.replace('.', 'pt')
    elif "thresh" in method:
        bt = params['vectorize']['bound_thresh']
        et = params['vectorize']['ext_thresh']
        threshs = f"{bt.replace('.', 'pt')}_{et.replace('.', 'pt')}"
    elif "water" in method:
        bt = params['vectorize']['bound_thresh']
        et = params['vectorize']['ext_thresh']
        ss = params['vectorize']['seed_size']
        threshs = f"{bt.replace('.', 'pt')}_{et.replace('.', 'pt')}_{ss.replace('.', 'pt')}"

    chip_acc (predpoly_dir=Path(version_dir)/f"infer_polys_{method}_{threshs}_2022", 
             pred_prefix=f"{method}_pred_polys_{threshs}th",
             refshp=params['segment']['seg_train-polys'],
             acc_id_file=Path(seg_dir_main) / params['segment']['acc_id_file'],
             out_acc_dir=accuracy_dir,
             target=params['segment']['target'],
             class_col="class", 
             pos_classes=['segement']['pos_classes'] )
