import xarray as xr

from ..handler import logger
from .filter_utils import FilterTsArgs, apply_condition_to_timeseries, store_count


def _retouch_sugar_palm(ts, idx, ctx: FilterTsArgs):
    logger.info('correcting illogical palm forest / sugar sequences...')
    npalm_for = store_count('palm_for', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['palm_for'])).sum(dim="time").astype('uint8'))
    ## change sugar to palm forest if classified as palm forest for at least 25% of ts or is palm forest on either side of ts
    unlikely_sugar_cond = (
        (ts == ctx.lc_cats['sugar']) &
        (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['palm_for'])) &
        (ts.shift(time=1).fillna(0).isin(ctx.lc_cats['palm_for'])) &
        (npalm_for >= 2)
    )
    return apply_condition_to_timeseries(ts, unlikely_sugar_cond, ctx.lc_cats['palm_for'][0],
                                         idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_sugar_grass(ts, idx, ctx: FilterTsArgs):
    logger.info('correcting sudden sugar blips (grass-sugar-grass where most of the ts is grass)')
    ngrass = store_count('grass', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['Grass_all'])).sum(dim="time").astype('uint8'))
    ## change sugar if classified as grass in observation on either side and majority of time series is grass
    unlikely_sugar_cond = (
        (ts == ctx.lc_cats['sugar']) &
        (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['Grass_all'])) &
        (ts.shift(time=1).fillna(0).isin(ctx.lc_cats['Grass_all'])) &
        (ngrass >= 4)
    )
    ##  fill with the majority grass type from the specific ts sequence
    grass_ras = next(key for key in ctx.base_rasters if key.startswith('grass'))
    return apply_condition_to_timeseries(ts, unlikely_sugar_cond, ctx.base_rasters[grass_ras],
                                         idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')
    
def _retouch_banana_wet(ts, idx, ctx: FilterTsArgs):
    logger.info('correcting illogical banana grass sequences...')
    npalm_for = store_count('palm_for', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['palm_for'])).sum(dim="time").astype('uint8'))
    nwet_grass = store_count('wet_grass', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['LowVeg_wet'])).sum(dim="time").astype('uint8'))
    nwet_med = store_count('wet_med', ctx.count_cache, lambda: (ts.isin(ctx['MedVeg_wet'])).sum(dim="time").astype('uint8'))
    unlikely_banana_cond = (ts == ctx.lc_cats['banana']) & ((nwet_grass + nwet_med + npalm_for) >= 5)
    return apply_condition_to_timeseries(ts, unlikely_banana_cond, ctx.lc_cats['MedVeg_wet'][0],
                                       idx, ctx.ts_files[0], ctx.params, region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_palmfor_grass_aggressive(ts, idx, ctx: FilterTsArgs):
    ## Especially a problem in the Cerrado in Py. Assumes anything palm forest at/near end of ts was always palm forest.
    ## TODO: check -- this may be too aggressive!
    logger.info('correcting illogical palm forest to grass sequences...')
    ts_yrs = ts['time'].values
    gtmix = ctx.lc_cats['gtmix'][0]
    palm_for = ctx.lc_cats['palm_for'][0]
    base_cond = (ts.sel(time=ts_yrs[-1]) == palm_for) | (ts.sel(time=ts_yrs[-2]) == palm_for)
    fill_val = xr.where(ctx.base_rasters['forest'] == palm_for, palm_for, gtmix)
    return apply_condition_to_timeseries(ts, base_cond, fill_val,
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_palmfor_wetgrass(ts, idx, ctx: FilterTsArgs):
    logger.info('correcting illogical palm_forest and wet_grass mixtures...')
    ## corrects cases where wetter low veg is misclassified as palm forest (similar to the banana case above)
    nwet_grass = store_count('wet_grass', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['LowVeg_wet'])).sum(dim="time").astype('uint8'))
    nwet_med = store_count('wet_med', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['MedVeg_wet'])).sum(dim="time").astype('uint8'))
    unlikely_palm_forest_cond = ts.isin(ctx.lc_cats['palm_for']) & ((nwet_grass + nwet_med) > 5)
    return apply_condition_to_timeseries(ts, unlikely_palm_forest_cond, ctx.lc_cats['MedVeg_wet'][0], 
                                       idx, ctx.ts_files[0], ctx.params, region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_grass_forest_aggressive(ts, idx, ctx: FilterTsArgs): 
    ## Assumes anything forest at end of ts was always forest. TODO: check -- this may be too aggressive!
    logger.info('aggressively correcting illogical grass to forest sequences...')
    ts_yrs = ts['time'].values
    ts_end = ts.sel(time=ts_yrs[-1])
    end_forest_cond = ts_end.isin(ctx.lc_cats['Trees_nat']) & (ts.sel(time=ts_yrs[-2]) == ts_end)
    return apply_condition_to_timeseries(ts, end_forest_cond, ts_end,
                                       idx, ctx.ts_files[0], ctx.params, region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_treeplant_medcrop(ts, idx, ctx: FilterTsArgs):
    logger.info('correcting illogical tree plantation / med_crop sequences...')
    ntreePlant = store_count('tree_plant', ctx.count_cache, lambda: (ts.isin(ctx.lc_catsS['Trees_plantation'])).sum(dim="time").astype('uint8'))
    unlikely_medcrop_cond = ts.isin(ctx.lc_cats['med_crops']) & (ntreePlant >= 3)
    return apply_condition_to_timeseries(ts, unlikely_medcrop_cond, ctx.lc_cats['Trees_plantation'][0], 
                                       idx, ctx.ts_files[0], ctx.params, region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_rice_water(ts, idx, ctx: FilterTsArgs):
    logger.info('correcting illogical rice-water sequences...')
    nwater = store_count('water', ctx.count_cache, lambda: (ts.isin(ctx.lc_catsS['NoVeg_water'])).sum(dim="time").astype('uint8'))
    ## if classified as rice, but classified as water at least half of the time series, change to water. 
    ##    Note: rice will be reinstated later if within segmented polygon
    unlikely_rice_cond = (ts == ctx.lc_cats['rice']) & (nwater >= 4)
    ts = apply_condition_to_timeseries(ts, unlikely_rice_cond, ctx.lc_cats['NoVeg_water'][0]
                                       ,idx, ctx.ts_files[0], ctx.params, region_key='illogical_regions', region_file_key='illogical_region_file')
    ## if classified as water but rice before and after, change back to rice
    flooded_rice_cond = (
        ts.isin(ctx.lc_cats['NoVeg_water']) &
        (ts.shift(time=-1).fillna(0) == ctx.lc_cats['rice']) &
        (ts.shift(time=1).fillna(0) == ctx.lc_cats['rice'])
    )
    return apply_condition_to_timeseries(ts, flooded_rice_cond, ctx.lc_cats['rice'],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_rice_built(ts, idx, ctx: FilterTsArgs):
    logger.info('correcting illogical rice-built sequences...')
    nrice = store_count('rice', ctx.count_cache, lambda: (ts == ctx.lc_cats['rice']).sum(dim="time").astype('uint8'))
    nwater = store_count('water', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['NoVeg_water'])).sum(dim="time").astype('uint8'))
    unlikely_built_cond = ts.isin(ctx.lc_cats['NoVeg_built']) & (nrice >= 2) & ((nrice + nwater) >= 4)
    return apply_condition_to_timeseries(ts, unlikely_built_cond, ctx.lc_cats['rice'],
                                       idx, ctx.ts_files[0], ctx.params, region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_wetgrass_crop(ts, idx, ctx: FilterTsArgs):
    logger.info("removing very brief crop blips with wet grass on either side...")
    ## fill crop blips: if a pixel is wetgrass at time t-1 and t+1 but low crop inbetween, fill with wetgrass
    crop_blip_cond = (
        ts.isin(ctx.lc_cats['LowVeg_crop']) &
        (ts.shift(time=1).fillna(0).isin(ctx.lc_cats['LowVeg_wet'])) &
        (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['LowVeg_wet']))
    )
    return apply_condition_to_timeseries(ts, crop_blip_cond, ctx.lc_cats['LowVeg_wet'][0],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_grass_crop(ts, idx, ctx: FilterTsArgs):
    ##TODO: Maybe do this outside so that the fill can be dynamic -- dominant grass or grass at point t+1 rather than static fill.
    logger.info("removing very brief crop blips in grass-dominated time series...")
    ## fill crop blips: if a pixel is mostly grass across the time series and grass at time t-1 or t+1 but low crop inbetween, fill with grass
    ##    Note: will retouch after to retain crop in segmented polys 
    ##          but analysis WARNING: will remove true grass --> crop gorwth in final year of ts unless in segmented polygon
    ngrass = store_count('grass', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['Grass_all'])).sum(dim="time").astype('uint8'))
    crop_blip_cond = (
        ts.isin(ctx.lc_cats['LowVeg_crop']) &
        (ngrass >=4) &
        (((ts.shift(time=1).fillna(0).isin(ctx.lc_cats['Grass_all']))) |
        (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['Grass_all'])))
    )
    return apply_condition_to_timeseries(ts, crop_blip_cond, ctx.lc_cats['Grass_all'][0],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')
    
def _retouch_wetgrass_built(ts, idx, ctx: FilterTsArgs):
    logger.info("removing very brief built blips with wet grass on either side...")
    ## fill built blips: if a pixel is wetgrass at time t-1 and t+1 but built inbetween, fill with wetgrass
    built_blip_cond = (
        ts.isin(ctx.lc_cats['NoVeg_built']) &
        (ts.shift(time=1).fillna(0).isin(ctx.lc_cats['LowVeg_wet'])) &
        (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['LowVeg_wet']))
    )
    return apply_condition_to_timeseries(ts, built_blip_cond, ctx.lc_cats['LowVeg_wet'][0],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_medcrop_wetshrub(ts, idx, ctx: FilterTsArgs):
    logger.info("fixing medium crop that is probably actually wet shrub...")
    ## reclass medium crop to wet shrub if wet shrub or wet grass at time t+1 and t-1. Note: will retouch after to retain crop in segmented polys
    unlikely_medcrop_cond = (
        ts.isin(ctx.lc_cats['med_crops']) &
        ((ts.shift(time=1).fillna(0).isin(ctx.lc_cats['MedVeg_wet'])) | (ts.shift(time=1).fillna(0).isin(ctx.lc_cats['LowVeg_wet']))) &
        ((ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['MedVeg_wet'])) | (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['LowVeg_wet'])))
    )
    return apply_condition_to_timeseries(ts, unlikely_medcrop_cond, ctx.lc_cats['MedVeg_wet'][0],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_medcrop_shrub(ts, idx, ctx: FilterTsArgs):
    logger.info("fixing medium crop that is probably actually other shrub...")
    ## reclass medium crop to shrub if dry grass at time t+1 and t-1. Note: will retouch after to retain crop in segmented polys
    unlikely_medcrop_cond = (
        ts.isin(ctx.lc_cats['med_crops']) &
        ((ts.shift(time=1).fillna(0).isin(ctx.lc_cats['dry_grass'])) | (ts.shift(time=1).fillna(0)==(ctx.lc_cats['MedVeg_noncrop'][0]))) &
        (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['dry_grass']) | (ts.shift(time=-1).fillna(0)==ctx.lc_cats['MedVeg_noncrop'][0]))
    )
    return apply_condition_to_timeseries(ts, unlikely_medcrop_cond, ctx.lc_cats['MedVeg_noncrop'][0],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_single_crop_blip(ts, idx, ctx: FilterTsArgs):
    logger.info("removing time-series crop specs...")
    ## reclass crop to grass if only crop once in time series.  Note, will retouch after to retain crops within segmented polygons.
    ## analysis WARNING: this will remove new crops in the last year of ts unless in segmented polygons 
    ##    (so true smallholder area will likely be reduced) Recommend running only in problem regions (e.g. 2 in Py) 
    ncrop = store_count('LowVeg_crop', ctx.count_cache, lambda: (ts == ctx.lc_cats['LowVeg_crop']).sum(dim="time").astype('uint8'))
    crop_spec_cond = ( 
        ts.isin(ctx.lc_cats['LowVeg_crop']) & (ncrop == 1))
    return apply_condition_to_timeseries(ts, crop_spec_cond, ctx.lc_cats['dry_grass'][0],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')
    
def _retouch_grass_built(ts, idx, ctx: FilterTsArgs):
    logger.info("removing very brief built blips where grass dominates time series and is on at least one side...")
    ## fill built blips: if a pixel is classified as built but is dominantly grass in the time series and is grass at time t-1 or t+1, fill with bare
    ## analysis WARNING: this will remove legitimate grass --> built growth in last year of ts
    ngrass = store_count('grass', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['Grass_all'])).sum(dim="time").astype('uint8'))
    built_blip_cond = (
        ts.isin(ctx.lc_cats['NoVeg_built']) &
        (ngrass >=4) &
        ((ts.shift(time=1).fillna(0).isin(ctx.lc_cats['Grass_all']))) |
        (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['Grass_all']))
    )
    return apply_condition_to_timeseries(ts, built_blip_cond, ctx.lc_cats['NoVeg_bare'][0],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_built_non_built(ts, idx, ctx: FilterTsArgs):
    logger.info("filling in built gaps...")
    ## fill built gaps: if a pixel is classified as built at time t-1 and t+1, it should be built at time t
    built_gap_cond = (
        ~ts.isin(ctx.lc_cats['NoVeg_built']) &
        (ts.shift(time=1).fillna(0).isin(ctx.lc_cats['NoVeg_built'])) &
        (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['NoVeg_built']))
    )
    return apply_condition_to_timeseries(ts, built_gap_cond, ctx.lc_cats['NoVeg_built'][0],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')
    
def _retouch_forest_treeplant(ts, idx, ctx: FilterTsArgs):
    logger.info("removing switches from tree-plantation to forest and vice-versa")
    ntreePlant = store_count('tree_plant', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['Trees_plantation'])).sum(dim="time").astype('uint8'))
    nforest = store_count('forest_nat', ctx.count_cache, lambda: (ts.isin(ctx.lc_cats['Trees_nat'])).sum(dim="time").astype('uint8'))

    unlikely_natfor_cond = (
        ts.isin(ctx.lc_cats['Trees_nat']) &
        (ntreePlant > nforest) &
        (ts.shift(time=1).fillna(0) > ctx.lc_cats['first_mature']) &
        (ts.shift(time=2).fillna(0) > ctx.lc_cats['first_mature'])
    )
    ts = apply_condition_to_timeseries(ts, unlikely_natfor_cond, ctx.lc_cats['Trees_plantation'][0], 
                                       idx, ctx.ts_files[0], ctx.params, region_key='illogical_regions', region_file_key='illogical_region_file')

    unlikely_treeplant_cond = (
        ts.isin(ctx.lc_cats['Trees_plantation']) &
        (ntreePlant < nforest) &
        (ts.shift(time=1).fillna(0) > ctx.lc_cats['first_mature']) &
        (ts.shift(time=2).fillna(0) > ctx.lc_cats['first_mature'])
    )
    return apply_condition_to_timeseries(ts, unlikely_treeplant_cond, ctx.base_rasters['forest'],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_forest_brieflow(ts, idx, ctx: FilterTsArgs):
    logger.info("removing very brief forest blips...")
    ## fill forest blips: if a pixel is forest at time t-1 and t+1 but low veg inbetween, fill with value from time t-1
    stable_open_forest = ctx.lc_cats['forest_open_stable']  ## open forest that should not fluctuate w/ mature/undisturbed
    blip_filler_stable = ts.shift(time=1).where(
        (ts < ctx.lc_catsS['first_medveg']) &
        (ts.shift(time=1).fillna(0).isin(stable_open_forest)) &
        (ts.shift(time=-1).fillna(0).isin(stable_open_forest)),
        ts
    )
    ## do not treat time=0, as this will cause NAs with shift
    ts = blip_filler_stable.where(blip_filler_stable.time != blip_filler_stable.time[0], ts)

    unstable_for = ctx.lc_cats['dense_for'] + ctx.lc_cats['open_for']
    ## if dense forest, fill with open forest. TODO: consider demoting to med_veg or removing -- may erase real disturbances
    dense_forest_blip_cond = (
        (ts < ctx.lc_cats['first_medveg']) &
        (ts.shift(time=1).fillna(0).isin(unstable_for)) &
        (ts.shift(time=-1).fillna(0).isin(unstable_for))
    )
    return apply_condition_to_timeseries(ts, dense_forest_blip_cond, ctx.lc_cats['open_for'][0], 
                                       idx, ctx.ts_files[0], ctx.params, region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_grass_to_forest(ts, idx, ctx: FilterTsArgs):
    logger.info("retouching grass that grew into forest in a year...")
    ## Correct illogical regrowth: grass straight to forest corrected to grass to shrub
    illogical_growth_cond = (
        ts.isin(ctx.lc_cats['Trees_nat']) &
        (ts.shift(time=1).fillna(255) < ctx.lc_cats['first_highveg']) &
        (ts.shift(time=2).fillna(255) < ctx.lc_cats['first_medveg'])
    )
    return apply_condition_to_timeseries(ts, illogical_growth_cond, ctx.lc_cats['MedVeg_noncrop'][0],
                                       idx, ctx.ts_files[0], ctx.params, region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_grass_to_palmforest(ts, idx, ctx: FilterTsArgs):
    ## change palm forest to grass-tree-mix if two preceding observations are lower vegetation
    illogical_palm_growth_cond = (
        (ts == ctx.lc_cats['palm_for'][0]) &
        (ts.shift(time=1).fillna(255) < ctx.lc_cats['first_highveg']) &
        (ts.shift(time=2).fillna(255) < ctx.lc_cats['first_medveg'])
    )
    return apply_condition_to_timeseries(ts, illogical_palm_growth_cond, ctx.lc_cats['gtmix'][0],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_twmix_medwet(ts, idx, ctx: FilterTsArgs):
    logger.info("retouching grass that grew into palm forest in a year...")
    ## demote tree-water-mix to med wetveg if lower veg for previous two observations
    demote_wet_trees_cond = (
        (ts == ctx.lc_cats['tree_water_mix']) &
        (ts.shift(time=1).fillna(255) < ctx.lc_cats['first_highveg']) &
        (ts.shift(time=2).fillna(255) < ctx.lc_cats['first_medveg'])
    )
    return apply_condition_to_timeseries(ts, demote_wet_trees_cond, ctx.lc_cats['MedVeg_wet'][0],
                                       idx, ctx.ts_files[0], ctx.params, region_key='illogical_regions', region_file_key='illogical_region_file')

def _retouch_noplant_plant(ts, idx, ctx: FilterTsArgs):
    logger.info("retouching young tree plantations...")
    ## Any med_veg to tree_plant corrected to young tree_plant to tree_plant
    young_plant_cond = (
        (ts >= ctx.lc_cats['first_medveg']) &
        (ts < ctx.lc_cats['first_mature']) &
        (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['Trees_plantation']))
    )
    corrected1= apply_condition_to_timeseries(ts, young_plant_cond, ctx.lc_cats['TreePlant_young'][0],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

    ## recompute against the now-updated ts to catch 2-yr growth that the first pass didn't

    baby_plant_cond = (
        (ts < ctx.lc_cats['first_medveg']) &
        (ts >= ctx.lc_cats['first_veg']) &
        (ts.shift(time=-1).fillna(0).isin(ctx.lc_cats['Trees_plantation']))
    )
    corrected2 =  apply_condition_to_timeseries(corrected1, baby_plant_cond, ctx.lc_cats['TreePlant_'],
                                       idx, ctx.ts_files[0], ctx.params,region_key='illogical_regions', region_file_key='illogical_region_file')

    return corrected2

CORRECTIONS = {
    'sugar-palm': _retouch_sugar_palm,
    'sugar-grass':_retouch_sugar_grass,
    'banana-wet': _retouch_banana_wet,
    'palm_for-grass-aggressive': _retouch_palmfor_grass_aggressive,
    'palm_for-wetgrass': _retouch_palmfor_wetgrass,
    'grass-forest-aggressive': _retouch_grass_forest_aggressive,
    'tree_plant-med_crop': _retouch_treeplant_medcrop,
    'rice-water': _retouch_rice_water,
    'rice-built': _retouch_rice_built,
    'forest-treeplant': _retouch_forest_treeplant,
    'forest-brieflow': _retouch_forest_brieflow,
    'grass-to-forest': _retouch_grass_to_forest,
    'grass-to-palmforest': _retouch_grass_to_palmforest,
    'twmix-medwet': _retouch_twmix_medwet,
    'noplant-plant': _retouch_noplant_plant,
    'wetgrass-crop': _retouch_wetgrass_crop,
    'wetgrass-built':_retouch_wetgrass_built,
    'built-non-built':_retouch_built_non_built,
    'grass-built':_retouch_grass_built,
    'grass-crop':_retouch_grass_crop,
    'medcrop-wetshrub':_retouch_medcrop_wetshrub,
    'medcrop-shrub':_retouch_medcrop_shrub,
    'singleton-crop': _retouch_single_crop_blip
    
}

    