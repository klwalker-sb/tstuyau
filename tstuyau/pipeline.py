#!/usr/bin/env python

"""
This code processes Landsat and Sentinel cubes

1. Move or flag images with no data
2. Co-register Sentinel images to Landsat
3. Mask Landsat and Sentinel images
4. (Optional) Fuse Landsat and Sentinel images with StarFM
5. (Optional) topographic correction
6. (Optional) Segment (SACFEI) the pan-sharpened images
7. Generate feature stack for modelling
"""

import os
from pathlib import Path
import argparse
from . import __version__
from .handler import logger
from . import steps


import yaml

class Config(object):
    def load(self):
        with open(self.config_file, 'r') as pf:
            self.params = yaml.load(pf, Loader=yaml.FullLoader)

class Tasks(Config):
    def __init__(
        self,
        config_file=None,
        config_updates=None
    ):
        self.params = None
        self.config_file = config_file

        if not self.config_file:

            p = Path(os.path.abspath(os.path.dirname(__file__)))
            self.config_file = str(p / 'config' / 'config.yaml')

        # Load the parameters
        self.load()

        if config_updates:

            # ['key:arg', 'key:arg']
            for config_pair in config_updates:

                items = config_pair.split(':')

                def check_for_list(eval_items):
                    if eval_items.startswith('['):
                        try:
                            eval_items = eval_items[1:-1].split(',')
                            eval_items = list(map(int, eval_items))
                        except:
                            pass
                    elif ',' in eval_items:
                        try:
                            eval_items = eval_items.split(',')
                            eval_items = list(map(str, eval_items))
                        except:
                            pass
                    return eval_items
                
                def check_for_fake_str(eval_items):
                    if eval_items == "True":
                        eval_items = True
                    elif eval_items == "False":
                        eval_items = False
                    if eval_items == "None":
                        eval_items = None
                    
                    return eval_items

                items[-1] = check_for_list(items[-1])
                items[-1] = check_for_fake_str(items[-1])

                if len(items) == 2:

                    try:
                        self.params[items[0]] = eval(items[1])
                    except:
                        self.params[items[0]] = items[1]

                elif len(items) == 3:

                    try:
                        self.params[items[0]][items[1]] = eval(items[2])
                    except:
                        self.params[items[0]][items[1]] = items[2]

                elif len(items) == 4:

                    try:
                        self.params[items[0]][items[1]][items[2]] = eval(items[3])
                    except:
                        self.params[items[0]][items[1]][items[2]] = items[3]

        main_path = self.params['main_path']
        self.params['main_path'] = Path(main_path)

        backup_path = self.params['backup_path']
        self.params['backup_path'] = Path(backup_path)

        scratch_path = self.params['scratch_dir']
        self.params['scratch_path'] = Path(scratch_path)

    def preprocess(self):
        logger.info('  Co-registering Sentinel images ...')
        #self.move_no_data()
        self.coregister()

    def masking(self):
        logger.info('  Creating cloud masks ...')
        self.mask_clouds()

    def move_nodata(self):
        logger.info('  Moving images with no data ...')
        steps.move_nodata(self.params)

    def coregister(self):
        logger.info('  Co-registering images ....')
        steps.coregister(self.params)

    def mask_clouds(self):
        logger.info('  Making masks to remove clouds, etc ...')
        steps.mask_clouds(self.params)

    def topo(self):
        logger.info('  Normalizing topography ...')
        steps.adjust_topo(self.params)

    def fusion(self):
        logger.info('  Fusing Landsat images ...')
        steps.fuse_sensors(self.params)

    def reconstruct(self):
        logger.info('  Reconstructing time series ...')
        steps.reconstruct(self.params)

    def reindex_si(self):
        logger.info('  Reindexing time series ...')
        steps.reindex_si(self.params)

    #def segment(self):
    #    logger.info('  Segmenting objects ...')
    #    steps.segment(self.params)

    def reclassify_raster(self):
        logger.info('   Reclassifying raster...')
        steps.reclassify_raster(self.params)

    def make_polygon_features(self):
        logger.info('   Calculating aggregate polygon stats...')
        steps.make_polygon_features(self.params)
        
    def prep_training_ts_for_segmentation(self):
        logger.info('   prepping training data...')
        steps.prep_training_ts_for_segmentation(self.params)
        
    def vectorize_seg_results(self):
        logger.info('   Vectorizing segmentation results...')
        steps.vectorize_seg_results(self.params)

    def segmentation_accuracy(self):
        logger.info('   Assessing segmentation accuracy...')
        steps.segmentation_accuracy(self.params)  
    
    def make_var_stack(self):
        logger.info('  Making variable stack ...')
        steps.make_variable_stack(self.params)

    def make_var_dataframe(self):
        logger.info('  Making variable dataframe ...')
        steps.make_var_dataframe(self.params)

    def format_ptfeat_set(self):
        logger.info('  Formatting sample and variable dataframe ...')
        steps.format_ptfeat_set(self.params)

    def make_and_score_model(self):
        logger.info('  Making new model ...')
        steps.make_and_score_model(self.params)
        
    def iterate_sample_model(self):
        logger.info('    Iterating over models...')
        steps.iterate_sample_model(self.params)
        
    def iterate_all_model_components(self):
        logger.info('    Iterating over models...')
        steps.iterate_all_model_components(self.params)

    def optimize_feature_model(self):
        logger.info('    Optimizing feature model...')
        steps.optimize_feature_model(self.params)

    def make_ts_composite(self):
        logger.info('    Making composite image ...')
        steps.make_ts_composite(self.params)

    def classify_timestep(self):
        logger.info('   Classifying land cover for single time step...')
        steps.classify_timestep(self.params)
        
    def classify_CRF(self):
        logger.info('   Classifying land cover with conditional random fileds...')
        steps.classify_CRF(self.params)

    def mosaic(self):
        logger.info('   Mosaicking classified images...')
        steps.mosaic_cells(self.params)

    #def assess(self):
    #    logger.info('  Assessing land cover predictions ...')
    #    steps.assess(self.params)

    def clean(self):
        logger.info('  Cleaning data ...')
        steps.clean(self.params)

    def compress(self):
        logger.info('  Compressing data ...')
        steps.compress(self.params)

    def sample_timeseries(self):
        logger.info('  Getting time series data for sample points ...')
        steps.sample_timeseries(self.params)

    def pre_post_df(self):
        logger.info('  Getting pre- post- event obs ...')
        steps.pre_post_df(self.params)

    def pre_post_separability(self):
        logger.info('  Getting summary stats and separability for pre- post- observations ...')
        steps.pre_post_separability(self.params)
        
    def plot_timeseries(self):
        logger.info('  Plotting time series ...')
        steps.plot_timeseries(self.params)

    def status(self):
        logger.info('  Checking step status ...')
        steps.status(self.params)

    def dl_check(self):
        logger.info('  Checking dl logs ...')
        steps.check_dl_logs(self.params)

    def make_thumbnails(self):
        logger.info('  making thumbnails ...')
        steps.make_thumbnails(self.params)


def main():

    parser = argparse.ArgumentParser(description='A pipeline for image processing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    avail_tasks = ['preprocess', 'move_nodata', 'mask', 'topo', 'fusion', 'reconstruct', 'reindex_si', 'reclassify_raster', 
                   'make_polygon_features', 'segment', 'make_ts_composite', 'make_var_stack', 'make_var_dataframe', 
                   'format_ptfeat_set', 'make_and_score_model','iterate_sample_model', 'iterate_all_model_components', 
                   'optimize_feature_model', 'classify_timestep', 'classify_CRF', 'mosaic', 'clean', 'assess', 'compress', 
                   'sample_timeseries', 'plot_timeseries', 'pre_post_df', 'pre_post_separability', 'dl_check', 'status', 
                   'version', 'make_thumbnails', 'vectorize_seg_results', 'segmentation_accuracy', 'prep_training_ts_for_segmentation']

    parser.add_argument('tasks', metavar='task', nargs='+', help='The tasks to submit', default=None,
                        choices=avail_tasks)

    parser.add_argument('--config', dest='config_file', help='The config file', default=None)
    parser.add_argument('--config-updates', dest='config_updates',
                        help='Updates for the configuration file (format should be key:arg key:arg key:arg)',
                        default=None, nargs='+')

    args = parser.parse_args()

    tasks = Tasks(config_file=args.config_file,
                  config_updates=args.config_updates)

    for task in args.tasks:

        if task == 'version':
            print(__version__)
            return

        if task == 'preprocess':
            tasks.preprocess()
        if task == 'move_nodata':
            tasks.move_nodata()
        elif task == 'mask':
            tasks.masking()
        elif task == 'topo':
            tasks.topo()
        elif task == 'fusion':
            tasks.fusion()
        elif task == 'reconstruct':
            tasks.reconstruct()
        elif task == 'reindex_si':
            tasks.reindex_si()
        #elif task == 'segment':
        #    tasks.segment()
        elif task == 'reclassify_raster':
            tasks.reclassify_raster()
        elif task == 'prep_training_ts_for_segmentation':
            tasks.prep_training_ts_for_segmentation()
        elif task == 'make_polygon_features':
            tasks.make_polygon_features()
        elif task == 'vectorize_seg_results':
            tasks.vectorize_seg_results()
        elif task == 'segmentation_accuracy':
            tasks.segmentation_accuracy()
        elif task == 'make_var_stack':
            tasks.make_var_stack()
        elif task == 'make_var_dataframe':
            tasks.make_var_dataframe()
        elif task == 'format_ptfeat_set':
            tasks.format_ptfeat_set()
        elif task == 'make_and_score_model':
            tasks.make_and_score_model()
        elif task == 'iterate_sample_model':
            tasks.iterate_sample_model()
        elif task == 'iterate_all_model_components':
            tasks.iterate_all_model_components()
        elif task == 'optimize_feature_model':
            tasks.optimize_feature_mode()
        elif task == 'make_ts_composite': 
            tasks.make_ts_composite()
        elif task == 'classify_timestep':
            tasks.classify_timestep()
        elif task == 'classify_CRF':
            tasks.classify_CRF()
        elif task == 'mosaic':
            tasks.mosaic()
        #elif task == 'assess':
        #    tasks.assess()
        elif task == 'clean':
            tasks.clean()
        elif task == 'compress':
            tasks.compress()
        elif task == 'plot_timeseries':
            tasks.plot_timeseries()
        elif task == 'sample_timeseries':
            tasks.sample_timeseries()
        elif task == 'pre_post_df':
            tasks.pre_post_df() 
        elif task == 'pre_post_separability':
            tasks.pre_post_separability()
        elif task == 'dl_check':
            tasks.dl_check()
        elif task == 'status':
            tasks.status()
        elif task == 'make_thumbnails':
            tasks.make_thumbnails()


if __name__ == '__main__':
    main()
