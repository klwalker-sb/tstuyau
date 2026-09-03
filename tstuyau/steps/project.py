def get_tsdir_name(params):   
    if (params['project_ver'] == 'Py_0') or (params['project_ver'] == 'Biltong_0'):  ## legacy
        dirstr = 'ms'
    elif not params['compare_imgtype'] and not params['compare_res'] and not params['compare_procseq']:
        dirstr = ''
    elif not params['compare_procseq']:
        dirstr = f"{params['image_type']}-{int(params['res'])}" 
    elif not (params['compare_imgtype'] or params['compare_res']):
        dirstr = f"{params['procseq']}"
    else:
        dirstr = f"{params['image_type']}-{int(params['res'])-{params['procseq']}}"

    return dirstr

class ProjectPaths(object):

    def __init__(self, params, grid=None):

        #######################################################################################################################
        ## Universal paths
        #######################################################################################################################
        ## Neither URL is working. TODO: find current url
        #self.hgt_url = 'https://e4ftl01.cr.usgs.gov/provisional/MEaSUREs/NASADEM'
        #self.hgt_url = 'https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11'

        ## note: main path and backup path for cell directories are usually <project>/stac/grids
        ##  so shared files will be in grandparent directory (parents[1]. If file dept changes, this needs to be changed here)
        self.srtm = params['main_path'].parents[1] /'data'/'srtm'
        self.test = params['main_path'].parents[1] / 'testing'
        self.figs = params['backup_path'].parents[1] /'outputs'/'visualizations'
        self.datasum = params['backup_path'].parents[1] /'outputs'/'summary_data'
        self.tiles = params['backup_path'].parents[1] /'tiles'
        self.dldb = params['backup_path'].parents[1] / 'cell_processing_dl.csv'
        self.logfiles = params['backup_path'].parents[1] / 'archived_logfiles'
        self.tssigs = params['backup_path'].parents[1] / 'vector'/'sample_ts_signatures'
        self.fmoddict = params['backup_path'].parents[1] / 'Feature_Models.json'
        self.singfeatdict = params['backup_path'].parents[1] / 'ancillary_var_dict.json'
        self.training = params['backup_path'].parents[1] / 'vector'/'pts_training'
        self.calval = params['backup_path'].parents[1] /'vector'/'pts_calval'
        self.valid = self.calval / 'validation_data'
        self.optimization = params['backup_path'].parents[1] /'optimization'
        self.political = params['main_path'].parents[1] / 'vector' / 'political'
        self.segmentation = params['main_path'].parents[1] / 'segmentation'
        self.segdir_temp = params['scratch_path'] / 'cnet'
        
        if params['classify']['temp_mod'] and (params['sample_model']['num_subsamples'] > 0):
            self.temp_modeling = params['scratch_dir'] / 'modeling'
            self.trainfeatsets = self.temp_modeling / 'fsets'
            self.trainptsets = self.temp_modeling /'pt_subsets'
            self.hos = self.temp_modeling / 'hos'
            self.classification = self.temp_modeling / 'classification'
        else:
            self.trainfeatsets = self.training /'features'
            self.hos = self.calval / 'fixedHOs'
            self.classification = params['backup_path'].parents[1] /'classification'
        
        if params['classify']['temp_mod']:
            self.fulltrainsets = self.temp_modeling_dir / 'vardfs'
        else:
            self.fulltrainsets = params['backup_path'].parents[1] /'classification' / 'inputs'
            self.trainptsets = self.training /'pt_subsets'

        ########################################################################################################################
        ## Individual cell paths
        ########################################################################################################################
        self.grid = grid
        if grid:
            ## for legacy purposes. TODO remove when no longer needed (but may need to change Eostac too)
            if (params['project_ver'] == 'Py_0') or (params['project_ver'] == 'Biltong_0'):
                self.ms = params['main_path'] / 'ms' / f'{grid:06d}' / 'brdf'
                self.bk = params['backup_path'] / 'bk' / f'{grid:06d}'
                if not self.ms.is_dir():
                    self.ms = params['main_path'] / f'{grid:06d}' / 'brdf'
                if not self.bk.is_dir():
                    self.bk = params['backup_path'] / f'{grid:06d}'
            
            ## ms is default locations for products processed therough brdf by Eostac.
            ##     the backup path is optional for storing later-stage (e.g. time-series products & classification results)
            ##     if a separate backup space does not exist, can set equal to main path in parameters.  
            else:
                self.ms = params['main_path'] / f'{grid:06d}' / 'brdf'
                self.bk = params['backup_path'] / f'{grid:06d}'
            
            ## tstuyau picks up from here. 
            ## The proc directory is where it will look for images for time-series analyses 
            ##     if the default pre-processing is used (qc and coreg only after brdf), the files stay in the 
            ##     original 'brdf' folder.If other steps are run and/or comparative processing is being conducted,
            ##     the final output folder is named based on the relevant processing information
            if not params['compare_res']: 
                endstr = ''
            else:
                endstr = f"_{params['res']}m"
                     
            if not params['compare_procseq']: 
                ## this will be the same as the default self.ms above if compare_res is also False
                self.proc = params['main_path'] / f"{grid:06d}/brdf{endstr}"
            else:
                self.proc = params['main_path'] / f"{grid:06d}/{params['procseq']}{endstr}"

            ## The ts directory is where the final time-series products are stored
            dirstr = get_tsdir_name(params)
            if dirstr == '':
                self.ts = self.bk / 'brdf_ts'
            else:
                self.ts = self.bk / 'brdf_ts' / f'{dirstr}'
                
            self.scratch = params['scratch_path'] / f'{grid:06d}'

            ## these help with the cleanup process
            self.gee = self.ms.parent / 'gee'
            self.nodata = self.ms.parent / 'brdf_nodata'
            self.s2_nocoreg = self.ms.parent / 'brdf_s2_nocoreg'
            self.masks = self.ms.parent / 'brdf_masks'
            self.fusion = self.ms.parent / 'brdf_fusion'
            self.comp = self.bk / 'comp'
            self.cls = self.bk / 'cls'

            ## can prep all directories here, but prefer to do it as needed in code
            ##     (this risks making a lot of empty directories if script settings are entered wrong)
    
            #self.nodata.mkdir(parents=True, exist_ok=True)
            #self.s2_nocoreg.mkdir(parents=True, exist_ok=True)
            #self.ref.mkdir(parents=True, exist_ok=True)
            #self.masks.mkdir(parents=True, exist_ok=True)
            #self.fusion.mkdir(parents=True, exist_ok=True)
            #self.ts.mkdir(parents=True, exist_ok=True)
            #self.seg.mkdir(parents=True, exist_ok=True)
            #self.comp.mkdir(parents=True, exist_ok=True)
            #self.cls.mkdir(parents=True, exist_ok=True)
            #self.scratch.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def clean_temp(clean_path):

        for temp_file in clean_path.glob('*temp*'):
            if temp_file.is_file():
                temp_file.unlink()
