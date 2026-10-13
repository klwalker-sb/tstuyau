from pathlib import Path
from ..handler import logger

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

def find_project_root(path, project_name):
    """Return the part of `path` up to and including the `project_name` folder, which is where all shared folders live"""
    path = Path(path)
    parts = path.parts
    if project_name not in parts:
        if 'paraguay_lc' in parts:  ##legacy code
            project_name = 'paraguay_lc'
        else:
            logger.warning(f"ERROR -- {project_name}' not found in path: {path}. Probably need to add project_name param to script")
    # use the last occurrence in case the name repeats higher up the tree
    idx = len(parts) - 1 - parts[::-1].index(project_name)
    return Path(*parts[:idx + 1])

class SharedPaths:
    def __init__(self, params):
        
        #######################################################################################################################
        ## Universal paths
        #######################################################################################################################
        ## Neither URL is working. TODO: find current url
        #self.hgt_url = 'https://e4ftl01.cr.usgs.gov/provisional/MEaSUREs/NASADEM'
        #self.hgt_url = 'https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11'

        ########################################################################################################################
        ## shared paths for all cells
        ########################################################################################################################
        self.root = find_project_root(params['main_path'], params['project_name'])
        self.backup_root = find_project_root(params['backup_path'], params['project_name'])
        self.scratch_root = Path(params['scratch_dir'])

        self.political = self.root/ 'vector' / 'political'
        self.segmentation = self.root / 'segmentation'
        self.test =self.root / 'testing'

        self.optimization = self.backup_root /'optimization'
        self.ancillary = self.backup_root /'ancillary'
        self.figs =self.backup_root /'outputs'/'visualizations'
        self.datasum = self.backup_root /'outputs'/'summary_data'
        self.mosaics = self.backup_root /'outputs'/'mosaics'
        self.tiles = self.backup_root /'tiles'
        self.dldb = self.backup_root / 'cell_processing_dl.csv'
        self.logfiles = self.backup_root / 'archived_logfiles'
        self.tssigs = self.backup_root / 'vector'/'sample_ts_signatures'
        self.fmoddict = self.backup_root / 'Feature_Models.json'
        self.singfeatdict = self.backup_root / 'ancillary_var_dict.json'
        self.training = self.backup_root / 'vector'/'pts_training'
        self.calval = self.backup_root /'vector'/'pts_calval'
        self.valid = self.calval / 'validation_data'
        self.optimization = self.backup_root /'optimization'

        self.segdir_temp =self.scratch_root / 'cnet'
        
        if params['classify']['temp_mod'] and (params['sample_model']['num_subsamples'] > 0):
            self.temp_modeling = params['scratch_dir'] / 'modeling'
            self.trainfeatsets = self.temp_modeling / 'fsets'
            self.trainptsets = self.temp_modeling /'pt_subsets'
            self.hos = self.temp_modeling / 'hos'
            self.classification = self.temp_modeling / 'classification'
        else:
            self.trainfeatsets = self.training /'features'
            self.hos = self.calval / 'fixedHOs'
            self.classification = self.backup_root /'classification'
        
        if params['classify']['temp_mod']:
            self.fulltrainsets = self.temp_modeling / 'vardfs'
        else:
            self.fulltrainsets = self.backup_root /'classification' / 'inputs'
            self.trainptsets = self.training /'pt_subsets'

class ProjectPaths(SharedPaths):
    """Shared paths plus cell-level paths when grid is given."""
    def __init__(self, params, grid=None):
        super().__init__(params)
        self.grid = grid
        if grid:

            ########################################################################################################################
            ## Individual cell paths
            ########################################################################################################################
            ## ms is default locations for products processed therough brdf by Eostac.
            ##     the backup path is optional for storing later-stage (e.g. time-series products & classification results)
            ##     if a separate backup space does not exist, can set equal to main path in parameters.  
            self.ms = params['main_path'] / f'{grid:06d}' / 'brdf'
            self.bk = params['backup_path'] / f'{grid:06d}'
            self.dem = params['main_path'] / f'{grid:06d}' /'terrain'/'dem'
            
            ## for legacy purposes. TODO remove when no longer needed (but may need to change Eostac too)
            if (params['project_ver'] == 'Py_0') or (params['project_ver'] == 'Biltong_0'):
                self.ms = params['main_path'] / 'ms' / f'{grid:06d}' / 'brdf'
                self.bk = params['backup_path'] / 'bk' / f'{grid:06d}'
                if not self.ms.is_dir():
                    self.ms = params['main_path'] / f'{grid:06d}' / 'brdf'
                if not self.bk.is_dir():
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
                
            self.scratch = params['scratch_dir'] / f'{grid:06d}'

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
