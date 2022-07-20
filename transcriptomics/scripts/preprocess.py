import os
import pandas as pd

# import libraries
from transcriptomics import data as preprocess
from transcriptomics.constants import Defaults
from transcriptomics import utils as cio

def get_default_config():
        config = {
                'atlas': None,
                'structure': None,
                # 'exact': True,
                'tolerance': 2,
                'ibf_threshold': 0.5, 
                'probe_selection': 'diff_stability',
                'lr_mirror': None,
                'sim_threshold': None,
                'missing': None,
                'gene_norm': 'zscore', # default is 'srs'
                'sample_norm': 'zscore', # default is 'srs'
                'no_corrected_mni': False,
                'norm_all': True, 
                'norm_structures': True,
                'no_reannotated': False,
                'save_counts': True,
                'save_donors': True,
                'region_agg': 'donors',# other option is 'samples'
                'agg_metric': 'mean',
                'donors': 'all',
                'data_dir': None,
                'verbose': 1,
                'n_proc': 1,
                # 'return_samples': True, # I added in this parameter to return an unaggregated dataframe (labelled samples x genes)
                }
        
        return config

def run(
        atlas_name='MDTB10',
        structure='Cerebellum'
        ):

        """ Run preprocessing routine for `atlas`

        Args: 
                atlas_name (str): default is 'Yeo7'. options are 'SUIT10', 'SUIT26', 'Buckner7'
        'Buckner17', 'MDTB10', 'MDTB20', 'Yeo7', 'Yeo17'
                structure (str): default is 'Cortex'. other options are 'Cerebellum'
        """
        # check whether dirs are true
        cio.make_dirs(Defaults.INTERIM_DIR)
        cio.make_dirs(Defaults.ATLAS_DIR)

        # get default config
        config = get_default_config()
        config.update({'atlas': atlas_name, 'structure': structure, 'data_dir': str(Defaults.RAW_DATA_DIR)})

        # download AHBA files
        # preprocess.download_AHBA()
        
        # download atlases for `atlas_name` and return gifti(s)
        gifti = preprocess.get_atlases(atlas_name, suffix='gii')
        nifti = preprocess.get_atlases(atlas_name, suffix='nii', space='MNI')

        # get atlas (and atlas info) if it doesn't already exist
        fname_atlas = os.path.join(Defaults.ATLAS_DIR, f'{atlas_name}-info.csv')
        if not os.path.isfile(fname_atlas):
                atlas_info = preprocess.get_atlas_info(gifti=gifti[0])
                atlas_info.to_csv(fname_atlas)
                print(f'saving atlas info at {fname_atlas} \n')

        # run main abagen preprocessing routine
        fname_data = os.path.join(Defaults.INTERIM_DIR, f'expression-alldonors-{atlas_name}-cleaned.csv')
        if not os.path.isfile(fname_data):
                data = preprocess.process_gene_expression(config, img=nifti, atlas_info=pd.read_csv(fname_atlas))
                data.to_csv(fname_data)
                print(f'preprocessing gene expression data using abagen and saved here: {fname_data}')
        
        # write out expression data to file for all donors
        cio.save_dict_as_JSON(os.path.join(Defaults.INTERIM_DIR, f"config_{atlas_name}.json"), config)
        print(f'saved out config_{atlas_name} to json') 

if __name__ == "__main__":
    run()