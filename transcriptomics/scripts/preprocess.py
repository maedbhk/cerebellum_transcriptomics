from SUITPy import atlas as catlas
from imageUtils import atlas, helper_functions

# import libraries
from transcriptomics import data as preprocess

def download_atlases(atlas_name='Yeo_7Networks', suffix='gii'):
        """download atlases that will be used in this script 
        (only downloads once)
        """
        if 'Yeo' in atlas_name:
                files = atlas.fetch_yeo_2011()
        elif 'Buckner' in atlas_name:
                files = catlas.fetch_buckner_2011()
        elif 'MDTB' in atlas_name:
                files = catlas.fetch_king_2019(data='atl')
        elif 'SUIT' in atlas_name:
                files = catlas.fetch_diedrichsen_2009()

        fpaths_all = []
        for fpath in files['files']:
                if  all(x in fpath for x in [atlas_name, suffix]):
                        fpaths_all.append(fpath)
        
        return fpaths_all

def config_defaults():
        config = {
                'atlas': None,
                'exact': True,
                'tolerance': 2,
                'ibf_threshold': 0.5,
                'probe_selection': 'diff_stability',
                'lr_mirror': False,
                'gene_norm': 'zscore', # default is 'srs'
                'sample_norm': 'zscore', # default is 'srs'
                'corrected_mni': True,
                'reannotated': True,
                'return_counts': True,
                'return_donors': True,
                'region_agg': 'donors', # other option is 'samples'
                'agg_metric': 'mean',
                'donors': 'all',
                'verbose': 1,
                'n_proc': 1,
                'return_samples': False
                } # I added in this parameter to return an unaggregated dataframe (labelled samples x genes)

        return config

def run(
        atlas_name='Yeo_7Networks',
        structure='cortex'
        ):

        """ Run preprocessing routine for `atlas`: options are 'Yeo_7Networks', 
        'Yeo_17Networks', 'MDTB10_dseg', 'MDTB10-subregions', 'SUIT', 'Buckner-7', 'Buckner-17'
        """

        # download AHBA files
        preprocess.download_data()
        
        # download atlases for `atlas_name` and return gifti(s)
        gifti = download_atlases(atlas_name, suffix='gii')
        nifti = download_atlases(atlas_name, suffix='nii')

        # get atlas (and atlas info)
        atlas_info = preprocess.get_atlas_info(gifti=gifti)

        # get default config for preprocessing
        config = config_defaults()

        # run main abagen preprocessing routine
        if structure=='cortex':
                preprocess.process_gene_expression(config, img=gifti, atlas_info=atlas_info)
        elif structure=='cerebellum':
                preprocess.process_gene_expression(config, img=nifti, atlas_info=atlas_info)

if __name__ == "__main__":
    run()