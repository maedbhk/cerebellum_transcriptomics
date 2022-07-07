from SUITPy import atlas as catlas
from imageUtils import atlas, helper_functions

# import libraries
from transcriptomics.data import DataSet

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

def run(
        atlas_name='Yeo_7Networks',
        structure='cortex'
        ):

        """ Run preprocessing routine for `atlas`: options are 'Yeo_7Networks', 
        'Yeo_17Networks', 'MDTB10_dseg', 'MDTB10-subregions', 'SUIT', 'Buckner-7', 'Buckner-17'
        """

        data = DataSet()

        # download AHBA files
        data.download_data()
        
        # download atlases for `atlas_name` and return gifti(s)
        gifti = download_atlases(atlas_name, suffix='gii')
        nifti = download_atlases(atlas_name, suffix='nii')

        # get atlas (and atlas info)
        atlas_info = data.get_atlas_info(gifti=gifti)

        # run main abagen preprocessing routine
        if structure=='cortex':
                data.process_gene_expression(img=gifti, atlas_info=atlas_info)
        elif structure=='cerebellum':
                data.process_gene_expression(img=nifti, atlas_info=atlas_info)

if __name__ == "__main__":
    run()