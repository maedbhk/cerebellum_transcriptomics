# import libraries
from pathlib import Path
from pickle import FALSE
from pkg_resources import resource_filename
import os
import re 
import glob
from collections import namedtuple
import seaborn as sns
from matplotlib import pyplot as plt

import abagen # library for preprocessing AHBA data
from sklearn.utils import Bunch
import nibabel as nib

import pandas as pd
import numpy as np

from SUITPy import atlas as catlas

from transcriptomics.constants import Defaults
from imageUtils import atlas, helper_functions
from transcriptomics import clustering 

GeneSubset = namedtuple("GeneSubset", ["threshold", "goi_idx"])

class DataSet: 

    def __init__(
        self, atlas='MDTB10', cutoff='top', percentile=1, 
        region_agg=True, threshold=True, remove_outliers=True, 
        normalize=True, reorder_labels=True, method='differential_stability'
        ):
        self.atlas = atlas
        self.cutoff = cutoff # 'top' or 'bottom'
        self.percentile = percentile
        self.method = method # 'differential_stability' or 'dimensionality_reduction'
        self.region_agg = region_agg
        self.threshold = threshold
        self.remove_outliers = remove_outliers
        self.normalize = normalize
        self.reorder_labels = reorder_labels

    def get_data(self):
        """ Primary function to return gene expression data filtered by various parameters: `threshold`, `cutoff`, `percentile`, `region_agg`, `normalize` etc.

        Returns: 
            dataframe (pd dataframe): filtered dataframe
        """
        # load preprocessed data
        dataframe = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{self.atlas}-cleaned.csv')

        if self.threshold:
            dataframe = self.threshold_data(dataframe)

        # remove atlas-specific outliers
        if self.remove_outliers:
            dataframe = self._remove_outliers(dataframe)

        # group data by region
        gene_id = dataframe.filter(regex=("[A-Z0-9].*")).columns.to_list()
        info = dataframe[dataframe.filter(regex=("(?!^.*[A-Z_]{2,}.*$)^[A-Za-z_]*$")).columns.to_list()]
        if self.region_agg: # group by region
            gene_id.append("region_id")  # append "region_id" to list of genes
            dataframe = dataframe[gene_id].groupby('region_id', as_index=True).mean() # # group by region
        else:
            dataframe = dataframe[gene_id]

        # normalize the data
        dataframe = dataframe.T
        if self.normalize:
            df_scaled = dataframe - np.mean(dataframe, axis=0)
            dataframe = df_scaled / df_scaled.std() # scale the data

        if self.reorder_labels and self.atlas=='MDTB20':
            dataframe = self._reorder_labels(dataframe, atlas)

        return dataframe, info
    
    def threshold_data(self, dataframe, save=False):
        """ This function thresholds gene expression data and optionally saves it out to ./PROCESSED_DIR

        Args: 
            dataframe (pd dataframe): expression data ('expression-alldonors-<atlas>-cleaned.csv')
            save (bool): default is False
            method (str): default is 'differential_stability'. other options: 'dimensionality_reduction'
        Returns: 
            dataframe (pd dataframe): thresholded by `cutoff` and `percentile`
        """
        if self.method=='differential_stability':
            gene_symbols = self._differential_stability(dataframe)
        elif self.method=='dimensionality_reduction':
            gene_symbols = self._dimensionality_reduction(dataframe)
        
        # apply threshold to gene expression and get subset
        dataframe = dataframe[list(gene_symbols) + list(dataframe.filter(regex=("[_].*")).columns)]

        # add method-specific cols
        if self.method=='differential_stability':
            dataframe['cutoff'], dataframe['percentile'] = [self.cutoff, self.percentile]
        
        dataframe['threshold_method'], dataframe['atlas'] = [self.method, self.atlas]
        
        # save out thresholded dataframe
        if save:
            out_name = f"expression-alldonors-{self.atlas}-{self.method}.csv"
            dataframe.to_csv(Defaults.PROCESSED_DIR / out_name, index=None, header=True)
        
        return dataframe

    def _differential_stability(self, dataframe):
        """ Donors are split into two groups (donors 1-3 and 4-6)
        and a correlation is taken across the averaged gene expression

        Args: 
            dataframe (pd dataframe):
        Returns: 
            gene_symbols (list of str): list of gene symbols
        """
        group1_idx = dataframe['donor_num']<=2
        group2_idx = dataframe['donor_num']>=3

        # do some clean up (get average across regions and drop labelled cols)
        df1 = dataframe[(group1_idx)].drop({'donor_id', 'sample_counts', 'donor_num', 'region_num'},1).groupby("region_id").mean()
        df2 = dataframe[(group2_idx)].drop({'donor_id', 'sample_counts', 'donor_num', 'region_num'},1).groupby("region_id").mean()
        
        # get correlation between two gene expression groups
        ds = abs(df1.corrwith(df2))

        # get back thresholded expression data
        if self.cutoff == "top":
            threshold = np.percentile(ds, 100 - self.percentile, interpolation='linear')
            goi_idx = np.argwhere(ds >= threshold)
        elif self.cutoff == "bottom":
            threshold = np.percentile(ds, 1 + self.percentile, interpolation='linear')
            goi_idx = np.argwhere(ds <= threshold)
        else:
            raise ValueError(f"Invalid percentile: {self.cutoff}")
        
        # get gene symbols
        gene_symbols = ds.index[goi_idx]

        return gene_symbols

    def _dimensionality_reduction(self, dataframe):
        """ get rank ordered genes from dimensionality reduction

        Args: 
            dataframe (pd dataframe):
        Returns: 
            gene_symbols (list of str): list of gene symbols
        """
        gene_id = dataframe.filter(regex=("[A-Z0-9].*")).columns.to_list()
        if self.region_agg: # group by region
            gene_id.append("region_id")  # append "region_id" to list of genes
            dataframe = dataframe[gene_id].groupby('region_id', as_index=True).mean()
        else:
            dataframe = dataframe[gene_id]

        # normalize the data
        dataframe = dataframe.T
        if self.normalize:
            df_scaled = dataframe - np.mean(dataframe, axis=0)
            dataframe = df_scaled / df_scaled.std() # scale the data

        # u, s, vt, all_pcs = _compute_svd(df[gene_symbols].T)
        u, s, vt, all_pcs = clustering.compute_svd(dataframe)

        # figure out top genes for PC1 only
        gene_symbols = all_pcs['pc0'].sort_values(ascending=False)

        if self.cutoff=='top':
            threshold = np.percentile(gene_symbols, 100 - self.percentile, interpolation='linear')
            gene_symbols = gene_symbols[gene_symbols >= threshold]
        elif self.cutoff=='bottom':
            threshold = np.percentile(gene_symbols, 1 + self.percentile, interpolation='linear')
            gene_symbols = gene_symbols[gene_symbols <= threshold]
        else:
            raise ValueError(f"Invalid percentile: {self.cutoff}")

        gene_symbols = gene_symbols.index.tolist()

        return gene_symbols

    def _get_donor_files(self, donor_id):
        """ This function returns individual donor files.

        Args:
            donor_id (str): the donor id to call.
        """
        data_files = self.get_all_files()
        regex = re.compile(f"{donor_id}")
        filtered_files = {k: list(filter(regex.findall, v)) for k, v in data_files.items()}
        return Bunch(**filtered_files)

    def _reorder_labels(dataframe): 
        """Reorder labels for `MDTB20` atlas

        Args: 
            dataframe (pd dataframe):
        Returns: 
            dataframe (pd dataframe) - reordered column labels
        """
        regex = re.compile(r"(-)(\w+)")

        order = []
        for col in dataframe.columns:
            match = regex.search(col)
            order.append(match.group()[1])

        reorder = sorted(range(len(order)), key=lambda k: order[k])
        cols_reorder = dataframe.columns[reorder]

        return dataframe[cols_reorder]
    
    def _remove_outliers(self, dataframe):
        """ removes outliers from atlases
        """
        if self.atlas=="SUIT10":
            idx = dataframe.query('region_id=="R09-IX" and donor_id=="donor9861"').index.values
            # remove this row from dataframe
            dataframe.drop(dataframe.index[idx], inplace=True)
            dataframe = dataframe.query('region_id!="R10-X"')
        elif self.atlas=="MDTB10":
            dataframe = dataframe.query('region_id!="R09"')
        elif self.atlas=="MDTB20":
            dataframe = dataframe.query('region_id!="R09-P" and region_id!="R02-P"')
        elif self.atlas=="Ji10":
            dataframe = dataframe.query('region_id!="R06" and region_id!="R08"')
        elif self.atlas=="Buckner17":
            dataframe = dataframe.query('region_id!="R05"')

        return dataframe

def download_AHBA():
    """ This downloads all files from AHBA. 
    If these files already exist, they aren't redownloaded.
    """
    fdir = os.path.join(Defaults.RAW_DATA_DIR / "microarray")
    if os.path.isdir(fdir):
        files = abagen.fetch_microarray(data_dir=fdir)
    else:
        files = abagen.fetch_microarray(data_dir=Defaults.RAW_DATA_DIR, donors='all')
    return files

def get_atlases(atlas_name='SUIT10', suffix='nii', space='MNI', hemisphere=None):
        """download atlases that will be used in this script 
        (only downloads once)
        Args: 
            atlas_name (str): name of parcellation: 'SUIT10', 'Yeo_17', 'MDTB10', 'MDTB20', 'Buckner7', 'Buckner17'
            suffix (str): 'nii' or 'gii'
            space (str): 'MNI' or 'SUIT'
            hemisphere (str or None): None if cerebellar atlas or 'R' or 'L'
        """
        # check to see if atlas already exists
        fpaths = [os.path.join(Defaults.ATLAS_DIR, f'atl-{atlas_name}_space-{space}_dseg.nii')]
        if suffix=='gii':
            fpaths = [os.path.join(Defaults.ATLAS_DIR, f'atl-{atlas_name}_dseg.label.gii')]
            if hemisphere is not None:
                fpaths = [os.path.join(Defaults.ATLAS_DIR, f'atl-{atlas_name}_dseg.{hemisphere}.label.gii')]

        # download atlas if it doesn't exist
        if not os.path.isfile(fpaths[0]):
            if 'Yeo' in atlas_name:
                    files = atlas.fetch_yeo_2011(data_dir=Defaults.ATLAS_DIR)
            elif 'Buckner' in atlas_name:
                    files = catlas.fetch_buckner_2011(data_dir=Defaults.ATLAS_DIR)
            elif 'MDTB' in atlas_name:
                    files = catlas.fetch_king_2019(data='atl', data_dir=Defaults.ATLAS_DIR)
            elif 'SUIT' in atlas_name:
                    files = catlas.fetch_diedrichsen_2009(data_dir=Defaults.ATLAS_DIR)

            fpaths = []
            for fpath in files['files']:
                    if  all(x in fpath for x in [atlas_name, suffix, space]):
                            fpaths.append(fpath)
        
        return fpaths

def join_atlases(
    atlas_cortex="Yeo7", 
    atlas_cerebellum="Buckner7", 
    num_regs=7
    ):

    # load in cortical atlas
    img_cortex = nib.load(os.path.join(Defaults.ATLAS_TEMPLATE_DIR, f'{atlas_cortex}.nii'), mmap=False)

    # load in cerebellar atlas
    img_cerebellum = nib.load(os.path.join(Defaults.ATLAS_TEMPLATE_DIR, f'{atlas_cerebellum}.nii'), mmap=False)

    # set affine matrix (should be same for cortex and cerebellum atlas)
    affine = img_cerebellum.affine

    # assign new labels to cerebellum so that 
    # cortex and cerebellar networks have unique labels
    data_cerebellum = img_cerebellum.get_fdata()
    data_relabelled = []
    for x in data_cerebellum.ravel():
        if x>0:
            value = x+num_regs
        else:
            value = x
        data_relabelled.append(value)

    # reshape the data so that it is in x,y,z coords
    data_relabelled = np.array(data_relabelled).reshape(img_cerebellum.shape)
        
    # get new data
    data = data_relabelled + img_cortex.get_fdata()
    data[data>num_regs*2]=0 

    # save out new image
    new_image = nib.Nifti1Image(data, affine)
    nib.save(new_image, os.path.join(Defaults.ATLAS_DIR, f'Yeo-Buckner-{num_regs}.nii'))

def get_atlas_info(gifti):
    """ Get information about atlas from gifti file

    Args:
        gifti (gifti): full path to gifti image
    """

    # get colors
    rgba, cpal, cmap =  helper_functions.get_gifti_colors(gifti)

    # get labels
    labels = helper_functions.get_gifti_labels(gifti)
    n_regions = len(labels)

    # get structure
    roi_id = np.arange(1,n_regions+1)
    structure = helper_functions.get_gifti_structure(gifti)

    # get hemisphere
    hemisphere = helper_functions.get_gifti_hemisphere(gifti)

    if 'Cortex' in structure:
        structure = 'cortex'

    df = pd.DataFrame({'region_num': roi_id, 'region_id': labels, 'r': rgba[:,0], 'g': rgba[:,1], 'b': rgba[:,2],
            'hemisphere': np.repeat(hemisphere, n_regions), 'structure': np.repeat(structure, n_regions)})

    return df

def save_mni_coords(self):
    file_dir = Defaults.RAW_DATA_DIR / "allenbrain"
    regex = r"(donor\d+)"

    colors = Defaults.donor_colors
    
    # get fpath to donor folders
    donor_files = []
    for donor in Defaults.donors:
        donor_files.append(os.path.join(file_dir, f'normalized_microarray_{donor}'))

    df_all = pd.DataFrame()
    for i, donor_file in enumerate(donor_files):
        os.chdir(donor_file)
        df = pd.read_csv("SampleAnnot.csv")
        df = df.query('slab_type=="CB"')[['structure_acronym', 'mni_x', 'mni_y', 'mni_z']]
        df['donor_id'] = re.findall(regex, donor_file)[0]
        df['donor_num'] = i+1
        df['r'] = colors[i][0]
        df['g'] = colors[i][1]
        df['b'] = colors[i][2]
        df_all = pd.concat([df_all, df])

    df_all.to_csv(Defaults.RAW_DATA_DIR / "mni_coords_all_donors.csv") 

def process_gene_expression(img, info=None, config=None):
    """ This function runs the main abagen routine (get_expression_data), 
    merges expression data with info and saves it out to ./INTERIM_DIR

    Args:
        img (str): fullpath to atlas (nifti or gifti)
        info (pd dataframe): containing information about `img`
        config (dict or None): data dict of parameters for processing gene expression
    
    Returns: 
        df_concat (pd dataframe): output dataframe
    """

    # get expression data
    if config is None:
        dataframes, counts = abagen.get_expression_data(img, atlas_info=info)
    else:
        dataframes, counts = abagen.get_expression_data(
            img, atlas_info=info, donors=config.donors, tolerance=config.tolerance, 
            ibf_threshold=config.ibf_threshold, probe_selection=config.probe_selection, 
            lr_mirror=config.lr_mirror, gene_norm=config.gene_norm,
            sample_norm=config.sample_norm, no_corrected_mni=config.no_corrected_mni, 
            no_reannotated=config.no_reannotated, save_counts=config.return_counts, 
            save_donors=config.return_donors, region_agg=config.region_agg,
            agg_metric=config.agg_metric, data_dir=config.data_dir, 
            verbose=config.verbose, n_proc=config.n_proc) # return_samples=config.return_samples
        
    # make sure `config.donors` is correctly specified
    if config.donors is 'all':
        config.donors = Defaults.donors

    # get expression data
    df_concat = pd.DataFrame()
    for i, df in enumerate(dataframes):

        df = df.reset_index()
        
        # add atlas info to expression data
        df_merge = info.merge(df, left_on='region_num', right_on='label')
        
        # add some new columns
        df_merge['donor_id'] = config.donors[i]
        df_merge['donor_num'] = i+1
        df_merge['sample_counts'] = counts[i].to_list()
        
        # concat dataframes and drop nan values
        df_concat = pd.concat([df_concat, df_merge]).dropna()

    return df_concat

    # list of available variable and methods
    # __all__ = ["save_expression_data", "save_atlas_info", "save_thresholded_data"]
