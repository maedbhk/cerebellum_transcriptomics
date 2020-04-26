# import libraries
# all of the libraries have been installed in a virutal environment
# you can find all of the requirements in the requirements.txt file in the repo
from pathlib import Path
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

from transcriptomics.constants import Defaults

ATLAS_TEMPLATE_DIR = Defaults.EXTERNAL_DIR / "atlas_templates"

GeneSubset = namedtuple("GeneSubset", ["threshold", "goi_idx"])

class Parameters: 
    atlas_info = None
    exact = True
    tolerance = 2
    ibf_threshold = 0.5 
    probe_selection = 'diff_stability'
    lr_mirror = False
    gene_norm = 'zscore' # default is 'srs'
    sample_norm = 'zscore' # default is 'srs'
    corrected_mni = True
    reannotated = True
    return_counts = True
    return_donors = True
    region_agg = 'donors' # other option is 'samples'
    agg_metric = 'mean'
    donors = 'all'
    data_dir = Defaults.RAW_DATA_DIR
    verbose = 1
    n_proc = 1
    return_samples = True # I added in this parameter to return an unaggregated dataframe (labelled samples x genes)

def get_all_files():
    """ This downloads all files from AHBA. 
    If these files already exist, they aren't redownloaded.
    """
    exists = os.path.isfile(Defaults.DATA_DIR / "allenbrain")
    if exists:
        data_files = abagen.fetch_microarray(data_dir=str(Defaults.DATA_DIR / "allenbrain"), donors='all')
    else:
        data_files = abagen.fetch_microarray(data_dir=str(Defaults.RAW_DATA_DIR), donors='all')
    return data_files

def save_expression_data(atlas):
    """ This function runs the main abagen routine (get_expression_data), 
    merges expression data with info and saves it out to ./INTERIM_DIR

    Args:
        atlas (str): the name of the atlas to use
    """
    # get info and nii filenames for each atlas
    atlas_files = _get_atlas_files(atlas)
    atlas_info = pd.read_csv(atlas_files.info)

    # get expression data
    dataframes, counts = abagen.get_expression_data(
        str(atlas_files['image']), atlas_info=Parameters.atlas_info, exact=Parameters.exact,
        tolerance=Parameters.tolerance, ibf_threshold=Parameters.ibf_threshold, 
        probe_selection=Parameters.probe_selection, lr_mirror=Parameters.lr_mirror, gene_norm=Parameters.gene_norm,
        sample_norm=Parameters.sample_norm, corrected_mni=Parameters.corrected_mni, reannotated=Parameters.reannotated,
        return_counts=Parameters.return_counts, return_donors=Parameters.return_donors, region_agg=Parameters.region_agg,
        agg_metric=Parameters.agg_metric, donors=Parameters.donors, data_dir=Parameters.data_dir, 
        verbose=Parameters.verbose, n_proc=Parameters.n_proc, return_samples=Parameters.return_samples)

    # what expression dataframe is being saved out. aggregate or all samples?
    if Parameters.return_samples:
        out_name = f"expression-alldonors-{atlas}-samples.csv"
    else:
        out_name = f"expression-alldonors-{atlas}-cleaned.csv"

    # get expression data
    df_concat = pd.DataFrame()
    for i, df in enumerate(dataframes):

        df = df.reset_index()
        
        # add atlas info to expression data
        df_merge = atlas_info.merge(df, left_on='region_num', right_on='label')
        
        # add some new columns
        df_merge['donor_id'] = Defaults.donors[i]
        df_merge['donor_num'] = i+1

        # return sample counts if returning aggregate dataframe
        if Parameters.return_samples:
            pass
        else:
            df_merge['sample_counts'] = counts[i].to_list()
        
        # concat dataframes and drop nan values
        df_concat = pd.concat([df_concat, df_merge]).dropna()

    # write out expression data to file for all donors
    output_dir = Defaults.INTERIM_DIR / out_name
    df_concat.to_csv(output_dir, index=None, header=True)  

def save_atlas_info(atlas):
    """ This function saves info file for specified atlas to ./EXTERNAL_DIR

    Args:
        atlas (str): the name of the atlas to use
    """
    if atlas=="Desikan-Killiany-83": # abagen comes with an info for Desikan-Killiany
        pass

    info_dataframe = _get_atlas_info(atlas)

    # get output dir
    output_dir = Defaults.EXTERNAL_DIR / "atlas_templates" / f"{atlas}-info.csv"

    # write to csv file
    info_dataframe.to_csv(output_dir, index=None, header=True)

def save_thresholded_data(atlas, which_genes='top', percentile=1):
    """ This function thresholds expression data and saves it out to ./PROCESSED_DIR

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1
    """
    
    # load in cleaned data (aggregated across samples)
    expression_cleaned = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-cleaned.csv') 
    out_name = f"expression-alldonors-{atlas}-{which_genes}-{percentile}.csv"

    # return differential stability results based on two groups (donors 1-3; donors 4-6)
    ds = _get_differential_stability(atlas) # this function assumes that you're providing all 6 donors

    # get back thresholded expression data
    threshold, gene_symbols = _threshold_genes_ds(ds, which_genes=which_genes, percentile=percentile) # choose top or bottom 
    
    # apply threshold to gene expression and get subset
    expression_thresholded = expression_cleaned[list(gene_symbols) + list(expression_cleaned.filter(regex=("[_].*")).columns)]
    expression_thresholded["threshold_type"] = f"{which_genes}_{percentile}%"
    # expression_thresholded["threshold"] = threshold
    
    # save out thresholded dataframe
    expression_thresholded.to_csv(Defaults.PROCESSED_DIR / out_name, index=None, header=True)

def resample_to_mni(atlas): 
    """ This function resamples atlas into mni space and overwrites original atlas.

    Args:
        atlas (str): the name of the atlas to use. 'MDTB-10', 'MDTB-10-subRegions', and 'Ji-10'
        need to be resampled.
    """
    # use 'Buckner-7' as target atlas
    target_img = nib.load(str(Defaults.EXTERNAL_DIR / "atlas_templates" / "Buckner-7.nii"))

    # transform atlases (SUIT-10 and MDTB-10 into same space as Buckner/Yeo atlases)
    img_to_transform = nib.load(str(Defaults.EXTERNAL_DIR / "atlas_templates" / f"{atlas}.nii"))

    img_to_transform = img_to_transform.__class__(img_to_transform.get_data().astype(np.int32), img_to_transform.affine,
                        header=img_to_transform.header)

    resampled = image.resample_img(img_to_transform, target_img.affine, target_shape=[256,256,256])

    # overwrites original file
    resampled.to_filename(str(Defaults.EXTERNAL_DIR / "atlas_templates" / f"{atlas}.nii"))

def save_mni_coords():
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

def join_atlases(atlas_cortex="Yeo-7", atlas_cerebellum="Buckner-7", num_regs=7):
    ATLAS_TEMPLATE_DIR = Defaults.EXTERNAL_DIR / "atlas_templates"

    # load in cortical atlas
    img_cortex = nib.load(os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", f'{atlas_cortex}.nii'), mmap=False)

    # load in cerebellar atlas
    img_cerebellum = nib.load(os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", f'{atlas_cerebellum}.nii'), mmap=False)

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
    nib.save(new_image, os.path.join(ATLAS_TEMPLATE_DIR, f'Yeo-Buckner-{num_regs}.nii'))

def _get_atlas_info(atlas):
    """ This function makes an info csv file that includes region_num, region_id and region_color.

    Args:
        atlas (str): the name of the atlas to use. 'MDTB-10', 'MDTB-10-subRegions', and 'Ji-10'
        need to be resampled.
    """
    # get roi labels for an atlas name
    labels_roi = _get_roi_labels(atlas)

    if atlas in list(Defaults.colour_info.keys()):
        # get color info
        colours = Defaults.colour_info[atlas]
    else:
        colours = np.zeros((len(labels_roi),3))
        colours = [list(np.random.choice(range(256), size=3)) for c in colours]

   # get num of rois
    n_roi = len(labels_roi)

    color_r = []
    color_g = []
    color_b = []
    for i in np.arange(len(colours)):
        color_r.append(colours[i][0])
        color_g.append(colours[i][1])
        color_b.append(colours[i][2])
    
    # intialise dict
    # info = {'region_num':list(range(1,n_roi+1)), 'region_id': labels_roi, 'colours': colours}
    info = {'region_num':list(range(1,n_roi+1)), 'region_id': labels_roi, 'r': color_r, 'g': color_g, 'b': color_b}

    # create dataframe
    info_dataframe = pd.DataFrame(info) 
    return info_dataframe

def _get_roi_labels(atlas):
    """ This function returns roi labels for the specified atlas.

    Args:
        atlas (str): the name of the atlas to use.
    """
    if atlas in list(Defaults.labels.keys()):
        labels_roi = Defaults.labels[atlas]
    else:
        num_roi_str = re.findall(r'\d+', atlas) # find number in a string
        labels_roi = [f'R{k:02d}' for k in range(1, int(num_roi_str[0]) + 1)]
    return labels_roi

def _get_atlas_files(atlas):
    """ This function returns nifti and info file names for specified atlas.

    Args:
        atlas (str): the name of the atlas to use. 'MDTB-10', 'MDTB-10-subRegions', and 'Ji-10'
        need to be resampled.
    """
    outname_image = ATLAS_TEMPLATE_DIR / f'{atlas}.nii'
    outname_info =  ATLAS_TEMPLATE_DIR / f'{atlas}-info.csv'
    atlas = Bunch(image=outname_image, info=outname_info)
    return atlas 

def _get_donor_files(donor_id):
    """ This function returns individual donor files.

    Args:
        donor_id (str): the donor id to call.
    """
    data_files = get_all_files()
    regex = re.compile(f"{donor_id}")
    filtered_files = {k: list(filter(regex.findall, v)) for k, v in data_files.items()}
    return Bunch(**filtered_files)

def _threshold_genes_ds(ds, which_genes, percentile): 
    """ This function returns thresholded genes based on differential stability analysis.

    Args:
        ds (list): output from  'get_differential_stability' function
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
    """
    if which_genes == "top":
        threshold = np.percentile(ds, 100 - percentile, interpolation = 'linear')
        goi_idx = np.argwhere(ds >= threshold)
    elif which_genes == "bottom":
        threshold = np.percentile(ds, 1 + percentile, interpolation = 'linear')
        goi_idx = np.argwhere(ds <= threshold)
    else:
        raise ValueError(f"Invalid percentile: {which_genes}")

    return GeneSubset(threshold, ds.index[goi_idx])

def _get_differential_stability(atlas): # outputs results of differential stability analysis - correlations between genes across donors
    """ This function returns differential stability results. Donors are split into two groups (donors 1-3 and 4-6)
    and a correlation is taken across the averaged gene expression

    Args:
        atlas (str): the name of the atlas to use
    """
    # load in cleaned data 
    dataframe = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-cleaned.csv') 
    
    # get indices for two groups (donors 1-3; donors 4-6)
    group1_idx = dataframe['donor_num']<=2
    group2_idx = dataframe['donor_num']>=3

    # do some clean up (get average across regions and drop labelled cols)
    df1 = dataframe[(group1_idx)].drop({'donor_id', 'sample_counts', 'donor_num', 'region_num'},1).groupby("region_id").mean()
    df2 = dataframe[(group2_idx)].drop({'donor_id', 'sample_counts', 'donor_num', 'region_num'},1).groupby("region_id").mean()
    
    # get correlation between two gene expression groups
    ds = abs(df1.corrwith(df2))
    
    return ds

# list of available variable and methods
# __all__ = ["save_expression_data", "save_atlas_info", "save_thresholded_data"]
