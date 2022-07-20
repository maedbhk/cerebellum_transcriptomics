import re
import os

import numpy as np
import scipy as sp
import pandas as pd

import nibabel as nib
import abagen
from sklearn import cluster

from transcriptomics.constants import Defaults

def compute_rank_k_approximation(dataframe):
    """This function returns the output of svd (u, s, vt).

    Args:
        dataframe: given by return_grouped_data
    """
    u, s, vt = np.linalg.svd(dataframe, full_matrices=False)
    # reconstructed_data = pd.DataFrame(u[:, 0:k] @ np.diag(s[0:k]) @ vt[0:k, :], columns = dataframe.columns)
    return u, s, vt

def compute_svd(dataframe, normalize=False):
    """This function returns the output from svd and pcs.

    Args:
        dataframe: given by return_grouped_data
        normalize (bool): default is True
    """
    if normalize:
        df_scaled = dataframe - np.mean(dataframe, axis=0)
        dataframe = df_scaled / df_scaled.std() # scale the data

    # do svd
    u, s, vt = np.linalg.svd(dataframe, full_matrices=False)

    # get the pcs (P=XV or P=US)
    # pcs = u * s 
    pcs = dataframe @ vt.T

    # get the first n pcs
    pcs= pd.DataFrame(pcs).add_prefix('pc')
    
    return u, s, vt, pcs

def pcs_winner_take_all(dataframe, num_pcs): 
    """This function returns pcs labelled by winner-take-all.

    Args:
        dataframe: given by return_grouped_data
        num_pcs (int): number of pcs to include in the winner-take-all.
    """
    u, s, vt, pcs = compute_svd(dataframe)
    
    pc_region_loading = pcs.iloc[:,:num_pcs].values @ vt[:num_pcs,:]

    num_genes = np.arange(len(pc_region_loading))

    # col names
    region_names = dataframe.columns

    reg_idx_all = []
    reg_names = []
    for g in num_genes:
        reg_idx = np.argmax(pc_region_loading[g,:])
        reg_idx_all.append(reg_idx)
        reg_names.append(region_names[reg_idx])
        
    pcs_labelled = pcs.copy()

    # add cols of jittered data
    for pc in pcs.columns:
        pcs_labelled[f'{pc}_jittered'] = pcs_labelled[pc] + np.random.normal(loc = 0, scale = 0.1, size = len(pcs))

    # add region to dataset
    pcs_labelled['region'] = reg_names
    
    return pcs_labelled

def corr_matrix(dataframe):
    """This function returns a correlation matrix.

    Args:
        dataframe: given by return_grouped_data
    """

    corr_matrix = dataframe.corr()
    labels = dataframe.columns

    return corr_matrix, labels

def _correct_indices_residualized_matrix(dataframe, atlas):
    """ get labels for regions included in residualized matrix.
    this function is necessary for `_corr_matrix_residualized`
        Args: 
            dataframe (pandas dataframe)
            atlas (str): name of atlas
        Returns:
            correct indices for labels to keep for residualized matrix
    """
    labels = dataframe.columns.to_list()

    atlas_info = os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", f'{atlas}-info.csv')
    if os.path.isfile(atlas_info):
        df = pd.read_csv(atlas_info)
        default_labels = df['region_id']
    else: 
        default_labels = Defaults.labels[atlas]      
    
    _, _, comm2 = np.intersect1d(labels, default_labels, assume_unique=True, return_indices=True)
    indices = sorted(comm2+1)

    return indices

def corr_matrix_residualized(dataframe, atlas):
    """ returns a residualized correlation matrix. removes spatial autocorr between regions
        Args: 
            dataframe: (pandas dataframe)
            atlas (str): atlas name
        Returns residualized correlation matrix + labels
    """
    corr_matrix = np.corrcoef(dataframe.T)
    labels = dataframe.columns
    atlas_obj = nib.load(os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", f'{atlas}.nii'))

    # figure out which labels from `atlas` are included in coexpression matrix
    indices = _correct_indices_residualized_matrix(dataframe=dataframe, atlas=atlas)

    corr_matrix_residualized = abagen.correct.remove_distance(coexpression=corr_matrix, atlas=atlas_obj, labels=indices)

    return corr_matrix_residualized, labels

def compute_k_means_n_dims(dataframe, num_clusters):
    """This function returns results of n-dimensional k-means.
    Args:
        dataframe: given by return_grouped_data
    """
    clusterer = cluster.KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_labels = clusterer.fit_predict(dataframe)

    dataframe['km_labels'] = kmeans_labels

    # split df into cluster groups
    grouped = dataframe.groupby(['km_labels'], sort=True)

    # compute sums for every column in every group
    df_n_dims = grouped.sum()
    
    return df_n_dims