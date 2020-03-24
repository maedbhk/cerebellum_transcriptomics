import re
from collections import namedtuple

import numpy as np
import scipy as sp
import pandas as pd

import matplotlib as mpla
from matplotlib import cm
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import cluster

from transcriptomics.constants import Defaults
from transcriptomics.visualization import visualize

GeneSubset = namedtuple("GeneSubset", ["threshold", "goi_idx"])

def return_grouped_data(atlas, which_genes='top', percentile=1, **kwargs):
    """This function returns grouped and thresholded data for a specified atlas.

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
                normalize (bool): whether or not to normalize (center and scale) the data
                atlas_other (str): returns thresholded data using genes from another atlas
                donor_num (int): any one of the 6 donors 
                remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
                reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
                unthresholded (bool): returns grouped data for unthresholded data.
    """
    # option to get genes from another atlas
    if kwargs.get("atlas_other"):
        dataframe = _threshold_data(atlas, which_genes, percentile, kwargs["atlas_other"])
    
    # option to get thresholded or unthresholded data
    if kwargs.get("unthresholded"):
        dataframe = return_unthresholded_data(atlas)
    else:
        dataframe = return_thresholded_data(atlas, which_genes=which_genes, percentile=percentile)

    # option to remove outliers
    if kwargs.get("remove_outliers"):
        dataframe = _remove_outliers(dataframe, atlas, **kwargs)

    # option to get subject-specific data
    if kwargs.get("donor_num"):
        donor_num = kwargs["donor_num"]
        dataframe = dataframe.query(f'donor_num=={donor_num}')

    # group data by regions
    dataframe = _group_by_region(dataframe)

    # option to center and scale
    if kwargs.get("normalize"):
        dataframe = _center_scale(dataframe.T)
    else:
        dataframe = dataframe.T

    # option to reorder labels
    if kwargs.get("reorder_labels"):
        dataframe = _reorder_labels(dataframe, atlas)

    return dataframe

    # df = do_something(df, **kwargs)

def return_thresholded_data(atlas, which_genes='top', percentile=1, **kwargs):
    """This function returns thresholded data for a specified atlas. 
       By default, returns the aggregate data (across regions) but there's also an option
       to return all samples

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
            kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
                normalize (bool): whether or not to normalize (center and scale) the data
                remove_outliers (bool): removes outliers from certain atlases
                donor_num (int): any one of the 6 donors
                all_samples (bool): returns aggregate samples (across regions) or all samples
    """
    
    dataframe = _threshold_data(atlas, which_genes, percentile, **kwargs)

    if kwargs.get("atlas_other"):
        genes = _get_gene_symbols(kwargs["atlas_other"], which_genes, percentile)
    else:
        genes = _get_gene_symbols(atlas, which_genes, percentile)
    
    if kwargs.get("remove_outliers"):
        dataframe = _remove_outliers(dataframe, atlas, **kwargs)

    # get subject-specific data
    if kwargs.get("donor_num"):
        donor_num = kwargs["donor_num"]
        dataframe = dataframe.query(f'donor_num=={donor_num}')

    if kwargs.get("normalize"):
        dataframe[genes] = _center_scale(dataframe[genes])

    return dataframe

def return_unthresholded_data(atlas, **kwargs):
    """This function returns unthresholded data for a specified atlas.

    Args:
        atlas (str): the name of the atlas to use

        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
                all_samples (bool): option to return dataframe with all samples or aggregated across regions
    """
    if kwargs.get("all_samples"):
        out_name = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-samples.csv') 
    else:
        out_name = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-cleaned.csv') 

    return out_name

def return_concatenated_data(atlas_cerebellum, atlas_cortex, which_genes='top', percentile=1, normalize=True, **kwargs):
    """This function returns concatenated dataframe for grouped and thresholded cortical and cerebellar data.

    Args:
        atlas_cerebellum (str): the name of the cerebellar atlas to use
        atlas_cortex (str): the name of the cortical atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        normalize (bool): whether or not to normalize (center and scale) the data
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
                atlas_other (str): returns thresholded data using genes from another atlas
                donor_num (int): any one of the 6 donors 
                remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
    """

    dataframe_1 = return_grouped_data(atlas_cortex, which_genes=which_genes, percentile=percentile, normalize=normalize, **kwargs)
    dataframe_2 = return_grouped_data(atlas_cerebellum, which_genes=which_genes, percentile=percentile, normalize=normalize, **kwargs)
    
    # add prefix to col names and reset index
    dataframe_1 = dataframe_1.add_prefix(f"{atlas_cortex}-").reset_index().rename({'index': f'gene_symbols-{atlas_cortex}'},axis=1)
    dataframe_2 = dataframe_2.add_prefix(f"{atlas_cerebellum}-").reset_index().rename({'index': f'gene_symbols-{atlas_cerebellum}'},axis=1)

    df_concat = pd.concat([dataframe_1, dataframe_2], axis=1)
    
    # center and scale concatenated dataframe
    if normalize:
        df_concat = _center_scale(df_concat.drop({f'gene_symbols-{atlas_cortex}', f'gene_symbols-{atlas_cerebellum}'}, axis=1))
    else:
        df_concat = df_concat.drop({f'gene_symbols-{atlas_cortex}', f'gene_symbols-{atlas_cerebellum}'}, axis=1)

    return df_concat

def _bootstrap_dendrogram(atlas, num_iter=10000):

    dataframe = return_grouped_data(atlas=atlas)

    # get true order
    R = visualize.dendrogram_plot(dataframe.T)
    true_order = R['leaves']

    distances = []
    for b in np.arange(10000):

        # get bootstrapped order
        for c in dataframe.columns:
            dataframe[c] = np.random.permutation(dataframe[c].values)

        R = visualize.dendrogram_plot(dataframe.T)
        bootstrap_order = R['leaves']

        # get euclidean difference between both vectors
        # explanation for why this works: https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        distances.append(np.linalg.norm(np.array(true_order)-np.array(bootstrap_order)))

    # calculate p-value as probability of observing Euclidean distance of zero
    matches = [d for d in distances if d==0]  
    p_val = len(matches) / len(distances)

    return p_val

def _remove_outliers(dataframe, atlas, **kwargs):
    if atlas=="SUIT-10":
        idx = dataframe.query('region_id=="R09-IX" and donor_id=="donor9861"').index.values
        # remove this row from dataframe
        dataframe.drop(dataframe.index[idx], inplace=True)
        if kwargs.get("extreme_removal"):
            # remove all rows containing R10-X
            dataframe = dataframe.query('region_id!="R10-X"')
    elif atlas=="MDTB-10":
        dataframe = dataframe.query('region_id!="R09"')
    elif atlas=="MDTB-10-subRegions":
        dataframe = dataframe.query('region_id!="R09-P" and region_id!="R02-P"')
    elif atlas=="Ji-10":
        dataframe = dataframe.query('region_id!="R06" and region_id!="R08"')
    elif atlas=="Buckner-17":
        dataframe = dataframe.query('region_id!="R05"')

    return dataframe

def _reorder_labels(dataframe, atlas):
    if atlas=="MDTB-10-subRegions":
        regex = re.compile(r"(-)(\w+)")

        order = []
        for col in dataframe.columns:
            match = regex.search(col)
            order.append(match.group()[1])

        reorder = sorted(range(len(order)), key=lambda k: order[k])
        cols_reorder = dataframe.columns[reorder]

    return dataframe[cols_reorder]

def _get_roi_labels(atlas, which_genes='top', percentile=1):
    """This function returns roi labels for a specified atlas

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
    """
    dataframe = pd.read_csv(Defaults.PROCESSED_DIR / f'expression-alldonors-{atlas}-{which_genes}-{percentile}.csv')

    roi_labels = dataframe['region_id'].unique()

    return sorted(roi_labels)

def _get_gene_symbols(atlas, which_genes, percentile):
    """This function returns subset of gene symbols for a specified atlas.

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
    """

    dataframe = pd.read_csv(Defaults.PROCESSED_DIR / f'expression-alldonors-{atlas}-{which_genes}-{percentile}.csv')

    # use regex to find gene columns
    gene_symbols = dataframe.filter(regex=("[A-Z0-9].*")).columns

    return gene_symbols

def _threshold_data(atlas, which_genes, percentile, **kwargs):
    """This function returns thresholded data for a specified atlas
    using a subset of genes from another atlas

    Args:
        atlas (str): the name of the atlas to return
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        kwargs (dict): dictionary of additional (optional) kwargs.
            atlas_other (str): option to use this atlas for gene extraction
            all_samples (bool): returns aggregate samples (across regions) or all samples
    """
    if kwargs.get("all_samples"):
        dataframe = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-samples.csv') 
    else:
        dataframe = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-cleaned.csv') 

    if kwargs.get("atlas_other"):
        gene_symbols = _get_gene_symbols(kwargs["atlas_other"], which_genes, percentile)
    else:
        gene_symbols = _get_gene_symbols(atlas, which_genes, percentile)

    return dataframe[list(gene_symbols) + list(dataframe.filter(regex=("[_].*")).columns)]

def _group_by_region(dataframe):
    """This function returns a dataframe grouped by region.
        Called by return_grouped_data

    Args:
        dataframe: given by return_grouped_data
    """
    # get list of genes from this dataframe
    cols = dataframe.filter(regex=("[A-Z0-9].*")).columns.to_list()

    # append "region_id" to list of genes
    cols.append("region_id")

    # get dataframe region_id x genes
    df_grouped = dataframe[cols]
    
    # group by region
    df_grouped = df_grouped.groupby('region_id', as_index=True).mean() # change as_index
    
    return df_grouped

def _center_scale(dataframe): 
    """This function returns a normalized dataframe.

    Args:
        dataframe: given by return_grouped_data
    """

    keyboard
    
    # center the dataframe
    df_center_scale = dataframe - np.mean(dataframe, axis = 0)
    
    # scale the data
    df_center_scale = df_center_scale / df_center_scale.std()
    
    return df_center_scale

def _compute_rank_k_approximation(dataframe):
    """This function returns the output of svd (u, s, vt).

    Args:
        dataframe: given by return_grouped_data
    """
    u, s, vt = np.linalg.svd(dataframe, full_matrices = False)
    # reconstructed_data = pd.DataFrame(u[:, 0:k] @ np.diag(s[0:k]) @ vt[0:k, :], columns = dataframe.columns)
    return u, s, vt

def _compute_svd(dataframe):
    """This function returns the output from svd and pcs.

    Args:
        dataframe: given by return_grouped_data
    """
    # do svd
    u, s, vt = _compute_rank_k_approximation(dataframe)

    # get the pcs (P=XV or P=US)
    # pcs = u * s 
    pcs = dataframe @ vt.T

    # get the first n pcs
    pcs= pd.DataFrame(pcs).add_prefix('pc')
    
    return u, s, vt, pcs

def _pcs_winner_take_all(dataframe, num_pcs): 
    """This function returns pcs labelled by winner-take-all.

    Args:
        dataframe: given by return_grouped_data
        num_pcs (int): number of pcs to include in the winner-take-all.
    """
    u, s, vt, pcs = _compute_svd(dataframe)
    
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

def _corr_matrix(dataframe):
    """This function returns a correlation matrix.

    Args:
        dataframe: given by return_grouped_data
    """
    return dataframe.corr()

def _compute_k_means_n_dims(dataframe, num_clusters):
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

# __all__ = ["save_expression_data", "save_atlas_info", "save_thresholded_data"]
