import re
import os
from collections import namedtuple
import itertools

import numpy as np
import scipy as sp
import pandas as pd

import matplotlib as mpla
from matplotlib import cm
from matplotlib import pyplot as plt
import seaborn as sns

import abagen # library for preprocessing AHBA data

import nibabel as nib

from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import cluster
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize

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
    # if kwargs.get("atlas_other"):
    #     dataframe = _threshold_data(atlas=atlas, which_genes=which_genes, percentile=percentile, **kwargs) #  kwargs["atlas_other"]
    
    # option to get thresholded or unthresholded data
    if kwargs.get("unthresholded"):
        dataframe = return_unthresholded_data(atlas)
    else:
        # dataframe = return_thresholded_data(atlas, which_genes=which_genes, percentile=percentile, **kwargs)
        dataframe = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-cleaned.csv') 

    # threshold based on gene symbols
    if kwargs.get("atlas_other"):
        genes = _get_gene_symbols(kwargs["atlas_other"], which_genes, percentile)
    else:
        genes = _get_gene_symbols(atlas, which_genes, percentile)
    dataframe = dataframe[list(genes) + list(dataframe.filter(regex=("[_].*")).columns)]

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

def return_thresholded_data(atlas, which_genes='top', percentile=1, **kwargs):
    """This function returns thresholded (ungrouped) data for a specified atlas. 
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
    # dataframe = _threshold_data(atlas=atlas, which_genes=which_genes, percentile=percentile, **kwargs)

    if kwargs.get("all_samples"):
        dataframe = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-samples.csv') 
    else:
        dataframe = pd.read_csv(Defaults.INTERIM_DIR / f'expression-alldonors-{atlas}-cleaned.csv') 

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

def return_concatenated_data(atlas_cerebellum, atlas_cortex, which_genes='top', percentile=1, **kwargs):
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
                normalize (bool): whether or not to normalize (center and scale) the data
    """

    dataframe_1 = return_grouped_data(atlas_cortex, which_genes=which_genes, percentile=percentile, **kwargs)
    dataframe_2 = return_grouped_data(atlas_cerebellum, which_genes=which_genes, percentile=percentile, **kwargs)
    
    # add prefix to col names and reset index
    dataframe_1 = dataframe_1.add_prefix(f"{atlas_cortex}-").reset_index().rename({'index': f'gene_symbols-{atlas_cortex}'},axis=1)
    dataframe_2 = dataframe_2.add_prefix(f"{atlas_cerebellum}-").reset_index().rename({'index': f'gene_symbols-{atlas_cerebellum}'},axis=1)

    df_concat = pd.concat([dataframe_1, dataframe_2], axis=1)
    
    # center and scale concatenated dataframe
    if kwargs.get("normalize"):
        df_concat = _center_scale(df_concat.drop({f'gene_symbols-{atlas_cortex}', f'gene_symbols-{atlas_cerebellum}'}, axis=1))
    else:
        df_concat = df_concat.drop({f'gene_symbols-{atlas_cortex}', f'gene_symbols-{atlas_cerebellum}'}, axis=1)

    return df_concat

def save_colors_transcriptomic_atlas(atlas="MDTB-10-subRegions", atlas_other="MDTB-10", remove_outliers=True, normalize=True):
    """ this function saves colors + labels based on dendrogram clustering for MDTB-10-subRegions not customised for any other atlas
        Args: 
            atlas (str): "MDTB-10-subRegions"
            atlas_other (str): "MDTB-10"
            remove_outliers (bool): default is True
            normalize (bool): default is True
    """

    df = return_grouped_data(atlas=atlas, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)
    
    R = visualize.dendrogram_plot(df.T)
    
    regex = r"(\d+)-(\w+)"

    # get atlas labels
    groups = []
    for p in R['ivl']:
        match = re.findall(regex, p)[0]
        groups.append(match)

    # get indices for labels
    index = []
    for group in groups:
        if group[1]=='A':
            index.append(int(group[0]))
        else:
            index.append(int(group[0]) + 10)

    # zero index the regions
    index = [i-1 for i in index] # zero index

    # figure out which regions are missing
    res = [ele for ele in range(max(index)+1) if ele not in index]

    for reorder in [True, False]:
        # assign colors to clusters
        colors = sns.color_palette("coolwarm", len(index))
        # convert to list
        colors = [list(ele) for ele in colors]
        if reorder:
            # append NaN color values to missing regions
            for ele in res:
                colors.append(np.tile(float("NaN"),3).tolist())
            # put the rgb colors in sorted order
            colors_dendrogram = [x[1] for x in sorted(zip(index+res, colors), key=lambda x: x[0])]
            labels = Defaults.labels[atlas] 
            outname = f"{atlas}-transcriptomic-info.csv"
        else:
            # don't sort the rgb colors
            colors_dendrogram = colors[::-1]
            labels = R['ivl'][::-1]
            outname = f"{atlas}-transcriptomic-dendrogram-ordering-info.csv"

        color_r = []
        color_g = []
        color_b = []
        for i in np.arange(len(colors_dendrogram)):
            color_r.append(np.round(colors_dendrogram[i][0],2))
            color_g.append(np.round(colors_dendrogram[i][1],2))
            color_b.append(np.round(colors_dendrogram[i][2],2))

        data = {'region_num':list(range(1,len(labels)+1)), 'region_id': labels, 'r': color_r, 'g':color_g, 'b':color_b}

        # create dataframe
        df_new = pd.DataFrame(data) 

        df_new.to_csv(os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", outname))

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
    """This function returns thresholded data for a specified atlas. 
    option to use a subset of genes from another atlas

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

    corr_matrix = dataframe.corr()
    labels = dataframe.columns

    return corr_matrix, labels

def _corr_matrix_residualized(dataframe, atlas):
    """ returns a residualized correlation matrix. removes spatial autocorr between regions
        Args: 
            dataframe: (pandas dataframe)
            atlas (str): atlas name
        Returns residualized correlation matrix + labels
    """
    corr_matrix = np.corrcoef(dataframe.T)
    labels = dataframe.columns
    atlas_obj = nib.load(os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", f'{atlas}.nii'))
    corr_matrix_residualized = abagen.correct.remove_distance(coexpression=corr_matrix, atlas=atlas_obj)

    return corr_matrix_residualized, labels

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

def _split_train_test(X, y, test_size):
    """ divides X and y into train and test sets
        Args:
            X (matrix): training data
            y (vector): labelled data
            test_size (int): size of test size, usually .2
        Returns:
            x_train, x_test, y_train, y_test
    """
    # y = label_binarize(y, classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

def _run_classifier_multiclass(X, y, classifier='logistic', test_size=.2): 
    """ fit logistic regression model - crossvalidated
        Args:
            X_train (matrix):
            y_train (vector):
            classifier (str): 'logistic', 'svm' etc
            test_size (int): default is .2
            fit_intercept (bool): default is True

            Kwargs:
                solver (str): default is 'lbfgs'
        Return:
            logistic model
    """
    random_state = np.random.seed(47)

    X_train, X_test, y_train, y_test = _split_train_test(X, y, test_size=test_size)

    if classifier=='logistic':
        classifier = lm.LogisticRegression(fit_intercept=True, multi_class='ovr', solver='lbfgs')  #multi_class='ovr'  
        classifier.fit(X_train, y_train)
        classifier_probs = classifier.predict_proba(X_test)
    elif classifier=="svm":
        classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
        classifier.fit(X_train, y_train)
        classifier_probs = classifier.decision_function(X_test)

    # predict class values
    y_pred = classifier.predict(X_test)

    return classifier, classifier_probs, y_pred, y_test

def _run_classifier_binary(X, y, classifier='logistic', test_size=.2):
    """ fit logistic regression model - crossvalidated
        Args:
            X_train (matrix):
            y_train (vector):
            classifier (str): 'logistic', 'svm' etc
            test_size (int): default is .2
            fit_intercept (bool): default is True

            Kwargs:
                solver (str): default is 'lbfgs'
        Return:
            logistic model
    """
    random_state = np.random.seed(47)

    X_train, X_test, y_train, y_test = _split_train_test(X, y, test_size=test_size)

    if classifier=='logistic':
        classifier = lm.LogisticRegression(fit_intercept=True, solver='lbfgs') # solver='lbfgs'
        classifier.fit(X_train, y_train)
        classifier_probs = classifier.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        classifier_probs = classifier_probs[:,1]
    elif classifier=="svm":
        classifier = svm.LinearSVC(random_state=random_state)
        classifier.fit(X_train, y_train)
        classifier_probs = classifier.decision_function(X_test)

    # predict class values
    y_pred = classifier.predict(X_test)

    return classifier, classifier_probs, y_pred, y_test

def _confusion_matrix(X, y, classifier='logistic', label_type='multi-class', test_size=.2):
    """ Returns confusion matrix for either accuracy or precision_recall
        Args:
            X (matrix): training data
            y (vector): labelled data
            test_size (int): default is .w
            type (str): "accuracy" or "precision_recall"
        Returns:
            confusion matrix
    """
    # run classifier
    if label_type=="multi-class":
        _, _, y_pred, y_test = _run_classifier_multiclass(X, y, classifier, test_size)
        f1 = f1_score(y_test, y_pred, average='micro')
    elif label_type=="binary":
        _, _, y_pred, y_test = _run_classifier_binary(X, y, classifier, test_size)
        f1 = f1_score(y_test, y_pred)
    else: 
        print(f'{label_type} does not exist. options are multi-class or binary')

    cnf_matrix = confusion_matrix(y_test, y_pred)

    return cnf_matrix, f1

def _recall_precision(X, y, classifier='logistic', label_type='multi-class', test_size=.2):
    if label_type=="multi-class":
        # run classifier on multi-class data
        classifier, classifier_probs, y_pred, y_test = _run_classifier_multiclass(X, y, classifier, test_size)

        # calculate precision and recall for multi-class
        y_binarized = label_binarize(y_test, y_test.unique())
        n_classes = y_binarized.shape[1]
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_binarized[:, i], classifier_probs[:, i])
            average_precision[i] = average_precision_score(y_binarized[:, i], classifier_probs[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_binarized.ravel(),
        classifier_probs.ravel())
        average_precision["micro"] = average_precision_score(y_binarized, classifier_probs,
                                                        average="micro")
        f1 = f1_score(y_test, y_pred, average='micro')
        # f1 = f1_score(y_test, y_pred, average=None) # f1 for all classes

    elif label_type=="binary":
        # run classifier on binary data
        classifier, classifier_probs, y_pred, y_test = _run_classifier_binary(X, y, classifier, test_size)

        precision, recall, _ = precision_recall_curve(y_test, classifier_probs) 
        average_precision = average_precision_score(y_test, classifier_probs) 
        n_classes = 1 
        f1, auc_score = f1_score(y_test, y_pred), auc(recall, precision)            
    else:
        print(f'{label_type} does not exist. options are multi-class or binary')               

    return precision, recall, average_precision, n_classes, f1

def _get_X_y(atlas, dataframe, label_type):
    """ returns training data (X), labelled data (Y), and classes
        Args:
            dataframe: dataframe containing X, Y, and classes
            atlas (str): which atlas are we working with? defines classes based on atlas
            label_type (str): 'multi-class' or 'binary'
        Returns:
            X, y, classes
    """
    # x_cols = _get_gene_symbols(atlas="MDTB-10", which_genes='top', percentile=1)
    x_cols = dataframe.filter(regex=("[A-Z0-9].*")).columns

    if label_type=="binary":
        if atlas=="MDTB-10-subRegions":
            dataframe['class_num'] = dataframe['region_num'].apply(lambda x: 0 if x<11 else 1)
            dataframe['class_name'] = dataframe['region_id'].str.extract(r'-(\w+)')
        else:
            print(f'binary option not available for {atlas}')

    if label_type=="multi-class":
            dataframe['class_num'] = dataframe['region_num']
            dataframe['class_name'] = dataframe['region_id']

    X = dataframe[x_cols]
    y = dataframe['class_num']

    classes = dataframe['class_name'].unique()

    return X, y, classes

def _compute_CV_error(model, X, y, test_size=.2):
    '''
    Split the training data into 4 subsets.
    For each subset, 
        fit a model holding out that subset
        compute the MSE on that subset (the validation set)
    You should be fitting 4 models total.
    Return the average MSE of these 4 folds.

    Args:
        model: an sklearn model with fit and predict functions 
        X_train (data_frame): Training data
        y_train (data_frame): Label 

    Return:
        the average validation MSE for the 4 splits.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    kf = KFold(n_splits=4)
    validation_errors = []
    
    for train_idx, valid_idx in kf.split(X_train):

        # split the data
        split_X_train, split_X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        split_y_train, split_y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]
      
        # Fit the model on the training split
        model.fit(split_X_train, split_y_train)
        
        error = _rmse(model.predict(split_X_valid), split_y_valid)

        validation_errors.append(error)
        
    return np.mean(validation_errors)

def _compute_test_train_error(model, X, y, test_size=.2):
    train_error_vs_N = []
    test_error_vs_N = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    range_of_num_features = range(1, X_train.shape[1] + 1)

    for N in range_of_num_features:
        X_train_first_N_features = X_train.iloc[:, :N]    
        
        model.fit(X_train_first_N_features, y_train)
        train_error = _rmse(model.predict(X_train_first_N_features), y_train)
        train_error_vs_N.append(train_error)
        
        X_test_first_N_features = X_test.iloc[:, :N]
        test_error = _rmse(model.predict(X_test_first_N_features), y_test)    
        test_error_vs_N.append(test_error)

    return test_error_vs_N, train_error_vs_N, range_of_num_features

def _rmse(y_pred, y_test, test_size=.2):
    """
    Args:
        y_pred: an array of the prediction from the model
        y_test: an array of the groudtruth label
        
    Returns:
        The root mean square error between the prediction and the groudtruth
    """
    return np.sqrt(np.mean((y_test - y_pred) ** 2)) 

def _get_model(model_type='linear'):
    if model_type=="linear":
        model = lm.LinearRegression()

    return model

def _fit_linear_model_optimal_features(X, y, test_size=.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    range_of_num_features = range(1, X_train.shape[1] + 1)

    errors = []
    for N in range_of_num_features:
        print(f"Trying first {N} features")
        model = lm.LinearRegression()
        
        # compute the cross validation error
        error = _compute_CV_error(model, X_train.iloc[:, :N], y_train) 
        
        print("\tRMSE:", error)
        errors.append(error)

    best_num_features = np.argmin(errors) + 1
    best_err = errors[best_num_features - 1]

    print(f"Best choice, use the first {best_num_features} features")

    # Fit linear model with best features
    model = lm.LinearRegression()
    model.fit(X_train.iloc[:, :best_num_features], y_train)
    train_rmse = _rmse(model.predict(X_train.iloc[:, :best_num_features]), y_train) 
    test_rmse = _rmse(model.predict(X_test.iloc[:, :best_num_features]), y_test)

    print("Train RMSE", train_rmse)
    print("KFold Validation RMSE", best_err)
    print("Test RMSE", test_rmse)

    return best_num_features, train_rmse, best_err, test_rmse

# __all__ = ["save_expression_data", "save_atlas_info", "save_thresholded_data"]
