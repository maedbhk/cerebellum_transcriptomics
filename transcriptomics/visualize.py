import os
import re

import pandas as pd
import numpy as np
import seaborn as sns
import ast
import random
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.image as mpimg
from PIL import Image
import itertools

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, set_link_color_palette
from sklearn import cluster
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score

from nilearn import plotting, datasets, surface

from transcriptomics import gec_functions_ana as ana
from transcriptomics import gec_functions_preprocess as preprocess
from transcriptomics.constants import Defaults

import plotly
import plotly.offline as py
py.init_notebook_mode(connected=False)

import plotly.graph_objs as go
import plotly.figure_factory as ff
import cufflinks as cf

from itertools import cycle

cf.set_config_file(offline=False, world_readable=True, theme='ggplot')

def plotting_style():
    """This function sets the style for plotting. It should be called before any plotting is done. 
        No args. 
    """
    plt.style.use('seaborn-poster') # ggplot
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Helvetica Neue') 
    plt.rc('text', usetex='false') 
    plt.rcParams['lines.linewidth'] = 2
    plt.rc('xtick', labelsize=20)     
    plt.rc('ytick', labelsize=20)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["savefig.format"] = 'png'

def diff_stability_plot(atlas, which_genes='top', percentile=1, ax=None, method='ds', **kwargs): 
    """This function plots the results of the differential stability analysis. 
    Main purpose of this graph is to demonstrate that only a small subset of genes 
    are taken to be used by further analyses. The top 1% of the most stable genes (across donor groups)
    as indicated by high R values. 
    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        ax (bool): figure axes. Default is None
        method (str): `ds` or `dr`. different ways of selecting genes. (`ds`=differnetial stability and `dr`=dimensionality reduction)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any graphical argument relevant to seaborn "distplot"
    """
    # genes from differential stability
    ds = preprocess._get_differential_stability(atlas=atlas)

    # threshold, _ = preprocess._threshold_genes_ds(genes_ds, which_genes=which_genes, percentile=percentile)
    # genes from dimensionality reduction
    dr = ana._threshold_genes_dr(atlas=atlas)

    if method=="ds":
        all_genes = ds
        gene_subset_n = np.round(len(all_genes)*(percentile/100)).astype(int)
        top_genes = dr.sort_values(ascending=False)[:gene_subset_n] #determine number of genes corresponding to `percentile`
        xlabel = 'Correlation (R)'
        # title = 'Gene selection (differential stability)'
    elif method=="dr":
        all_genes = dr
        gene_subset_n = np.round(len(all_genes)*(percentile/100)).astype(int)
        top_genes = ds.sort_values(ascending=False)[:gene_subset_n]
        xlabel = 'Dimensionality reduction'
        # title = 'Gene selection (dimensionality reduction)'
    else: 
        print(f'{method} does not exist as a method')

    # get threshold for `method`
    if which_genes == "top":
        threshold = np.percentile(all_genes, 100 - percentile, interpolation = 'linear')
    elif which_genes == "bottom":
        threshold = np.percentile(all_genes, 1 + percentile, interpolation = 'linear')

    # set bin size
    bins = 500

    # visualise thresholded expression data
    if ax is None:
        plt.figure(num=2, figsize=[20,8])
    ax = sns.distplot(all_genes, bins=bins, ax=ax, **kwargs)
    ax.axvline(x=threshold, color='r', linestyle='--')
    ax.set_ylabel('Gene Count')
    ax.set_xlabel(xlabel)
    # ax.set_title(title)

    # highlight genes from other method to demonstrate similarity (if any) 
    # indices = [all_genes.index.get_loc(i) for i in ['ACP2']]
    # indices = [all_genes.index.get_loc(i) for i in top_genes.index]
    # binlen = len(all_genes) // bins
    # binidx = [idx // binlen for idx in indices]
    # for idx in binidx:
    #     idx = idx - 1
    #     idx = np.clip(idx, 0, bins-1)
    #     ax.patches[idx].set_color('red')

    # find overlap of methods
    set1 = dr.sort_values(ascending=False)[:gene_subset_n]
    set2 = ds.sort_values(ascending=False)[:gene_subset_n]
    common_genes = set(set1.index).intersection(set2.index)
    # print(f'The gene selection methods contain {len(common_genes)/gene_subset_n}% of the same genes')

    return ax, all_genes, top_genes

def _reorder_colors_x_axis(dataframe, cpal):
    """
    reorder the colors along the x axis when regions are missing
    atlas labels need to be in the format "R01" to work
    """
    try:
        labels = sorted(dataframe['region_id'].unique())
    except:
        labels = dataframe.columns

    regex = r"(\d+)-?(\w+)?"

    try:
        # get atlas labels
        groups = []
        for p in labels:
            match = re.findall(regex, p)[0]
            groups.append(match)
    except:
        print('atlas labels are not in the correct format, example format is R01 or R01-<any char>')

    index = []
    for group in groups:
        if group[1]=='A':
            index.append(int(group[0])-1)
        elif group[1]=='P':
            index.append(int(group[0]) + int(len(cpal)/ 2) - 1)
        else:
            index.append(int(group[0])-1)
            
    cpal_reordered = []
    for i in index:
        cpal_reordered.append(cpal[i])

    return labels, cpal_reordered

def sample_counts_roi_donor(dataframe, ax=None, **kwargs):
    """ Plots sample counts per roi and donor
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
                may include any graphical argument relevant to seaborn "barplot"
    """

    if ax is None:
        plt.figure(num=2, figsize=[20,8])
    ax = sns.barplot(x="region_id", y="sample_counts", hue="donor_id", data=dataframe, palette="Set2", order=sorted(dataframe['region_id'].unique()), ax=ax, **kwargs) #  bw=.2, inner="stick"
    ax.set_ylabel('Sample Count')
    ax.set_xlabel('')
    ax.legend(loc='upper right')

def sample_counts_roi(dataframe, ax=None, **kwargs):
    """ Plots sample counts per roi
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
                atlas (str): atlas name to color the plot with atlas colors
                may include any graphical argument relevant to seaborn "violinplot"
            
    """

    if kwargs.get("atlas"):
        cpal = _make_colorpalette(kwargs['atlas'])
    else:
        cpal = "Set2"

    # gets the correct color for each region (only a problem when there are missing regions)
    labels, cpal_reordered = _reorder_colors_x_axis(dataframe, cpal)

    if ax is None:
        plt.figure(num=2, figsize=[20,8])
    
    if len(dataframe['donor_id'].unique())>1:
        ax = sns.violinplot(x="region_id", y="sample_counts", data=dataframe, palette=cpal_reordered, scale="count", order=labels, ax=ax, **kwargs) #  bw=.2, inner="stick"
    else:
        ax = sns.barplot(x="region_id", y="sample_counts", data=dataframe, palette=cpal_reordered, order=labels, ax=ax) #  bw=.2, inner="stick"
    ax.set_ylabel('Sample Count')
    ax.set_xlabel('')

def sample_counts_donor_mean(dataframe, ax=None, **kwargs):
    """ Plots sample counts per donor (mean)
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
            may include any graphical argument relevant to seaborn "barplot"
    """
    if ax is None:
        plt.figure(num=2, figsize=[20,8])

    # colors for barplot (seems like a lot of hassle to make plot colors ...)
    labels = Defaults.donors
    colors_rgb = Defaults.donor_colors # given as rgb between 0 and 1
    cmap_name = "donors"
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors_rgb, N=len(colors_rgb))
    mpl.cm.register_cmap("mycolormap", cmap)
    cpal = sns.color_palette("mycolormap", n_colors=len(labels))

    ax = sns.violinplot(x="donor_id", y="sample_counts", data=dataframe, palette=cpal, scale="count", ax=ax, **kwargs) #  bw=.2, inner="stick"
    ax.set_ylabel('Sample Count')
    ax.set_xlabel('')

def sample_counts_donor_sum(dataframe, ax=None, **kwargs):
    """ Plots sample counts per donor (sum)
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
            may include any graphical argument relevant to seaborn "barplot"
    """
    if ax is None:
        plt.figure(num=2, figsize=[20,8])

    # colors for barplot (seems like a lot of hassle to make plot colors ...)
    labels = Defaults.donors
    colors_rgb = Defaults.donor_colors # given as rgb between 0 and 1
    cmap_name = "donors"
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors_rgb, N=len(colors_rgb))
    mpl.cm.register_cmap("mycolormap", cmap)
    cpal = sns.color_palette("mycolormap", n_colors=len(labels))

    sample_total = dataframe.groupby('donor_id').sum()['sample_counts']
    x = sample_total.index
    y = sample_total.values
    ax = sns.barplot(x, y, ax=ax, palette=cpal, **kwargs)
    ax.set_ylabel('Sample Count')
    ax.set_xlabel('')

def png_plot(filename, ax=None):
    """ Plots a png image from the "atlas_templates" folder.
        
        Args:
            filename (str): the name of the image to plot
            ax (bool): figure axes. Default is None
    """
    fpath = str(Defaults.EXTERNAL_DIR / "atlas_templates" / "atlases_png" / f"{filename}.png")
    if os.path.isfile(fpath):
        img = mpimg.imread(fpath)
    else:
        print("image does not exist. run concat_png to create desired image")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.imshow(img, origin='upper', vmax=abs(img).max(), vmin=-abs(img).max(), aspect='equal')

def _make_colormap(atlas):
    """ Helper function to make a matplotlib colormap. Color info csv file must have been created.
    If atlas does not exist, a default colormap is provided.
        
        Args:
            atlas (str): the name of the atlas to use
    """

    atlas_info = Defaults.EXTERNAL_DIR / "atlas_templates" / f"{atlas}-info.csv"
    if os.path.isfile(atlas_info): 
        df = pd.read_csv(atlas_info)

        colors = df.apply(lambda x: list([x['r'],
                                        x['g'],
                                        x['b']]), axis=1) 

        regions = df['region_id']

        # if transcriptomic, don't divide by 255
        if atlas.find('transcriptomic')>-1:
            colors_rgb = [[np.round(y, 2) for y in x] for x in colors]
        else:
            colors_rgb = [[np.round(y / 255, 2) for y in x] for x in colors]

        # remove any nan colors (applicable for transcriptomic parcellation)
        colors_all = []
        labels = []
        for i, rgb in enumerate(colors_rgb):
            if any([np.isnan(x) for x in rgb]):
                pass
            else:
                colors_all.append(rgb)
                labels.append(regions[i])

        cm = LinearSegmentedColormap.from_list(atlas, colors_all, N=len(colors_all))
    else: 
        cm = "Accent"
        labels = []

    return cm, labels

def _make_colorpalette(atlas):

    cm, labels = _make_colormap(atlas)

    mpl.cm.register_cmap("mycolormap", cm)
    cpal = sns.color_palette("mycolormap", n_colors=len(labels))

    return cpal

def _make_colorbar(atlas, ax=None):
    """ Helper function to make a matplotlib colorbar - specifically for atlas. Color info csv file must have been created.
        
        Args:
            atlas (str): the name of the atlas to use
            info_file (bool): does this colorbar have an info file?
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(1, 10))

    cmap, labels = _make_colormap(atlas)

    bounds = np.arange(cmap.N + 1)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cmap.reversed(cmap), 
                                    norm=norm,
                                    ticks=bounds,
                                    format='%s',
                                    orientation='vertical',
                                    )
    cb3.set_ticklabels(labels[::-1])  
    cb3.ax.tick_params(size=0)
    cb3.set_ticks(bounds+.5)
    cb3.ax.tick_params(axis='y', which='major', labelsize=30)

    plt.savefig(str(Defaults.EXTERNAL_DIR / "atlas_templates" / "atlases_png" / f"{atlas}-colorbar.png"), bbox_inches='tight')

def _make_colorbar_donor(ax=None):
    """ Helper function to make a matplotlib colorbar - specifically for donor.
        Donor names and colors are given in constants.py
        
        Args:
            atlas (str): the name of the atlas to use
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(1, 10))

    labels = Defaults.donors
    colors_rgb = Defaults.donor_colors # given as rgb between 0 and 1

    cmap_name = "donors"
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors_rgb, N=len(colors_rgb))

    bounds = np.arange(cmap.N + 1)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cmap.reversed(cmap), 
                                    norm=norm,
                                    ticks=bounds,
                                    format='%s',
                                    orientation='vertical',
                                    )
    cb3.set_ticklabels(labels[::-1])  
    cb3.ax.tick_params(size=0)
    cb3.set_ticks(bounds+.5)
    cb3.ax.tick_params(axis='y', which='major', labelsize=30)

    plt.savefig(str(Defaults.EXTERNAL_DIR / "atlas_templates" / "atlases_png" / f"{cmap_name}-colorbar.png"), bbox_inches='tight')

def interactive_cortex(atlas, info_file=True, mesh='very_inflated', hemisphere="L"):
    """ Plots an interactive cortical surface map. Options for mesh and map can be found in 
        'fs_LR_32'. Options for mesh end with 'surf.gii' and options for atlas end with 'label.gii'
        
        Args:
            atlas (str): the name of the atlas to use for mapping labelled data onto surface
            info_file (bool): does the atlas have an associated info file
            mesh (str): the name of the mesh to use for visualizing labelled data on surface.
            Default is 'very_inflated'
            hemisphere (str): hemisphere to visualize. Options are "L" and "R". Default is "L".
    """
    # get mesh
    surf_mesh = os.path.join(Defaults.SURF_DIR, f"fs_LR.32k.{hemisphere}.{mesh}.surf.gii")

    # get surface
    surf_map = _get_cortex_atlas(atlas, hemisphere=hemisphere)

    # get colors
    cm, labels = _make_colormap(atlas)

    # visualize surface data
    surf_data = surface.load_surf_data(str(os.path.join(Defaults.SURF_DIR, surf_map)))
    view = plotting.view_surf(str(surf_mesh), surf_data, cmap=cm, symmetric_cmap=False, colorbar=False)     
    # view.resize(500,500)

    view.open_in_browser()

def interactive_cerebellum(atlas, info_file=True, surf_mesh = "PIAL_SUIT.surf.gii"):
    os.chdir(Defaults.EXTERNAL_DIR /  "atlas_templates")

    surf_map_data = surface.load_surf_data(f"{atlas}.label.gii").astype('int32')

    # get colors
    cm, labels = _make_colormap(atlas)

    # plotting.plot_surf_roi(str(os.path.join(Defaults.EXTERNAL_DIR, "atlas_templates", surf_mesh)), roi_map=surf_map_data, view='medial')

    view = plotting.view_surf(surf_mesh, surf_map_data, symmetric_cmap=False, cmap=cm, colorbar=False)     #cmap=cm
    # view.resize(500,500)

    view.open_in_browser()

def _make_png_cortex(atlas, info_file=True, mesh='very_inflated', hemisphere="L", view="lateral", ax=None, save=True, resize=True):
    """ Plots a cortical surface map. Options for mesh and map can be found in 
        'fs_LR_32'. Options for mesh end with 'surf.gii' and options for atlas end with 'label.gii'
        
        Args:
            atlas (str): the name of the atlas to use for mapping labelled data onto surface
            info_file (bool): does the atlas have an info file
            mesh (str): the name of the mesh to use for visualizing labelled data on surface.
            Default is 'very_inflated'
            hemisphere (str): hemisphere to visualize. Options are "L" and "R". Default is "L".
            view (str): visualize atlas from different views. Default is "lateral".
            ax (bool): figure axes. Default is None
            save (bool): save out plot. Default is True. If False, image is visualised in window.
    """

    # get mesh
    surf_mesh = os.path.join(Defaults.SURF_DIR, f"fs_LR.32k.{hemisphere}.{mesh}.surf.gii")

    # get surface
    surf_map = _get_cortex_atlas(atlas, hemisphere=hemisphere)

    # get colors
    cm, labels = _make_colormap(atlas)

    surf_data = surface.load_surf_data(str(os.path.join(Defaults.SURF_DIR, surf_map)))

    filename = Defaults.EXTERNAL_DIR / "atlas_templates" / "atlases_png" / f"{atlas}-{hemisphere}-{view}.png"

    if hemisphere is "L":
        hem = "left"
    else:
        hem = "right"

    if save:
        plotting.plot_surf_roi(str(surf_mesh), roi_map=surf_data, hemi=hem, view=view, 
                cmap=cm, darkness=.5, output_file=str(filename))
        # if resize:
        #     image = Image.open(str(filename))
        #     atlas_size = image.size
        #     newsize = (int(atlas_size[0]/2), int(atlas_size[1]/2)) 
        #     image = image.resize(newsize)
        #     image.save(str(filename))
    else:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        plotting.plot_surf_roi(str(surf_mesh), roi_map=surf_data,
                hemi=hem, view=view, cmap=cm, darkness=.5, axes=ax)
        plotting.show()

def _make_all_png():
    atlases = ["Yeo-7", "Yeo-17"]
    hemispheres = ["L", "R"]
    views = ["lateral", "medial"]

    # make cortex 
    for atlas in atlases:
        for hem in hemispheres:
            for view in views:
                _make_png_cortex(atlas=atlas, hemisphere=hem, view=view)

    # make colorbar
    for atlas in Defaults.colour_info:
        _make_colorbar(atlas)

    # # resize cerebellar atlases
    # atlases = ["SUIT-10-flat", "MDTB-10-flat", "Buckner-7-flat", "Buckner-17-flat", "Ji-10-flat"]
    # for atlas in atlases:
    #     _resize_png(atlas=atlas)

def _resize_png(atlas):
    os.chdir(Defaults.EXTERNAL_DIR / "atlas_templates")
    image = Image.open(f"{str(atlas)}.png")
    atlas_size = image.size
    newsize = (int(atlas_size[0]/4), int(atlas_size[1]/4)) 
    image = image.resize(newsize)
    out_dir = Defaults.EXTERNAL_DIR / "atlas_templates" / f"{atlas}.png"
    image.save(str(out_dir))

def concat_png(filenames, outname, offset=0):

    os.chdir(Defaults.EXTERNAL_DIR / "atlas_templates" / "atlases_png")

    # join images together
    try:
        images = [Image.open(f"{str(x)}.png") for x in filenames]
    except:
        _make_all_png()
        images = [Image.open(f"{str(x)}.png") for x in filenames]

    # resize all images (keep ratio aspect) based on size of min image
    sizes = ([(np.sum(i.size), i.size ) for i in images])
    min_sum = sorted(sizes)[0][0]

    images_resized = []
    for s, i in zip(sizes, images):
        resize_ratio = int(np.floor(s[0] / min_sum))
        orig_size = list(s[1])
        if resize_ratio>1:
            resize_ratio = resize_ratio - 1.5
        new_size = tuple([int(np.round(x / resize_ratio)) for x in orig_size])
        images_resized.append(Image.fromarray(np.asarray(i.resize(new_size))))

    widths, heights = zip(*(i.size for i in images_resized))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for im in images_resized:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0] - offset

    out_dir = Defaults.EXTERNAL_DIR / "atlas_templates" / "atlases_png" / f"{outname}.png"
    new_im.save(str(out_dir))

def _get_cortex_atlas(atlas, hemisphere="L"):
    """ Helper function to retrieve cortical surface map. 
        
        Args:
            atlas (str): the name of the atlas to use for mapping labelled data onto surface. 
            Options are "Yeo-7", "Yeo-17", "Glasser", "Desikan-Killiany-83", and "Dextrieux". 
            If the atlas does not exist, an error will be printed.
            hemisphere (str): hemisphere to visualize. Options are "L" and "R". Default is "L".
    """
    # get surface 
    if atlas == "Yeo-7":
        surf_map = f"Yeo_JNeurophysiol11_7Networks.32k.{hemisphere}.label.gii"

    elif atlas == "Yeo-17":
        surf_map = f"Yeo_JNeurophysiol11_17Networks.32k.{hemisphere}.label.gii"

    elif atlas == "Glasser":
        surf_map = f"Glasser_2016.32k.{hemisphere}.label.gii"

    elif atlas == "Desikan-Killiany-83":
        surf_map = f"Desikan.32k.{hemisphere}.label.gii"

    elif atlas == "Dextrieux":
        surf_map = f"Dextrieux.32k.{hemisphere}.label.gii"

    else:
        print("Not an option. Options are yeo-7, yeo-17, Glass, Desikan-Killiany-83 or Dextrieux. Not formatted yet for cerebellar atlases")

    return surf_map

def k_means_n_dims_heatmap(dataframe, num_clusters=2, ax=None, **kwargs):
    """ Plots n-dim k means heatmap
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            num_clusters (int): number of clusters for k-means. Default is 2
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
            may include any graphical argument relevant to plotly go.Heatmap
    """
    df_n_dims = ana._compute_k_means_n_dims(dataframe, num_clusters=num_clusters)

    # get x and y labels
    x_labels = df_n_dims.columns
    y_labels = []
    for y in df_n_dims.index.values:
        y_labels.append('cluster' + str(y+1))

    if ax is None:
        plt.figure(num=2, figsize=[20,8])
    # visualise heatmap
    data = [go.Heatmap(z=df_n_dims.values.tolist(), 
                       y=y_labels,
                       x=x_labels,
                       colorscale='Viridis', 
                       ax=ax, **kwargs)]
    plotly.offline.iplot(data, filename='pandas-heatmap')

def silhouette_plot(dataframe, range_n_clusters=[2]):
    """ Plots silhouette plot for k means analysis
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            range_n_clusters (list): number of clusters for k-means. Default is [2]
    """
    # code borrowed from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    for n_clusters in range_n_clusters:
        fig = plt.figure()
        ax1 = plt.axes()
        fig.set_size_inches(18, 7)

        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(dataframe) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(dataframe)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(dataframe, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(dataframe, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        ax1.set_title(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    plt.show()

def pcs_2D_plot(dataframe):
    """ Plots 2D plot of first two PCs
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
    """
    u, s, vt, pcs = ana._compute_svd(dataframe)

    colors = np.array(["rgba({0},{1},{2},1)".format(*c) for c in sns.color_palette("Blues", len(pcs))])
    
    plt.figure(num=2, figsize=[20,8])
    py.iplot(
        go.Figure(
            data=[go.Scatter(x = pcs['pc0'], y = pcs['pc1'], mode='markers', marker=dict(color=colors, size=10), text=pcs.reset_index().rename({'index': 'gene_symbol'}, axis=1)['gene_symbol'])
                 ],
            layout=go.Layout(xaxis=dict(title="PC1"), yaxis=dict(title="PC2"))))
    plotting_style()

    # table of genes per pc
    cluster_1 = pcs.query('pc0>0 and pc1>-4')
    cluster_2 = pcs.query('pc0<0 and pc1>-4')

    fig = go.Figure(data=[go.Table(header=dict(values=['PC1', 'PC2']),
                     cells=dict(values=[cluster_1.index.to_list(), cluster_2.index.to_list()]))
                         ])
    fig.show()

def pcs_winner_2D_plot(dataframe, num_pcs=2, ax=None, **kwargs):
    """ Plots 2D plot. Each data point is labelled with "winner" region label.
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            num_pcs (int): number of pcs to include in winner-take-all. Default is 2.
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
            may include any graphical argument relevant to seaborn's scatterplot
    """

    pcs_labelled = ana._pcs_winner_take_all(dataframe, num_pcs=num_pcs)
    
    if ax is None:
        plt.figure(num=2, figsize=[20,8])
    ax = sns.scatterplot(data = pcs_labelled, x = "pc0", y = "pc1", hue='region', palette="Set2", s=100, hue_order=sorted(pcs_labelled['region'].unique()), ax=ax, **kwargs) # palette='coolwarm'
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=15)  
    ax.set_ylabel('PC2')
    ax.set_xlabel('PC1')

def relative_weight_genes(dataframe, pc_num=0):
    
    u, s, vt, pcs = ana._compute_svd(dataframe)

    relative_weight = []
    for pc in pcs[f"pc{pc_num}"]:
        total_weight = sum(abs(pcs[f"pc{pc_num}"]))
        relative_weight.append(abs(pc) / total_weight)

    ax = sns.distplot(relative_weight, bins=20)
    ax.set_ylabel('Genes')
    ax.set_xlabel('Relative Weight')
    ax.set_title(f"PC{pc_num+1}")

def pcs_table(dataframe, num_pcs=2):
    pcs_labelled = ana._pcs_winner_take_all(dataframe, num_pcs=num_pcs)

    pcs_labelled = pcs_labelled.reset_index().rename({'index': 'genes'}, axis=1)

    # table of genes per region
    region_genes = pcs_labelled.reset_index().groupby('region')['index'].apply(list)
    fig = go.Figure(data=[go.Table(header=dict(values=region_genes.keys().to_list()),
                     cells=dict(values=region_genes.to_list()))
                         ])
    fig.show()

    return pcs_labelled

def pcs_winner_3D_plot(dataframe, num_pcs=2):
    """ Plots 3D plot. Each data point is labelled with "winner" region label.
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            num_pcs (int): number of pcs to include in winner-take-all. Default is 2.
    """
    pcs_labelled = ana._pcs_winner_take_all(dataframe, num_pcs=num_pcs)
    
    regions = pcs_labelled['region'].unique()

    colors = _color_list()
    colors = random.sample(colors, len(regions))

    clusters = []
    scatters = []
    for i, (reg, color) in enumerate(zip(regions, colors)): 
        clusters.append(
            dict(
            alphahull = 5, 
    #         name = 'cluster' + str(i+1),
            name = reg,
            opacity = .3,
            type = "mesh3d",
            x = pcs_labelled.query(f'region == "{reg}"')['pc0'].values,
            y = pcs_labelled.query(f'region == "{reg}"')['pc1'].values,
            z = pcs_labelled.query(f'region == "{reg}"')['pc2'].values,
            color = color,
            showscale = True
        )
        )
        scatters.append(
            dict(
            mode = "markers",
    #         name = 'cluster' + str(i+1),
            name = reg,
            type = "scatter3d",    
            x = pcs_labelled.query(f'region == "{reg}"')['pc0'].values, 
            y = pcs_labelled.query(f'region == "{reg}"')['pc1'].values, 
            z = pcs_labelled.query(f'region == "{reg}"')['pc2'].values, 
            marker = dict(size=3, color=color)  
        )
        )

    layout = dict(
        title = 'Interactive PCs in 3D',
        scene = dict(
            xaxis = dict( zeroline=True ),
            yaxis = dict( zeroline=True ),
            zaxis = dict( zeroline=True ),
            xaxis_title = "PC1",
            yaxis_title = "PC2",
            zaxis_title = "PC3",
        )
    )
    plt.figure(num=2, figsize=[20,8])
    fig = dict( data=scatters + clusters, layout=layout )
    #Use py.iplot() for IPython notebook
    plotly.offline.iplot(fig, filename='mesh3d_sample')

def pcs_loading_plot(dataframe, pcs=[1], ax=None, **kwargs):
    """ Plots pcs loading plot for n pcs. 
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            pcs (list of int): pcs to plot.
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
                group_pcs (bool): plot multiple pcs on graph
                atlas (str): colour graph with atlas colors
                may include any graphical argument relevant to seaborn's barplot
    """

    if kwargs.get("atlas"):
        cpal = _make_colorpalette(kwargs['atlas'])
    else:
        cpal = "Set2"

    # gets the correct color for each region (only a problem when there are missing regions)
    labels, cpal_reordered = _reorder_colors_x_axis(dataframe, cpal)

    u, s, vt, _ = ana._compute_svd(dataframe)

    # zero index the pcs
    pcs = [x-1 for x in pcs]

    if kwargs.get("group_pcs"):
        for pc in pcs:
            if ax is None:
                plt.figure(num=2, figsize=[20,8])
            ax = sns.barplot(x=labels, y=vt[pc, :], palette=cpal_reordered, alpha=0.7, ax=ax)
            # ax.set_xticks(labels) # rotation=90
            ax.set_ylabel(f'PC{pc+1} Loading')
            ax.set_xlabel('Regions')
    else:
        if ax is None:
            plt.figure(num=2, figsize=[20,8])
        ax = sns.barplot(x=labels, y=vt[pcs[0], :], palette=cpal_reordered, alpha=0.7, ax=ax) # indexs into list
        # ax.set_xticks(labels) # rotation=90
        ax.set_ylabel(f'PC{pcs[0]+1} Loading')
        ax.set_xlabel('')

        # print(f'Positive Loading on pc {num_pcs+1}: {dataframe.columns[vt[num_pcs, :]>0].to_list()}')
        # print(f'Negative Loading on pc {num_pcs+1}: {dataframe.columns[vt[num_pcs, :]<0].to_list()}')
        
        variance_explained(dataframe)

def scree_plot(dataframe, ax=None, **kwargs):
    """ Plots scree plot for n pcs
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
            may include any graphical argument relevant to seaborn's lineplot
    """
    u, s, vt, pcs = ana._compute_svd(dataframe)

    if ax is None:
        plt.figure(num=2, figsize=[20,8])
    ax = sns.lineplot(x=np.arange(len(s)), y=s**2 / sum(s**2), ax=ax, **kwargs)
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Variance Explained')
    # ax.set_title('Scree Plot')
    plt.show()

def _reorder_dendrogram_leaves(dataframe):
    # calculate euclidean distances between regions
    Y = cdist(dataframe.T, dataframe.T, 'euclidean')
    Y[Y == 0] = 'nan'

    n_regions = len(dataframe.columns) - 1
    distances = []
    for i in np.arange(n_regions):
        distances.append(np.nanmean(Y[0] - np.nanmean(Y[i+1])))

    # get indices of distances
    idx = np.argsort(distances)
    idx = list(idx+1)
    idx.insert(0,0)

    # reorder dataframe using sorted indices
    cols = dataframe.columns.tolist()
    cols_reorder = []
    for i in idx:
        cols_reorder.append(cols[i])
    
    return dataframe[cols_reorder]

def dendrogram_plot(dataframe, method='ward', metric='euclidean', reorder=True, orientation='top', color_leaves=True, ax=None, **kwargs):
    """ Plots dendrogram plot.
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            method (str): 'ward' # there are other options given by linkage
            metric (str): 'euclidean' # there are other options given by linkage
            ax (bool): figure axes. Default is None
            color_leaves (bool): whether or not leaf nodes should be colored
            orientation (str): how the dendrogram should be oriented. Default is 'top'. Other options are 'right', 'left'
            reorder (bool): reorders the expression matrix before being input to clustering algo. See Gomez et al. (2019) for details
                kwargs (dict): dictionary of additional (optional) kwargs.
                may include any graphical argument relevant to scipy's dendrogram
    """
    if ax is None:
        plt.figure(num=1, figsize=[25,8])

    if reorder:
        dataframe = _reorder_dendrogram_leaves(dataframe)

    if color_leaves:
        set_link_color_palette(['b', 'r', 'y', 'm'])
    else:
        set_link_color_palette(['k', 'k', 'k', 'k'])

    R = dendrogram(
        Z=linkage(dataframe, method, metric),
        orientation=orientation,
        get_leaves=True,
        color_threshold=35.0,
        labels=dataframe.index.to_list(),
        distance_sort='ascending',
        above_threshold_color='black', 
        ax=ax, 
        **kwargs
        )

    # plt.title("Hierarchical Clustering Dendrogram", fontsize=20)
    plt.xlabel('')
    plt.ylabel(f"{metric.capitalize()} Distance")

    return R

def genes_dendrogram_print(dataframe):

    R = dendrogram_plot(dataframe)

    cluster1 = [color for color in R['color_list'] if color=='m']
    print(f'there are {len(cluster1)} genes in cluster1: {R["ivl"][0:len(cluster1)]}')

    cluster2 = [color for color in R['color_list'] if color=='y']
    print(f'there are {len(cluster2)} genes in cluster2: {R["ivl"][len(cluster1):len(cluster1)+len(cluster2)]}')

def clustermap_plot(dataframe, ax=None, **kwargs):
    """ Plots clustermap plot (heatmap and dendrogram)
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
            may include any graphical argument relevant to seaborn's clustermap
    """

    if ax is None:
        plt.figure(num=2, figsize=[20,8])
    sns.clustermap(dataframe, ax=ax, **kwargs)

def pairplot(dataframe):
    """ Plots pairplot (correlations between columns in a dataframe)
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            ax (bool): figure axes. Default is None
            kwargs (dict): dictionary of additional (optional) kwargs.
            may include any graphical argument relevant to seaborn's lineplot
    """
    sns.pairplot(dataframe, kind='reg')
    plt.tick_params(axis='both', which='major', labelsize=20)

def variance_explained(dataframe, pcs=[1]):
    """ Prints variance explained for n pcs

        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            pcs (list of int): which pcs should be analyzed
            
    """
    pcs_var_fraction = ana._variance_explained(dataframe, pcs=pcs)

    print(f"The first {pcs} pcs account for {np.round(pcs_var_fraction*100,2)}% of the overall variance")

def simple_corr_heatmap(dataframe, ax=None, **kwargs):
    """ Plots simple correlation heatmap.
        
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            ax (bool): figure axes. Default is None

        Kwargs (dict): dictionary of additional (optional) kwargs.
                may include any graphical argument relevant to seaborn's heatmap
                distance_correct (bool): correct for spatial autocorrelation between regions
                atlas (str): necessary to give atlas name if you're getting `distance_correct` matrix
                simple_labels (bool): make simpler labels for figures (i.e. remove first part of prefix "Buckner" or "Yeo")
        Returns: 
            heatmap visualization
    """
    if ax is None:
        plt.figure(num=2, figsize=[20,8])

    if kwargs.get("distance_correct"):
        if kwargs.get("atlas"):
            atlas = kwargs["atlas"]
        else:
            print('you must give an atlas name as kwarg to return residualized matrix')
        corr_matrix, labels = ana._corr_matrix_residualized(dataframe, atlas)
    else:
        corr_matrix, labels = ana._corr_matrix(dataframe)

    ax = sns.heatmap(
        corr_matrix, 
        vmin=-1, vmax=1, center=0,
        # cmap=sns.cubehelix_palette(8),
        cmap=sns.diverging_palette(220, 20, sep=20, as_cmap=True),
        square=True, 
        ax=ax, 
        linewidths=2.0,
        cbar=True,
        # **kwargs
    )
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=30)

    # figure out labels
    if kwargs.get("simple_labels"):
        regex = '(.*)-(.*)' # get everything after the second '-'
        labels_heat = []
        for label in labels:
            # labels_heat.append(re.findall(regex, label)[0])
            labels_heat.append(re.findall(regex, label)[0][1])
    else:
        labels_heat = labels

    ax.set_xticklabels(
        labels_heat,
        rotation=45,
        horizontalalignment='right'
        )
    ax.set_yticklabels(
        labels_heat,
        rotation=360,
        horizontalalignment='right'
    )
    ax.set_xlabel('')
    ax.set_ylabel('')

def raster_plot(dataframe, ax=None, gene_reorder=True, cbar=True, **kwargs):
    """ Plots raster plot, rows are genes and columns are rois
        Args:
            dataframe: dataframe is output from ana.return_grouped_data or ana.return_thresholded_data
            ax (bool): figure axes. Default is None
            gene_reorder (bool): reorders rows of raster plot according to dendrogram results. default is True.
            cbar (bool): option to plot a colorbar. default is True.
                kwargs (dict): dictionary of additional (optional) kwargs.
                may include any graphical argument relevant to seaborn's heatmap
    """
    if ax is None:
        plt.figure(num=1, figsize=[25,15])

    # reorder genes in raster plot based on clustering from dendrogram
    if gene_reorder:
        dataframe = dataframe.set_index(dataframe.index)
        R = dendrogram_plot(dataframe, color_leaves=True)
        dataframe = dataframe.reindex(R['ivl'])
        dataframe = dataframe.reindex(index=dataframe.index[::-1])

    ax = sns.heatmap(
        dataframe, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(220, 20, n=7), # sns.color_palette("YlOrRd", 10)
        square=False, 
        ax=ax,
        # linewidths=.5,
        cbar=cbar,
        cbar_kws={"shrink": 0.8},
        yticklabels=False
    )
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=30)

    # ax.tick_params(axis=u'both', which=u'both', length=0, bottom=False, top=False, labelbottom=False) 
    ax.set_xlabel('')
    ax.set_ylabel('Genes')

def complex_corr_heatmap(x, y, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 

def corrplot(corr_matrix, size_scale=500, marker='s'):
    corr = pd.melt(corr_matrix.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    complex_corr_heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=corr_matrix.columns,
        y_order=corr_matrix.columns[::-1],
        size_scale=size_scale
    )

def confusion_matrix_plot(atlas, dataframe, classifier='logistic', label_type='multi-class', normalize=True, test_size=.2, ax=None):
    """ This function prints and plots the confusion matrix.
        Args:
            atlas (str): "SUIT-10" or "MDTB-10-subRegions"
            dataframe: dataframe. should have x_cols, y_col, class_col
            x_cols (list): list of col names for training data
            y_col: col name for labelled data. default is 'region_num'
            class_col: default is 'region_id'
            normalize (bool): normalization can be applied by setting `normalize=True`
    """
    if ax is None:
        plt.figure(num=1, figsize=[10,10])

    X, y, classes = ana._get_X_y(atlas, dataframe, label_type)

    cm, f1 = ana._confusion_matrix(X, y, classifier, label_type, test_size)
    print(f'f1 = {f1}')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized confusion matrix"
    else:
        title = 'Confusion matrix, without normalization'

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def precision_recall_plot(atlas, dataframe, classifier='logistic', label_type='multi-class', test_size=.2, ax=None):
    """ This function plots precision-recall curves for binary and multi-class classification
    Args:
        atlas (str): "SUIT-10" or "MDTB-10-subRegions"
        dataframe: dataframe. should have x_cols, y_col, class_col
        x_cols (list): list of col names for training data
        y_col: col name for labelled data. default is 'region_num'
        class_col: default is 'region_id'
        normalize (bool): normalization can be applied by setting `normalize=True`
    """
    X, y, classes = ana._get_X_y(atlas, dataframe, label_type)

    _, _, _, y_test = ana._split_train_test(X, y, test_size=test_size)

    precision, recall, average_precision, n_classes, f1 = ana._recall_precision(X, y, classifier, label_type, test_size)

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    if ax is None:
        plt.figure(num=1, figsize=[10,10])

    if label_type=="multi-class":
        # plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                    ''.format(average_precision["micro"]))

        # for i, color in zip(range(n_classes), colors):
        #     l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        #     lines.append(l)
        #     labels.append(f'Precision-recall for class {classes[i]} (area = {average_precision[i]})')

        plt.legend(lines, labels, loc=(0, .5), prop=dict(size=14)) # bbox_to_anchor=(1, 0.5)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        print(f'Precision-recall (area = {average_precision["micro"]}')
        print(f'f1 = {f1}')

    elif label_type=="binary":
        # plot the precision-recall curves
        no_skill = len(y_test[y_test==1]) / len(y_test)

        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(recall, precision, marker='.', label='Classifier')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        print(f'Precision-recall (area = {average_precision})')
        print(f'f1 = {f1}')

    else:
        print('label type does not exist. options are binary or multi-class')

    # summarize scores
     
def residual_error_plot():
    plt.scatter(np.arange(len(X_test)), Y_test - model.predict(X_test.iloc[:, :best_num_features]))
    plt.xlabel('x')
    plt.ylabel('residual (true y - estimated y)')
    plt.title('Residual vs x for Linear Model')
    plt.axhline(y = 0, color='r');

def test_train_error_plot(atlas, dataframe, model_type='linear', label_type='multi-class', test_size=.2, ax=None):
    X, y, _ = ana._get_X_y(atlas, dataframe, label_type=label_type)

    model = ana._get_model(model_type=model_type)

    test_error_vs_N, train_error_vs_N, range_of_num_features = ana._compute_test_train_error(model, X, y, test_size=test_size)

    best_num_features, train_rmse, best_err, test_rmse = ana._fit_linear_model_optimal_features(X, y, test_size=test_size)

    miny = np.min(train_error_vs_N)
    maxy = np.max(test_error_vs_N)

    plt.plot(range_of_num_features, train_error_vs_N)
    plt.plot(range_of_num_features, test_error_vs_N)
    plt.vlines(best_num_features, miny, maxy, colors='k', linestyles='dashed')
    plt.legend(["training error", "test error", "best # of features"], loc='best', fontsize=11)
    plt.xlabel("number of features")
    plt.ylabel("RMSE")

    print("Train RMSE", train_rmse)
    print("KFold Validation RMSE", best_err)
    print("Test RMSE", test_rmse)

def _color_list():
    """ Helper function, returns rgb colorlist. Doesn't take args.
    """
    colors_all = dict(**mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors_all.items())
    colors = [name for hsv, name in by_hsv]
    return colors
