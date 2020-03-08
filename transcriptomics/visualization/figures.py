from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

import numpy as np

from transcriptomics.visualization import visualize
from transcriptomics import gec_functions_ana as ana 
from transcriptomics.constants import Defaults

def plotting_style():
    # fig = plt.figure(num=2, figsize=[20,8])
    plt.style.use('seaborn-poster') # ggplot
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Helvetica Neue') 
    plt.rc('text', usetex='false') 
    plt.rcParams['lines.linewidth'] = 2
    plt.rc('xtick', labelsize=14)     
    plt.rc('ytick', labelsize=14)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["savefig.format"] = 'png'
    plt.rc("axes.spines", top=False, right=False) # removes certain axes

    # sets the line weight for border of graph
    # for axis in ['top','bottom','left','right']:
    #     ax2.spines[axis].set_linewidth(4)

def fig_methods(atlas='SUIT-10', which_genes='top', percentile=1, remove_outliers=True, **kwargs):
    """This function plots figure 1. This is a descriptive figure so it 
    can either go in the methods section or the supp.
    Panel A: Suit parcellation + labels. Flatmap and Volume.
    Panel B: Differential stability analysis. Histogram. 
    Panel C: Sample count per ROI (mean across donors). Violinplot.
    Panel D: Sample count per donor (sum across donors). Barplot.

    - Methods section: outliers are removed (Region IX for donor9861). Results do not change when outliers are included.
    (I don't think the sample count figures need to be included in supp). Differential stability analysis comes before sample
    count because sample count is done on thresholded genes (although it actually shouldn't make a difference?)

    Supp:
    - Individual sample count per donor per roi.
    - What about visualizing the sample count on the flatmap and/or volume?

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
                atlas_other (str): returns thresholded data using genes from another atlas
                donor_num (int): any one of the 6 donors 
                reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """
    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    df = ana.return_thresholded_data(atlas=atlas, which_genes=which_genes, percentile=percentile, **kwargs)
    
    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)

    x_pos = -0.18
    y_pos = 1.02
    
    # 1a
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(x_pos, 1.39, 'A', transform=ax1.transAxes, fontsize=40,
    verticalalignment='top')
    visualize.png_plot(atlas=atlas, ax=ax1)
    ax1.axis('off')

    # 1b
    ax2 = fig.add_subplot(gs[0, 1])
    visualize.sample_counts_roi(df, ax=ax2, atlas=atlas)
    ax2.tick_params(axis='x', which='major', labelsize=15)
    ax2.tick_params(axis='y', which='major', labelsize=20)
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=40,
        verticalalignment='top')

    # 1c
    ax3 = fig.add_subplot(gs[1, 0])
    visualize.sample_counts_donor_sum(df, ax=ax3)
    ax3.tick_params(axis='x', which='major', labelsize=17)
    ax3.tick_params(axis='y', which='major', labelsize=20)
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=40,
        verticalalignment='top')
   
    # 1d
    ax4 = fig.add_subplot(gs[1, 1])
    visualize.diff_stability_plot(atlas=atlas, which_genes=which_genes, percentile=percentile, ax=ax4)
    ax4.tick_params(axis='x', which='major', labelsize=20)
    ax4.tick_params(axis='y', which='major', labelsize=20)
    ax4.text(x_pos, y_pos, 'D', transform=ax4.transAxes, fontsize=40,
        verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)

    fig.show()
    
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "fig_methods"), bbox_inches="tight", dpi=300)

def fig_1(atlas='SUIT-10', which_genes='top', percentile=1, remove_outliers=True, reorder_genes=True, **kwargs):
    """This function plots figure 1. 
    Panel A: Raster plot for SUIT-10. Genes x ROIs
    Panel B: Dendrogram for SUIT-10. Clusters genes.

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
                atlas_other (str): returns thresholded data using genes from another atlas
                donor_num (int): any one of the 6 donors 
                reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """
    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.2
    y_pos = 1.02

    ax1 = fig.add_subplot(gs[0, 0])
    df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, reorder_genes=reorder_genes, **kwargs)
    df = df.set_index(df.index)
    visualize.dendrogram_plot(df, orientation='left', color_leaves=True, ax=ax1)
    ax1.tick_params(axis='x', which='both', labelbottom=False, bottom=False, top=False)
    ax1.axis('off')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=30,
    verticalalignment='top')
    # ax2.tick_params(axis='y', which='major', labelsize=20)

    ax2 = fig.add_subplot(gs[0, 1])
    df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, reorder_genes=reorder_genes, **kwargs)
    visualize.raster_plot(df, reorder_genes=reorder_genes, ax=ax2)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=30,
    verticalalignment='top')

    plt.subplots_adjust(left=0.01, bottom=0.001, right=0.4, top=2.0, wspace=.4, hspace=.8)

    fig.show()
    
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "fig_1"), bbox_inches="tight", dpi=300)

def fig_2(atlas='SUIT-10', which_genes='top', percentile=1, remove_outliers=True, **kwargs):
    """This function plots figure 2. 
    Panel A: Correlation plot for anatomical regions. Shows distinct anatomical clustering of data.
    Panel B: Dendrogram.
    Panel C: PC1 Loading.
    Panel D: PC2 Loading. 

    Methods section: Results are the same if X is removed (small number of samples, could be biasing results. Supp?)

    Supp:
    - Need to also show that this structure falls apart if bottom 1% of genes are taken or an equivalent
    number of genes are randomly drawn. Repeat exact same figure -- seems like overkill?
    Just show one of either the corr plot, dendrogram, or the pc loading plots
    - Table of genes clustered along the first two pcs (ordered according to strength).
    Which genes are explaining the variance? Or another way to show this might be a heatmap of genes x rois
    can the crossover effect be nicely visualized?

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
                atlas_other (str): returns thresholded data using genes from another atlas
                donor_num (int): any one of the 6 donors 
                reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """
    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, **kwargs)

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)

    x_pos = -0.18
    y_pos = 1.02

    # 2a
    ax1 = fig.add_subplot(gs[0, 0])
    visualize.simple_corr_heatmap(df, ax=ax1)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.text(-0.38, y_pos, 'A', transform=ax1.transAxes, fontsize=40,
    verticalalignment='top')

    # 2b
    ax2 = fig.add_subplot(gs[0, 1])
    visualize.dendrogram_plot(df.T, ax=ax2, color_leaves=False)
    ax2.tick_params(axis='x', which='major', labelsize=15)
    ax2.tick_params(axis='y', which='major', labelsize=20)
    plt.setp(ax2.lines, linewidth=10) # THIS ISN'T WORKING
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=40,
    verticalalignment='top')

    # load in new unthresholded grouped data
    # 2c
    df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, unthresholded=False, **kwargs)
    ax3 = fig.add_subplot(gs[1,0])
    visualize.pcs_loading_plot(df, num_pcs=0, group_pcs=False, atlas=atlas, ax=ax3)
    ax3.tick_params(axis='x', which='major', labelsize=15)
    ax3.tick_params(axis='y', which='major', labelsize=20)
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=40,
    verticalalignment='top')

    # 2d
    ax4 = fig.add_subplot(gs[1,1])
    visualize.pcs_loading_plot(df, num_pcs=1, group_pcs=False, atlas=atlas, ax=ax4)
    ax4.tick_params(axis='x', which='major', labelsize=15)
    ax4.tick_params(axis='y', which='major', labelsize=20)
    ax4.text(x_pos, y_pos, 'D', transform=ax4.transAxes, fontsize=40,
    verticalalignment='top')

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)

    fig.show()
    
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "fig_2"), bbox_inches="tight", dpi=300)

def fig_3(atlas1='MDTB-10-subRegions', atlas2='SUIT-10', which_genes='top', percentile=1, remove_outliers=True, **kwargs):
    """This function plots figure 3. 
    Panel A: MDTB parcellation + labels. Flatmap and volume.
    Panel B: Dendrogram. 
    Panel C: Parcellation labelled according to dendrogram. 

    TO-DO: 
    - Make parcellation labelled according to dendrogram
    - MAYBE MOVE MDTB PARCELLATION TO SUPP.

    Methods section: Anatomical genes were used to cluster the functional networks. Outliers removed
    because certain subregions didn't have any samples.

    Supp: 
    - Individual donor results for dendrogram and/or parcellation. 
    - Sample roi count and donor count for each functional region (counts will be different eventhough genes are the same). 
    Individual count for roi and donor instead of group?

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
                atlas_other (str): returns thresholded data using genes from another atlas
                donor_num (int): any one of the 6 donors 
                reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """
    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig)

    x_pos = -0.18
    y_pos = 1.02

    # 2a
    ax1 = fig.add_subplot(gs[0, 0])
    visualize.png_plot(atlas=atlas1, ax=ax1)
    ax1.axis('off')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=40,
    verticalalignment='top')

    # # 2b
    # ax2 = fig.add_subplot(gs[0, 1])
    # visualize.sample_counts_roi(df, ax=ax2, atlas=atlas1)
    # ax2.tick_params(axis='x', which='major', labelsize=13)
    # ax2.tick_params(axis='y', which='major', labelsize=20)

    # 2c
    df = ana.return_grouped_data(atlas=atlas1, which_genes=which_genes, atlas_other=atlas2, percentile=percentile, **kwargs)
    ax2 = fig.add_subplot(gs[0, 1])
    visualize.dendrogram_plot(df.T, ax=ax2, color_leaves=True)
    ax2.tick_params(axis='x', which='major', labelsize=12)
    ax2.tick_params(axis='y', which='major', labelsize=20)
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=40,
    verticalalignment='top')

    # 2c
    # ax3 = fig.add_subplot(gs[1, :1])
    # ADD IN TRANSCRIPTOMIC PARCELLATION

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)

    fig.show()
    
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "fig_3"), bbox_inches="tight", dpi=300)

def fig_4(atlas_cerebellum='Buckner-7', atlas_cortex='Yeo-7', atlas_other='SUIT-10',which_genes='top', percentile=1, remove_outliers=True, **kwargs):
    """This function plots figure 4. 
    Panel A: Parcellations - Yeo-7 and Buckner-7.
    Panel B: Correlation matrix for Buckner and Yeo regions. 

    Methods section: Anatomical genes were used to cluster the functional networks. Outliers removed
    because certain networks didn't have any samples.

    Supp. figures: 
    - Sample roi and donor count for Yeo-7 and Buckner-7 networks. Individual subject results? 

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
            atlas_other (str): returns thresholded data using genes from another atlas
            donor_num (int): any one of the 6 donors 
            reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """

    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    df = ana.return_concatenated_data(atlas_cerebellum=atlas_cerebellum, atlas_cortex=atlas_cortex, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, **kwargs)

    fig = plt.figure()

    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.18
    y_pos = 1.02

    # a
    ax1 = fig.add_subplot(gs[0, 0])
    visualize.png_plot(atlas=f"{atlas_cortex}", ax=ax1)
    ax1.axis('off')
    ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=40,
    verticalalignment='top')

    # e
    ax2 = fig.add_subplot(gs[0, 1])
    visualize.simple_corr_heatmap(df, ax=ax2)
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=40,
    verticalalignment='top')

    plt.subplots_adjust(left=0.02, bottom=0.001, right=2.0, top=1.0, wspace=.2, hspace=.3)

    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "fig_4"), bbox_inches="tight", dpi=300)

def supp_1(atlas='SUIT-10', which_genes='top', percentile=1, remove_outliers=True, **kwargs):
    """This function plots supp 1. 
    Panel A. Individual sample count per donor per roi.

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
            atlas_other (str): returns thresholded data using genes from another atlas
            donor_num (int): any one of the 6 donors 
            reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """

    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    fig = plt.figure()

    # gs = GridSpec(1, 1, figure=fig)

    # ax1 = fig.add_subplot(gs[0, 0])
    # visualize.sample_counts_roi_donor(df, ax=ax1)
    # ax1.tick_params(axis='x', which='major', labelsize=13)
    # ax1.legend(fontsize=12)

    gs = GridSpec(2, 3, figure=fig)

    for i in np.arange(0,6):
        df = ana.return_thresholded_data(atlas=atlas, which_genes=which_genes, percentile=percentile, donor_num=i+1, **kwargs)

        if i < 3:
            r = 0
            c = i
        else:
            r = 1
            c = i - 3

        ax1 = fig.add_subplot(gs[r, c])
        visualize.sample_counts_roi(df, ax=ax1, atlas=atlas)
        ax1.tick_params(axis='x', which='major', labelsize=10)
        ax1.tick_params(axis='y', which='major', labelsize=20)
        ax1.set_title(str(df["donor_id"].unique()[0]))

        plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)

        plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_1"), bbox_inches="tight", dpi=300)

def supp_2(atlas='SUIT-10', which_genes='top', percentile=1, remove_outliers=True, **kwargs): 
    """This function plots supp 2. 
    Panel A. Individual dendrogram and pc loading plots and sample roi counts.

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
            atlas_other (str): returns thresholded data using genes from another atlas
            donor_num (int): any one of the 6 donors 
            reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """

    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    fig = plt.figure()

    gs = GridSpec(6, 3, figure=fig)
    x_pos = -0.18
    y_pos = 1.1

    for i in np.arange(0,6):
        df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, donor_num=i+1, **kwargs)

        ax1 = fig.add_subplot(gs[i, 0])
        if i==0:
            ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=40,
            verticalalignment='top')
        visualize.pcs_loading_plot(df, num_pcs=0, group_pcs=False, ax=ax1, atlas='SUIT-10')
        ax1.tick_params(axis='x', which='major', labelsize=10)
        ax1.set_title(str(Defaults.donors[i]))

        ax2 = fig.add_subplot(gs[i, 1])
        if i==0:
            ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=40,
            verticalalignment='top')
        visualize.pcs_loading_plot(df, num_pcs=1, group_pcs=False, ax=ax2, atlas=atlas)
        ax2.tick_params(axis='x', which='major', labelsize=10)

        ax3 = fig.add_subplot(gs[i, 2])
        if i==0:
            ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=40,
            verticalalignment='top')
        visualize.dendrogram_plot(df.T, ax=ax3)
        ax2.tick_params(axis='x', which='major', labelsize=10)

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)

    fig.show()

    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_2"), bbox_inches="tight", dpi=300)

def supp_3(atlas="MDTB-10-subRegions", which_genes='top', percentile=1, remove_outliers=True, **kwargs):
    """This function plots supp 3. 
    Panel A. ROI sample counts + individual dendrogram.

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
            atlas_other (str): returns thresholded data using genes from another atlas
            donor_num (int): any one of the 6 donors 
            reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """

    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    fig = plt.figure()

    gs = GridSpec(6, 2, figure=fig)
    x_pos = -0.18
    y_pos = 1.1

    for i in np.arange(0,6):

        ax1 = fig.add_subplot(gs[i, 0])
        if i==0:
            ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=40,
            verticalalignment='top')
        df_thresh = ana.return_thresholded_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other="SUIT-10", donor_num=i+1, **kwargs)
        visualize.sample_counts_roi(df_thresh, ax=ax1, atlas=atlas)
        ax1.tick_params(axis='x', which='major', labelsize=13)
        ax1.tick_params(axis='y', which='major', labelsize=20)
        ax1.set_title(str(df_thresh["donor_id"].unique()[0]))

        ax2 = fig.add_subplot(gs[i, 1])
        if i==0:
            ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=40,
            verticalalignment='top')
        df_group = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, atlas_other="SUIT-10", percentile=percentile, donor_num=i+1, **kwargs)
        visualize.dendrogram_plot(df_group.T, ax=ax2)
        ax2.tick_params(axis='x', which='major', labelsize=13)

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)

    fig.show()
    
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_3"), bbox_inches="tight", dpi=300)

def supp_4(atlas_cerebellum="Buckner-7", atlas_cortex="Yeo-7", which_genes='top', percentile=1, remove_outliers=True, **kwargs):
    """This function plots supp 5. 
    Panel A. ROI sample counts for cerebellum and cortex and individual corr matrix.

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
            atlas_other (str): returns thresholded data using genes from another atlas
            donor_num (int): any one of the 6 donors 
            reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """

    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    fig = plt.figure()

    gs = GridSpec(6, 2, figure=fig)

    for i in np.arange(0,6):
        
        ax1 = fig.add_subplot(gs[i, 0])
        df_thresh = ana.return_thresholded_data(atlas=atlas_cerebellum, which_genes=which_genes, percentile=percentile, atlas_other="SUIT-10", donor_num=i+1, **kwargs)
        visualize.sample_counts_roi(df_thresh, ax=ax1, atlas=atlas_cerebellum)
        ax1.tick_params(axis='x', which='major', labelsize=13)
        ax1.tick_params(axis='y', which='major', labelsize=20)

        ax2 = fig.add_subplot(gs[i, 1]) 
        df_thresh = ana.return_thresholded_data(atlas=atlas_cortex, which_genes=which_genes, percentile=percentile, atlas_other="SUIT-10", donor_num=i+1, **kwargs)
        visualize.sample_counts_roi(df_thresh, ax=ax2, atlas=atlas_cortex)
        ax2.tick_params(axis='x', which='major', labelsize=13)
        ax2.tick_params(axis='y', which='major', labelsize=20)

        # ax3 = fig.add_subplot(gs[i, 2])
        # df_concat = ana.return_concatenated_data(atlas_cerebellum=atlas_cerebellum, atlas_cortex=atlas_cortex, which_genes=which_genes, percentile=percentile, donor_num=i+1, **kwargs)
        # visualize.simple_corr_heatmap(df_concat, ax=ax3)

    plt.subplots_adjust(left=0.2, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)

    fig.show()
    
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_4"), bbox_inches="tight", dpi=300)

def supp_5(atlas='SUIT-26', which_genes='top', percentile=1, remove_outliers=True, **kwargs):
    """This function plots supp 6. 
    Panel A. Plot heatmap for SUIT-26.

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
            atlas_other (str): returns thresholded data using genes from another atlas
            donor_num (int): any one of the 6 donors 
            reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """

    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    fig = plt.figure()

    gs = GridSpec(1, 1, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    dataframe = ana.return_grouped_data(atlas=atlas)
    visualize.simple_corr_heatmap(dataframe, ax=ax1)

    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_5"), bbox_inches="tight", dpi=300)

def supp_additional(atlas='SUIT-10', which_genes='top', percentile=1, remove_outliers=True, extreme_removal=True, **kwargs):
    """This function plots supp 3 for top 1% of genes. Outliers (region X) removed. 
    Panel A: Correlation plot for anatomical regions. Shows distinct anatomical clustering of data.
    Panel B: Dendrogram.
    Panel C: PC1 Loading.
    Panel D: PC2 Loading. 

    Args:
        atlas (str): the name of the atlas to use
        which_genes (str): 'top' or 'bottom' % of genes to threshold
        percentile (int): any % of changes to threshold. Default is 1 
        remove_outliers (bool): certain atlases have outliers that should be removed (i.e. SUIT-10)
        kwargs (dict): dictionary of additional (optional) kwargs.
            may include any of the following:
                atlas_other (str): returns thresholded data using genes from another atlas
                donor_num (int): any one of the 6 donors 
                reorder_labels (bool): certain atlases have labels that need to be reordered for visual presentation (i.e. MDTB-10-subRegions)
    """
    if remove_outliers:
        kwargs["remove_outliers"] = remove_outliers

    if extreme_removal:
        kwargs["extreme_removal"] = extreme_removal

    df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, **kwargs)

    fig = plt.figure()

    gs = GridSpec(2, 2, figure=fig)

    # 2a
    ax1 = fig.add_subplot(gs[0, 0])
    visualize.simple_corr_heatmap(df, ax=ax1)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    # 2b
    ax2 = fig.add_subplot(gs[0, 1])
    visualize.dendrogram_plot(df.T, ax=ax2)
    ax2.tick_params(axis='x', which='major', labelsize=15)
    ax2.tick_params(axis='y', which='major', labelsize=20)
    plt.setp(ax2.lines, linewidth=10) 

    # 2c
    ax3 = fig.add_subplot(gs[1,0])
    visualize.pcs_loading_plot(df, num_pcs=0, group_pcs=False, atlas='SUIT-10', ax=ax3)
    ax3.tick_params(axis='x', which='major', labelsize=15)
    ax3.tick_params(axis='y', which='major', labelsize=20)

    # 2d
    ax4 = fig.add_subplot(gs[1,1])
    visualize.pcs_loading_plot(df, num_pcs=1, group_pcs=False, atlas='SUIT-10', ax=ax4)
    ax4.tick_params(axis='x', which='major', labelsize=15)
    ax4.tick_params(axis='y', which='major', labelsize=20)

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)

    fig.show()
    
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_3"), bbox_inches="tight")