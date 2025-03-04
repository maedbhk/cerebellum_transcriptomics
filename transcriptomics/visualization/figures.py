from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

import numpy as np
import os
import pandas as pd

from transcriptomics.visualization import visualize
from transcriptomics import gec_functions_ana as ana 
from transcriptomics.constants import Defaults

def plotting_style():
    # fig = plt.figure(num=2, figsize=[20,8])
    plt.style.use('seaborn-poster') # ggplot
    plt.rc('font', family='sans-serif') 
    plt.rc('font', serif='Helvetica Neue') 
    plt.rc('text', usetex='false') 
    plt.rcParams['lines.linewidth'] = 6
    plt.rc('xtick', labelsize=14)     
    plt.rc('ytick', labelsize=14)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["savefig.format"] = 'svg'
    plt.rc("axes.spines", top=False, right=False) # removes certain axes

    # sets the line weight for border of graph
    # for axis in ['top','bottom','left','right']:
    #     ax2.spines[axis].set_linewidth(4)

def fig_1(atlas='MDTB-10', which_genes='top', percentile=1, remove_outliers=True, normalize=True):
    """
    This function plots figure 1 of the paper
    Panel A: 
    Panel B: 
    Panel C:
    Panel D:
    """
    plt.clf()

    plotting_style()

    # fig = plt.figure()
    fig = plt.figure(figsize=(15,15))
    gs = GridSpec(3, 2, figure=fig)

    x_pos = -0.2
    y_pos = 1.02

    df = ana.return_thresholded_data(atlas=atlas, which_genes=which_genes, percentile=percentile, remove_outliers=remove_outliers, normalize=normalize)

    # 1a
    ax1 = fig.add_subplot(gs[0, 0])
    visualize.png_plot(filename=atlas, ax=ax1)
    ax1.axis('off')
    ax1.text(x_pos, 1.0, 'A', transform=ax1.transAxes, fontsize=60,
    verticalalignment='top')
    ax1.yaxis.label.set_size(50)

    # 1b
    ax2 = fig.add_subplot(gs[0, 1])
    visualize.png_plot(filename="atlas-samples-donors-colorbar", ax=ax2) # png_plot will append .png
    ax2.axis('off')
    ax2.text(-0.25, .87, 'B', transform=ax2.transAxes, fontsize=60,
        verticalalignment='top')
    ax2.yaxis.label.set_size(50)

    # 1c
    ax3 = fig.add_subplot(gs[1, 0])
    visualize.sample_counts_roi(df, ax=ax3, atlas=atlas)
    ax3.tick_params(axis='x', which='major', labelsize=40, rotation=45)
    ax3.tick_params(axis='y', which='major', labelsize=40)
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=60,
        verticalalignment='top')
    ax3.yaxis.label.set_size(50)

    # 1d
    ax4 = fig.add_subplot(gs[1, 1])
    visualize.diff_stability_plot(atlas=atlas, which_genes=which_genes, percentile=percentile, ax=ax4)
    ax4.tick_params(axis='x', which='major', labelsize=40)
    ax4.tick_params(axis='y', which='major', labelsize=40)
    ax4.text(x_pos, y_pos, 'D', transform=ax4.transAxes, fontsize=60,
        verticalalignment='top')
    ax4.yaxis.label.set_size(50)
    ax4.xaxis.label.set_size(50)

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "fig_1"), bbox_inches="tight", dpi=300)

def fig_2(which_genes='top', percentile=1, remove_outliers=True, reorder_labels=True, atlas_other="MDTB-10", normalize=True):
    plt.clf()
    
    plotting_style()
    
    fig = plt.figure(figsize=(20, 30))
    gs = GridSpec(3, 3, figure=fig)

    x_pos = -0.2
    y_pos = 1.02

    # 2a
    ax1 = fig.add_subplot(gs[0, 0])
    visualize.png_plot(filename='MDTB-10-subRegions', ax=ax1)
    ax1.axis('off')
    ax1.text(x_pos-0.02, y_pos, 'A', transform=ax1.transAxes, fontsize=60,
    verticalalignment='top')
    ax1.yaxis.label.set_size(30)

    # 2b
    ax2 = fig.add_subplot(gs[0, 1:3])
    df = ana.return_grouped_data(atlas='MDTB-10-subRegions', which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)
    df = df.set_index(df.index)
    visualize.dendrogram_plot(df.T, orientation='top', color_leaves=True, ax=ax2)
    ax2.tick_params(axis='x', which='major', labelsize=40, rotation=45)
    ax2.tick_params(axis='y', which='major', labelsize=40)
    # ax2.axis('off')
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=60,
    verticalalignment='top')
    ax2.yaxis.label.set_size(50)

    # 2c
    df = ana.return_grouped_data(atlas='MDTB-10-subRegions', which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, reorder_labels=reorder_labels, remove_outliers=remove_outliers, normalize=normalize)
    ax3 = fig.add_subplot(gs[1, 0])
    visualize.simple_corr_heatmap(df, atlas='MDTB-10-subRegions', distance_correct=True, ax=ax3)
    ax3.tick_params(axis='x', which='major', labelsize=15)
    ax3.tick_params(axis='y', which='major', labelsize=15)
    ax3.text(x_pos-.03, y_pos, 'C', transform=ax3.transAxes, fontsize=60,
    verticalalignment='top')
    ax3.yaxis.label.set_size(30)

    # 2d
    df = ana.return_concatenated_data(atlas_cerebellum="Buckner-7", atlas_cortex="Yeo-7", which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=remove_outliers, normalize=normalize)
    # df = ana.return_grouped_data(atlas="Yeo-Buckner-7", percentile=percentile, reorder_labels=reorder_labels, remove_outliers=remove_outliers, normalize=normalize)
    ax4 = fig.add_subplot(gs[1, 1])
    visualize.simple_corr_heatmap(df, atlas="Yeo-Buckner-7", distance_correct=True, simple_labels=True, ax=ax4)
    ax4.tick_params(axis='x', which='major', labelsize=20)
    ax4.tick_params(axis='y', which='major', labelsize=20)
    ax4.text(x_pos-.16, y_pos, 'D', transform=ax4.transAxes, fontsize=60,
    verticalalignment='top')
    ax4.yaxis.label.set_size(30)

    # 2e
    df = ana.return_concatenated_data(atlas_cerebellum="Buckner-17", atlas_cortex="Yeo-17", which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=remove_outliers, normalize=normalize)
    # df = ana.return_grouped_data(atlas="Yeo-Buckner-17", which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, reorder_labels=reorder_labels, remove_outliers=remove_outliers, normalize=normalize)
    ax5 = fig.add_subplot(gs[1, 2])
    visualize.simple_corr_heatmap(df, atlas="Yeo-Buckner-17", distance_correct=True, simple_labels=True, ax=ax5) # ax=ax5
    ax5.tick_params(axis='x', which='major', labelsize=20)
    ax5.tick_params(axis='y', which='major', labelsize=20)
    ax5.text(x_pos-.16, y_pos, 'E', transform=ax5.transAxes, fontsize=60,
    verticalalignment='top')
    ax5.yaxis.label.set_size(30)

    plt.subplots_adjust(left=0.02, bottom=0.001, right=2.0, top=1.0, wspace=.2, hspace=.3)

    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "fig_2"), bbox_inches="tight", dpi=300)

def fig_3(atlas='SUIT-10', which_genes='top', percentile=1, remove_outliers=True, atlas_other="MDTB-10", normalize=True):
    # plt.clf()
    
    plotting_style()

    x_pos = -0.2
    y_pos = 1.02
    
    #3a
    fig = plt.figure(figsize=(14, 15))
    gs = GridSpec(7, 8, figure=fig)

    ax1 = fig.add_subplot(gs[:, :2])
    visualize.png_plot(filename='SUIT-10_v2', ax=ax1)
    ax1.axis('off')
    ax1.text(x_pos-0.02, y_pos, 'A', transform=ax1.transAxes, fontsize=40,
    verticalalignment='top')
    ax1.yaxis.label.set_size(30)

    # 3bi
    ax1 = fig.add_subplot(gs[:, 2])
    df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)
    df = df.set_index(df.index)
    plt.rcParams['lines.linewidth'] = 2
    visualize.dendrogram_plot(df, orientation='left', color_leaves=False, ax=ax1)
    ax1.tick_params(axis='x', which='major', labelbottom=False, bottom=False, top=False)
    ax1.axis('off')
    ax1.text(x_pos, y_pos, 'B', transform=ax1.transAxes, fontsize=40,
    verticalalignment='top')
    ax1.yaxis.label.set_size(30)
    plt.rcParams['lines.linewidth'] = 6

    # 3bii
    ax2 = fig.add_subplot(gs[:, 3])
    df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)
    visualize.raster_plot(df,  ax=ax2)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    # ax2.text(x_pos, y_pos, 'C', transform=ax2.transAxes, fontsize=40,
    # verticalalignment='top')
    ax2.yaxis.label.set_size(30)

    # 3c
    ax3 = fig.add_subplot(gs[0:2, 3:7])
    visualize.simple_corr_heatmap(df, atlas=atlas, distance_correct=True, ax=ax3)
    ax3.tick_params(axis='both', which='major', labelsize=20)
    ax3.text(-0.4, 1.06, 'C', transform=ax3.transAxes, fontsize=40,
    verticalalignment='top')
    ax3.yaxis.label.set_size(30)

    # 3d
    ax4 = fig.add_subplot(gs[3:5, 5:7])
    visualize.dendrogram_plot(df.T, ax=ax4, color_leaves=False)
    ax4.tick_params(axis='x', which='major', labelsize=15, rotation=45)
    ax4.tick_params(axis='y', which='major', labelsize=25)
    plt.setp(ax4.lines, linewidth=10) # THIS ISN'T WORKING
    ax4.text(-0.28, 1.09, 'D', transform=ax4.transAxes, fontsize=40,
    verticalalignment='top')
    ax4.yaxis.label.set_size(20)

    # 3e
    ax5 = fig.add_subplot(gs[5:7,5:7])
    visualize.pcs_loading_plot(df, pcs=[1], group_pcs=False, atlas=atlas, ax=ax5)
    ax5.tick_params(axis='x', which='major', labelsize=15, rotation=45)
    ax5.tick_params(axis='y', which='major', labelsize=25)
    ax5.text(-0.28, 1.04, 'E', transform=ax5.transAxes, fontsize=40,
    verticalalignment='top')
    ax5.yaxis.label.set_size(20)

    plt.subplots_adjust(left=0.02, bottom=0.001, right=2.0, top=1.0, wspace=.2, hspace=.3)

    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "fig_3"), bbox_inches="tight", dpi=300)

def fig_4(image1="rat_zones", image2="MDTB-10-subRegions-transcriptomic", atlas='MDTB-10-subRegions', which_genes='top', percentile=1, remove_outliers=True, atlas_other="MDTB-10", normalize=True):
    plt.clf()
    
    plotting_style()

    # fig = plt.figure(figsize=(15,15))
    fig = plt.figure()
    gs = GridSpec(1, 2, figure=fig)

    x_pos = -0.2
    y_pos = 1.02 

    # # get colors for transcriptomic atlas
    # ana.save_colors_transcriptomic_atlas()
    # # make colorbar for atlas
    # visualize._make_colorbar(atlas="MDTB-10-subRegions-transcriptomic-dendrogram-ordering")
    # print('make the atlas in SUIT (matlab)')

    # 4a
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(x_pos, 1.0, 'A', transform=ax1.transAxes, fontsize=40,
    verticalalignment='top')
    visualize.png_plot(filename=image1, ax=ax1)
    ax1.axis('off')

    # 4b
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(x_pos, 1.077, 'B', transform=ax2.transAxes, fontsize=40,
    verticalalignment='top')
    visualize.png_plot(filename=image2, ax=ax2)
    ax2.axis('off')

    plt.subplots_adjust(left=0.02, bottom=0.001, right=2.0, top=1.0, wspace=.2, hspace=.3)

    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "fig_4"), bbox_inches="tight", dpi=300)

def supp_1(which_genes='top', percentile=1, remove_outliers=True, atlas_other="MDTB-10", normalize=True):
    plt.clf()

    atlas_anat = "SUIT-10"
    atlas_cerebellum_task = "MDTB-10-subRegions"
    atlas_cerebellum_rs_7 = "Buckner-7"
    atlas_cortex_rs_7 = "Yeo-7"
    atlas_cerebellum_rs_17 = "Buckner-17"
    atlas_cortex_rs_17 = "Yeo-17"
    
    plotting_style()
    
    # fig = plt.figure()
    fig = plt.figure(figsize=(35,20))
    gs = GridSpec(3, 3, figure=fig)

    x_pos = -0.2
    y_pos = 1.02

    # 1a
    ax1 = fig.add_subplot(gs[0, 0])
    visualize.png_plot(filename="Yeo-Buckner-7-v1", ax=ax1)
    ax1.axis('off')
    ax1.text(x_pos+.15, y_pos+.4, 'A', transform=ax1.transAxes, fontsize=100,
    verticalalignment='top')
    ax1.yaxis.label.set_size(50)

    # 1b
    df = ana.return_thresholded_data(atlas=atlas_cortex_rs_7, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=remove_outliers, normalize=normalize)
    ax2 = fig.add_subplot(gs[0, 1])
    visualize.sample_counts_roi(df, ax=ax2, atlas=atlas_cortex_rs_7)
    ax2.tick_params(axis='x', which='major', labelsize=50, rotation=45)
    ax2.tick_params(axis='y', which='major', labelsize=50)
    ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=100,
        verticalalignment='top')
    ax2.yaxis.label.set_size(50)

    # 1c
    df = ana.return_thresholded_data(atlas=atlas_cerebellum_rs_7, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=remove_outliers, normalize=normalize)
    ax3 = fig.add_subplot(gs[0, 2])
    visualize.sample_counts_roi(df, ax=ax3, atlas=atlas_cerebellum_rs_7)
    ax3.tick_params(axis='x', which='major', labelsize=50, rotation=45)
    ax3.tick_params(axis='y', which='major', labelsize=50)
    ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=100,
        verticalalignment='top')
    ax3.yaxis.label.set_size(50)

    # 1d
    ax4 = fig.add_subplot(gs[1, 0])
    visualize.png_plot(filename="Yeo-Buckner-17-v1", ax=ax4)
    ax4.axis('off')
    ax4.text(x_pos+.15, y_pos+.4, 'D', transform=ax4.transAxes, fontsize=100,
    verticalalignment='top')
    ax4.yaxis.label.set_size(50)

    # 1e
    df = ana.return_thresholded_data(atlas=atlas_cortex_rs_17, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=remove_outliers, normalize=normalize)
    ax5 = fig.add_subplot(gs[1, 1])
    visualize.sample_counts_roi(df, ax=ax5, atlas=atlas_cortex_rs_17)
    ax5.tick_params(axis='x', which='major', labelsize=50, rotation=45)
    ax5.tick_params(axis='y', which='major', labelsize=50)
    ax5.text(x_pos, y_pos, 'E', transform=ax5.transAxes, fontsize=100,
        verticalalignment='top')
    ax5.yaxis.label.set_size(50)

    # 1f
    df = ana.return_thresholded_data(atlas=atlas_cerebellum_rs_17, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=remove_outliers, normalize=normalize)
    ax6 = fig.add_subplot(gs[1, 2])
    visualize.sample_counts_roi(df, ax=ax6, atlas=atlas_cerebellum_rs_17)
    ax6.tick_params(axis='x', which='major', labelsize=50, rotation=45)
    ax6.tick_params(axis='y', which='major', labelsize=50)
    ax6.text(x_pos, y_pos, 'F', transform=ax6.transAxes, fontsize=100,
        verticalalignment='top')
    ax6.yaxis.label.set_size(50)

    df = ana.return_thresholded_data(atlas=atlas_cerebellum_task, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=remove_outliers, normalize=normalize)
    ax7 = fig.add_subplot(gs[2, 0:2])
    visualize.sample_counts_roi(df, ax=ax7, atlas=atlas_cerebellum_task)
    ax7.tick_params(axis='x', which='major', labelsize=50, rotation=45)
    ax7.tick_params(axis='y', which='major', labelsize=50)
    ax7.text(x_pos, y_pos, 'G', transform=ax7.transAxes, fontsize=100,
        verticalalignment='top')
    ax7.yaxis.label.set_size(50)

    df = ana.return_thresholded_data(atlas=atlas_anat, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=remove_outliers, normalize=normalize)
    ax8 = fig.add_subplot(gs[2, 2:3])
    visualize.sample_counts_roi(df, ax=ax8, atlas=atlas_anat)
    ax8.tick_params(axis='x', which='major', labelsize=50, rotation=45)
    ax8.tick_params(axis='y', which='major', labelsize=50)
    ax8.text(x_pos, y_pos, 'H', transform=ax8.transAxes, fontsize=100,
        verticalalignment='top')
    ax8.yaxis.label.set_size(50)

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.2, hspace=.3)
    
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_1"), bbox_inches="tight", dpi=300)

def supp_2(atlas1="MDTB-10-subRegions", atlas2="SUIT-10", which_genes='top', atlas_other="MDTB-10", percentile=1, classifier='logistic', remove_outliers=True, normalize=True):
    plt.clf()
    
    plotting_style()

    x_pos = -0.18
    y_pos = 1.1
    
    # return all samples dataframe
    df = ana.return_thresholded_data(atlas=atlas1, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=False, normalize=normalize, all_samples=True)
    plt.clf()
    plt.figure()
    visualize.confusion_matrix_plot(atlas1, df, classifier=classifier, label_type="binary") # ax=ax1
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_4a"), bbox_inches="tight", dpi=300)
    plt.text(0, 0, 'A', fontsize=40, verticalalignment='top')
    plt.clf()

    # return all samples dataframe
    df = ana.return_thresholded_data(atlas=atlas2, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, remove_outliers=False, normalize=normalize, all_samples=True)
    plt.figure()
    visualize.confusion_matrix_plot(atlas2, df, classifier=classifier, label_type="multi-class") # ax=ax1
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_4b"), bbox_inches="tight", dpi=300)
    plt.text(0, 0, 'B', fontsize=40, verticalalignment='top')
    plt.clf()

    # plt.figure()
    # visualize.precision_recall_plot(atlas, df, classifier=classifier, label_type=label_type) #ax=ax2
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_2"), bbox_inches="tight", dpi=300)

def supp_3(atlas="SUIT-10", which_genes='top', percentile=1, remove_outliers=True, atlas_other="MDTB-10", normalize=True):
    plt.clf()
    
    plotting_style()
    
    fig = plt.figure(figsize=(15,15))

    gs = GridSpec(6, 3, figure=fig)
    x_pos = -0.18
    y_pos = 1.1

    for i in np.arange(0,6):
        df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, donor_num=i+1, remove_outliers=remove_outliers, normalize=normalize)

        ax1 = fig.add_subplot(gs[i, 0])
        if i==0:
            ax1.text(x_pos, y_pos, 'A', transform=ax1.transAxes, fontsize=50,
            verticalalignment='top')
        visualize.pcs_loading_plot(df, pcs=[1], group_pcs=False, ax=ax1, atlas=atlas)
        ax1.tick_params(axis='x', which='major', labelsize=30, rotation=35)
        ax1.tick_params(axis='y', which='major', labelsize=30)
        
        ax1.set_title(str(Defaults.donors[i]), fontsize=40)
        ax1.yaxis.label.set_size(40)

        ax2 = fig.add_subplot(gs[i, 1])
        if i==0:
            ax2.text(x_pos, y_pos, 'B', transform=ax2.transAxes, fontsize=50,
            verticalalignment='top')
        visualize.pcs_loading_plot(df, pcs=[2], group_pcs=False, ax=ax2, atlas=atlas)
        ax2.tick_params(axis='x', which='major', labelsize=30, rotation=35)
        ax2.tick_params(axis='y', which='major', labelsize=30)
        ax2.yaxis.label.set_size(40)

        ax3 = fig.add_subplot(gs[i, 2])
        if i==0:
            ax3.text(x_pos, y_pos, 'C', transform=ax3.transAxes, fontsize=50,
            verticalalignment='top')
        visualize.dendrogram_plot(df.T, ax=ax3, color_leaves=False)
        ax3.tick_params(axis='x', which='major', labelsize=30, rotation=35)
        ax3.tick_params(axis='y', which='major', labelsize=30)
        ax3.yaxis.label.set_size(40)

    plt.subplots_adjust(left=0.125, bottom=0.001, right=2.0, top=2.0, wspace=.5, hspace=1.0)
    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_3"), bbox_inches="tight", dpi=300)

def supp_4(atlas="SUIT-26", which_genes='top', percentile=1, remove_outliers=True, atlas_other="MDTB-10", normalize=True):
    
    plt.clf()

    plotting_style()
    
    fig = plt.figure()

    gs = GridSpec(1, 1, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    dataframe = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)
    visualize.simple_corr_heatmap(dataframe, atlas=atlas, distance_correct=True, ax=ax1)

    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_4"), bbox_inches="tight", dpi=300)

def supp_5(atlas1='SUIT-10', atlas2='MDTB-10-subRegions', which_genes='top', atlas_other="MDTB-10", percentile=1, model_type='linear', label_type='multi-class', normalize=True):
    plt.clf()

    plotting_style()
    
    # return all samples dataframe
    df = ana.return_thresholded_data(atlas1, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, normalize=normalize, all_samples=True)

    visualize.test_train_error_plot(atlas1, df)

    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_5a"), bbox_inches="tight", dpi=300)

    plt.clf()
    # return all samples dataframe
    df = ana.return_thresholded_data(atlas2, which_genes=which_genes, atlas_other=atlas_other, percentile=percentile, normalize=normalize, all_samples=True)

    visualize.test_train_error_plot(atlas2, df)

    plt.savefig(str(Defaults.PROCESSED_DIR / "figures" / "supp_5"), bbox_inches="tight", dpi=300)

def supp_6(atlas='SUIT-10', which_genes='top', atlas_other="MDTB-10", percentile=1, remove_outliers=True, normalize=True):
    """ save gene table
    """
    plt.clf()

    df = ana.return_grouped_data(atlas=atlas, which_genes=which_genes, percentile=percentile, atlas_other=atlas_other, remove_outliers=remove_outliers, normalize=normalize)

    pcs_labelled = visualize.pcs_table(df, num_pcs=2)

    # sort genes on pc0
    cols_to_keep = [col for col in pcs_labelled.columns if 'jittered' not in col]
    pcs_labelled = pcs_labelled.sort_values(by='pc0', ascending=False) 

    FIG_DIR = os.path.join(Defaults.PROCESSED_DIR, 'figures')
    if percentile==100:
        try: 
            df = pd.read_csv(os.path.join(FIG_DIR, f'table_1_{atlas}-{which_genes}-1.csv'))
            pcs_labelled = pcs_labelled[~pcs_labelled['genes'].isin(df['genes'])]
        except: 
            print(f'table_1_{atlas}-{which_genes}-1.csv does not exist, create')

    pcs_labelled[cols_to_keep].to_csv(os.path.join(FIG_DIR, f'table_1_{atlas}-{which_genes}-{percentile}.csv'))