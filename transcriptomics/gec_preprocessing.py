
# coding: utf-8

# ## TO-DO List
# * Incorporate colour information into the csv info file

# # Preprocessing script for AHBA data
# 
# In 2013, the Allen Institute for Brain Science released the Allen Human Brain Atlas, a dataset containing microarray expression data collected from six human brains. This dataset has offered an unprecedented opportunity to examine the genetic underpinnings of the human brain.
# 
# However, in order to be effectively used in most analyses, the AHBA microarray expression data often needs to be (1) collapsed into regions of interest (e.g., parcels or networks), and (2) combined across donors.
# 
# The Python package, abagen, was used to download and preprocess the data. Abagen is a reproducible pipeline for processing and preparing the AHBA microarray expression data for analysis.
# 
# Checkout the abagen toolbox here: https://abagen.readthedocs.io/en/stable/index.html
# 
# Warning - there are some bugs in the abagen toolbox that I have corrected. I will highlight them in the relevant steps.
# 
# There are a few different ways to install abagen. <br />
# For this project, I used the following: 
# - git clone https://github.com/rmarkello/abagen.git
# - cd abagen
# - pip install .[io]

# # Import Libraries and set base directories
# First step is to import necessary libraries and set base directories.
# Flexibly sets base directory to your parent directory. <br />
# - Raw data are stored in data/raw.<br />
# - Intermediate data are stored in data/intermediate.<br />
# - Processed data are stored in data/processed.

# In[2]:


# import extensions
# warnings: suppresses any warning messages
# (auto)reload functions allows for changes in any functions you write and 
# automatically reloads them into the notebook to reflect changes (methods etc)
# import warnings
# warnings.filterwarnings("ignore")

# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[107]:


# import libraries
# all of the libraries have been installed in a virutal environment
# you can find all of the requirements in the requirements.txt file in the repo
from pathlib import Path
from pkg_resources import resource_filename
import os
import re 

import abagen # library for preprocessing AHBA data
from sklearn.utils import Bunch
import pandas as pd
import numpy as np

from transcriptomics.constants import * # set base directories
from transcriptomics import gec_functions_preprocess as preprocess


# # Fetch the AHBA data
# Downloads all 6 subjects from AHBA website but we first check if they have been downloaded. 
# "allenbrain" is the name of the folder that is downloaded.
# 'fetch_microarray' gets the file names for all the data. 

# FETCH
# download all 6 subjects from ahba
# check first if the data have already been downloaded
# exists = os.path.isfile(DATA_DIR / "allenbrain")
# if exists:
#     data_files = abagen.fetch_microarray(data_dir=str(DATA_DIR / "allenbrain"), donors='all')
# else:
#     data_files = abagen.fetch_microarray(data_dir=str(RAW_DATA_DIR), donors='all')

data_files = preprocess.get_all_files()


# # Define a parcellation
# In order to process the microarray expression data, you’ll need a parcellation. A parcellation is an atlas image that contains regions or parcels denoted by unique integer IDs. You can use whatever atlas you’d like as long as it is a volumetric atlas in MNI space.
# 
# The desikan_killiany parcellation is included in the abagen package. 
# 
# The returned object 'atlas' is a dictionary with two keys: <br />
# - Image: filepath to the Nifti containing the atlas data <br />
# - Info: filepath to a CSV file containing auxilliary information about the parcellation.
# 
# I have created a folder in data/external/atlas_templates containing cerebellar parcellations in MNI space along with csv files containing auxilliary information about the parcellation. 
# Here are the parcellation options for the cerebellum that you can call: <br />
# - SUIT-10
# - Bucker-7
# - Buckner-17
# - Ji-10
# - MDTB-10
# 
# Here is the parcellation option for the cerebral cortex that you can call: <br />
# - Desikan-Killiany

# MAKE INFO CSV FOR CEREBELLAR PARCELLATIONS
# THESE FILES HAVE ALREADY BEEN CREATED SO NO NEED TO RUN THIS CELL

# parcellations = ['SUIT-10', 'Buckner-7', 'Buckner-17', 'Ji-10', 'MDTB-10']

# for atlas_name in parcellations:

# # get roi labels for an atlas name
# labels_roi = preprocess.get_roi_labels(atlas_name)

# # get num of rois
# n_roi = len(labels_roi)

# # intialise dict
# info = {'id':list(range(1,n_roi+1)), 'label': labels_roi, 'hemisphere': ['none']*n_roi, 'structure':['cerebellum']*n_roi}

# # create dataframe
# df = pd.DataFrame(info) 
# # print(df.head())

# # get output dir
# output_dir = EXTERNAL_DIR / "atlas_templates" / "{}-info.csv".format(atlas_name)

# # write to csv file
# df.to_csv(output_dir, index = None, header=True)

# LOAD DATA
# microarray = abagen.io.read_microarray(data_files.microarray[0], parquet=True)
# annot = abagen.io.read_microarray(data_files.annotation[0], parquet=True)
# ontology = abagen.io.read_microarray(data_files.ontology[0], parquet=True)
# pacall = abagen.io.read_microarray(data_files.pacall[0], parquet=True)
# probes = abagen.io.read_microarray(data_files.probes[0], parquet=True)


# # Process data
# The expression object returned is a pandas dataframe, where rows correspond to region labels as defined in our atlas image, columns correspond to genes, and entry values are normalized microarray expression data averaged across donors. 
# 
# Unfortunately, due to how tissue samples were collected from the donor brains, it is possible that some regions in an atlas may not be represented by any expression data. This is likely due to the fact that not all six donors had any tissue samples taken from a particular ROI.
# 
# If you require a full matrix with expression data for every region, you can specify the following: <br />
# Expression = abagen.get_expression_data(files, atlas.image, atlas.info, exact=False). 
# - See documentation for more details on the parameters: https://abagen.readthedocs.io/en/stable/api.html or query abagen.get_expression_data?
# 
# Parameters I chose: 
# - return_counts = True
# - exact = False
# - tolerance = 3
# - metric = mean
# - ibf_threshold = 0.5
# - corrected_mni = True
# 
# Bugs in the abagen code: 
# - It seems that the abagen code only tested one atlas ('Desikan-Killiany') and as such, there were some bugs when running other (i.e. cerebellar) atlases.
# - The bugs were in allen.py and process.py. 
# - The abagen toolbox is not being maintained (i.e. pull requests cannot be submitted). Therefore, it is unlikely that these bugs will be fixed in the near future. Therefore, as a fix, please replace allen.py and process.py with my corrected code (same names). 
# - In the code, I have highlighted the fixes if you are curious. 

# PROCESS GROUP AND INDIVIDUAL DATA 
parcellations = ['Desikan-Killiany']

for atlas_name in parcellations:

    # get info and nii filenames for each atlas
    atlas_files = preprocess.get_atlas_files(atlas_name)
    print(pd.read_csv(atlas_files.info))

    # get expression data for the individual
    expression, counts = abagen.get_expression_data(data_files, str(atlas_files.image), str(atlas_files.info), return_counts=True, aggregate=None)
 
    # aggregate data
    group_expression = expression.groupby("label").aggregate("mean")
    
    # write out expression data to file for group
    output_dir = PROCESSED_DIR / "expression-group-{}.csv".format(atlas_name)
    group_expression.to_csv(output_dir, index=None, header=True)

    # write out expression data to file for all donors
    output_dir = PROCESSED_DIR / "expression-alldonors-{}.csv".format(atlas_name)
    expression.to_csv(output_dir, index=None, header=True)
