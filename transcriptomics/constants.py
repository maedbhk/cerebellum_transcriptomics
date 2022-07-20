from pathlib import Path
import os
import numpy as np
import matplotlib.colors as mc
import colorsys

class Defaults:

    # set base directories
    BASE_DIR = Path(__file__).absolute().parent.parent # Path(__file__).absolute().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    INTERIM_DIR = DATA_DIR / "interim"
    PROCESSED_DIR = DATA_DIR / "processed"
    # SURF_DIR =  os.path.abspath(os.path.join(BASE_DIR, '..', 'fs_LR_32'))
    ATLAS_DIR = DATA_DIR / "atlases"

    # list of donor ids
    donors = ['donor9861', 'donor10021', 'donor12876', 'donor14380', 'donor15496', 'donor15697']
    donor_colors = [[1,1,0], [0,0,1], [1,0,1], [0,1,0], [0,1,1], [1,0,0]]
    
    parcellations = ['SUIT10', 'SUIT26', 'Buckner7', 'Buckner17', 'MDTB10', 'MDTB20', 'Yeo7', 'Yeo17']

