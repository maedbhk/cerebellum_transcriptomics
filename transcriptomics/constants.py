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
    EXTERNAL_DIR = DATA_DIR / "external"
    INTERIM_DIR = DATA_DIR / "interim"
    PROCESSED_DIR = DATA_DIR / "processed"
    SURF_DIR =  os.path.abspath(os.path.join(BASE_DIR, '..', 'fs_LR_32'))

    # list of donor ids
    donors = ['donor9861', 'donor10021', 'donor12876', 'donor14380', 'donor15496', 'donor15697']
    donor_colors = [[1,1,0], [0,0,1], [1,0,1], [0,1,0], [0,1,1], [1,0,0]]
    
    parcellations = ['SUIT-10', 'Buckner-7', 'Buckner-17', 'MDTB-10', 'Yeo-7', 'Yeo-17', 'MDTB-10-subRegions', 'Ji-10', 'Desikan-Killiany-83', 'SUIT-26']
                    # 'NNMF_dim2_group1', 'NNMF_dim2_group2','NNMF_dim3_group1', 'NNMF_dim3_group2','NNMF_dim4_group1', 
                    # 'NNMF_dim4_group2','NNMF_dim4_group1','NNMF_dim4_group2','NNMF_dim5_group1', 
                    # 'NNMF_dim5_group2','NNMF_dim6_group1','NNMF_dim6_group2','NNMF_dim7_group1',
                    # 'NNMF_dim7_group2','NNMF_dim8_group1','NNMF_dim8_group2','NNMF_dim9_group1',
                    # 'NNMF_dim9_group2','NNMF_dim10_n26381', 'Desikan-Killiany-83', 'Retinotopy-10']

    # rgb colours for each 
    colour_info = dict([('SUIT-10', [[204, 255, 0], [0, 230, 92], [0, 102, 255],[204, 0, 255],[255, 0, 0], [255, 153, 0],[51, 255, 0], [0, 255, 255], [51, 0, 255], [255, 0, 153]]),
                    ('MDTB-10', [[46, 166, 152], [85, 151, 32], [51, 102, 147], [15, 25, 126], [165, 24, 162], [175, 44, 71], [225, 126, 176], [236, 161, 8], [252, 218, 118], [119, 118, 246]]),
                    ('Buckner-7', [[120, 18, 134], [70, 130, 180], [0, 118, 14], [196, 58, 250], [220, 248, 164], [230, 148, 34], [205, 62, 78]]),
                    ('Buckner-17', [[120, 18, 134], [255, 0, 0], [70, 130, 180], [42, 204, 164], [74, 155, 60], [0, 118, 14], [196, 58, 250], [255, 152, 213], [220, 248, 164], [122, 135, 50], [119, 140, 176], [230, 148, 34], [135, 50, 74], [12, 48, 255], [0, 0, 130], [255, 255, 0], [205, 62, 78]]),
                    ('Yeo-7', [[120, 18, 134], [70, 130, 180], [0, 118,  14], [196,  58, 250], [220, 248, 164], [230, 148,  34], [205,  62,  78]]),
                    ('Yeo-17', [[120, 18, 134], [255, 0,  0], [70, 130, 180], [42, 204, 164], [74, 155,  60], [0, 118,  14], [196,  58, 250], [255, 152, 213], [220, 248, 164], [122, 135,  50], [119, 140, 176], [230, 148,  34], [135,  50,  74], [12,  48, 255], [0,  0, 130], [255, 255, 0], [205, 62,  78]]),
                    ('SUIT-26', [[204, 255, 0], [184, 230, 0], [0, 230, 92], [0, 255, 102], [0, 102, 255], [0, 82, 204], [0, 102, 255], [204, 0, 255], [163, 0, 204], [204, 0, 255], [255, 0, 0], [204, 0, 0], [255, 0, 0], [255, 153, 0], [204, 122, 0], [255, 153, 0], [51, 255, 0], [41, 204, 0], [51, 255, 0], [0, 255, 255], [0, 204, 204], [0, 255, 255], [51, 0, 255], [41, 0, 204], [51, 0, 255], [255, 0, 153], [204, 0, 122], [255, 0, 153], [245, 74, 59], [214, 33, 17], [217, 120, 30], [148, 81, 9], [34, 96, 150], [37, 111, 207]]),
                    ('MDTB-10-subRegions', [[23, 83, 76], [43, 76, 16], [26, 51, 74], [8, 13, 63], [83, 12, 81], [88, 22, 36], [113, 63, 88], [118, 81, 4], [126, 109, 59], [60, 59, 123], [46, 166, 152], [85, 151, 32], [51, 102, 147], [15, 25, 126], [165, 24, 162], [175, 44, 71], [225, 126, 176], [236, 161, 8], [252, 218, 118], [119, 118, 246]]),
    ])

    labels = {'SUIT-10': ['R01-I-IV', 'R02-V', 'R03-VI', 'R04-CrusI', 'R05-CrusII', 'R06-VIIb', 'R07-VIIIa', 'R08-VIIIb', 'R09-IX', 'R10-X'],
            'SUIT-26': ['R01-I-IV-L', 'R01-I-IV-R', 'R02-V-L', 'R02-V-R', 'R03-VI-L', 'R03-VI-Ve', 'R03-VI-R', 'R04-CrusI-L', 'R04-CrusI-Ve', 'R04-CrusI-R', 'R05-CrusII-L','R05-CrusII-Ve', 'R05-CrusII-R', 'R06-VIIb-L', 'R06-VIIb-Ve', 'R06-VIIb-R', 'R07-VIIIa-L', 'R07-VIIIa-Ve', 'R07-VIIIa-R', 'R08-VIIIb-L', 'R08-VIIIb-Ve', 'R08-VIIIb-R', 'R09-IX-L', 'R09-IX-Ve', 'R09-IX-R', 'R10-X-L', 'R10-X-Ve', 'R10-X-R', 'R11-Dentate-L', 'R11-Dentate-R', 'R12-Globose-L', 'R12-Globose-R', 'R13-Fastigial-L', 'R13-Fastigial-R'],
            'MDTB-10-subRegions': ['R01-A', 'R02-A', 'R03-A', 'R04-A', 'R05-A', 'R06-A', 'R07-A', 'R08-A', 'R09-A', 'R10-A', 'R01-P', 'R02-P', 'R03-P', 'R04-P', 'R05-P', 'R06-P', 'R07-P', 'R08-P', 'R09-P', 'R10-P'],
            'Retinotopy-10': ['R01-right_lVIIIb', 'R02-right_mVIIIb', 'R03-left_mVIIIb', 'R04-left_lVIIIb', 'R05-left_VIIb', 'R06-right_VIIb', 'R07-left_mOMV', 'R08-left_lOMV', 'R09-right_mOMV', 'R10-right_lOMV'],
            'Yeo-Buckner-7': ['Yeo-7-R01', 'Yeo-7-R02', 'Yeo-7-R03', 'Yeo-7-R04', 'Yeo-7-R05', 'Yeo-7-R06', 'Yeo-7-R07', 'Buckner-7-R01', 'Buckner-7-R02', 'Buckner-7-R03', 'Buckner-7-R04', 'Buckner-7-R05', 'Buckner-7-R06', 'Buckner-7-R07'],
            'Yeo-Buckner-17': ['Yeo-17-R01', 'Yeo-17-R02', 'Yeo-17-R03', 'Yeo-17-R04', 'Yeo-17-R05', 'Yeo-17-R06', 'Yeo-17-R07', 'Yeo-17-R08', 'Yeo-17-R09', 'Yeo-17-R10', 'Yeo-17-R11', 'Yeo-17-R12', 'Yeo-17-R13', 'Yeo-17-R14','Yeo-17-R15', 'Yeo-17-R16', 'Yeo-17-R17', 'Buckner-17-R01', 'Buckner-17-R02', 'Buckner-17-R03', 'Buckner-17-R04', 'Buckner-17-R05', 'Buckner-17-R06', 'Buckner-17-R07', 'Buckner-17-R08', 'Buckner-17-R09', 'Buckner-17-R10', 'Buckner-17-R11', 'Buckner-17-R12', 'Buckner-17-R13', 'Buckner-17-R14', 'Buckner-17-R15', 'Buckner-17-R16', 'Buckner-17-R17']
            }
