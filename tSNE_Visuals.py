import umap.umap_ as umap
import pandas as pd
import datetime as dt
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Image
import os
import numpy as np

cmap_hcp = {
  'tfMRI_MOTOR':'#9F2936',
  'tfMRI_WM':'#6A76FC', 
  'tfMRI_EMOTION':'#1B587C', 
  'tfMRI_GAMBLING':'#FE00CE', 
  'tfMRI_LANGUAGE':'#4E8542', 
  'tfMRI_RELATIONAL':'#604878',
  'tfMRI_SOCIAL':'#EEA6FB'
}

cmap_ucla = {
  'scap':'#9F2936',
  'bart':'#6A76FC', 
  'taskswitch':'#1B587C', 
  'stopsignal':'#FE00CE', 
  'rest':'#4E8542', 
  'bht':'#604878',
  'pamret':'#EEA6FB', 
  'pamenc':'#C19859'
}

feature_set_dict = {
  'parcel_connection':{
  }
}

sep = os.path.sep
local_path = '/data/hx-hx1/kbaacke/datasets/ucla_analysis_output/'
run_uid = '89952a'
fs_outpath = f'{local_path}{run_uid}{sep}FeatureSelection{sep}'
job_cap = 14

for k in feature_set_dict.keys():
  for target_df in ['train_x','test_x','train_y','test_y']:
    feature_set_dict[k][target_df] = pd.DataFrame(np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}_colnames.npy', allow_pickle=True))

length_list3 = [
  19900,
  19877,
  18386,
  13745,
  9857,
  7402,
  5773,
  4586,
  3754,
  3125,
  2656,
  2271,
  1958,
  1727,
  1522,
  1331,
  1184,
  1064,
  947,
  857,
  780,
  709,
  667,
  616,
  560,
  513,
  479,
  450,
  418,
  390,
  361,
  342,
  316,
  294,
  282,
  271,
  251,
  242,
  229,
  214,
  201,
  191,
  181,
  172,
  167,
  162,
  157,
  152,
  146,
  142,
  136,
  132,
  126,
  124,
  121,
  116,
  109,
  105,
  101,
  96,
  93,
  90,
  85,
  79,
  78,
  76,
  75,
  74,
  71,
  70,
  66,
  65,
  63,
  61,
  60,
  57,
  55,
  54,
  53,
  50,
  49,
  47,
  46,
  45,
  44,
  42,
  41,
  39,
  38,
  36,
  35,
  34,
  32,
  31,
  30,
  29,
  27,
  26,
  24,
  23,
  22,
  21,
  20,
  19,
  18,
  17,
  16,
  15,
  14,
  13,
  12,
  10,
  9,
  8,
  7,6,5,4,3,2,1
]

# Read in feature selection 
for k in feature_set_dict.keys():
  # PCA
  # logging.info(f'\tHierarchical Feature Selection ({k}) Done: {sub_end_time}')
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k]['train_pca'] = np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_pca.npy')
    feature_set_dict[k]['test_pca'] = np.load(f'{fs_outpath}{k}{sep}{run_uid}_test_pca.npy')
    # feature_set_dict[k]['pca'] = pk.load(open(f'{fs_outpath}{k}{sep}{run_uid}_pca.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    # logging.info('\tPrevious PCA Output imported: {sub_end_time}')
  except Exception as e:
    sub_end_time = dt.datetime.now()
    # logging.info(f'Error reading {k} PCA: {e}, {sub_end_time}')
  # logging.info(f'\tPCA import Done: {sub_end_time}')
  ## KPCA
  for kernel in ['rbf', 'linear']:
    try:
      sub_start_time = dt.datetime.now()
      feature_set_dict[k][f'train_kpca-{kernel}'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_kpca-{kernel}.npy')
      feature_set_dict[k][f'test_kpca-{kernel}'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_kpca-{kernel}.npy')
      # feature_set_dict[k][f'kpca-{kernel}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_kpca-{kernel}.pkl', 'rb'))
      sub_end_time = dt.datetime.now()
      # logging.info('\tPrevious KernelPCA-{kernel} Output imported: {sub_end_time}')
    except Exception as e:
      print(e)
  ## TruncatedSVD
  for component_size in [
    #50, 100, 150, 200, 300, 400, 
    500
    #, 600, 700, 800, 900, 1000
    ]:
    sub_end_time = dt.datetime.now()
    # logging.info(f'\tTruncatedSVD Feature Extraction ({k}) Started: {sub_end_time}')
    try:
      sub_start_time = dt.datetime.now()
      feature_set_dict[k][f'train_tSVD-{component_size}'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_tSVD-{component_size}.npy')
      feature_set_dict[k][f'test_tSVD-{component_size}'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_tSVD-{component_size}.npy')
      # feature_set_dict[k][f'tSVD-{component_size}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_tSVD-{component_size}.pkl', 'rb'))
      sub_end_time = dt.datetime.now()
      # logging.info('\tPrevious TruncatedSVD-{component_size} Output imported: {sub_end_time}')
    except Exception as e:
      print(e)
  ## ICA
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k][f'train_ICA'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_ICA.npy')
    feature_set_dict[k][f'test_ICA'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_ICA.npy')
    # feature_set_dict[k][f'ICA'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_ICA.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    # logging.info('\tPrevious ICA Output imported: {sub_end_time}')
  except Exception as e:
      print(e)
  ## LDA
  sub_end_time = dt.datetime.now()
  # logging.info(f'\LDA Feature Extraction ({k}) Started: {sub_end_time}')
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k][f'train_LDA'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_LDA.npy')
    feature_set_dict[k][f'test_LDA'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_LDA.npy')
    # feature_set_dict[k][f'LDA'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_LDA.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    # logging.info('\tPrevious LDA Output imported: {sub_end_time}')
  except Exception as e:
    print(e)
    

# Full Feature Set

# PCA

# kPCA

# TSVD

# ICA

# LDA

