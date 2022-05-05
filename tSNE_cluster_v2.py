#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script to run TSNE on HCP task data
# Downloads data from R2 (on CANOPY CLuster)
# Reads in data
# Runs TSNE
# Saves TSNE variables as output

# Imports
try:
  import platform
  import logging
  import pandas as pd
  import os
  import datetime as dt
  import json 
  # import paramiko
  # from scp import SCPClient
  # import getpass
  import numpy as np
  from sklearn.manifold import TSNE
  # import matplotlib.pyplot as plt
  import pickle as pk
  # import seaborn as sb
  import time
except Exception as e:
  print(f'Error loading libraries: ')
  raise Exception(e)




# Global Variables
try:
  sep = os.path.sep
  # source_path = '/home/kbaacke/HCP_Analyses/'
  source_path = '/data/hx-hx1/kbaacke/hcp_analysis_output/'
  # source_path = os.path.dirname(os.path.abspath(__file__)) + sep
  sys_name = platform.system() 
  hostname = platform.node()
  output_path = '/data/hx-hx1/kbaacke/datasets/hcp_analysis_output/'
  local_path = '/data/hx-hx1/kbaacke/datasets/hcp_analysis_output/'
  # output_path = 'C:\\Users\\Sarah Melissa\\Documents\\output\\'
  # local_path = 'C:\\Users\\Sarah Melissa\\Documents\\temp\\'
  run_uid = '8d2513'
  remote_outpath = '/data/hx-hx1/kbaacke/datasets/hcp_analysis_output/'
  # uname = 'solshan2'
  # datahost = 'r2.psych.uiuc.edu'
  source_path = '/data/hx-hx1/kbaacke/datasets/hcp_analysis_output/'
except Exception as e:
  print(e)


outpath = output_path + run_uid + sep
tsne_out = outpath + 't-SNE' + sep
# Make folder specific to this run's output
try:
    os.makedirs(tsne_out)
except:
    pass


# Start logging file
total_start_time = dt.datetime.now()
logging.basicConfig(filename=f'{outpath}{run_uid}_t-SNE_DEBUG.log', level=logging.DEBUG)
arch = str(platform.architecture()[0])
logging.debug(f'Architecture: {arch}')
machine = platform.machine()
logging.debug(f'Processor: {machine}')
node = platform.node()
logging.debug(f'Node Name: {node}')
logging.info(f'Started; {total_start_time}') #Adds a line to the logfile to be exported after analysis

# Read in Data
## meta_dict
meta_dict = json.load(open(local_path + run_uid + sep + run_uid + 'metadata.json'))
logging.debug('meta_dict: ')
logging.debug(meta_dict)

sessions = [
  "tfMRI_MOTOR",
  "tfMRI_WM",
  "tfMRI_EMOTION",
  "tfMRI_GAMBLING",
  "tfMRI_LANGUAGE",
  "tfMRI_RELATIONAL",
  "tfMRI_SOCIAL"
]
feature_set_dict = {
  'parcel_connection':{
  }
}

fs_outpath = f'{local_path}{run_uid}{sep}FeatureSelection{sep}'
sub_start_time = dt.datetime.now()
logging.info(f'Attempting to read data from {fs_outpath}: {sub_start_time}')
try:
  for k in feature_set_dict.keys():
    for target_df in ['train_x','test_x','train_y','test_y']:
      feature_set_dict[k][target_df] = pd.DataFrame(np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}_colnames.npy', allow_pickle=True))
  sub_end_time = dt.datetime.now()
  logging.info(f'Premade raw data successfully imported from {fs_outpath}: {sub_end_time}')
except Exception as e:
  print(f'Error reading in raw data: {e}')
  logging.info(f'Error reading in raw data: {e}')

# Read in feature selection 
for k in feature_set_dict.keys():
  # Hierarchical
  sub_start_time = dt.datetime.now()
  hierarchical_start = 1
  hierarchical_end = 30
  feature_set_dict[k]['hierarchical_selected_features'] = {}
  try:
    for n in range(hierarchical_start, hierarchical_end):
      feature_set_dict[k]['hierarchical_selected_features'][n] = np.load(f'{fs_outpath}{k}{sep}{run_uid}_hierarchical-{n}.npy')
    sub_end_time = dt.datetime.now()
    logging.info('Previous Hierarchical Feature Selection Output imported: {sub_end_time}')
  except Exception as e:
    print(f'Error reading {k} Hierarchical Features, n = {n}, Error: {e}')
    logging.info(f'Error reading {k} Hierarchical Features, n = {n}, Error: {e}')
  # PCA
  logging.info(f'\tHierarchical Feature Selection ({k}) Done: {sub_end_time}')
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k]['train_pca'] = np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_pca.npy')
    feature_set_dict[k]['test_pca'] = np.load(f'{fs_outpath}{k}{sep}{run_uid}_test_pca.npy')
    feature_set_dict[k]['pca'] = pk.load(open(f'{fs_outpath}{k}{sep}{run_uid}_pca.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    logging.info('\tPrevious PCA Output imported: {sub_end_time}')
  except Exception as e:
    sub_end_time = dt.datetime.now()
    logging.info(f'Error reading {k} PCA: {e}, {sub_end_time}')
  logging.info(f'\tPCA import Done: {sub_end_time}')
  # RFC feature selection
  ## Select from model
  sub_start_time_outer = dt.datetime.now()
  logging.info(f'\tSelectFromModel on FRC on {k} started: {sub_start_time_outer}')
  for x in feature_set_dict[k]['hierarchical_selected_features'].keys():
    if x>1 and x<len(feature_set_dict[k]['train_x'].columns):
      # This can be optimized, return to this later
      sub_start_time = dt.datetime.now()
      try:
        feature_set_dict[k][f'rf_selected_{x}'] = np.load(f'{fs_outpath}{k}{sep}{run_uid}_rf_selected_{x}.npy')
        sub_end_time = dt.datetime.now()
        logging.info(f'\t\tSelectFromModel on RFC for {x} max features read from previous run')
      except Exception as e:
        sub_end_time = dt.datetime.now()
        logging.info(f'\t\tError reading SelectFromModel on RFC for {x} max features read from previous run: {e}, {sub_end_time}')
  sub_end_time_outer = dt.datetime.now()
  logging.info(f'\tSelectFromModel on FRC on {k} Done: {sub_end_time_outer}')
  ## Permutation importance
  sub_start_time_outer = dt.datetime.now()
  n_estimators = 500
  n_repeats = 50
  try:
    feature_set_dict[k][f'feature_importances_{n_estimators}'] = np.load(f'{fs_outpath}{k}{sep}{run_uid}_feature_importances_est-{n_estimators}.npy')
    logging.info('\tFRC Feature importance and permutation importance on {k} read in from prior run.')
  except Exception as e:
    logging.info(f'\tError reading {k} permutation importance features: {e}')


k = 'parcel_connection'
parcel_connection_features = feature_set_dict[k]


# Use to keep track of length of feature sets
length_list = [
  19900,
  19884,
  18540,
  13981,
  10027,
  7482,
  5847,
  4638,
  3771,
  3149,
  2679,
  2297,
  1980,
  1725,
  1533,
  1354,
  1205,
  1078,
  960,
  876,
  807,
  734,
  676,
  622,
  569,
  522,
  480,
  443,
  414
]


# Setup outcome label dataframe for plotting later
plotting_df = pd.DataFrame()
plotting_df['task'] = parcel_connection_features['train_y']['task']
numeric_task_ref = pd.DataFrame(
  {
    'label':["MOTOR", "WM", "EMOTION", "GAMBLING", "LANGUAGE", "RELATIONAL", "SOCIAL"],
    'task':[4, 7, 1, 2, 3, 5, 6]
  }
)
plotting_df = pd.merge(plotting_df, numeric_task_ref, on='task')

# Create function to run tSNE
def run_tsne(data, label, perplexity = 77, learning_rate=50, init = 'random', verbose=1, n_iter=2000, n_jobs = 1):
  start_time = dt.datetime.now()
  # Run tSNE
  try:
    tSNE_output = TSNE(
      n_components=2,
      perplexity=perplexity,
      learning_rate=learning_rate,
      init=init,
      verbose=verbose,
      n_iter=n_iter,
      n_jobs = n_jobs
      ).fit_transform(data)
    np.save(f'{tsne_out}{label}_tSNE.npy',tSNE_output)
    # sb.scatterplot(tSNE_output[:,0], tSNE_output[:,1], hue = plotting_df['label'], s=3).set(title=label)
    # plt.savefig(f'{tsne_out}{label} tSNE.png', transparent=True)
    end_time = dt.datetime.now()
    runtime = end_time - start_time
    print(f'tSNE on {label} complete. Runtime: {runtime}')
  except Exception as e:
    print(f'Error running tSNE for {label}: {e}')

# Create function to run tSNE and save plots
def run_plt_tsne(data, label, perplexity = 77, learning_rate=50, init = 'random', verbose=1, n_iter=2000):
  start_time = dt.datetime.now()
  # Run tSNE
  try:
    tSNE_output = TSNE(
      n_components=2,
      perplexity=perplexity,
      learning_rate=learning_rate,
      init=init,
      verbose=verbose,
      n_iter=n_iter,
      ).fit_transform(data)
    sb.scatterplot(tSNE_output[:,0], tSNE_output[:,1], hue = plotting_df['label'], s=3).set(title=label)
    plt.savefig(f'{tsne_out}{label} tSNE.png', transparent=True)
    end_time = dt.datetime.now()
    runtime = end_time - start_time
    print(f'tSNE on {label} complete. Runtime: {runtime}')
  except Exception as e:
    print(f'Error running tSNE for {label}: {e}')

#### Run to here to set-up ####

perp_list = [
  50, 
  # 55, 
  # 60, 
  # 70, 
  75, 
  # 80, 
  # 85, 
  90
]

full_data = feature_set_dict[k]['train_x']
for perp in perp_list:
  # The following sections will perform tSNE on the 4 different feature selection sets we have
  # 1) PCA
  run_tsne(
    data = np.array(parcel_connection_features['train_pca']),
    label = f'PCA_perp{perp}',
    perplexity = perp,
    n_jobs = 4
  )

for perp in perp_list:
  for n in length_list[6:]:
    run_tsne(
      data = parcel_connection_features['train_pca'][:, 0:n],
      label = f'PCA_{n}_perp{perp}',
      perplexity = perp,
      n_jobs = 4
    )

for perp in perp_list:
  # # 2) Hierarchical selected features
  for level in list(feature_set_dict[k]['hierarchical_selected_features'].keys())[:]:
    feature_index = feature_set_dict[k]['hierarchical_selected_features'][level]
    feature_len = len(feature_index)
    run_tsne(
      data = np.array(feature_set_dict[k]['train_x'][feature_set_dict[k]['train_x'].columns[feature_index]]),
      label = f'HFS level {level} {feature_len} features_perp{perp}'
    )

for perp in perp_list:
  # 3) Random Forest Select from Model
  for level in list(feature_set_dict[k]['hierarchical_selected_features'].keys())[:]:
    feature_index = feature_set_dict[k][f'rf_selected_{x}']
    feature_len = len(feature_index)
    run_tsne(
      data = np.array(feature_set_dict[k]['train_x'][feature_index]),
      label = f'RF Selected from Model {feature_len} features_perp{perp}'
    )

for perp in perp_list:
  # 4) Permutation Importance features
  n_estimators = 500
  n_repeats = 50
  perm_importances = feature_set_dict[k][f'feature_importances_{n_estimators}']
  ranked_indices = np.argsort(perm_importances)[::-1]
  for n in length_list[1:]: # Starting this at 1 so you don't run this on the full dataset (at position 1), 29 options
    feature_index = ranked_indices[:n]
    run_tsne(
      data = np.array(feature_set_dict[k]['train_x'][feature_set_dict[k]['train_x'].columns[feature_index]]),
      label = f'RF Permutation Importance {n} features_perp{perp}'
    )

