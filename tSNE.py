#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script to run TSNE on HCP task data
# Downloads data from R2 (on CANOPY CLuster)
# Reads in data
# Runs TSNE

# Imports
try:
  import platform
  # import argparse
  import logging
  import pandas as pd
  import os
  import datetime as dt
  import json # Not on Cluster
  # import hashlib # Not on Cluster
  import paramiko
  # import sys
  from scp import SCPClient ############################ Mising
  # import shutil
  import getpass
  # import nibabel as nib
  import numpy as np
  # from nilearn import datasets
  # from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
  # from sklearn.preprocessing import StandardScaler
  # from sklearn.ensemble import RandomForestClassifier
  # from sklearn.metrics import confusion_matrix, classification_report
  # from sklearn.linear_model import LogisticRegression
  from sklearn.manifold import TSNE
  # from sklearn.model_selection import cross_val_score, train_test_split, KFold, GridSearchCV
  # from sklearn.feature_selection import SelectFromModel
  # from sklearn.decomposition import PCA
  # from sklearn.inspection import permutation_importance
  # from itertools import compress
  # from sklearn import svm
  # from sklearn.svm import SVC
  # from pathlib import Path
  # from statsmodels.stats.outliers_influence import variance_inflation_factor
  import matplotlib.pyplot as plt
  # import seaborn as sns
  # from scipy.stats import spearmanr
  # from scipy.cluster import hierarchy
  # from collections import defaultdict
  import pickle as pk
  # from skopt import BayesSearchCV ############################ Mising
  # from skopt.space import Real, Categorical, Integer ############################ Mising
  # from operator import itemgetter
except Exception as e:
  print(f'Error loading libraries: ')
  raise Exception(e)


# Global Variables
try:
  sep = os.path.sep
  # source_path = '/home/kbaacke/HCP_Analyses/'
  source_path = os.path.dirname(os.path.abspath(__file__)) + sep
  sys_name = platform.system() 
  hostname = platform.node()
  output_path = 'C:\\Users\\Sarah Melissa\\Documents\\output\\'
  local_path = 'C:\\Users\\Sarah Melissa\\Documents\\temp\\'
  run_uid = '8d2513'
  remote_outpath = '/mnt/usb1/hcp_analysis_output/'
  uname = 'solshan2'
  datahost = 'r2.psych.uiuc.edu'
  source_path = '/mnt/usb1/hcp_analysis_output/'
except:
  pass



# Template Functions
try:
  def createSSHClient(server, user, password, port=22):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client
except Exception as e:
  print(f'Error defining template functions: ')
  raise Exception(e)

outpath = output_path + run_uid + sep
tsne_out = outpath + 't-SNE' + sep
# Make folder specific to this run's output
try:
    os.makedirs(tsne_out)
except:
    pass

# SCP data to temp location
if source_path!=None:
  # Interupt request for password and username if none passed
  if uname == None:
    uname = getpass.getpass(f'Username for {datahost}:')
  psswd = getpass.getpass(f'Password for {uname}@{datahost}:')
  src_basepath = source_path
  download_start_time = dt.datetime.now()
  print('Starting Data Transfer: ', download_start_time)
  try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(datahost, 22, uname, psswd)
    scp = SCPClient(ssh.get_transport())
    scp.get(source_path + run_uid, local_path + run_uid, recursive=True)
  except Exception as e:
    print(f'Error transferring data from {uname}@{datahost} ')
    raise Exception(e)

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


parcel_connection_features = feature_set_dict['parcel_connection']

# start with raw data
input_raw = np.array(parcel_connection_features['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'])

input_pca = np.array(
  parcel_connection_features['train_pca']
)
level = 9
k = 'parcel_connection'
feature_index = feature_set_dict[k]['hierarchical_selected_features'][level]
input_hfs_9_input = np.array(feature_set_dict[k]['train_x'][feature_set_dict[k]['train_x'].columns[feature_index]])


embedded_raw = TSNE(
  n_components=2,
  perplexity=77,
  learning_rate=50,
  init='random',
  verbose=1,
  n_iter=2000,
  ).fit_transform(input_raw)

colormap = np.array(['black','red', 'green', 'blue', 'yellow', 'purple',
                    'olive', 'saddlebrown'])
plt.scatter(embedded_raw[:,0], embedded_raw[:,1],c = np.array(parcel_connection_features['train_y']), s=1)
ploting_df = pd.DataFrame(embedded_raw, columns = ['x','y'])
ploting_df['task'] = parcel_connection_features['train_y']['task']

ploting_df['task'] = ploting_df['task'].astype(str)

import seaborn as sb
sb.scatterplot(embedded_raw[:,0], embedded_raw[:,1], hue = ploting_df['task'], s=3)
