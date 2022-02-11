#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports

# Imports
try:
  import platform
  import argparse
  import logging
  import pandas as pd
  import os
  import datetime as dt
  import json # Not on Cluster
  import hashlib # Not on Cluster
  import paramiko
  import sys
  from scp import SCPClient ############################ Mising
  import shutil
  import getpass
  import nibabel as nib
  import numpy as np
  from nilearn import datasets
  from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
  from sklearn.preprocessing import StandardScaler
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import confusion_matrix, classification_report
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import cross_val_score, train_test_split, KFold, GridSearchCV
  from sklearn.feature_selection import SelectFromModel
  from sklearn.decomposition import PCA
  from sklearn.inspection import permutation_importance
  from itertools import compress
  from sklearn import svm
  from sklearn.svm import SVC
  from pathlib import Path
  from statsmodels.stats.outliers_influence import variance_inflation_factor
  import matplotlib.pyplot as plt
  import seaborn as sns
  from scipy.stats import spearmanr
  from scipy.cluster import hierarchy
  from collections import defaultdict
  import pickle as pk
  from skopt import BayesSearchCV ############################ Mising
  from skopt.space import Real, Categorical, Integer ############################ Mising
  from operator import itemgetter
except Exception as e:
  print(f'Error loading libraries: ')
  raise Exception(e)


# Global Variables
sep = os.path.sep
# source_path = '/home/kbaacke/HCP_Analyses/'
source_path = os.path.dirname(os.path.abspath(__file__)) + sep
sys_name = platform.system() 
hostname = platform.node()

# Template Functions
try:
  def parse_args(args):
    #Presets
    parser = argparse.ArgumentParser(
        description='Analysis Script for task decoding.'
      )
    parser.add_argument(
      "-source_path", help='Full base-path on the Data Node where the data is stored.', required=False, default=None
    )
    parser.add_argument(
      "-uname", help='Username to use when requesting files from the data node via scp.', required=False
    )
    parser.add_argument(
      "-datahost", help='Name of the source node. For example: \'r2.psych.uiuc.edu\'', required=False
    )
    parser.add_argument(
      "-local_path", help='Full base-path on the local machine where data will be stored (or is already stored).', required=True
    )
    parser.add_argument(
      "--output", help=f'Full local path to the location ot store the output. Defaults to the source path fo this python file + \'Output{sep}\'', required=False
    )
    parser.add_argument(
      "--remote_output", help=f'Full remote path to the location ot store the output. By default will not send the output to the data host.', required=False
    )
    parser.add_argument(
      "--n_jobs", help="Specify number of CPUs to use in the analysis. Default is 1. Set to -1 to use all available cores.", required=False, default=1
    )
    parser.add_argument(
        "-run_uid", help="Unique identifier from feature selection step.",
        required=True
    )
    return parser.parse_known_args(args)
  def createSSHClient(server, user, password, port=22):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client
except Exception as e:
  print(f'Error defining template functions: ')
  raise Exception(e)

# # Custom Functions
# try:
#     def BayesCVWrapper_V01(etsimator, search_spaces, n_jobs, optimizer_kwargs): # TODO

#         pass
# except Exception as e:
#   print(f'Error defining custom functions: ')
#   raise Exception(e)

v1_argslist = [ # for running from L2
  '-source_path','/mnt/usb1/hcp_analysis_output/',
  '-uname','kbaacke',
  '-datahost','r2.psych.uiuc.edu',
  '-local_path','C:\\Users\\kyle\\temp\\',
  '--output','C:\\Users\\kyle\\output\\',
  '--remote_output','/mnt/usb1/hcp_analysis_output/',
  '--n_jobs','4',
  '-run_uid','8d2513'
]

# Read args
args, leforvers = parse_args(v1_argslist)

# SCP data to temp location
# if args.source_path!=None:
#   # Interupt request for password and username if none passed
#   if args.uname == None:
#     args.uname = getpass.getpass(f'Username for {args.datahost}:')
#   psswd = getpass.getpass(f'Password for {args.uname}@{args.datahost}:')
#   src_basepath = args.source_path
#   download_start_time = dt.datetime.now()
#   print('Starting Data Transfer: ', download_start_time)
#   try:
#     ssh = paramiko.SSHClient()
#     ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     ssh.connect(args.datahost, 22, args.uname, psswd)
#     scp = SCPClient(ssh.get_transport())
#     scp.get(args.source_path + args.run_uid, args.local_path + args.run_uid, recursive=True)
#   except Exception as e:
#     print(f'Error transferring data from {args.uname}@{args.datahost} ')
#     raise Exception(e)

outpath = args.output + args.run_uid + sep
try:
    os.mkdir(outpath)
except:
    pass
run_uid = args.run_uid
total_start_time = dt.datetime.now()
logging.basicConfig(filename=f'{outpath}{run_uid}_DEBUG.log', level=logging.DEBUG)

arch = str(platform.architecture()[0])
logging.debug(f'Architecture: {arch}')
machine = platform.machine()
logging.debug(f'Processor: {machine}')
node = platform.node()
logging.debug(f'Node Name: {node}')
logging.info(f'Started; {total_start_time}') #Adds a line to the logfile to be exported after analysis
logging.debug('args: ')
logging.debug(args)

# Read in Data
## meta_dict
meta_dict = json.load(open(args.local_path + run_uid + sep + run_uid + 'metadata.json'))
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
  'parcel_sum':{
  },
  'network_sum':{
  },
  'parcel_connection':{
  },
  'network_connection':{
  }
}
fs_outpath = f'{args.local_path}{run_uid}{sep}FeatureSelection{sep}'
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
    logging.info('Previous Hierarchical Feaure Selection Output imported: {sub_end_time}')
  except Exception as e:
    print(f'Error reading {k} Hierarchical Feaures, n = {n}, Error: {e}')
    logging.info(f'Error reading {k} Hierarchical Feaures, n = {n}, Error: {e}')
  # PCA
  logging.info(f'\tHierarchical Feaure Selection ({k}) Done: {sub_end_time}')
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

# Select data set (parcel vs network and _sum and _connection)
k = 'parcel_connection'
train_x = feature_set_dict[k]['train_x']
train_y = feature_set_dict[k]['train_y'].values.ravel()
test_x = feature_set_dict[k]['test_x']
test_y = feature_set_dict[k]['test_y'].values.ravel()
# Iterate through feature sets with same sizes 
# Levels of hierarchical selection:
h_levels = list(feature_set_dict[k]['hierarchical_selected_features'].keys())
h_levels.reverse()
# subset h_levels
h_levels = h_levels[:3]
for level in h_levels:
  feature_index = feature_set_dict[k]['hierarchical_selected_features'][level]
  sub_train_x = feature_set_dict[k]['train_x'][feature_set_dict[k]['train_x'].columns[feature_index]]
  sub_test_x = feature_set_dict[k]['test_x'][feature_set_dict[k]['train_x'].columns[feature_index]]
  n_folds = 5
  opt = BayesSearchCV(
    SVC(),
    {
      'C': Real(1e-6, 1e+6, prior='log-uniform'),
      'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
      'degree': Integer(1,8),
      'kernel': Categorical(['linear', 'poly', 'rbf']),
    },
    n_iter=32,
    random_state=0,
    refit=True,
    cv=n_folds,
    n_jobs = int(args.n_jobs)
  )
  res = opt.fit(sub_train_x, train_y)
  print(opt.score(sub_test_x, test_y))