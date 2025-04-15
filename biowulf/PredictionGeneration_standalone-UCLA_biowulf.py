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
  from operator import itemgetter
  from sklearn.model_selection import GroupShuffleSplit
  from joblib import Parallel, delayed
except Exception as e:
  print(f'Error loading libraries: ')
  raise Exception(e)


# Global Variables
sep = os.path.sep
# source_path = '/home/kbaacke/HCP_Analyses/'
source_path = os.path.dirname(os.path.abspath(__file__)) + sep
sys_name = platform.system() 
hostname = platform.node()
job_cap = 7

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
  '-source_path','/data/hx-hx1/kbaacke/datasets/ucla_analysis_output/',#'/mnt/usb1/ucla_analysis_output/'
  '-uname','kbaacke',
  '-datahost','r2.psych.uiuc.edu',
  '-local_path','/data/hx-hx1/kbaacke/datasets/ucla_analysis_output/',#'C:\\Users\\kyle\\temp\\',#
  '--output','/data/hx-hx1/kbaacke/datasets/ucla_analysis_output/',#'C:\\Users\\kyle\\output\\',#
  '--remote_output','/data/hx-hx1/kbaacke/datasets/ucla_analysis_output/',#'/mnt/usb1/ucla_analysis_output/'
  '--n_jobs','4',
  '-run_uid','89952a'
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
  'parcel_connection':{
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

length_list2 = [
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  18,
  19,
  20,
  21,
  22,
  23,
  24,
  25,
  26,
  27,
  28,
  29,
  33,
  34,
  35,
  38,
  39,
  40,
  42,
  43,
  44,
  48,
  50,
  51,
  52,
  54,
  57,
  60,
  62,
  63,
  64,
  67,
  69,
  73,
  75,
  76,
  79,
  81,
  82,
  83,
  84,
  91,
  95,
  97,
  101,
  103,
  110,
  115,
  122,
  125,
  129,
  134,
  141,
  149,
  152,
  156,
  161,
  169,
  179,
  187,
  195,
  208,
  216,
  223,
  242,
  254,
  264,
  275,
  297,
  320,
  340,
  365,
  397,
  414,
  443,
  480,
  522,
  569,
  622,
  676,
  734,
  807,
  876,
  960,
  1078,
  1205,
  1354,
  1533,
  1725,
  1980,
  2297,
  2679,
  3149,
  3771,
  4638,
  5847,
  7482,
  10027,
  13981,
  18540,
  19884,
  19900
]

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
  # Hierarchical
  sub_start_time = dt.datetime.now()
  hierarchical_start = 1
  hierarchical_end = 200
  feature_set_dict[k]['hierarchical_selected_features'] = {}
  for n in range(hierarchical_start, hierarchical_end):
    try:
      feature_set_dict[k]['hierarchical_selected_features'][n] = np.load(f'{fs_outpath}{k}{sep}{run_uid}_hierarchical-{n}.npy')
    except Exception as e:
      print(f'Error reading {k} Hierarchical Features, n = {n}, Error: {e}')
      logging.info(f'Error reading {k} Hierarchical Features, n = {n}, Error: {e}')
  sub_end_time = dt.datetime.now()
  logging.info('Previous Hierarchical Feature Selection Output imported: {sub_end_time}')
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
  for x_len in length_list3:
    # This can be optimized, return to this later
    sub_start_time = dt.datetime.now()
    try:
      feature_set_dict[k][f'rf_selected_n{x_len}'] = np.load(f'{fs_outpath}{k}{sep}{run_uid}_rf_selected_n{x_len}.npy')
      sub_end_time = dt.datetime.now()
      logging.info(f'\t\tSelectFromModel on RFC for {x_len} max features read from previous run')
    except Exception as e:
      sub_end_time = dt.datetime.now()
      logging.info(f'\t\tError reading SelectFromModel on RFC for {x_len} max features read from previous run: {e}, {sub_end_time}')
  sub_end_time_outer = dt.datetime.now()
  logging.info(f'\tSelectFromModel on FRC on {k} Done: {sub_end_time_outer}')
  # Permutation importance
  sub_start_time_outer = dt.datetime.now()
  n_estimators = 500
  n_repeats = 50
  try:
    feature_set_dict[k][f'feature_importances_{n_estimators}'] = np.load(f'{fs_outpath}{k}{sep}{run_uid}_feature_importances_est-{n_estimators}.npy')
    logging.info('\tFRC Feature importance and permutation importance on {k} read in from prior run.')
  except Exception as e:
    logging.info(f'\tError reading {k} permutation importance features: {e}')
  try:
    feature_set_dict[k][f'permutation_importances_est-{n_estimators}_rep-{n_repeats}'] = np.load(f'{fs_outpath}{k}/{run_uid}_permutation_importances_est-{n_estimators}_rep-{n_repeats}.npy')
    logging.info('\tFRC Feature importance and permutation importance on {k} read in from prior run.')
  except Exception as e:
    logging.info(f'\tError reading {k} permutation importance features: {e}')
  ## KPCA
  for kernel in ['rbf', 'linear']:
    try:
      sub_start_time = dt.datetime.now()
      feature_set_dict[k][f'train_kpca-{kernel}'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_kpca-{kernel}.npy')
      feature_set_dict[k][f'test_kpca-{kernel}'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_kpca-{kernel}.npy')
      feature_set_dict[k][f'kpca-{kernel}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_kpca-{kernel}.pkl', 'rb'))
      sub_end_time = dt.datetime.now()
      logging.info('\tPrevious KernelPCA-{kernel} Output imported: {sub_end_time}')
    except Exception as e:
      print(e)
  ## TruncatedSVD
  for component_size in [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    sub_end_time = dt.datetime.now()
    logging.info(f'\tTruncatedSVD Feature Extraction ({k}) Started: {sub_end_time}')
    try:
      sub_start_time = dt.datetime.now()
      feature_set_dict[k][f'train_tSVD-{component_size}'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_tSVD-{component_size}.npy')
      feature_set_dict[k][f'test_tSVD-{component_size}'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_tSVD-{component_size}.npy')
      feature_set_dict[k][f'tSVD-{component_size}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_tSVD-{component_size}.pkl', 'rb'))
      sub_end_time = dt.datetime.now()
      logging.info('\tPrevious TruncatedSVD-{component_size} Output imported: {sub_end_time}')
    except Exception as e:
      print(e)
  ## ICA
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k][f'train_ICA'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_ICA.npy')
    feature_set_dict[k][f'test_ICA'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_ICA.npy')
    feature_set_dict[k][f'ICA'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_ICA.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    logging.info('\tPrevious ICA Output imported: {sub_end_time}')
  except Exception as e:
      print(e)
  ## LDA
  sub_end_time = dt.datetime.now()
  logging.info(f'\LDA Feature Extraction ({k}) Started: {sub_end_time}')
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k][f'train_LDA'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_LDA.npy')
    feature_set_dict[k][f'test_LDA'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_LDA.npy')
    feature_set_dict[k][f'LDA'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_LDA.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    logging.info('\tPrevious LDA Output imported: {sub_end_time}')
  except Exception as e:
    print(e)
    

atlases_used = {
  'c1720102':{# !with ICA-Aroma!
    "atlas_name":"MNINonLinear/aparc+aseg.nii.gz",
    "Labels": [
      "Left-Cerebellum-Cortex", "Left-Thalamus", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
      "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", "Left-VentralDC", "Left-choroid-plexus", 
      "Right-Cerebellum-Cortex", "Right-Thalamus", "Right-Caudate", "Right-Putamen", "Right-Pallidum", 
      "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area", "Right-VentralDC", "Right-choroid-plexus"
      ], 
    "confounds": None
    },
  'd8a41be9':{# !with ICA-Aroma!
    "atlas_name": "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm", 
    "confounds": None
    },
  '69354adf':{
    "atlas_name": "Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm", 
    "confounds": None, 
    "Note_1": "No Smoothing, no Confounds"
  },
  '056537de':{
    "atlas_name": "MNINonLinear/aparc+aseg.nii.gz",
    "Labels": [
      "Left-Cerebellum-Cortex", "Left-Thalamus", "Left-Caudate", "Left-Putamen", "Left-Pallidum",
      "Left-Hippocampus", "Left-Amygdala", "Left-Accumbens-area", "Left-VentralDC", "Left-choroid-plexus", 
      "Right-Cerebellum-Cortex", "Right-Thalamus", "Right-Caudate", "Right-Putamen", "Right-Pallidum", 
      "Right-Hippocampus", "Right-Amygdala", "Right-Accumbens-area", "Right-VentralDC", "Right-choroid-plexus"
      ], 
      "confounds": None, 
    "Note_1": "No ICA-Aroma"
  }
}

# Select data set (parcel vs network and _sum and _connection)
k = 'parcel_connection'
random_state = 42

curr_date_str = dt.datetime.now().strftime('%Y-%m-%d_%H:%M')

def generate_uid(metadata, length = 8):
  dhash = hashlib.md5()
  encoded = json.dumps(metadata, sort_keys=True).encode()
  dhash.update(encoded)
  # You can change the 8 value to change the number of characters in the unique id via truncation.
  run_uid = dhash.hexdigest()[:length]
  return run_uid

metadata = meta_dict

metadata['random_state'] = random_state

input_dir = f'{outpath}inputs/'
meta_path = f'{outpath}metadata/'
confusion_path = f'{outpath}confusion/'
classification_path = f'{outpath}classification/'
weight_path = f'{outpath}weights/'
for pth in [meta_path, confusion_path, classification_path, weight_path]:
  try:
    os.mkdir(pth)
  except:
    pass

def run_svc(train_x, train_y, test_x, test_y, dataset, subset, split_ind, method, C=1):
  start_time = dt.datetime.now()
  clf_svm = SVC(kernel='linear', random_state=random_state, C = C)
  clf_svm.fit(train_x, train_y)
  y_pred = clf_svm.predict(test_x)
  training_accuracy = clf_svm.score(train_x, train_y)
  test_accuracy = clf_svm.score(test_x, test_y)
  classification_rep = classification_report(test_y, y_pred)
  confusion_mat = confusion_matrix(test_y, y_pred)
  # Create a UID without accuracy to prevent repeat runs due to mis-aligned accuracies
  meta_dict = {
    'dataset':dataset,
    'subset':subset,
    'split_ind':split_ind,
    'Classifier':'Support Vector Machine',
    #'with_que':metadata['with_que'],
    'ICA_Aroma':metadata['ICA-Aroma'],
    'C':C,
    'kernal':'linear',
    'max_depth': None
  }
  pred_uid = generate_uid(meta_dict)
  clf_param_dict = clf_svm.get_params()
  for k in clf_param_dict.keys():
    clf_param_dict[k] = [clf_param_dict[k]]
  clf_param_dict['metadata_ref'] = [pred_uid]
  clf_param_df = pd.DataFrame(clf_param_dict)
  meta_dict['train_accuracy'] = training_accuracy
  meta_dict['test_accuracy'] = test_accuracy
  end_time = dt.datetime.now()
  with open(f'{meta_path}{pred_uid}_metadata.json', 'w') as outfile:
    json.dump(metadata, outfile)
  try:
    feature_len = len(train_x.columns)
  except:
    feature_len = train_x.shape[1]
  results_dict = {
    'dataset':[dataset],
    'subset':[subset],
    'split_ind':[split_ind],
    'Classifier':['Support Vector Machine'],
    'train_accuracy':[training_accuracy],
    'test_accuracy':[test_accuracy],
    #'with_que':[metadata['with_que']],
    'ICA_Aroma':[metadata['ICA-Aroma']],
    'FS/FR Method':[method],
    'N_Features':[feature_len],
    'metadata_ref':[pred_uid],
    'runtime':(end_time - start_time).total_seconds()
    # 'classification_report':classification_rep,
    # 'confusion_matrix':confusion_mat
  }
  clf_param_df.to_csv(f'{weight_path}{pred_uid}_weights.csv', index=False)
  np.savetxt(f'{confusion_path}{pred_uid}_confusion_matrix.csv', confusion_mat, delimiter=",")
  np.savetxt(f'{classification_path}{pred_uid}_classification_report.csv', confusion_mat, delimiter=",")
  return results_dict

def run_rfc(train_x, train_y, test_x, test_y, dataset, subset, split_ind, method, n_estimators=500, max_depth = None):
  start_time = dt.datetime.now()
  forest = RandomForestClassifier(random_state=random_state ,n_estimators=n_estimators, max_depth = None)
  forest.fit(train_x, train_y)
  y_pred = forest.predict(test_x)
  training_accuracy = forest.score(train_x, train_y)
  test_accuracy = forest.score(test_x, test_y)
  classification_rep = classification_report(test_y, y_pred)
  confusion_mat = confusion_matrix(test_y, y_pred)
  # Create a UID without accuracy to prevent repeat runs due to mis-aligned accuracies
  meta_dict = {
    'dataset':dataset,
    'subset':subset,
    'split_ind':split_ind,
    'Classifier':'Random Forest',
    #'with_que':metadata['with_que'],
    'ICA_Aroma':metadata['ICA-Aroma'],
    'random_state':random_state,
    'n_estimators':n_estimators,
    'max_depth': None
  }
  pred_uid = generate_uid(meta_dict)
  forest_param_dict = forest.get_params()
  for k in forest_param_dict.keys():
    forest_param_dict[k] = [forest_param_dict[k]]
  forest_param_dict['metadata_ref'] = [pred_uid]
  forest_param_df = pd.DataFrame(forest_param_dict)
  meta_dict['train_accuracy'] = training_accuracy
  meta_dict['test_accuracy'] = test_accuracy
  end_time = dt.datetime.now()
  with open(f'{meta_path}{pred_uid}_metadata.json', 'w') as outfile:
    json.dump(metadata, outfile)
  try:
    feature_len = len(train_x.columns)
  except:
    feature_len = train_x.shape[1]
  results_dict = {
    'dataset':[dataset],
    'subset':[subset],
    'split_ind':[split_ind],
    'Classifier':['Random Forest'],
    'train_accuracy':[training_accuracy],
    'test_accuracy':[test_accuracy],
    #'with_que':[metadata['with_que']],
    'ICA_Aroma':[metadata['ICA-Aroma']],
    'FS/FR Method':[method],
    'N_Features':[feature_len],
    'metadata_ref':[pred_uid],
    'runtime':(end_time - start_time).total_seconds()
    # 'classification_report':classification_rep,
    # 'confusion_matrix':confusion_mat
  }
  forest_param_df.to_csv(f'{weight_path}{pred_uid}_weights.csv', index=False)
  np.savetxt(f'{confusion_path}{pred_uid}_confusion_matrix.csv', confusion_mat, delimiter=",")
  np.savetxt(f'{classification_path}{pred_uid}_classification_report.csv', confusion_mat, delimiter=",")
  return results_dict

# gss_holdout = GroupShuffleSplit(n_splits=1, train_size = .9, random_state = random_state)
gss_cv = GroupShuffleSplit(n_splits=10, train_size = (8.0/9.0), random_state = random_state)


full_data = feature_set_dict[k]['train_x']
full_data_outcome = feature_set_dict[k]['train_y']
index_dict = {
  'parcel_connection':full_data.index
}

# info_df = pd.read_csv(f'{outpath}feature_length_index.csv')

info_index = {
  'subset':[],
  'N_features':[],
  'Method':[]
}


gss_holdout = GroupShuffleSplit(n_splits=1, train_size = .8, random_state = random_state)
idx_1 = gss_holdout.split(
    X = full_data,
    y = feature_set_dict[k]['train_y'],
    groups = full_data['Subject']
  )


for train, test in idx_1:
  train_data = train
  test_data = test


# Split into training and test sets and save the indices into a dict for later
data = full_data#.iloc[holdout_split_dict['train']]
idx_2 = gss_cv.split(
  X = data,
  y = feature_set_dict[k]['train_y'],#.iloc[holdout_split_dict['train']],
  groups = data['Subject']
)
# Will become a dict of tuples containing train, test indices
cv_split_dict = {}
ind = 0 # Arbitrary label for splits starting at 0
for train, test in idx_2:
  cv_split_dict[ind] = (train, test) # save as a tuple of train, test
  ind +=1 

for ind in cv_split_dict.keys():
  np.save(f'{fs_outpath}{k}{sep}{run_uid}_split_{ind}_train.npy', cv_split_dict[ind][0])
  np.save(f'{fs_outpath}{k}{sep}{run_uid}_split_{ind}_test.npy', cv_split_dict[ind][1])

########### Full Feature Set ###############
if False:
  data_dict = {}
  info_index['subset'].append('All')
  info_index['N_features'].append(len(list(full_data.columns)[1:]))
  info_index['Method'].append('None')
  for split_ind in cv_split_dict.keys(): # full set: n_splits from gss object e.g. [0,1,2,3,4] if n_splits = 5 or split_dict[dataset].keys()
    data_dict[split_ind] = {
      'train_x':full_data.iloc[cv_split_dict[split_ind][0]][list(full_data.columns)[1:]],
      'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
      'test_x':full_data.iloc[cv_split_dict[split_ind][1]][list(full_data.columns)[1:]],
      'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
    }
  results_rfc_list = Parallel(n_jobs = job_cap)(
    delayed(run_rfc)(
      train_x = data_dict[split_ind]['train_x'],
      train_y = data_dict[split_ind]['train_y'],
      test_x = data_dict[split_ind]['test_x'],
      test_y = data_dict[split_ind]['test_y'],
      dataset = 'UCLA',
      subset = 'All',
      split_ind = split_ind,
      method='None'
      ) for split_ind in data_dict.keys()
    )
  df_concat_list = []
  for res_dict in results_rfc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
    print(f'Dataset: {res_dict["dataset"]}, subset {res_dict["subset"]}, {res_dict["split_ind"]}')
    print(f'\tTraining Accuracy: {res_dict["train_accuracy"]}')
    print(f'\tTest Accuracy: {res_dict["test_accuracy"]}')
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)
  results_svc_list = Parallel(n_jobs = job_cap)(
    delayed(run_svc)(
      train_x = data_dict[split_ind]['train_x'],
      train_y = data_dict[split_ind]['train_y'],
      test_x = data_dict[split_ind]['test_x'],
      test_y = data_dict[split_ind]['test_y'],
      dataset = 'UCLA',
      subset = 'All',
      split_ind = split_ind,
      method='None'
      ) for split_ind in data_dict.keys()
    )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_svc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]

############ KPCA ##########
if False:
  n_components = 25
  data_dict = {}
  for kernel in ['rbf', 'linear']:
    data_dict[kernel] = {}
    for split_ind in cv_split_dict.keys():
      data_dict[kernel][split_ind] = {
        'train_x':feature_set_dict[k][f'train_kpca-{kernel}'][cv_split_dict[split_ind][0],0:n_components],
        'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
        'test_x':feature_set_dict[k][f'train_kpca-{kernel}'][cv_split_dict[split_ind][1],0:n_components],
        'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
      }
  results_rfc_list = Parallel(n_jobs = job_cap)(
  delayed(run_rfc)(
    train_x = data_dict[kernel][split_ind]['train_x'],
    train_y = data_dict[kernel][split_ind]['train_y'],
    test_x = data_dict[kernel][split_ind]['test_x'],
    test_y = data_dict[kernel][split_ind]['test_y'],
    dataset = 'UCLA',
    subset = f'kPCA_{kernel}-{n_components}',
    split_ind = split_ind,
    method='kPCA'
    ) for split_ind in cv_split_dict.keys() for kernel in ['rbf', 'linear']
  )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_rfc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
    print(f'Dataset: {res_dict["dataset"]}, subset {res_dict["subset"]}, {res_dict["split_ind"]}')
    print(f'\tTraining Accuracy: {res_dict["train_accuracy"]}')
    print(f'\tTest Accuracy: {res_dict["test_accuracy"]}')
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)
  results_svc_list = Parallel(n_jobs = job_cap)(
    delayed(run_svc)(
      train_x = data_dict[kernel][split_ind]['train_x'],
      train_y = data_dict[kernel][split_ind]['train_y'],
      test_x = data_dict[kernel][split_ind]['test_x'],
      test_y = data_dict[kernel][split_ind]['test_y'],
      dataset = 'UCLA',
      subset = f'kPCA_{kernel}-{n_components}',
      split_ind = split_ind,
      method='kPCA'
      ) for split_ind in cv_split_dict.keys() for kernel in ['rbf', 'linear']
    )
  for res_dict in results_svc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

######## TruncatedSVD ##############
if False:
  comp_sizes = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
  data_dict = {}
  for component_size in comp_sizes:
    data_dict[component_size] = {}
    for split_ind in cv_split_dict.keys():
      data_dict[component_size][split_ind] = {
        'train_x':feature_set_dict[k][f'train_tSVD-{component_size}'][cv_split_dict[split_ind][0]],
        'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
        'test_x':feature_set_dict[k][f'train_tSVD-{component_size}'][cv_split_dict[split_ind][1]],
        'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
      }
  results_rfc_list = Parallel(n_jobs = job_cap)(
  delayed(run_rfc)(
    train_x = data_dict[component_size][split_ind]['train_x'],
    train_y = data_dict[component_size][split_ind]['train_y'],
    test_x = data_dict[component_size][split_ind]['test_x'],
    test_y = data_dict[component_size][split_ind]['test_y'],
    dataset = 'UCLA',
    subset = f'TruncatedSVD_{component_size}',
    split_ind = split_ind,
    method='TruncatedSVD'
    ) for split_ind in cv_split_dict.keys() for component_size in comp_sizes
  )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_rfc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
    print(f'Dataset: {res_dict["dataset"]}, subset {res_dict["subset"]}, {res_dict["split_ind"]}')
    print(f'\tTraining Accuracy: {res_dict["train_accuracy"]}')
    print(f'\tTest Accuracy: {res_dict["test_accuracy"]}')
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)
  results_svc_list = Parallel(n_jobs = job_cap)(
    delayed(run_svc)(
      train_x = data_dict[component_size][split_ind]['train_x'],
      train_y = data_dict[component_size][split_ind]['train_y'],
      test_x = data_dict[component_size][split_ind]['test_x'],
      test_y = data_dict[component_size][split_ind]['test_y'],
      dataset = 'UCLA',
      subset = f'TruncatedSVD_{component_size}',
      split_ind = split_ind,
      method='TruncatedSVD'
      ) for split_ind in cv_split_dict.keys() for component_size in comp_sizes
    )
  for res_dict in results_svc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)

######### LDA ##########
if False:
  data_dict = {}
  for split_ind in cv_split_dict.keys():
    data_dict[split_ind] = {
      'train_x':feature_set_dict[k][f'train_LDA'][cv_split_dict[split_ind][0]],
      'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
      'test_x':feature_set_dict[k][f'train_LDA'][cv_split_dict[split_ind][1]],
      'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
    }
  results_rfc_list = Parallel(n_jobs = job_cap)(
  delayed(run_rfc)(
    train_x = data_dict[split_ind]['train_x'],
    train_y = data_dict[split_ind]['train_y'],
    test_x = data_dict[split_ind]['test_x'],
    test_y = data_dict[split_ind]['test_y'],
    dataset = 'UCLA',
    subset = f'LDA',
    split_ind = split_ind,
    method='LDA'
    ) for split_ind in cv_split_dict.keys()
  )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_rfc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
    print(f'Dataset: {res_dict["dataset"]}, subset {res_dict["subset"]}, {res_dict["split_ind"]}')
    print(f'\tTraining Accuracy: {res_dict["train_accuracy"]}')
    print(f'\tTest Accuracy: {res_dict["test_accuracy"]}')
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)
  results_svc_list = Parallel(n_jobs = job_cap)(
    delayed(run_svc)(
      train_x = data_dict[split_ind]['train_x'],
      train_y = data_dict[split_ind]['train_y'],
      test_x = data_dict[split_ind]['test_x'],
      test_y = data_dict[split_ind]['test_y'],
      dataset = 'UCLA',
      subset = f'LDA',
      split_ind = split_ind,
      method='LDA'
      ) for split_ind in cv_split_dict.keys()
    )
  for res_dict in results_svc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)

############ ICA ############### *****
if True:
  data_dict = {}
  for split_ind in cv_split_dict.keys():
    data_dict[split_ind] = {
      'train_x':feature_set_dict[k][f'train_ICA'][cv_split_dict[split_ind][0]],
      'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
      'test_x':feature_set_dict[k][f'train_ICA'][cv_split_dict[split_ind][1]],
      'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
    }
  results_rfc_list = Parallel(n_jobs = job_cap)(
  delayed(run_rfc)(
    train_x = data_dict[split_ind]['train_x'],
    train_y = data_dict[split_ind]['train_y'],
    test_x = data_dict[split_ind]['test_x'],
    test_y = data_dict[split_ind]['test_y'],
    dataset = 'UCLA',
    subset = f'ICA',
    split_ind = split_ind,
    method='ICA'
    ) for split_ind in cv_split_dict.keys()
  )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_rfc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
    print(f'Dataset: {res_dict["dataset"]}, subset {res_dict["subset"]}, {res_dict["split_ind"]}')
    print(f'\tTraining Accuracy: {res_dict["train_accuracy"]}')
    print(f'\tTest Accuracy: {res_dict["test_accuracy"]}')
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)
  results_svc_list = Parallel(n_jobs = job_cap)(
    delayed(run_svc)(
      train_x = data_dict[split_ind]['train_x'],
      train_y = data_dict[split_ind]['train_y'],
      test_x = data_dict[split_ind]['test_x'],
      test_y = data_dict[split_ind]['test_y'],
      dataset = 'UCLA',
      subset = f'ICA',
      split_ind = split_ind,
      method='ICA'
      ) for split_ind in cv_split_dict.keys()
    )
  for res_dict in results_svc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)

####### Hierarchical ################
if False:
  datacols = list(full_data.columns)[1:]
  # list(np.array(datacols)[feature_set_dict[k]['hierarchical_selected_features'][n]])
  data_dict = {}
  for split_ind in cv_split_dict.keys():
    data_dict[split_ind] = {}
    for n in feature_set_dict[k]['hierarchical_selected_features'].keys():
      try:
        data_dict[split_ind][n] = {
          'train_x':full_data.iloc[cv_split_dict[split_ind][0]][np.array(datacols)[feature_set_dict[k]['hierarchical_selected_features'][n]]],
          'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
          'test_x':full_data.iloc[cv_split_dict[split_ind][1]][np.array(datacols)[feature_set_dict[k]['hierarchical_selected_features'][n]]],
          'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
        }
        info_index['subset'].append(f'Hierarchical-{n}')
        info_index['N_features'].append(len(feature_set_dict[k]['hierarchical_selected_features'][n]))
        info_index['Method'].append('Hierarchical Clustering')
      except Exception as e:
        print(f'Error retreiving hierarchical_selected_features {n}:')
        print(e)
  results_rfc_list = Parallel(n_jobs = job_cap)(
    delayed(run_rfc)(
      train_x = data_dict[split_ind][n]['train_x'],
      train_y = data_dict[split_ind][n]['train_y'],
      test_x = data_dict[split_ind][n]['test_x'],
      test_y = data_dict[split_ind][n]['test_y'],
      dataset = 'UCLA',
      subset = f'Hierarchical-{n}',
      split_ind = split_ind,
      method='Hierarchical Clustering'
      ) for split_ind in data_dict.keys() for n in feature_set_dict[k]['hierarchical_selected_features'].keys()
    )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_rfc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
    print(f'Dataset: {res_dict["dataset"]}, subset {res_dict["subset"]}, {res_dict["split_ind"]}')
    print(f'\tTraining Accuracy: {res_dict["train_accuracy"]}')
    print(f'\tTest Accuracy: {res_dict["test_accuracy"]}')
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  results_svc_list = Parallel(n_jobs = job_cap)(
    delayed(run_svc)(
      train_x = data_dict[split_ind][n]['train_x'],
      train_y = data_dict[split_ind][n]['train_y'],
      test_x = data_dict[split_ind][n]['test_x'],
      test_y = data_dict[split_ind][n]['test_y'],
      dataset = 'UCLA',
      subset = f'Hierarchical-{n}',
      split_ind = split_ind,
      method='Hierarchical Clustering'
      ) for split_ind in data_dict.keys() for n in feature_set_dict[k]['hierarchical_selected_features'].keys()
    )
  for res_dict in results_svc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)

######### PCA ##########
if False:
  # info_index['subset'].append('PCA')
  # info_index['N_features'].append(len(list(feature_set_dict[k]['train_pca'].columns)))
  # info_index['Method'].append('PCA')
  data_dict = {}
  for split_ind in cv_split_dict.keys():
    data_dict[split_ind] = {
      'train_x':feature_set_dict[k]['train_pca'][cv_split_dict[split_ind][0]],
      'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
      'test_x':feature_set_dict[k]['train_pca'][cv_split_dict[split_ind][1]],
      'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
    }
  results_rfc_list = Parallel(n_jobs = job_cap)(
    delayed(run_rfc)(
      train_x = data_dict[split_ind]['train_x'],
      train_y = data_dict[split_ind]['train_y'],
      test_x = data_dict[split_ind]['test_x'],
      test_y = data_dict[split_ind]['test_y'],
      dataset = 'UCLA',
      subset = 'PCA Full',
      split_ind = split_ind,
      method='PCA'
      ) for split_ind in data_dict.keys()
    )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_rfc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
    print(f'Dataset: {res_dict["dataset"]}, subset {res_dict["subset"]}, {res_dict["split_ind"]}')
    print(f'\tTraining Accuracy: {res_dict["train_accuracy"]}')
    print(f'\tTest Accuracy: {res_dict["test_accuracy"]}')
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)
  results_svc_list = Parallel(n_jobs = job_cap)(
    delayed(run_svc)(
      train_x = data_dict[split_ind]['train_x'],
      train_y = data_dict[split_ind]['train_y'],
      test_x = data_dict[split_ind]['test_x'],
      test_y = data_dict[split_ind]['test_y'],
      dataset = 'UCLA',
      subset = 'PCA Full',
      split_ind = split_ind,
      method='PCA'
      ) for split_ind in data_dict.keys()
    )
  for res_dict in results_svc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)
  # # length_set = set(length_list3)

# Iterate through lengths with PCA data
if False:
  data_dict = {}
  for split_ind in cv_split_dict.keys():
    data_dict[split_ind] = {}
    for length in set(length_list3):
      if length<feature_set_dict[k]['train_pca'].shape[1]:
        data_dict[split_ind][length] = {
          'train_x':feature_set_dict[k]['train_pca'][cv_split_dict[split_ind][0], 0:length],
          'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
          'test_x':feature_set_dict[k]['train_pca'][cv_split_dict[split_ind][1], 0:length],
          'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
        }
  results_rfc_list = Parallel(n_jobs = job_cap)(
    delayed(run_rfc)(
      train_x = data_dict[split_ind][length]['train_x'],
      train_y = data_dict[split_ind][length]['train_y'],
      test_x = data_dict[split_ind][length]['test_x'],
      test_y = data_dict[split_ind][length]['test_y'],
      dataset = 'UCLA',
      subset = f'PCA_{length}',
      split_ind = split_ind,
      method='PCA'
      ) for split_ind in data_dict.keys() for length in data_dict[split_ind].keys()
    )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_rfc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
    print(f'Dataset: {res_dict["dataset"]}, subset {res_dict["subset"]}, {res_dict["split_ind"]}')
    print(f'\tTraining Accuracy: {res_dict["train_accuracy"]}')
    print(f'\tTest Accuracy: {res_dict["test_accuracy"]}')
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)
  results_svc_list = Parallel(n_jobs = job_cap)(
    delayed(run_svc)(
      train_x = data_dict[split_ind][length]['train_x'],
      train_y = data_dict[split_ind][length]['train_y'],
      test_x = data_dict[split_ind][length]['test_x'],
      test_y = data_dict[split_ind][length]['test_y'],
      dataset = 'UCLA',
      subset = f'PCA_{length}',
      split_ind = split_ind,
      method='PCA'
      ) for split_ind in data_dict.keys() for length in data_dict[split_ind].keys()
    )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_svc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)

######### SelectFromModel ###########
if False:
  data_dict = {}
  for n in length_list3:
    data_dict[n] = {}
    try:
      # print(n, len(feature_set_dict[k][f'rf_selected_n{n}']))
      input_subset = full_data[feature_set_dict[k][f'rf_selected_n{n}']]
      info_index['subset'].append(f'rf_selected_n{n}')
      info_index['N_features'].append(n)
      info_index['Method'].append('SelectFromModel')
      for split_ind in cv_split_dict.keys():
        data_dict[n][split_ind] = {
          'train_x':input_subset.iloc[cv_split_dict[split_ind][0]],
          'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
          'test_x':input_subset.iloc[cv_split_dict[split_ind][1]],
          'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
        }
    except Exception as e:
      print(e)
  results_rfc_list = Parallel(n_jobs = job_cap)(
    delayed(run_rfc)(
      train_x = data_dict[n][split_ind]['train_x'],
      train_y = data_dict[n][split_ind]['train_y'],
      test_x = data_dict[n][split_ind]['test_x'],
      test_y = data_dict[n][split_ind]['test_y'],
      dataset = 'UCLA',
      subset = f'rf_selected_n{n}',
      split_ind = split_ind,
      method='Select From Model'
      ) for split_ind in cv_split_dict.keys() for n in data_dict.keys()
    )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_rfc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
    print(f'Dataset: {res_dict["dataset"]}, subset {res_dict["subset"]}, {res_dict["split_ind"]}')
    print(f'\tTraining Accuracy: {res_dict["train_accuracy"]}')
    print(f'\tTest Accuracy: {res_dict["test_accuracy"]}')
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)
  # Come back to this, some caused a 4hr wait so I cancelled with more than 90% of them run
  results_svc_list = Parallel(n_jobs = job_cap)(
    delayed(run_svc)(
      train_x = data_dict[n][split_ind]['train_x'],
      train_y = data_dict[n][split_ind]['train_y'],
      test_x = data_dict[n][split_ind]['test_x'],
      test_y = data_dict[n][split_ind]['test_y'],
      dataset = 'UCLA',
      subset = f'rf_selected_n{n}',
      split_ind = split_ind,
      method='Select From Model'
      ) for split_ind in cv_split_dict.keys() for n in data_dict.keys()
    )
  for res_dict in results_svc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
  accuracy_df = pd.concat(df_concat_list)
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

########### Permutation Importance ################
if False:
  n_estimators = 500
  n_repeats = 50
  ranked_features = feature_set_dict[k][f'feature_importances_{n_estimators}']
  feature_names = full_data.columns
  rank_df = pd.DataFrame({
    'feature':list(full_data.columns)[1:],
    'importance':ranked_features
  })
  rank_df.sort_values(by='importance', ascending=False,inplace=True)
  ordered_features = list(rank_df['feature'])
  data_dict = {}
  for length in length_list3:
    data_dict[length] = {}
    input_subset = full_data[ordered_features[:length]]
    info_index['subset'].append(f'Permutation-importance_{length}')
    info_index['N_features'].append(length)
    info_index['Method'].append('Permutation Importance')
    for split_ind in cv_split_dict.keys():
      data_dict[length][split_ind] = {
        'train_x':input_subset.iloc[cv_split_dict[split_ind][0]],
        'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
        'test_x':input_subset.iloc[cv_split_dict[split_ind][1]],
        'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
      }
  results_rfc_list = Parallel(n_jobs = job_cap)(
    delayed(run_rfc)(
      train_x = data_dict[length][split_ind]['train_x'],
      train_y = data_dict[length][split_ind]['train_y'],
      test_x = data_dict[length][split_ind]['test_x'],
      test_y = data_dict[length][split_ind]['test_y'],
      dataset = 'UCLA',
      subset = f'Permutation-Importance_{length}',
      split_ind = split_ind,
      method='Permutation Importance'
      ) for split_ind in cv_split_dict.keys() for length in length_list3
    )
  accuracy_df = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
  df_concat_list = [accuracy_df]
  for res_dict in results_rfc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
    print(f'Dataset: {res_dict["dataset"]}, subset {res_dict["subset"]}, {res_dict["split_ind"]}')
    print(f'\tTraining Accuracy: {res_dict["train_accuracy"]}')
    print(f'\tTest Accuracy: {res_dict["test_accuracy"]}')
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  # accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)
  results_svc_list = Parallel(n_jobs = job_cap)(
    delayed(run_svc)(
      train_x = data_dict[length][split_ind]['train_x'],
      train_y = data_dict[length][split_ind]['train_y'],
      test_x = data_dict[length][split_ind]['test_x'],
      test_y = data_dict[length][split_ind]['test_y'],
      dataset = 'UCLA',
      subset = f'Permutation-Importance_{length}',
      split_ind = split_ind,
      method='Permutation Importance'
      ) for split_ind in cv_split_dict.keys() for length in length_list3
    )
  for res_dict in results_svc_list:
    df_concat_list.append(pd.DataFrame(res_dict))
  accuracy_df = pd.concat(df_concat_list)
  print(accuracy_df)
  datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
  accuracy_df.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
  accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

#####################################################################################

LDA_result_rfc = run_rfc(
  train_x = feature_set_dict[k][f'train_LDA'],
  train_y = feature_set_dict[k]['train_y'].values.ravel().astype('int'),
  test_x = feature_set_dict[k][f'test_LDA'],
  test_y = feature_set_dict[k]['test_y'].values.ravel().astype('int'),
  dataset = 'UCLA',
  subset = f'LDA',
  split_ind = 'Validation',
  method='LDA'
)

LDA_result_svc = run_svc(
  train_x = feature_set_dict[k][f'train_LDA'],
  train_y = feature_set_dict[k]['train_y'].values.ravel().astype('int'),
  test_x = feature_set_dict[k][f'test_LDA'],
  test_y = feature_set_dict[k]['test_y'].values.ravel().astype('int'),
  dataset = 'UCLA',
  subset = f'LDA',
  split_ind = 'Validation',
  method='LDA'
)


accuracy_df[accuracy_df['FS/FR Method']=='PCA'] # wow
accuracy_df['FS/FR Method'].unique()