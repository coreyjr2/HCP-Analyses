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
  '-local_path','/mnt/usb1/hcp_analysis_output/',#'C:\\Users\\kyle\\temp\\',
  '--output','/mnt/usb1/hcp_analysis_output/',#'C:\\Users\\kyle\\output\\',
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
  # 'parcel_sum':{
  # },
  # 'network_sum':{
  # },
  'parcel_connection':{
  }#,
  # 'network_connection':{
  # }
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

def run_svc(train_x, train_y, test_x, test_y, dataset, subset, split_ind, C=1):
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
  results_dict = {
    'dataset':[dataset],
    'subset':[subset],
    'split_ind':[split_ind],
    'Classifier':['Support Vector Machine'],
    'train_accuracy':[training_accuracy],
    'test_accuracy':[test_accuracy],
    #'with_que':[metadata['with_que']],
    'ICA_Aroma':[metadata['ICA-Aroma']],
    'metadata_ref':[pred_uid],
    'runtime':(end_time - start_time).total_seconds()
    # 'classification_report':classification_rep,
    # 'confusion_matrix':confusion_mat
  }
  clf_param_df.to_csv(f'{weight_path}{pred_uid}_weights.csv', index=False)
  np.savetxt(f'{confusion_path}{pred_uid}_confusion_matrix.csv', confusion_mat, delimiter=",")
  np.savetxt(f'{classification_path}{pred_uid}_classification_report.csv', confusion_mat, delimiter=",")
  return results_dict

def run_rfc(train_x, train_y, test_x, test_y, dataset, subset, split_ind, n_estimators=500, max_depth = None):
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
  results_dict = {
    'dataset':[dataset],
    'subset':[subset],
    'split_ind':[split_ind],
    'Classifier':['Random Forest'],
    'train_accuracy':[training_accuracy],
    'test_accuracy':[test_accuracy],
    #'with_que':[metadata['with_que']],
    'ICA_Aroma':[metadata['ICA-Aroma']],
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

# idx_1 = gss_holdout.split(
#     X = full_data,
#     y = feature_set_dict[k]['train_y'],
#     groups = full_data['Subject']
#   )
# holdout_split_dict = {}
# for train, test in idx_1:
#   holdout_split_dict['train'] = train
#   holdout_split_dict['test'] = test


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


# Full Feature Set
data_dict = {}
for split_ind in cv_split_dict.keys(): # full set: n_splits from gss object e.g. [0,1,2,3,4] if n_splits = 5 or split_dict[dataset].keys()
  data_dict[split_ind] = {
    'train_x':full_data.iloc[cv_split_dict[split_ind][0]][list(full_data.columns)[1:]],
    'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
    'test_x':full_data.iloc[cv_split_dict[split_ind][1]][list(full_data.columns)[1:]],
    'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
  }

results_rfc_list = Parallel(n_jobs = 14)(
  delayed(run_rfc)(
    train_x = data_dict[split_ind]['train_x'],
    train_y = data_dict[split_ind]['train_y'],
    test_x = data_dict[split_ind]['test_x'],
    test_y = data_dict[split_ind]['test_y'],
    dataset = 'HCP_1200',
    subset = 'All',
    split_ind = split_ind
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
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

results_svc_list = Parallel(n_jobs = 14)(
  delayed(run_svc)(
    train_x = data_dict[split_ind]['train_x'],
    train_y = data_dict[split_ind]['train_y'],
    test_x = data_dict[split_ind]['test_x'],
    test_y = data_dict[split_ind]['test_y'],
    dataset = 'HCP_1200',
    subset = 'All',
    split_ind = split_ind
    ) for split_ind in data_dict.keys()
  )

for res_dict in results_svc_list:
  df_concat_list.append(pd.DataFrame(res_dict))

accuracy_df = pd.concat(df_concat_list)

print(accuracy_df)
datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

# Hierarchical
datacols = list(full_data.columns)[1:]
list(np.array(datacols)[feature_set_dict[k]['hierarchical_selected_features'][n]])
hierarchical_start = 1
hierarchical_end = 30
data_dict = {}
for split_ind in cv_split_dict.keys():
  data_dict[split_ind] = {}
  for n in range(hierarchical_start, hierarchical_end):
    data_dict[split_ind][n] = {
      'train_x':full_data.iloc[cv_split_dict[split_ind][0]][np.array(datacols)[feature_set_dict[k]['hierarchical_selected_features'][n]]],
      'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
      'test_x':full_data.iloc[cv_split_dict[split_ind][1]][np.array(datacols)[feature_set_dict[k]['hierarchical_selected_features'][n]]],
      'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
    }

results_rfc_list = Parallel(n_jobs = 14)(
  delayed(run_rfc)(
    train_x = data_dict[split_ind][n]['train_x'],
    train_y = data_dict[split_ind][n]['train_y'],
    test_x = data_dict[split_ind][n]['test_x'],
    test_y = data_dict[split_ind][n]['test_y'],
    dataset = 'HCP_1200',
    subset = f'Hierarchical-{n}',
    split_ind = split_ind
    ) for split_ind in data_dict.keys() for n in range(hierarchical_start, hierarchical_end)
  )

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
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

results_svc_list = Parallel(n_jobs = 14)(
  delayed(run_svc)(
    train_x = data_dict[split_ind][n]['train_x'],
    train_y = data_dict[split_ind][n]['train_y'],
    test_x = data_dict[split_ind][n]['test_x'],
    test_y = data_dict[split_ind][n]['test_y'],
    dataset = 'HCP_1200',
    subset = f'Hierarchical-{n}',
    split_ind = split_ind
    ) for split_ind in data_dict.keys() for n in range(hierarchical_start, hierarchical_end)
  )

for res_dict in results_svc_list:
  df_concat_list.append(pd.DataFrame(res_dict))

accuracy_df = pd.concat(df_concat_list)

print(accuracy_df)
datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)


# accuracy_df_1 = pd.read_csv(f'{outpath}Prediction_Accuracies_03-30-2022_08-21-51.csv')
# accuracy_df = pd.concat([accuracy_df_1, accuracy_df])

# PCA
data_dict = {}
for split_ind in cv_split_dict.keys():
  data_dict[split_ind] = {
    'train_x':feature_set_dict[k]['train_pca'][cv_split_dict[split_ind][0]],
    'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
    'test_x':feature_set_dict[k]['train_pca'][cv_split_dict[split_ind][1]],
    'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
  }


results_rfc_list = Parallel(n_jobs = 10)(
  delayed(run_rfc)(
    train_x = data_dict[split_ind]['train_x'],
    train_y = data_dict[split_ind]['train_y'],
    test_x = data_dict[split_ind]['test_x'],
    test_y = data_dict[split_ind]['test_y'],
    dataset = 'HCP_1200',
    subset = 'PCA',
    split_ind = split_ind
    ) for split_ind in data_dict.keys()
  )

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
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

results_svc_list = Parallel(n_jobs = 10)(
  delayed(run_svc)(
    train_x = data_dict[split_ind]['train_x'],
    train_y = data_dict[split_ind]['train_y'],
    test_x = data_dict[split_ind]['test_x'],
    test_y = data_dict[split_ind]['test_y'],
    dataset = 'HCP_1200',
    subset = 'PCA',
    split_ind = split_ind
    ) for split_ind in data_dict.keys()
  )

for res_dict in results_svc_list:
  df_concat_list.append(pd.DataFrame(res_dict))

accuracy_df = pd.concat(df_concat_list)

print(accuracy_df)
datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

length_list = [
  # 19900, #1
  # 19884, #2
  # 18540, #3
  # 13981, #4
  # 10027, #5
  # 7482, #6
  5847, #7
  4638, #8
  3771, #9
  3149, #10
  2679, #11
  2297, #12
  1980, #13
  1725, #14
  1533, #15
  1354, #16
  1205, #17
  1078, #18
  960, #19
  876, #20
  807, #21
  734, #22
  676, #23
  622, #24
  569, #25
  522, #26
  480, #27
  443, #28
  414 #29
]

# Iterate through lengths with PCA data
data_dict = {}
for split_ind in cv_split_dict.keys():
  data_dict[split_ind] = {}
  for length in length_list:
    if length<feature_set_dict[k]['train_pca'].shape[1]:
      data_dict[split_ind][length] = {
        'train_x':feature_set_dict[k]['train_pca'][cv_split_dict[split_ind][0], 0:length],
        'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
        'test_x':feature_set_dict[k]['train_pca'][cv_split_dict[split_ind][1], 0:length],
        'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
      }

results_rfc_list = Parallel(n_jobs = 10)(
  delayed(run_rfc)(
    train_x = data_dict[split_ind][length]['train_x'],
    train_y = data_dict[split_ind][length]['train_y'],
    test_x = data_dict[split_ind][length]['test_x'],
    test_y = data_dict[split_ind][length]['test_y'],
    dataset = 'HCP_1200',
    subset = f'PCA_{length}',
    split_ind = split_ind
    ) for split_ind in data_dict.keys() for length in length_list
  )
#################################
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
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

results_svc_list = Parallel(n_jobs = 10)(
  delayed(run_svc)(
    train_x = data_dict[split_ind][length]['train_x'],
    train_y = data_dict[split_ind][length]['train_y'],
    test_x = data_dict[split_ind][length]['test_x'],
    test_y = data_dict[split_ind][length]['test_y'],
    dataset = 'HCP_1200',
    subset = f'PCA_{length}',
    split_ind = split_ind
    ) for split_ind in data_dict.keys() for length in length_list
  )

for res_dict in results_svc_list:
  df_concat_list.append(pd.DataFrame(res_dict))

accuracy_df = pd.concat(df_concat_list)

print(accuracy_df)
datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)


# SelectFromModel
feature_set_dict[k][f'rf_selected_{n}']
data_dict = {}
for split_ind in cv_split_dict.keys():
  data_dict[split_ind] = {}
  for n in feature_set_dict[k]['hierarchical_selected_features'].keys():
    data_dict[split_ind][n] = {
      'train_x':full_data.iloc[cv_split_dict[split_ind][0]][np.array(datacols)[feature_set_dict[k][f'rf_selected_{n}']]],
      'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
      'test_x':full_data.iloc[cv_split_dict[split_ind][1]][np.array(datacols)[feature_set_dict[k][f'rf_selected_{n}']]],
      'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
    }


results_rfc_list = Parallel(n_jobs = 10)(
  delayed(run_rfc)(
    train_x = data_dict[split_ind][n]['train_x'],
    train_y = data_dict[split_ind][n]['train_y'],
    test_x = data_dict[split_ind][n]['test_x'],
    test_y = data_dict[split_ind][n]['test_y'],
    dataset = 'HCP_1200',
    subset = f'rf_selected_{n}',
    split_ind = split_ind
    ) for split_ind in data_dict.keys() for n in feature_set_dict[k]['hierarchical_selected_features'].keys()
  )

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
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

results_svc_list = Parallel(n_jobs = 10)(
  delayed(run_svc)(
    train_x = data_dict[split_ind][n]['train_x'],
    train_y = data_dict[split_ind][n]['train_y'],
    test_x = data_dict[split_ind][n]['test_x'],
    test_y = data_dict[split_ind][n]['test_y'],
    dataset = 'HCP_1200',
    subset = f'rf_selected_{n}',
    split_ind = split_ind
    ) for split_ind in data_dict.keys() for n in feature_set_dict[k]['hierarchical_selected_features'].keys()
  )

for res_dict in results_svc_list:
  df_concat_list.append(pd.DataFrame(res_dict))

accuracy_df = pd.concat(df_concat_list)

print(accuracy_df)
datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

length_list = [
  19900, #1
  19884, #2
  18540, #3
  13981, #4
  10027, #5
  7482, #6
  5847, #7
  4638, #8
  3771, #9
  3149, #10
  2679, #11
  2297, #12
  1980, #13
  1725, #14
  1533, #15
  1354, #16
  1205, #17
  1078, #18
  960, #19
  876, #20
  807, #21
  734, #22
  676, #23
  622, #24
  569, #25
  522, #26
  480, #27
  443, #28
  414 #29
]
# Permutation Importance
n_estimators = 500
n_repeats = 50
ranked_features = feature_set_dict[k][f'feature_importances_{n_estimators}']

data_dict = {}
for split_ind in cv_split_dict.keys():
  data_dict[split_ind] = {}
  for length in length_list:
    data_dict[split_ind][length] = {
      'train_x':full_data.iloc[cv_split_dict[split_ind][0]][list(ranked_features)[:length]],
      'train_y':full_data_outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
      'test_x':full_data.iloc[cv_split_dict[split_ind][1]][list(ranked_features)[:length]],
      'test_y':full_data_outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
    }


results_rfc_list = Parallel(n_jobs = 10)(
  delayed(run_rfc)(
    train_x = data_dict[split_ind][length]['train_x'],
    train_y = data_dict[split_ind][length]['train_y'],
    test_x = data_dict[split_ind][length]['test_x'],
    test_y = data_dict[split_ind][length]['test_y'],
    dataset = 'HCP_1200',
    subset = f'Permutation-Importance_{length}',
    split_ind = split_ind
    ) for split_ind in data_dict.keys() for length in length_list
  )

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
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies_{datetime_str}.csv', index=False)

results_svc_list = Parallel(n_jobs = 10)(
  delayed(run_svc)(
    train_x = data_dict[split_ind][length]['train_x'],
    train_y = data_dict[split_ind][length]['train_y'],
    test_x = data_dict[split_ind][length]['test_x'],
    test_y = data_dict[split_ind][length]['test_y'],
    dataset = 'HCP_1200',
    subset = f'Permutation-Importance_{length}',
    split_ind = split_ind
    ) for split_ind in data_dict.keys() for length in length_list
  )

for res_dict in results_svc_list:
  df_concat_list.append(pd.DataFrame(res_dict))

accuracy_df = pd.concat(df_concat_list)

print(accuracy_df)
datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
accuracy_df.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)



if False:
  # ## ! DO NOT RUN; WILL RUN FOREVER ! ##
  # train_x = feature_set_dict[k]['train_x']
  # train_y = feature_set_dict[k]['train_y'].values.ravel()
  # test_x = feature_set_dict[k]['test_x']
  # test_y = feature_set_dict[k]['test_y'].values.ravel()
  # # Iterate through feature sets with same sizes 
  # # Levels of hierarchical selection:
  # h_levels = list(feature_set_dict[k]['hierarchical_selected_features'].keys())
  # h_levels.reverse()
  # # subset h_levels
  # h_levels = h_levels[:3]
  # for level in h_levels:
  #   feature_index = feature_set_dict[k]['hierarchical_selected_features'][level]
  #   sub_train_x = feature_set_dict[k]['train_x'][feature_set_dict[k]['train_x'].columns[feature_index]]
  #   sub_test_x = feature_set_dict[k]['test_x'][feature_set_dict[k]['train_x'].columns[feature_index]]
  #   n_folds = 5
  #   opt = BayesSearchCV(
  #     SVC(),
  #     {
  #       'C': Real(1e-6, 1e+6, prior='log-uniform'),
  #       'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
  #       'degree': Integer(1,8),
  #       'kernel': Categorical(['linear', 'poly', 'rbf']),
  #     },
  #     n_iter=32,
  #     random_state=0,
  #     refit=True,
  #     cv=n_folds,
  #     n_jobs = int(args.n_jobs)
  #   )
  #   res = opt.fit(sub_train_x, train_y)
  #   print(opt.score(sub_test_x, test_y))