#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Info:
  Outline:
    1. Computation node (COMP) sends an SCP request to the Data node (DAT)
    2. Either COMP received that data and stores it in a temporary directory OR the process will fail here.
    3. (INSERT ANALYSIS HERE)
    4. Any output data will be transfered back to the output path specified on DAT.
    5. COMP wipes the files downloaded in step 2 and any additional output files
  Important notes:
    - All data needed for the analysis (and nothing else) should be in a single directory for easiest use. 
      * TODO: Enable more dunamic file selection (from text file?)
    - If you can, you should use data stored locally already or a cluster-wide data source to reduce the time cost of transferring the data back and forth.
    - This pipeline allows for a net storage footprint of 0 on COMP after the conclusion of the analysis. If grid-level parallelization using CGE is not possible for some reason, this provides an alternative method without incurring additional storage costs.
    - This sript is expected to be used with command line arguements.
    - If you can paralellize at the command level (i.e. using CGE), you should do so. This is for use where there is a python implementation of paralellization naieve to the cluster infrastucture in your script and you would like to run different versions simmultaneously. 
    - The initial trigger of this analysis will require the user to input a password to SCP the source data to the local drive specified
  Authors:
    kabaacke-psy, cjrichier 

'''

'''
Runs:
  python3 HCP_analysis_vCluster_001.py \
    -source_path /mnt/usb1/HCP_1200/HCP_69354adf \
    -uname kbaacke \
    -datahost r2.psych.uiuc.edu \
    -local_path /home/kbaacke/temp/ \ #Change to one of the locations that Kevin specified or /mnt/usb1 in the case of r2
    --output /home/kbaacke/output/ \
    --n_jobs 7 \
    -atlas_path /mnt/usb1/HCP_1200/HCP_69354adf/69354adf_parcellation-metadata.json \
    --confounds_subset Subject "Age__22-25" "Age__26-30" "Age__31-35" "Age__36+" Gender__F Gender__M
    

'''
v1_argslist = [
  '-source_path', '/mnt/usb1/HCP_1200/HCP_69354adf',
  '-uname', 'kbaacke',
  '-datahost', 'r2.psych.uiuc.edu ',
  '-local_path', '/home/kbaacke/temp/',
  '--output', '/home/kbaacke/output/',
  '--n_jobs', '7',
  '-atlas_path', '/mnt/usb1/HCP_1200/HCP_69354adf/69354adf_parcellation-metadata.json',
  '--confounds_subset', 'Subject', 'Age__22-25', 'Age__26-30', 'Age__31-35', 'Age__36+', 'Gender__F', 'Gender__M'
]
# TODO: dafualt immplicit parcellation metadata search in source_path

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
  from scp import SCPClient
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
  from pathlib import Path
  from statsmodels.stats.outliers_influence import variance_inflation_factor
  import matplotlib.pyplot as plt
  import seaborn as sns
  from scipy.stats import spearmanr
  from scipy.cluster import hierarchy
  from collections import defaultdict
  import pickle as pk
except Exception as e:
  print(f'Error loading libraries: ')
  raise Exception(e)


# Global Variables
sep = os.path.sep
source_path = os.path.dirname(os.path.abspath(__file__)) + sep
sys_name = platform.system() 
hostname = platform.node()


# Template Functions
try:
  def parse_args(args):
    #Presets
    parser = argparse.ArgumentParser(
        description=''
      )
    parser.add_argument(
      "-source_path", help='Full base-path on the Data Node where the data is stored.', required=True
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
      "--output", help=f'Full path to the location ot store the output. Defaults to the source path fo this python file + \'Output{sep}\'', required=False
    )
    parser.add_argument(
      "--n_jobs", help="Specify number of CPUs to use in the analysis. Default is 1. Set to -1 to use all available cores.", required=False, default=1
    )
    # Context Specific
    parser.add_argument(
      "-atlas_path",
      help=f'''
        Full path to the metadata json file for the parcellation.\nEx: S:{sep}HCP{sep}HCP_83e395d0{sep}83e395d0_parcellation-metadata.json
      '''
    )
    parser.add_argument(
      '--confounds',
      help='Full base-path on the Data Node where the confounds/demographics .csv file is located. Use \'None\' to indicate that no confounds should be used.\nDefault is \'{source_path}demographics_with_dummy_vars.csv\'', required=False, default = source_path + 'demographics_with_dummy_vars.csv'
    )
    parser.add_argument(
      '--confound_subset',
      help='''
        Use to specify a list of columns present in the confounds file that you would like to include in the analysis pipeline. By default, all will be included.\nOptions in deafult file:
        \'Subject\',
        \'Age__22-25\',
        \'Age__26-30\',
        \'Age__31-35\',
        \'Age__36+\',
        \'Gender__F\',
        \'Gender__M\',
        \'Acquisition__Q01\',
        \'Acquisition__Q02\',
        \'Acquisition__Q03\',
        \'Acquisition__Q04\',
        \'Acquisition__Q05\',
        \'Acquisition__Q06\',
        \'Acquisition__Q07\',
        \'Acquisition__Q08\',
        \'Acquisition__Q09\',
        \'Acquisition__Q10\',
        \'Acquisition__Q11\',
        \'Acquisition__Q12\',
        \'Acquisition__Q13\'
      ''',nargs='*',required=False
    )
    parser.add_argument(
      '--movement_regressor',
      help='Full base-path on the Data Node where the movement regressor .csv file is located. Use \'None\' to indicate that no confounds should be used.\nDefault is \'{source_path}relative_RMS_means_collapsed.csv\'', required=False, default = source_path + 'relative_RMS_means_collapsed.csv'
    )
    parser.set_defaults(
      ica_aroma = False,
      smoothed=False,
      concatenate = True,
    )
    parser.add_argument(
      '--ica_aroma',
      help='''
        Indicates on metatdata json that ICA-Aroma has been performed on the data prior to parcellation.
      ''',
      dest = 'ica_aroma',
      action = 'store_true',
      required=False
    )
    parser.add_argument(
      '--smoothed',
      help='''
        Indicates on metatdata json that the data has been smoothed prior to parcellation.
      ''',
      dest = 'smoothed',
      action = 'store_true',
      required=False
    )
    parser.add_argument(
      '--no_concatenate',
      help='''
        Indicates that analysis should not be conducted on timeseries generated by concatenating each subjects RL and LR scans. Instead, LR and RL scans will be treated as seperates data points for an each subject. 
      ''',
      dest = 'concatenate',
      action = 'store_false',
      required=False
    )
    return parser.parse_known_args(args)

  def createSSHClient(server, user, password, port=22):
      client = paramiko.SSHClient()
      client.load_system_host_keys()
      client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
      client.connect(server, port, user, password)
      return client

  def generate_uid(meta_dict, hash_len=6):
    dhash = hashlib.md5()
    encoded = json.dumps(meta_dict, sort_keys=True).encode()
    dhash.update(encoded)
    run_uid = dhash.hexdigest()[:hash_len]
    return run_uid
except Exception as e:
  print(f'Error defining template functions: ')
  raise Exception(e)


# Custom Functions
try:
  def create_ordered_network_labels():
    gregions = pd.DataFrame(np.load(source_path + "glasser_regions.npy"), columns=['Label','network','unkown'])
    gregions = gregions[['Label','network']]
    glabels = pd.read_csv(source_path + 'Glasser_labels.csv')
    full_label_file = pd.merge(glabels, gregions, how='left',on='Label')
    full_label_file.to_csv(source_path + 'mni_glasser_info.csv', index=False)

  def load_parcellated_task_timeseries_v2(meta_dict,subjects, session, npy_template, run_names = ['RL','LR'], confounds_path = None): # un-Tested
    # New version that only takes parcellated data
    remove_mean = meta_dict['subtract parcel-wise mean']
    atlas_name = meta_dict['atlas_name']
    concatenate = meta_dict['concatenate']
    parcellated_dict = {}
    concat_dict = {}
    print('Loading in parcellated data for task: ', session)
    for subject in subjects:
      try:
        #print('\t',subject)
        sub_dict = {}
        for run in run_names:
          # First try to load in numpy file
          masked_timeseries = np.load(npy_template.format(subject=subject, session=session, run=run, atlas_name=atlas_name))
          if remove_mean:
            masked_timeseries -= masked_timeseries.mean(axis=1, keepdims=True)
          sub_dict[run] = masked_timeseries
        if concatenate:
          concat_dict[subject] = np.vstack((sub_dict[run_names[0]], sub_dict[run_names[1]]))
        parcellated_dict[subject] = sub_dict
      except Exception as e:
        #print(f'Subject {subject} is not available: {e}')
        pass
    if concatenate:
      return concat_dict
    else:
      return parcellated_dict

  def generate_parcel_input_features(parcellated_data, labels): # Tested
    out_dict = {}
    out_df_dict = {}
    for session in parcellated_data.keys():
      parcel_dict = parcellated_data[session]
      out_dict[session] = {}
      for subject in parcel_dict.keys():#(284, 78)
        ts = parcel_dict[subject]
        out_dict[session][subject] = list(np.mean(ts.T, axis=1))
        out_dict[session][subject].append(subject)
      labels2 = list(labels)
      labels2.append('Subject')
      out_df_dict[session] = pd.DataFrame.from_dict(out_dict[session], orient='index', columns = list(labels2))
      sub = out_df_dict[session]['Subject']
      out_df_dict[session].drop(labels=['Subject'], axis=1, inplace=True)
      out_df_dict[session].insert(0, 'Subject', sub)
      out_df_dict[session].insert(0, 'task',numeric_task_ref[session])
    parcels_full = pd.DataFrame(pd.concat(list(out_df_dict.values()), axis = 0))
    return parcels_full

  def generate_network_input_features(parcels_full, networks): # Tested
    X_network = parcels_full.copy()[parcels_full.columns[2:]]
    X_network.rename(
      columns={i:j for i,j in zip(X_network.columns,networks)}, inplace=True
    )
    X_network = X_network.groupby(lambda x:x, axis=1).sum()
    parcels_full.reset_index(drop=True, inplace=True)
    X_network.reset_index(drop=True, inplace=True)
    X_network.insert(0, 'Subject',parcels_full['Subject'])
    X_network['Subject'] = parcels_full['Subject']
    X_network.insert(0, 'task',parcels_full['task'])
    X_network['task'] = parcels_full['task']
    return X_network

  def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

  def scale_subset(df, cols_to_exclude):
    df_excluded = df[cols_to_exclude]
    df_temp = df.drop(cols_to_exclude, axis=1, inplace=False)
    df_temp = mean_norm(df_temp)
    df_ret = pd.concat([df_excluded, df_temp], axis=1, join='inner')
    return df_ret

  def connection_names(corr_matrix, labels): # Tested
    name_idx = np.triu_indices_from(corr_matrix, k=1)
    out_list = []
    for i in range(len(name_idx[0])):
      out_list.append(str(labels[name_idx[0][i]]) + '|' + str(labels[name_idx[1][i]]))
    return out_list

  def generate_parcel_connection_features(parcellated_data, labels): # Tested
    out_dict = {}
    out_df_dict = {}
    for session in parcellated_data.keys():
      parcel_dict = parcellated_data[session]
      out_dict[session] = {}
      for subject in parcel_dict.keys():
        ts = parcel_dict[subject]
        cor_coef = np.corrcoef(ts.T)
        out_dict[session][subject] = list(cor_coef[np.triu_indices_from(cor_coef, k=1)])
        out_dict[session][subject].append(subject)
        colnames = connection_names(cor_coef, labels)
        colnames.append('Subject')
      out_df_dict[session] = pd.DataFrame.from_dict(out_dict[session], orient='index', columns = colnames)
      sub = out_df_dict[session]['Subject']
      out_df_dict[session].drop(labels=['Subject'], axis=1, inplace=True)
      out_df_dict[session].insert(0, 'Subject', sub)
      out_df_dict[session].insert(0, 'task',numeric_task_ref[session])
    parcels_connections_full = pd.DataFrame(pd.concat(list(out_df_dict.values()), axis = 0))
    return parcels_connections_full

  def generate_network_connection_features(parcellated_data, networks): # Tested
    scaler = StandardScaler() 
    out_dict = {}
    out_df_dict = {}
    for session in parcellated_data.keys():
      parcel_dict = parcellated_data[session]
      out_dict[session] = {}
      for subject in parcel_dict.keys():
        ts = parcel_dict[subject]
        cor_coef = np.corrcoef(scaler.fit_transform(pd.DataFrame(ts, columns = networks).groupby(lambda x:x, axis=1).sum()).T)
        out_dict[session][subject] = list(cor_coef[np.triu_indices_from(cor_coef, k=1)])
        out_dict[session][subject].append(subject)
      colnames = connection_names(cor_coef, pd.DataFrame.from_dict({}, orient='index', columns = networks).groupby(lambda x:x, axis=1).sum().columns)
      colnames.append('Subject')
      out_df_dict[session] = pd.DataFrame.from_dict(out_dict[session], orient='index', columns = colnames)
      sub = out_df_dict[session]['Subject']
      out_df_dict[session].drop(labels=['Subject'], axis=1, inplace=True)
      out_df_dict[session].insert(0, 'Subject', sub)
      out_df_dict[session].insert(0, 'task',numeric_task_ref[session])
    network_connections_full = pd.DataFrame(pd.concat(list(out_df_dict.values()), axis = 0))
    return network_connections_full

  def XY_split(df, outcome_col, excluded = []): # Tested
    excluded.append(outcome_col)
    dfx = df.drop(labels=excluded, axis=1, inplace=False)
    dfy = df[[outcome_col]]
    return dfx, dfy

  def hierarchical_fs(x, n_sub):
    # Returns an index of features to be used
    corr = spearmanr(x).correlation
    corr_linkage = hierarchy.ward(corr)
    cluster_ids = hierarchy.fcluster(corr_linkage, n_sub, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    return selected_features

  def hierarchical_fs_v2(x, start_level, end_level):
    # Returns an index of features to be used
    corr = spearmanr(x).correlation
    corr_linkage = hierarchy.ward(corr)
    out = {}
    for n in range(start_level, end_level):
      cluster_ids = hierarchy.fcluster(corr_linkage, n, criterion='distance')
      cluster_id_to_feature_ids = defaultdict(list)
      for idx, cluster_id in enumerate(cluster_ids):
          cluster_id_to_feature_ids[cluster_id].append(idx)
      selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
      out[n] = selected_features
    return out

  def pca_fs(train_x, test_x, k_components=None):
    if k_components!=None:
      pca = PCA(k_components).fit(train_x)
    else:
      pca = PCA().fit(train_x)
    train_pca = pca.transform(train_x)
    test_pca = pca.transform(test_x)
    return train_pca, test_pca, pca

  def random_forest_fs(x, y, n_estimators, n_repeats=10, n_jobs=1, max_features = 500):
    #Returns a list of columns to use as features
    sel = SelectFromModel(RandomForestClassifier(n_estimators = n_estimators, n_jobs=n_jobs, random_state=42), max_features=max_features)
    sel.fit(x,y)
    return list(sel.get_support())

  def random_forest_fs_v2(x, y, n_estimators, n_repeats=10, n_jobs=1):
    #Returns a list of columns to use as features
    forest = RandomForestClassifier(random_state=42 ,n_estimators=n_estimators)
    forest.fit(x,y)
    result = permutation_importance(forest, x, y, n_repeats=10, random_state=42, n_jobs=n_jobs)
    forest_importances = pd.Series(result.importances_mean, index=x.columns)
    return forest_importances

  def random_forest_fs_v3(x, y, n_estimators, n_repeats=10, n_jobs=1):
    #Returns a list of columns to use as features
    forest = RandomForestClassifier(random_state=42 ,n_estimators=n_estimators)
    forest.fit(x,y)
    importances = forest.feature_importances_
    return importances

  def fetch_labels(meta_dict, json_path = None):
    if 'glasser' in meta_dict['atlas_name']:
      regions_file = np.load(source_path + "glasser_regions.npy").T
      parcel_labels = regions_file[0]
      network_labels = regions_file[1]
    elif meta_dict['atlas_name'] == 'msdl':
      atlas_MSDL = datasets.fetch_atlas_msdl()
      parcel_labels = atlas_MSDL['labels']
      network_labels = atlas_MSDL['networks']
    elif meta_dict['atlas_name'] == 'mni_glasser':
      info_file = pd.read_csv(source_path + 'mni_glasser_info.csv')
      parcel_labels = info_file['Label']
      network_labels = info_file['networks']
    else:
      # try:
      print(json_path + meta_dict['atlas_name'] + '_parcellation-metadata.json')
      parc_meta_dict = json.load(open(json_path + meta_dict['atlas_name'] + '_parcellation-metadata.json'))
      network_info = pd.read_csv(source_path + 'misc' + sep + parc_meta_dict['atlas_name'] + '.Centroid_RAS.csv')
      info_temp = network_info['ROI Name'].str.split('_', expand=True)
      info_temp.columns = ['Parcelation Name','Hemisphere','ICN','sub1','sub2']
      info_temp['Parcel Names'] = info_temp['sub1'] + '_' + info_temp['sub2']
      info_temp.drop(columns=['sub1','sub2'], axis=1, inplace=True)
      network_info_full = pd.DataFrame.join(info_temp, network_info)
      network_mapping = pd.read_csv(source_path + 'misc' + sep + '7NetworksOrderedNames.csv')
      network_info_full = pd.merge(network_info_full, network_mapping, how='left', on='ICN')
      parcel_labels = network_info_full['ROI Name']
      network_labels = network_info_full['Network Name']
      # except:
      #   raise NotImplementedError
    return parcel_labels, network_labels
except Exception as e:
  print(f'Error defining custom functions: ')
  raise Exception(e)

# Parse Args
# args = parse_args( # Uncomment before running in cli
#   sys.argv[1:]
# )

args, leftovers = parse_args( # Uncomment to get the documentation
  ['-h']
)

# args, leforvers = parse_args(v1_argslist)

# Generate meta-dict
# meta_dict = {
#   'atlas_name' : '69354adf',
#   'smoothed' : False,
#   'ICA-Aroma' : False,
#   'confounds': [],
#   'Random Forest Estimators': 1000,
#   'Random State':42,
#   'subtract parcel-wise mean': True, # Fixing this at true, no cli arg
#   'concatenate':True
# }

#### TODO: Change atlas_name to parcellation UID, import atlas_name from meta_dict
args.atlas_name = os.path.basename(args.atlas_path)[:8] # Pull the atlas_name from the atlas_path variable (UID)

parcellation_dict = json.load(open(args.atlas_path))

'''
actual_atlas_name = parcellation_dict['atlas_name']
'''

meta_dict = {
  'atlas_name' : args.atlas_name,
  'smoothed' : args.smoothed,
  'ICA-Aroma' : args.ica_aroma,
  # 'confounds': args.,
  'Random Forest Estimators': 1000,
  'Random State':42,
  'subtract parcel-wise mean': True, # Fixing this at true, no cli arg
  'concatenate':args.concatenate
}

if args.confounds is not "None":
  demographics = pd.read_csv(args.confounds)
  if args.confounds_subset is not None:
    demographics_dummy = demographics[args.confounds_subset]
    meta_dict['confounds'] = list(args.confounds_subset)
  else:
    meta_dict['confounds'] = demographics.columns
  if args.movement_regressor is "None":
    confounds = demographics

if args.movement_regressor is not "None":
  relative_RMS_means_collapsed = pd.read_csv(args.movement_regressor)
  meta_dict['confounds'] = meta_dict['confounds'] + list(relative_RMS_means_collapsed).columns[2:]
  if args.confounds is "None":
    confounds = relative_RMS_means_collapsed

if (args.movement_regressor is not "None") and (args.confounds is not "None"):
  confounds = pd.merge(relative_RMS_means_collapsed, demographics_dummy, how='left', on='Subject')

run_uid = generate_uid(meta_dict)
outpath = args.output + run_uid + sep
try:
  os.makedirs(outpath)
except Exception as e:
  print(e, 'Output directory already created.')

with open(outpath + run_uid + 'metadata.json', 'w') as outfile:
  json.dump(meta_dict, outfile)

# Interupt request for password and username if none passed
if args.uname == None:
  args.uname = getpass.getpass(f'Username for {args.datahost}:')
args.psswd = getpass.getpass(f'Password for {args.uname}@{args.datahost}:')

# SCP data to temp location
if args.source_path!=None:
  src_basepath = args.source_path
  download_start_time = dt.datetime.now()
  print('Starting Data Transfer: ', download_start_time)
  try:
    ssh = createSSHClient(args.datahost, args.uname, args.psswd)
    scp = SCPClient(ssh.get_transport())
    scp.get(args.source_path, args.localpath, recursive=True)
  except Exception as e:
    print(f'Error transferring data from {args.uname}@{args.datahost} ')
    raise Exception(e)
'''
# First, check output location to see if the analysis has been run before, prompt to continue if true
if os.path.exists(): #TODO, DOES NOT WORK
  print(f'An analysis with this same metadata dictionary has been run: {run_uid}')
  print('Would you like to re-run? (y/n)')
  if not 'y' in input().lower():
    raise Exception('Analyses halted.')
'''

# Run analysis
total_start_time = dt.datetime.now()
logging.basicConfig(filename=f'{run_uid}_DEBUG.log', level=logging.DEBUG) # Set level to DEBUG for detailed output

try:
  logging.debug(f'Architecture: ', platform.architecture()[0], ', ', platform.architecture()[1])
  logging.debug(f'Processor: ', platform.machine())
  logging.debug(f'Node Name: : ', platform.node())
  logging.info(f'Started; {total_start_time}') #Adds a line to the logfile to be exported after analysis
  logging.debug('args: ')
  logging.debug(args)
  logging.debug('meta_dict: ')
  logging.debug(meta_dict)
  '''
  DO THINGS HERE 
  '''

  numeric_task_ref = {
    "tfMRI_MOTOR":4,
    "tfMRI_WM":7,
    "tfMRI_EMOTION":1,
    "tfMRI_GAMBLING":2,
    "tfMRI_LANGUAGE":3,
    "tfMRI_RELATIONAL":5,
    "tfMRI_SOCIAL":6
  }
  basepath = args.local_path
  npy_template_hcp = '{basepath}HCP{sep}HCP_1200{sep}{subject}{sep}MNINonLinear{sep}Results{sep}{session}_{run}{sep}{atlas_name}_{session}_{run}.npy'
  HCP_1200 = f'{basepath}HCP_1200{sep}'
  parcel_labels, network_labels = fetch_labels(meta_dict, HCP_1200)

  subjects = []
  for f in os.listdir(HCP_1200):
    if len(f)==6:
      subjects.append(f)

  # Subset Subjects here to test runtimes:
  # subects = subjects[:500]
  
  sessions = [
    "tfMRI_MOTOR",
    "tfMRI_WM",
    "tfMRI_EMOTION",
    "tfMRI_GAMBLING",
    "tfMRI_LANGUAGE",
    "tfMRI_RELATIONAL",
    "tfMRI_SOCIAL"
  ]

  outpath = f'{args.output}{run_uid}/'
  fs_outpath = outpath + 'FeatureSelection/'

  feature_set_dict = {
    'parcel_sum':{
    },
    'network_sum':{
    },
    'parcel_connection':{
    },
    'network_conneciton':{
    }
  }

for k in feature_set_dict.keys():
  print(k)
  for x in feature_set_dict[k]['hierarchical_selected_features'].keys():
    print(x, len(feature_set_dict[k]['hierarchical_selected_features'][x]))


for n in range(7,20):
  feature_set_dict[k]['hierarchical_selected_features'][n] = hierarchical_fs(feature_set_dict[k]['train_x'],n)
  np.save(f'{fs_outpath}{k}/{run_uid}_hierarchical-{n}.npy',np.array(feature_set_dict[k]['hierarchical_selected_features'][n]))
  
  sub_start_time = dt.datetime.now()
  logging.info(f'Attempting to read data from {fs_outpath}: {sub_start_time}')
  try:
    for k in feature_set_dict.keys():
      for target_df in ['train_x','test_x','train_y','test_y']:
        feature_set_dict[k][target_df] = pd.DataFrame(np.load(f'{fs_outpath}{k}/{run_uid}_{target_df}_colnames.npy', allow_pickle=True).T, columns = np.load(f'{fs_outpath}{k}/{run_uid}_{target_df}_colnames.npy', allow_pickle=True))
    sub_end_time = dt.datetime.now()
    logging.info(f'Premade raw data successfully imported from {fs_outpath}: {sub_end_time}')
  except:
    sub_end_time = dt.datetime.now()
    logging.info(f'Premaide data failed to load. Importing from data from parcellated timeseries: {sub_end_time}')
    sub_start_time = dt.datetime.now()
    logging.info(f'Reading parcellated data Started: {sub_start_time}')
    parcellated_data = {}
    for session in sessions:
      #Read in parcellated data, or parcellate data if meta-data conditions not met by available data
      parcellated_data[session] = load_parcellated_task_timeseries_v2(meta_dict, subjects, session, npy_template = npy_template_hcp)
    sub_end_time = dt.datetime.now()
    logging.info(f'Reading parcellated data Done: {sub_end_time}')
      
    sub_start_time = dt.datetime.now()
    logging.info(f'generate_parcel_input_features Started: {sub_start_time}')
    parcels_sums = generate_parcel_input_features(parcellated_data, parcel_labels)
    sub_end_time = dt.datetime.now()
    logging.info(f'generate_parcel_input_features Done: {sub_end_time}')

    sub_start_time = dt.datetime.now()
    logging.info(f'generate_network_input_features Started: {sub_start_time}')
    network_sums = generate_network_input_features(parcels_sums, network_labels)
    sub_end_time = dt.datetime.now()
    logging.info(f'generate_network_input_features Done: {sub_end_time}')

    sub_start_time = dt.datetime.now()
    logging.info(f'generate_parcel_connection_features Started: {sub_start_time}')
    parcel_connection_task_data = generate_parcel_connection_features(parcellated_data, parcel_labels)
    sub_end_time = dt.datetime.now()
    logging.info(f'generate_parcel_connection_features Done: {sub_end_time}')

    sub_start_time = dt.datetime.now()
    logging.info(f'generate_network_connection_features Started: {sub_start_time}')
    network_connection_features = generate_network_connection_features(parcellated_data, network_labels)
    sub_end_time = dt.datetime.now()
    logging.info(f'generate_network_connection_features Done: {sub_end_time}')

    sub_start_time = dt.datetime.now()
    logging.info(f'Confound merging Started: {sub_start_time}')
    # Merge in any confounds
    if (args.movement_regressor is not "None") or (args.confounds is not "None"):
      def str_combine(x, y):
        return str(x) + str(y)
      confounds['temp_index'] = confounds.apply(lambda row: str_combine(row['Subject'], row['task']), axis=1)
      confounds.drop(columns=['Subject','task'], inplace=True)
      parcels_sums['temp_index'] = parcels_sums.apply(lambda row: str_combine(row['Subject'], row['task']), axis=1)
      network_sums['temp_index'] = network_sums.apply(lambda row: str_combine(row['Subject'], row['task']), axis=1)
      parcel_connection_task_data['temp_index'] = parcel_connection_task_data.apply(lambda row: str_combine(row['Subject'], row['task']), axis=1)
      network_connection_features['temp_index'] = network_connection_features.apply(lambda row: str_combine(row['Subject'], row['task']), axis=1)

      parcel_sum_input = pd.merge(confounds, parcels_sums, on='temp_index', how = 'right').dropna()
      network_sum_input = pd.merge(confounds, network_sums, on='temp_index', how = 'right').dropna()
      parcel_connection_input = pd.merge(confounds, parcel_connection_task_data, on='temp_index', how = 'right').dropna()
      network_connection_input = pd.merge(confounds, network_connection_features, on='temp_index', how = 'right').dropna()
    else:
      parcel_sum_input = parcels_sums
      network_sum_input = network_sums
      parcel_connection_input = parcel_connection_task_data
      network_connection_input = network_connection_features
    sub_end_time = dt.datetime.now()
    logging.info(f'Confound merging Done: {sub_end_time}')

    sub_start_time = dt.datetime.now()
    logging.info(f'XY Split Started: {sub_start_time}')
    # XY Split
    parcel_sum_x, parcel_sum_y = XY_split(parcel_sum_input, 'task')
    network_sum_x, network_sum_y = XY_split(network_sum_input, 'task')
    parcel_connection_x, parcel_connection_y = XY_split(parcel_connection_input, 'task')
    network_connection_x, network_connection_y = XY_split(network_connection_input, 'task')
    sub_end_time = dt.datetime.now()
    logging.info(f'XY Split Done: {sub_end_time}')

    sub_start_time = dt.datetime.now()
    logging.info(f'Training Test Split Started: {sub_start_time}')
    # Training Test Split
    parcel_sum_x_train, parcel_sum_x_test, parcel_sum_y_train, parcel_sum_y_test = train_test_split(parcel_sum_x, parcel_sum_y, test_size = 0.2)
    network_sum_x_train, network_sum_x_test, network_sum_y_train, network_sum_y_test = train_test_split(network_sum_x, network_sum_y, test_size = 0.2)
    parcel_connection_x_train, parcel_connection_x_test, parcel_connection_y_train, parcel_connection_y_test = train_test_split(parcel_connection_x, parcel_connection_y, test_size = 0.2)
    network_connection_x_train, network_connection_x_test, network_connection_y_train, network_connection_y_test = train_test_split(network_connection_x, network_connection_y, test_size = 0.2)
    sub_end_time = dt.datetime.now()
    logging.info(f'Training Test Split Done: {sub_end_time}')

    sub_start_time = dt.datetime.now()
    logging.info(f'Scaling non-categorical Variables Started: {sub_start_time}')
    # Scaling non-categorical Variables
    cols_to_exclude = list(confounds.columns)
    cols_to_exclude.remove('task')
    parcel_sum_x_train = scale_subset(parcel_sum_x_train, cols_to_exclude)
    parcel_sum_x_test = scale_subset(parcel_sum_x_test, cols_to_exclude)
    network_sum_x_train = scale_subset(network_sum_x_train, cols_to_exclude)
    network_sum_x_test = scale_subset(network_sum_x_test, cols_to_exclude)
    parcel_connection_x_train = scale_subset(parcel_connection_x_train, cols_to_exclude)
    parcel_connection_x_test = scale_subset(parcel_connection_x_test, cols_to_exclude)
    network_connection_x_train = scale_subset(network_connection_x_train, cols_to_exclude)
    network_connection_x_test = scale_subset(network_connection_x_test, cols_to_exclude)
    sub_end_time = dt.datetime.now()
    logging.info(f'Scaling non-categorical Variables Done: {sub_end_time}')

    feature_set_dict = {
      'parcel_sum':{
        'train_x': parcel_sum_x_train,
        'test_x': parcel_sum_x_test,
        'train_y': parcel_sum_y_train,
        'test_y': parcel_sum_y_test
      },
      'network_sum':{
        'train_x': network_sum_x_train,
        'test_x': network_sum_x_test,
        'train_y': network_sum_y_train,
        'test_y': network_sum_y_test
      },
      'parcel_connection':{
        'train_x': parcel_connection_x_train,
        'test_x': parcel_connection_x_test,
        'train_y': parcel_connection_y_train,
        'test_y': parcel_connection_y_test
      },
      'network_conneciton':{
        'train_x': network_connection_x_train,
        'test_x': network_connection_x_test,
        'train_y': network_connection_y_train,
        'test_y': network_connection_y_test
      }
    }
    for k in feature_set_dict.keys():
      try:
        os.makedirs(f'{fs_outpath}/{k}')
      except:
        pass
      for target_df in ['train_x','test_x','train_y','test_y']:
        np.save(f'{fs_outpath}{k}/{run_uid}_{target_df}.npy', np.array(feature_set_dict[k][target_df]))
        np.save(f'{fs_outpath}{k}/{run_uid}_{target_df}_colnames.npy', np.array(feature_set_dict[k][target_df].columns))
  # Feature Selection
  try:
    os.makedirs(fs_outpath)
  except:
    pass
  fs_start_time = dt.datetime.now()
  logging.info(f'Feature Selection Started: {fs_start_time}')
  for k in feature_set_dict.keys():
    sub_start_time = dt.datetime.now()
    try:
      for h in feature_set_dict[k]['hieratchical_selected_features'].keys():
        feature_set_dict[k]['hieratchical_selected_features'][h] = np.load(f'{fs_outpath}{k}/{run_uid}_hierarchical-{k}.npy')
      sub_end_time = dt.datetime.now()
      logging.info('Previous Hierarchical Feaure Selection Output imported: {sub_end_time}')
    except:
      sub_start_time = dt.datetime.now()
      logging.info(f'\tHierarchical Feaure Selection ({k}) Started: {sub_start_time}')
      feature_set_dict[k]['hieratchical_selected_features'] = {}
      for n in range(1,7):
        feature_set_dict[k]['hieratchical_selected_features'][n] = hierarchical_fs(feature_set_dict[k]['train_x'],n)
        np.save(f'{fs_outpath}{k}/{run_uid}_hierarchical-{n}.npy',np.array(feature_set_dict[k]['hieratchical_selected_features'][n]))
      sub_end_time = dt.datetime.now()
      logging.info(f'\tHierarchical Feaure Selection ({k}) Done: {sub_end_time}')
    
    try:
      sub_start_time = dt.datetime.now()
      feature_set_dict[k]['train_pca_auto'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_pca-auto.npy')
      feature_set_dict[k]['test_pca_auto'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_pca-auto.npy')
      feature_set_dict[k]['pca_auto'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_pca-auto.pkl', 'rb'))
      feature_set_dict[k]['pca_auto'].transform(feature_set_dict[k]['train_x'])
      sub_end_time = dt.datetime.now()
      logging.info('Previous PCA Output imported: {sub_end_time}')
    except:
      sub_start_time = dt.datetime.now()
      logging.info(f'\tPCA Started: {sub_start_time}')
      train_pca_auto, test_pca_auto, pca_auto = pca_fs(feature_set_dict[k]['train_x'], feature_set_dict[k]['test_x'], k_components=None)
      feature_set_dict[k]['train_pca_auto'] = train_pca_auto
      feature_set_dict[k]['test_pca_auto'] = test_pca_auto
      feature_set_dict[k]['pca_auto'] = pca_auto
      np.save(f'{fs_outpath}{k}/{run_uid}_train_pca-auto.npy',feature_set_dict[k]['train_pca_auto'])
      np.save(f'{fs_outpath}{k}/{run_uid}_test_pca-auto.npy',feature_set_dict[k]['test_pca_auto'])
      pk.dump(feature_set_dict[k]['pca_auto'], open(f'{fs_outpath}{k}/{run_uid}_pca-auto.pkl', "wb"))
      sub_end_time = dt.datetime.now()
      logging.info(f'\tPCA Done: {sub_end_time}')

imp1 = random_forest_fs_v3(feature_set_dict[k]['train_x'], np.array(feature_set_dict[k]['train_y']['task']), n_estimators = 500, n_repeats=10, n_jobs=10)

    try:
      sub_start_time = dt.datetime.now()
      feature_set_dict[k]['fr_features_v1'] = np.load(f'{fs_outpath}{k}/{run_uid}_fr_features_v1.npy')
      sub_end_time = dt.datetime.now()
      logging.info('Previous Forest FS V1 Output imported: {sub_end_time}')
    except:
      sub_start_time = dt.datetime.now()
      logging.info(f'\tRandom Forest FS V1 Started: {sub_start_time}')
      feature_set_dict[k]['fr_features_v1'] = random_forest_fs(feature_set_dict[k]['train_x'], feature_set_dict[k]['train_y'].to_numpy(), n_estimators = 500, n_repeats=10, n_jobs=10)
      np.save(f'{fs_outpath}{k}/{run_uid}_fr_features_v1.npy',feature_set_dict[k]['fr_features_v1'])
      sub_end_time = dt.datetime.now()
      logging.info(f'\tRandom Forest FS V1 Done: {sub_end_time}')

    try:
      sub_start_time = dt.datetime.now()
      feature_set_dict[k]['fr_features_v2'] = np.load(f'{fs_outpath}{k}/{run_uid}_fr_features_v2.npy')
      sub_end_time = dt.datetime.now()
      logging.info('Previous Forest FS V2 Output imported: {sub_end_time}')
    except:
      sub_start_time = dt.datetime.now()
      logging.info(f'\tRandom Forest FS V2 Started: {sub_start_time}')
      feature_set_dict[k]['fr_features_v2'] = random_forest_fs_v2(feature_set_dict[k]['train_x'], feature_set_dict[k]['train_y'], n_estimators = 500, n_repeats=10, n_jobs=2)
      np.save(f'{fs_outpath}{k}/{run_uid}_fr_features_v2.npy',feature_set_dict[k]['fr_features_v2'])
      sub_end_time = dt.datetime.now()
      logging.info(f'\tRandom Forest FS V2 Done: {sub_end_time}')


  fs_end_time = dt.datetime.now()
  logging.info(f'Feature Selection Done: {fs_end_time}')



  

  total_end_time = dt.datetime.now()
  runtime = total_end_time - total_start_time
  logging.info(f'Finished; {total_start_time}')
  logging.info(f'Runtime; {runtime}')

except Exception as e:
  logging.exception(f'ERROR: {e}')
  total_end_time = dt.datetime.now()
  runtime = total_end_time - total_start_time
  logging.info(f'Finished; {total_start_time}')
  logging.info(f'Runtime; {runtime}')

# Output Data to temp location
files_to_transfer = [ 
  f'{source_path}{run_uid}_DEBUG.log'
  # ,''
  # ,''
]



# SCP any outputs to SSD location
scp = SCPClient(ssh.get_transport())
ssh = createSSHClient(args.datahost, args.uname, args.psswd)
# rmtree temp files

