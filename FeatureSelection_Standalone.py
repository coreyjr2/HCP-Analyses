#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Made to run FS on R2 without any counfounds of motion varibales

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
  from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FastICA
  from sklearn.inspection import permutation_importance
  from sklearn.manifold import TSNE, SpectralEmbedding, MDS
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
  from skopt import BayesSearchCV ############################ Mising
  from skopt.space import Real, Categorical, Integer ############################ Mising
  from operator import itemgetter
  import random
  from sklearn.model_selection import GroupShuffleSplit
  # import brainconn.utils as bc
except Exception as e:
  print(f'Error loading libraries: ')
  raise Exception(e)



v3_argslist = [ # Used on dx and ran out of RAM on the parcell connection hierarchical clustering step
  # '-source_path', '/mnt/usb1/HCP_69354adf',
  # '-uname', 'kbaacke',
  # '-datahost', 'r2.psych.uiuc.edu', # not needed
  '-local_path', '/mnt/usb1/HCP_69354adf/HCP_69354adf', 
  '--output', '/mnt/usb1/hcp_analysis_output/',
  # '--remote_output','/mnt/usb1/Code/' # not needed
  '--n_jobs', '8',
  '-atlas_path', '/mnt/usb1/HCP_69354adf/HCP_69354adf/69354adf_parcellation-metadata.json',
  '--movement_regressor','None',
  '--confounds', 'None',
  # '--confound_subset','None'
]

# Global Variables
sep = os.path.sep
source_path = '/home/kbaacke/HCP_Analyses/'
# source_path = os.path.dirname(os.path.abspath(__file__)) + sep
sys_name = platform.system() 
hostname = platform.node()
numeric_task_ref = {
  "tfMRI_MOTOR":4,
  "tfMRI_WM":7,
  "tfMRI_EMOTION":1,
  "tfMRI_GAMBLING":2,
  "tfMRI_LANGUAGE":3,
  "tfMRI_RELATIONAL":5,
  "tfMRI_SOCIAL":6
}

try:
  def parse_args(args):
    #Presets
    parser = argparse.ArgumentParser(
        description='Feature Selection script.'
      )
    parser.add_argument(
      "-source_path", help='Full base-path on the Data Node where the data is stored.', required=False, default=None
    )
    # parser.add_argument(
    #   "-uname", help='Username to use when requesting files from the data node via scp.', required=False
    # )
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
    parser.add_argument(# Context Specific
      "-atlas_path",
      help=f'''
        Full path to the metadata json file for the parcellation.\nEx: S:{sep}HCP{sep}HCP_83e395d0{sep}83e395d0_parcellation-metadata.json
      '''
    )
    parser.add_argument(
      '--confounds',
      help='Full base-path on the Data Node where the confounds/demographics .csv file is located. Use \'None\' to indicate that no confounds should be used.\nDefault is \'None\'', required=False, default = 'None'
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
  def str_combine(x, y):
    return str(x) + str(y)
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
  def load_parcellated_task_timeseries_v2(meta_dict,subjects, session, npy_template, basepath, run_names = ['RL','LR'], confounds_path = None): # un-Tested
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
          masked_timeseries = np.load(npy_template.format(subject=subject, session=session, run=run, atlas_name=atlas_name, basepath = basepath, sep = sep))
          if remove_mean:
            masked_timeseries -= masked_timeseries.mean(axis=1, keepdims=True)
          sub_dict[run] = masked_timeseries
        if concatenate:
          concat_dict[subject] = np.vstack((sub_dict[run_names[0]], sub_dict[run_names[1]]))
        parcellated_dict[subject] = sub_dict
      except Exception as e:
        print(f'Subject {subject} is not available: {e}')
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
  def generate_parcel_connection_features_v2(parcellated_data, labels, threshold = None): 
    # threshold as proportion of values to keep
    out_dict = {}
    out_df_dict = {}
    for session in parcellated_data.keys():
      parcel_dict = parcellated_data[session]
      out_dict[session] = {}
      for subject in parcel_dict.keys():
        ts = parcel_dict[subject]
        cor_coef = np.corrcoef(ts.T)
        if threshold != None:
          cor_coef = bc.threshold_proportional(cor_coef, p = threshold)
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
  def kpca_fs(train_x, test_x, kernel='rbf', n_components = None, n_jobs = 1):
    if n_components!=None:
      kpca = KernelPCA(n_components = n_components, kernel = kernel, n_jobs = n_jobs).fit(train_x)
    else:
      kpca = KernelPCA(kernel = kernel, n_jobs = n_jobs).fit(train_x)
    train_kpca = kpca.transform(train_x)
    test_kpca = kpca.transform(test_x)
    return train_kpca, test_kpca, kpca
  def tSVD_fs(train_x, test_x, n_components, n_iter = 5, random_state = 812):
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_state)
    svd.fit(train_x)
    train_svd = svd.transform(train_x)
    test_svd = svd.transform(test_x)
    return train_svd, test_svd, svd
  def TSNE_fs(train_x, test_x, n_components, perplexity = 30, learning_rate=200.0, init = 'random', verbose=0, n_iter=1000, n_jobs = 1):
    tsne = TSNE(
      n_components=n_components,
      perplexity=perplexity,
      learning_rate=learning_rate,
      init=init,
      verbose=verbose,
      n_iter=n_iter,
      n_jobs = n_jobs
      )
    tsne.fit(train_x)
    train_tsne = tsne.transform(train_x)
    test_tsne = tsne.transform(test_x)
    return train_tsne, test_tsne, tsne
  def ICA_fs(train_x, test_x, n_components=None, max_iter=200, random_state=42):
    ica = FastICA(n_components=None, max_iter=max_iter, random_state=random_state)
    ica.fit(train_x)
    train_ica = ica.transform(train_x)
    test_ica = ica.transform(test_x)
    return train_ica, test_ica, ica
  def LE_fs(train_x, test_x, n_components=None, random_state=42, n_jobs=1,n_neighbors = None):
    print()
    if n_components is None:
      LE = SpectralEmbedding(random_state=random_state, n_jobs=n_jobs, n_neighbors = n_neighbors)
    else:
      LE = SpectralEmbedding(n_components = n_components, random_state=random_state, n_jobs=n_jobs, n_neighbors = n_neighbors)
    LE.fit(train_x)
    train_LE = LE.transform(train_x)
    test_LE = LE.transform(test_x)
    return train_LE, test_LE, LE
  def MDS_fs(train_x, test_x, n_components=2, max_iter=300, random_state=42, n_jobs = 1):
    MDS = MDS(n_components=n_components, random_state=random_state, n_jobs=n_jobs)
    MDS.fit(train_x)
    train_MDS = MDS.transform(train_x)
    test_MDS = MDS.transform(test_x)
    return train_MDS, test_MDS, MDS
  def LDA_fs(train_x, test_x, train_y):
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(train_x, train_y)
    train_LDA = LDA.transform(train_x)
    test_LDA = LDA.transform(test_x)
    return train_LDA, test_LDA, LDA
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

args, leforvers = parse_args(v3_argslist)

args.atlas_name = os.path.basename(args.atlas_path)[:8]

parcellation_dict = json.load(open(args.atlas_path))

meta_dict = {
  'atlas_name' : args.atlas_name,
  'smoothed' : args.smoothed,
  'ICA-Aroma' : args.ica_aroma,
  # 'confounds': args.,
  'subtract parcel-wise mean': True, # Fixing this at true, no cli arg
  'concatenate':args.concatenate
}

if args.confounds != "None":
  demographics = pd.read_csv(args.confounds)
  if args.confound_subset[0] != 'None':
    demographics = demographics[args.confound_subset]
  if args.movement_regressor == "None":
    confounds = demographics

if args.movement_regressor is not "None":
  relative_RMS_means_collapsed = pd.read_csv(args.movement_regressor)
  if args.confounds is "None":
    confounds = relative_RMS_means_collapsed

if (args.movement_regressor is not "None") and (args.confounds is not "None"):
  confounds = pd.merge(relative_RMS_means_collapsed, demographics, how='left', on='Subject')

try:
  meta_dict['confounds'] = list(confounds.columns)
except:
  meta_dict['confounds'] = None

run_uid = generate_uid(meta_dict)
outpath = args.output + run_uid + sep
try:
  os.makedirs(outpath)
except Exception as e:
  print(e, 'Output directory already created.')

with open(outpath + run_uid + 'metadata.json', 'w') as outfile:
  json.dump(meta_dict, outfile)

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
logging.debug('meta_dict: ')
logging.debug(meta_dict)

basepath = args.local_path
npy_template_hcp = '{basepath}HCP{sep}HCP_1200{sep}{subject}{sep}MNINonLinear{sep}Results{sep}{session}_{run}{sep}{atlas_name}_{session}_{run}.npy'
HCP_1200 = f'{basepath}{sep}HCP{sep}HCP_1200{sep}'
#parcel_labels, network_labels = fetch_labels(meta_dict, f'{basepath}HCP_{args.atlas_name}{sep}') # remote version
parcel_labels, network_labels = fetch_labels(meta_dict, f'{basepath}{sep}') # r2 version
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
  'network_connection':{
  }
}

sub_start_time = dt.datetime.now()
logging.info(f'Attempting to read data from {fs_outpath}: {sub_start_time}')
try:
  for k in feature_set_dict.keys():
    for target_df in ['train_x','test_x','train_y','test_y']:
      feature_set_dict[k][target_df] = pd.DataFrame(np.load(f'{fs_outpath}{k}/{run_uid}_{target_df}.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}/{run_uid}_{target_df}_colnames.npy', allow_pickle=True))
  sub_end_time = dt.datetime.now()
  logging.info(f'Premade raw data successfully imported from {fs_outpath}: {sub_end_time}')
except:
  sub_end_time = dt.datetime.now()
  logging.info(f'Premade data failed to load. Importing from data from parcellated timeseries: {sub_end_time}')

<<<<<<< HEAD
# sub_start_time = dt.datetime.now()
# logging.info(f'Reading parcellated data Started: {sub_start_time}')
# parcellated_data = {}
# for session in sessions:
#   #Read in parcellated data, or parcellate data if meta-data conditions not met by available data
#   parcellated_data[session] = load_parcellated_task_timeseries_v2(meta_dict, subjects, session, npy_template = npy_template_hcp, basepath = f'{basepath}{sep}') # remote version: f'{basepath}HCP_{args.atlas_name}{sep}'
=======
sub_start_time = dt.datetime.now()
logging.info(f'Reading parcellated data Started: {sub_start_time}')
parcellated_data = {}
for session in sessions:
  #Read in parcellated data, or parcellate data if meta-data conditions not met by available data
  parcellated_data[session] = load_parcellated_task_timeseries_v2(meta_dict, subjects, session, npy_template = npy_template_hcp, basepath = f'{basepath}{sep}') # remote version: f'{basepath}HCP_{args.atlas_name}{sep}'
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
>>>>>>> master

# sub_end_time = dt.datetime.now()
# logging.info(f'Reading parcellated data Done: {sub_end_time}')
# sub_start_time = dt.datetime.now()
# logging.info(f'generate_parcel_input_features Started: {sub_start_time}')
# parcels_sums = generate_parcel_input_features(parcellated_data, parcel_labels)
# sub_end_time = dt.datetime.now()
# logging.info(f'generate_parcel_input_features Done: {sub_end_time}')
# sub_start_time = dt.datetime.now()
# logging.info(f'generate_network_input_features Started: {sub_start_time}')
# network_sums = generate_network_input_features(parcels_sums, network_labels)
# sub_end_time = dt.datetime.now()
# logging.info(f'generate_network_input_features Done: {sub_end_time}')

<<<<<<< HEAD
# sub_start_time = dt.datetime.now()
# logging.info(f'generate_parcel_connection_features Started: {sub_start_time}')
# parcel_connection_task_data = generate_parcel_connection_features(parcellated_data, parcel_labels)
# sub_end_time = dt.datetime.now()
# logging.info(f'generate_parcel_connection_features Done: {sub_end_time}')

# sub_start_time = dt.datetime.now()
# logging.info(f'generate_parcel_connection_features Started: {sub_start_time}')
# parcel_connection_task_data_thresh_2 = generate_parcel_connection_features_v2(parcellated_data, parcel_labels, threshold = .2)
# sub_end_time = dt.datetime.now()
# logging.info(f'generate_parcel_connection_features Done: {sub_end_time}')

# sub_start_time = dt.datetime.now()
# logging.info(f'generate_network_connection_features Started: {sub_start_time}')
# network_connection_features = generate_network_connection_features(parcellated_data, network_labels)
# sub_end_time = dt.datetime.now()
# logging.info(f'generate_network_connection_features Done: {sub_end_time}')
# sub_start_time = dt.datetime.now()
# logging.info(f'Confound merging Started: {sub_start_time}')

# # Merge in any confounds
# if (args.movement_regressor != "None") or (args.confounds != "None"):
#   confounds['Subject'] = confounds['Subject'].astype(str)
#   confounds['task'] = confounds['task'].astype(int)
#   confounds['temp_index'] = confounds.apply(lambda row: str_combine(row['Subject'], row['task']), axis=1)
#   confounds.drop(columns=['Subject','task'], inplace=True)
#   parcels_sums['temp_index'] = parcels_sums.apply(lambda row: str_combine(row['Subject'], row['task']), axis=1)
#   network_sums['temp_index'] = network_sums.apply(lambda row: str_combine(row['Subject'], row['task']), axis=1)
#   parcel_connection_task_data['temp_index'] = parcel_connection_task_data.apply(lambda row: str_combine(row['Subject'], row['task']), axis=1)
#   network_connection_features['temp_index'] = network_connection_features.apply(lambda row: str_combine(row['Subject'], row['task']), axis=1)
#   parcel_sum_input = pd.merge(confounds, parcels_sums, on='temp_index', how = 'right').dropna()
#   network_sum_input = pd.merge(confounds, network_sums, on='temp_index', how = 'right').dropna()
#   parcel_connection_input = pd.merge(confounds, parcel_connection_task_data, on='temp_index', how = 'right').dropna()
#   network_connection_input = pd.merge(confounds, network_connection_features, on='temp_index', how = 'right').dropna()
# else:
#   parcel_sum_input = parcels_sums
#   network_sum_input = network_sums
#   parcel_connection_input = parcel_connection_task_data
#   network_connection_input = network_connection_features
#   sub_end_time = dt.datetime.now()
#   logging.info(f'Confound merging Done: {sub_end_time}')

# # XY Split
# # sub_start_time = dt.datetime.now()
# # logging.info(f'XY Split Started: {sub_start_time}')
# # parcel_sum_x, parcel_sum_y = XY_split(parcel_sum_input, 'task')
# # network_sum_x, network_sum_y = XY_split(network_sum_input, 'task')
# # parcel_connection_x, parcel_connection_y = XY_split(parcel_connection_input, 'task')
# # network_connection_x, network_connection_y = XY_split(network_connection_input, 'task')
# # sub_end_time = dt.datetime.now()
# # logging.info(f'XY Split Done: {sub_end_time}')

# # Training Test Split
# sub_start_time = dt.datetime.now()
# logging.info(f'Training Test Split Started: {sub_start_time}')
# random_state=42
# gss_holdout = GroupShuffleSplit(n_splits=1, train_size = .9, random_state = random_state)
# idx_1 = gss_holdout.split(
#     X = parcel_sum_input[parcel_sum_input.columns[6:]],
#     y = parcel_sum_input['task'],
#     groups = parcel_sum_input['Subject']
#   )

# idx_dict = {}
# for train, test in idx_1:
#   idx_dict['train'] = train
#   idx_dict['test'] = test

# # parcel_sum_x_train = parcel_sum_input.iloc[idx_dict['train']][parcel_sum_input.columns[1:]]
# # parcel_sum_x_test = parcel_sum_input.iloc[idx_dict['test']][parcel_sum_input.columns[1:]]
# # parcel_sum_y_train = parcel_sum_input.iloc[idx_dict['train']][['task']]
# # parcel_sum_y_test = parcel_sum_input.iloc[idx_dict['test']][['task']]

# # network_sum_x_train = network_sum_input.iloc[idx_dict['train']][network_sum_input.columns[1:]]
# # network_sum_x_test = network_sum_input.iloc[idx_dict['test']][network_sum_input.columns[1:]]
# # network_sum_y_train = network_sum_input.iloc[idx_dict['train']][['task']]
# # network_sum_y_test = network_sum_input.iloc[idx_dict['test']][['task']]

# parcel_connection_x_train = parcel_connection_input.iloc[idx_dict['train']][parcel_connection_input.columns[1:]]
# parcel_connection_x_test = parcel_connection_input.iloc[idx_dict['test']][parcel_connection_input.columns[1:]]
# parcel_connection_y_train = parcel_connection_input.iloc[idx_dict['train']][['task']]
# parcel_connection_y_test = parcel_connection_input.iloc[idx_dict['test']][['task']]

# # network_connection_x_train = network_connection_input.iloc[idx_dict['train']][network_connection_input.columns[1:]]
# # network_connection_x_test = network_connection_input.iloc[idx_dict['test']][network_connection_input.columns[1:]]
# # network_connection_y_train = network_connection_input.iloc[idx_dict['train']][['task']]
# # network_connection_y_test = network_connection_input.iloc[idx_dict['test']][['task']]

# # parcel_sum_x_train, parcel_sum_x_test, parcel_sum_y_train, parcel_sum_y_test = train_test_split(parcel_sum_x, parcel_sum_y, test_size = 0.2)
# # network_sum_x_train, network_sum_x_test, network_sum_y_train, network_sum_y_test = train_test_split(network_sum_x, network_sum_y, test_size = 0.2)
# # parcel_connection_x_train, parcel_connection_x_test, parcel_connection_y_train, parcel_connection_y_test = train_test_split(parcel_connection_x, parcel_connection_y, test_size = 0.2)
# # network_connection_x_train, network_connection_x_test, network_connection_y_train, network_connection_y_test = train_test_split(network_connection_x, network_connection_y, test_size = 0.2)
# sub_end_time = dt.datetime.now()
# logging.info(f'Training Test Split Done: {sub_end_time}')

# # Scaling non-categorical Variables
# sub_start_time = dt.datetime.now()
# logging.info(f'Scaling non-categorical Variables Started: {sub_start_time}')
# try:
#   cols_to_exclude = list(confounds.columns)
#   cols_to_exclude.append('Subject')
# except:
#   cols_to_exclude = ['Subject']

# # parcel_sum_x_train = scale_subset(parcel_sum_x_train, cols_to_exclude)
# # parcel_sum_x_test = scale_subset(parcel_sum_x_test, cols_to_exclude)
# # network_sum_x_train = scale_subset(network_sum_x_train, cols_to_exclude)
# # network_sum_x_test = scale_subset(network_sum_x_test, cols_to_exclude)
# parcel_connection_x_train = scale_subset(parcel_connection_x_train, cols_to_exclude)
# parcel_connection_x_test = scale_subset(parcel_connection_x_test, cols_to_exclude)
# # network_connection_x_train = scale_subset(network_connection_x_train, cols_to_exclude)
# # network_connection_x_test = scale_subset(network_connection_x_test, cols_to_exclude)
# sub_end_time = dt.datetime.now()
# logging.info(f'Scaling non-categorical Variables Done: {sub_end_time}')

# ##### TO SAVE #####
# feature_set_dict = {
#   # 'parcel_sum':{
#   #   'train_x': parcel_sum_x_train,
#   #   'test_x': parcel_sum_x_test,
#   #   'train_y': parcel_sum_y_train,
#   #   'test_y': parcel_sum_y_test
#   # },
#   # 'network_sum':{
#   #   'train_x': network_sum_x_train,
#   #   'test_x': network_sum_x_test,
#   #   'train_y': network_sum_y_train,
#   #   'test_y': network_sum_y_test
#   # },
#   'parcel_connection':{
#     'train_x': parcel_connection_x_train,
#     'test_x': parcel_connection_x_test,
#     'train_y': parcel_connection_y_train,
#     'test_y': parcel_connection_y_test
#   }#,
#   # 'network_connection':{
#   #   'train_x': network_connection_x_train,
#   #   'test_x': network_connection_x_test,
#   #   'train_y': network_connection_y_train,
#   #   'test_y': network_connection_y_test
#   # }
# }
# for k in feature_set_dict.keys():
#   try:
#     os.makedirs(f'{fs_outpath}/{k}')
#   except:
#     pass
#   for target_df in ['train_x','test_x','train_y','test_y']:
#     np.save(f'{fs_outpath}{k}/{run_uid}_{target_df}.npy', np.array(feature_set_dict[k][target_df]))
#     np.save(f'{fs_outpath}{k}/{run_uid}_{target_df}_colnames.npy', np.array(feature_set_dict[k][target_df].columns))


##### TO READ #####
fs_outpath = outpath + 'FeatureSelection/'
feature_set_dict = {
  'parcel_connection':{
=======
sub_start_time = dt.datetime.now()
logging.info(f'generate_network_connection_features Started: {sub_start_time}')
network_connection_features = generate_network_connection_features(parcellated_data, network_labels)
sub_end_time = dt.datetime.now()
logging.info(f'generate_network_connection_features Done: {sub_end_time}')
sub_start_time = dt.datetime.now()
logging.info(f'Confound merging Started: {sub_start_time}')
# Merge in any confounds
if (args.movement_regressor != "None") or (args.confounds != "None"):
  confounds['Subject'] = confounds['Subject'].astype(str)
  confounds['task'] = confounds['task'].astype(int)
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
# XY Split
sub_start_time = dt.datetime.now()
logging.info(f'XY Split Started: {sub_start_time}')
parcel_sum_x, parcel_sum_y = XY_split(parcel_sum_input, 'task')
network_sum_x, network_sum_y = XY_split(network_sum_input, 'task')
parcel_connection_x, parcel_connection_y = XY_split(parcel_connection_input, 'task')
network_connection_x, network_connection_y = XY_split(network_connection_input, 'task')
sub_end_time = dt.datetime.now()
logging.info(f'XY Split Done: {sub_end_time}')
# Training Test Split
sub_start_time = dt.datetime.now()
logging.info(f'Training Test Split Started: {sub_start_time}')
parcel_sum_x_train, parcel_sum_x_test, parcel_sum_y_train, parcel_sum_y_test = train_test_split(parcel_sum_x, parcel_sum_y, test_size = 0.2)
network_sum_x_train, network_sum_x_test, network_sum_y_train, network_sum_y_test = train_test_split(network_sum_x, network_sum_y, test_size = 0.2)
parcel_connection_x_train, parcel_connection_x_test, parcel_connection_y_train, parcel_connection_y_test = train_test_split(parcel_connection_x, parcel_connection_y, test_size = 0.2)
network_connection_x_train, network_connection_x_test, network_connection_y_train, network_connection_y_test = train_test_split(network_connection_x, network_connection_y, test_size = 0.2)
sub_end_time = dt.datetime.now()
logging.info(f'Training Test Split Done: {sub_end_time}')

# Scaling non-categorical Variables
sub_start_time = dt.datetime.now()
logging.info(f'Scaling non-categorical Variables Started: {sub_start_time}')
try:
  cols_to_exclude = list(confounds.columns)
  cols_to_exclude.append('Subject')
except:
  cols_to_exclude = ['Subject']
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
  'network_connection':{
    'train_x': network_connection_x_train,
    'test_x': network_connection_x_test,
    'train_y': network_connection_y_train,
    'test_y': network_connection_y_test
>>>>>>> master
  }
}

try:
  for k in feature_set_dict.keys():
    for target_df in ['train_x','test_x','train_y','test_y']:
      feature_set_dict[k][target_df] = pd.DataFrame(np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}_colnames.npy', allow_pickle=True))
  sub_end_time = dt.datetime.now()
  logging.info(f'Premade raw data successfully imported from {fs_outpath}: {sub_end_time}')
except Exception as e:
  print(f'Error reading in raw data: {e}')
  logging.info(f'Error reading in raw data: {e}')

# Feature Selection
try:
  os.makedirs(fs_outpath)
except:
  pass
fs_start_time = dt.datetime.now()
logging.info(f'Feature Selection Started: {fs_start_time}')
<<<<<<< HEAD
len_list = []
target_keys = ['parcel_connection']

# # Hierarchical
# for k in target_keys:
#   # Hierarchical
#   sub_start_time = dt.datetime.now()
#   hierarchical_start = 1
#   hierarchical_end = 250
#   # try:
#   #   for n in range(hierarchical_start, hierarchical_end):
#   #     feature_set_dict[k]['hierarchical_selected_features'][h] = np.load(f'{fs_outpath}{k}/{run_uid}_hierarchical-{k}.npy')
#   #   sub_end_time = dt.datetime.now()
#   #   logging.info('Previous Hierarchical Feature Selection Output imported: {sub_end_time}')
#   # except:
#   sub_start_time = dt.datetime.now()
#   logging.info(f'\tHierarchical Feature Selection ({k}) Started: {sub_start_time}')
#   feature_set_dict[k]['hierarchical_selected_features'] = hierarchical_fs_v2(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'],hierarchical_start, hierarchical_end)
#   for n in range(hierarchical_start, hierarchical_end):
#     #feature_set_dict[k]['hierarchical_selected_features'][n] = hierarchical_fs(feature_set_dict[k]['train_x'],n)
#     n_len = len(feature_set_dict[k]['hierarchical_selected_features'][n])
#     if n==1 or n_len!=len(feature_set_dict[k]['hierarchical_selected_features'][n-1]):
#       np.save(f'{fs_outpath}{k}/{run_uid}_hierarchical-{n}.npy',np.array(feature_set_dict[k]['hierarchical_selected_features'][n]))
#       print(n, n_len)
#       len_list.append(n_len)
# # PCA
# for k in target_keys:
#   # PCA
#   sub_end_time = dt.datetime.now()
#   logging.info(f'\tHierarchical Feature Selection ({k}) Done: {sub_end_time}')
#   try:
#     sub_start_time = dt.datetime.now()
#     feature_set_dict[k]['train_pca'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_pca.npy')
#     feature_set_dict[k]['test_pca'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_pca.npy')
#     feature_set_dict[k]['pca'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_pca.pkl', 'rb'))
#     sub_end_time = dt.datetime.now()
#     logging.info('\tPrevious PCA Output imported: {sub_end_time}')
#   except:
#     sub_start_time = dt.datetime.now()
#     logging.info(f'\tPCA Started: {sub_start_time}')
#     train_pca, test_pca, pca = pca_fs(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'], feature_set_dict[k]['test_x'].loc[:,feature_set_dict[k]['test_x'].columns != 'Subject'], k_components=None)
#     feature_set_dict[k]['train_pca'] = train_pca
#     feature_set_dict[k]['test_pca'] = test_pca
#     feature_set_dict[k]['pca'] = pca
#     np.save(f'{fs_outpath}{k}/{run_uid}_train_pca.npy',feature_set_dict[k]['train_pca'])
#     np.save(f'{fs_outpath}{k}/{run_uid}_test_pca.npy',feature_set_dict[k]['test_pca'])
#     pk.dump(feature_set_dict[k]['pca'], open(f'{fs_outpath}{k}/{run_uid}_pca.pkl', "wb"))
#     sub_end_time = dt.datetime.now()
#     logging.info(f'\tPCA Done: {sub_end_time}')


# # RFC feature selection
# ## Select from model
# for k in target_keys:
#   # RFC feature selection
#   ## Select from model
#   sub_start_time_outer = dt.datetime.now()
#   logging.info(f'\tSelectFromModel on FRC on {k} started: {sub_start_time_outer}')
#   # for x_len in len_list:
#   for x_len in [9,8,7,6,5,4,3,2,1]:
#     # This can be optimized, return to this later
#     sub_start_time = dt.datetime.now()
#     # try:
#     #   feature_set_dict[k][f'rf_selected_{x}'] = np.load(f'{fs_outpath}{k}/{run_uid}_rf_selected_{x}.npy')
#     #   sub_end_time = dt.datetime.now()
#     #   logging.info(f'\t\tSelectFromModel on FRC for {x} max features read from previous run')
#     # except:
#     logging.info(f'\t\tSelectFromModel FRC FS V1 Started: {sub_start_time}')
#     feature_set_dict[k][f'rf_selected_n{x_len}'] = list(
#       compress(
#         list(feature_set_dict[k]['train_x'].columns),
#         random_forest_fs(
#           feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'],
#           np.array(feature_set_dict[k]['train_y']['task']),
#           n_estimators = 500,
#           n_repeats=10,
#           n_jobs=10,
#           max_features = x_len
#         )
#       )
#     )
#     np.save(f'{fs_outpath}{k}/{run_uid}_rf_selected_n{x_len}.npy',feature_set_dict[k][f'rf_selected_n{x_len}'])
#     sub_end_time = dt.datetime.now()
#     logging.info(f'\t\tSelectFromModel on FRC for {x_len} max features Done: {sub_end_time}')
#   sub_end_time_outer = dt.datetime.now()
#   logging.info(f'\tSelectFromModel on RFC on {k} Done: {sub_end_time_outer}')

len_list = [
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

# # Select random features
# info_index = {
#   'subset':[],
#   'N_features':[],
#   'Method':[]
# }
=======
for k in feature_set_dict.keys():
  # Hierarchical
  sub_start_time = dt.datetime.now()
  hierarchical_start = 1
  hierarchical_end = 30
  # try:
  #   for n in range(hierarchical_start, hierarchical_end):
  #     feature_set_dict[k]['hierarchical_selected_features'][h] = np.load(f'{fs_outpath}{k}/{run_uid}_hierarchical-{k}.npy')
  #   sub_end_time = dt.datetime.now()
  #   logging.info('Previous Hierarchical Feaure Selection Output imported: {sub_end_time}')
  # except:
  sub_start_time = dt.datetime.now()
  logging.info(f'\tHierarchical Feaure Selection ({k}) Started: {sub_start_time}')
  feature_set_dict[k]['hierarchical_selected_features'] = hierarchical_fs_v2(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'],hierarchical_start, hierarchical_end)
  for n in range(hierarchical_start, hierarchical_end):
    #feature_set_dict[k]['hierarchical_selected_features'][n] = hierarchical_fs(feature_set_dict[k]['train_x'],n)
    if len(feature_set_dict[k]['hierarchical_selected_features'][n])>1:
      np.save(f'{fs_outpath}{k}/{run_uid}_hierarchical-{n}.npy',np.array(feature_set_dict[k]['hierarchical_selected_features'][n]))
      print(n)

for k in feature_set_dict.keys():
  # PCA
  sub_end_time = dt.datetime.now()
  logging.info(f'\tHierarchical Feaure Selection ({k}) Done: {sub_end_time}')
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k]['train_pca'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_pca.npy')
    feature_set_dict[k]['test_pca'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_pca.npy')
    feature_set_dict[k]['pca'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_pca.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    logging.info('\tPrevious PCA Output imported: {sub_end_time}')
  except:
    sub_start_time = dt.datetime.now()
    logging.info(f'\tPCA Started: {sub_start_time}')
    train_pca, test_pca, pca = pca_fs(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'], feature_set_dict[k]['test_x'].loc[:,feature_set_dict[k]['test_x'].columns != 'Subject'], k_components=None)
    feature_set_dict[k]['train_pca'] = train_pca
    feature_set_dict[k]['test_pca'] = test_pca
    feature_set_dict[k]['pca'] = pca
    np.save(f'{fs_outpath}{k}/{run_uid}_train_pca.npy',feature_set_dict[k]['train_pca'])
    np.save(f'{fs_outpath}{k}/{run_uid}_test_pca.npy',feature_set_dict[k]['test_pca'])
    pk.dump(feature_set_dict[k]['pca'], open(f'{fs_outpath}{k}/{run_uid}_pca.pkl', "wb"))
    sub_end_time = dt.datetime.now()
    logging.info(f'\tPCA Done: {sub_end_time}')

for k in feature_set_dict.keys():
  # RFC feature selection
  ## Select from model
  sub_start_time_outer = dt.datetime.now()
  logging.info(f'\tSelectFromModel on FRC on {k} started: {sub_start_time_outer}')
  for x in feature_set_dict[k]['hierarchical_selected_features'].keys():
    if x>1 and x<len(feature_set_dict[k]['train_x'].columns):
      # This can be optimized, return to this later
      sub_start_time = dt.datetime.now()
      try:
        feature_set_dict[k][f'rf_selected_{x}'] = np.load(f'{fs_outpath}{k}/{run_uid}_rf_selected_{x}.npy')
        sub_end_time = dt.datetime.now()
        logging.info(f'\t\tSelectFromModel on FRC for {x} max features read from previous run')
      except:
        logging.info(f'\t\tSelectFromModel FRC FS V1 Started: {sub_start_time}')
        feature_set_dict[k][f'rf_selected_{x}'] = list(
          compress(
            list(feature_set_dict[k]['train_x'].columns),
            random_forest_fs(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'],
            np.array(feature_set_dict[k]['train_y']['task']),
            n_estimators = 500,
            n_repeats=10,
            n_jobs=4,
            max_features = x)
          )
        )
        np.save(f'{fs_outpath}{k}/{run_uid}_rf_selected_{x}.npy',feature_set_dict[k][f'rf_selected_{x}'])
        sub_end_time = dt.datetime.now()
        logging.info(f'\t\tSelectFromModel on FRC for {x} max features Done: {sub_end_time}')
  sub_end_time_outer = dt.datetime.now()
  logging.info(f'\tSelectFromModel on FRC on {k} Done: {sub_end_time_outer}')
>>>>>>> master

# for x in len_list:
#   for y in range (10): # Make 10 random sets per feature set size
#     target_columns = random.sample(sorted(feature_set_dict[k]['train_x'].columns[1:]), k=x)
#     info_index['subset'].append(f'Random_{x}_v{y}')
#     info_index['N_features'].append(len(target_columns))
#     info_index['Method'].append('Random')
#     np.save(f'{fs_outpath}{k}/{run_uid}_Random_{x}_v{y}.npy', np.array(target_columns))

# Permutation importance
for k in feature_set_dict.keys():
  ## Permutation importance
  sub_start_time_outer = dt.datetime.now()
  n_estimators = 500
  n_repeats = 50
  try:
    feature_set_dict[k][f'feature_importances_{n_estimators}'] = np.load(f'{fs_outpath}{k}/{run_uid}_feature_importances_est-{n_estimators}.npy')
    logging.info('\tFRC Feature importance and permutation importance on {k} read in from prior run.')
  except:
    logging.info(f'\tFRC Feature importance and permutation importance on {k} started: {sub_start_time_outer}')
    forest = RandomForestClassifier(random_state=42 ,n_estimators=n_estimators)
    forest.fit(
      feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'],
      np.array(feature_set_dict[k]['train_y']['task'])
    )
    now = dt.datetime.now()
    logging.info(f'\tInitial FRC on {n_estimators} estimators from {k} Done: {now}')
    importances = forest.feature_importances_
    np.save(f'{fs_outpath}{k}/{run_uid}_feature_importances_est-{n_estimators}.npy', importances)
    feature_set_dict[k][f'feature_importances_est-{n_estimators}'] = importances
    permutation_importances_result = permutation_importance(
      forest, 
      feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'],
      np.array(feature_set_dict[k]['train_y']['task']),
      n_repeats=n_repeats,
      random_state=42, 
      n_jobs=int(args.n_jobs)
    )
    permutation_importances = pd.Series(
      permutation_importances_result.importances_mean,
      index=feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'].columns)
    feature_set_dict[k][f'permutation_importances_est-{n_estimators}_rep-{n_repeats}'] = permutation_importances
    np.save(f'{fs_outpath}{k}/{run_uid}_permutation_importances_est-{n_estimators}_rep-{n_repeats}.npy', permutation_importances)
<<<<<<< HEAD
    logging.info(f'\tPermutation Importance on {n_estimators} estimators and {n_repeats} repeats from {k} Done: {now}')

# KPCA
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
for k in target_keys:
  for kernel in ['rbf', 'linear']:
    sub_end_time = dt.datetime.now()
    logging.info(f'\tKernelPCA Feature Extraction ({k}) Done: {sub_end_time}')
    try:
      sub_start_time = dt.datetime.now()
      feature_set_dict[k][f'train_kpca-{kernel}'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_kpca-{kernel}.npy')
      feature_set_dict[k][f'test_kpca-{kernel}'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_kpca-{kernel}.npy')
      feature_set_dict[k][f'kpca-{kernel}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_kpca-{kernel}.pkl', 'rb'))
      sub_end_time = dt.datetime.now()
      logging.info('\tPrevious KernelPCA-{kernel} Output imported: {sub_end_time}')
    except:
      sub_start_time = dt.datetime.now()
      logging.info(f'\tKernelPCA-{kernel} Started: {sub_start_time}')
      train_kpca, test_kpca, kpca = kpca_fs(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'], feature_set_dict[k]['test_x'].loc[:,feature_set_dict[k]['test_x'].columns != 'Subject'], n_components=None, kernel = kernel, n_jobs=14)
      feature_set_dict[k]['train_kpca-{kernel}'] = train_kpca
      feature_set_dict[k]['test_kpca-{kernel}'] = test_kpca
      feature_set_dict[k]['kpca-{kernel}'] = kpca
      np.save(f'{fs_outpath}{k}/{run_uid}_train_kpca-{kernel}.npy',feature_set_dict[k]['train_kpca-{kernel}'])
      np.save(f'{fs_outpath}{k}/{run_uid}_test_kpca-{kernel}.npy',feature_set_dict[k]['test_kpca-{kernel}'])
      pk.dump(feature_set_dict[k]['kpca-{kernel}'], open(f'{fs_outpath}{k}/{run_uid}_kpca-{kernel}.pkl', "wb"))
      sub_end_time = dt.datetime.now()
      logging.info(f'\tKernelPCA-{kernel} Done: {sub_end_time}')

##### Visualizing Eigenvector to select N vars
# import pickle as pk
# import plotly.express as px
# run_uid = '89952a'
# fs_outpath = 'S:\\hcp_analysis_output\\89952a\\FeatureSelection\\parcel_connection\\'
# k = 'parcel_connection'
# for kernel in ['rbf', 'linear']:
#   kpca = pk.load(open(f'{fs_outpath}{k}\\{run_uid}_kpca-{kernel}.pkl', 'rb'))
#   print(kpca.lambdas_[:60])
#   fig = px.line(kpca.lambdas_[:60])
#   fig.show()

# TruncatedSVD
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
for k in target_keys:
  for component_size in len_list:
    try:
      sub_end_time = dt.datetime.now()
      logging.info(f'\tTruncatedSVD Feature Extraction ({k}) Started: {sub_end_time}')
      try:
        sub_start_time = dt.datetime.now()
        feature_set_dict[k][f'train_tSVD-{component_size}'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_tSVD-{component_size}.npy')
        feature_set_dict[k][f'test_tSVD-{component_size}'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_tSVD-{component_size}.npy')
        feature_set_dict[k][f'tSVD-{component_size}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_tSVD-{component_size}.pkl', 'rb'))
        sub_end_time = dt.datetime.now()
        logging.info('\tPrevious TruncatedSVD-{component_size} Output imported: {sub_end_time}')
      except:
        sub_start_time = dt.datetime.now()
        logging.info(f'\tTruncatedSVD-{component_size} Started: {sub_start_time}')
        train_tSVD, test_tSVD, tSVD = tSVD_fs(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'], feature_set_dict[k]['test_x'].loc[:,feature_set_dict[k]['test_x'].columns != 'Subject'], n_components=component_size, )
        feature_set_dict[k]['train_tSVD-{component_size}'] = train_tSVD
        feature_set_dict[k]['test_tSVD-{component_size}'] = test_tSVD
        feature_set_dict[k]['tSVD-{component_size}'] = tSVD
        np.save(f'{fs_outpath}{k}/{run_uid}_train_tSVD-{component_size}.npy',feature_set_dict[k]['train_tSVD-{component_size}'])
        np.save(f'{fs_outpath}{k}/{run_uid}_test_tSVD-{component_size}.npy',feature_set_dict[k]['test_tSVD-{component_size}'])
        pk.dump(feature_set_dict[k]['tSVD-{component_size}'], open(f'{fs_outpath}{k}/{run_uid}_tSVD-{component_size}.pkl', "wb"))
        sub_end_time = dt.datetime.now()
        logging.info(f'\tTruncatedSVD-{component_size} Done: {sub_end_time}')
    except:
      print(f'tsvd failed for size {component_size}')

# TSNE 
# (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
# BAD; no transform function, only fit_transform
for k in target_keys:
  for component_size in [2]:
    sub_end_time = dt.datetime.now()
    logging.info(f'\TSNE Feature Extraction ({k}) Done: {sub_end_time}')
    try:
      sub_start_time = dt.datetime.now()
      feature_set_dict[k][f'train_TSNE-{component_size}'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_TSNE-{component_size}.npy')
      feature_set_dict[k][f'test_TSNE-{component_size}'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_TSNE-{component_size}.npy')
      feature_set_dict[k][f'TSNE-{component_size}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_TSNE-{component_size}.pkl', 'rb'))
      sub_end_time = dt.datetime.now()
      logging.info('\tPrevious TruncatedSVD-{component_size} Output imported: {sub_end_time}')
    except:
      sub_start_time = dt.datetime.now()
      logging.info(f'\TSNE-{component_size} Started: {sub_start_time}')
      train_TSNE, test_TSNE, TSNE = TSNE_fs(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'], feature_set_dict[k]['test_x'].loc[:,feature_set_dict[k]['test_x'].columns != 'Subject'], n_components=component_size, n_jobs = 14)
      feature_set_dict[k]['train_TSNE-{component_size}'] = train_TSNE
      feature_set_dict[k]['test_TSNE-{component_size}'] = test_TSNE
      feature_set_dict[k]['TSNE-{component_size}'] = TSNE
      np.save(f'{fs_outpath}{k}/{run_uid}_train_TSNE-{component_size}.npy',feature_set_dict[k]['train_TSNE-{component_size}'])
      np.save(f'{fs_outpath}{k}/{run_uid}_test_TSNE-{component_size}.npy',feature_set_dict[k]['test_TSNE-{component_size}'])
      pk.dump(feature_set_dict[k]['TSNE-{component_size}'], open(f'{fs_outpath}{k}/{run_uid}_TSNE-{component_size}.pkl', "wb"))
      sub_end_time = dt.datetime.now()
      logging.info(f'\TSNE-{component_size} Done: {sub_end_time}')

# ICA 
# (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
for k in target_keys:
  sub_end_time = dt.datetime.now()
  logging.info(f'\ICA Feature Extraction ({k}) Done: {sub_end_time}')
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k][f'train_ICA'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_ICA.npy')
    feature_set_dict[k][f'test_ICA'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_ICA.npy')
    feature_set_dict[k][f'ICA'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_ICA.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    logging.info('\tPrevious ICA Output imported: {sub_end_time}')
  except:
    sub_start_time = dt.datetime.now()
    logging.info(f'\ICA Started: {sub_start_time}')
    train_ICA, test_ICA, ICA = ICA_fs(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'], feature_set_dict[k]['test_x'].loc[:,feature_set_dict[k]['test_x'].columns != 'Subject'])
    feature_set_dict[k]['train_ICA'] = train_ICA
    feature_set_dict[k]['test_ICA'] = test_ICA
    feature_set_dict[k]['ICA'] = ICA
    np.save(f'{fs_outpath}{k}/{run_uid}_train_ICA.npy',feature_set_dict[k]['train_ICA'])
    np.save(f'{fs_outpath}{k}/{run_uid}_test_ICA.npy',feature_set_dict[k]['test_ICA'])
    pk.dump(feature_set_dict[k]['ICA'], open(f'{fs_outpath}{k}/{run_uid}_ICA.pkl', "wb"))
    sub_end_time = dt.datetime.now()
    logging.info(f'\ICA Done: {sub_end_time}')

# LE 
# (SpectralEmbedding: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html)
# Did not work, no transform function, only fit_transform
for k in target_keys:
  sub_end_time = dt.datetime.now()
  logging.info(f'\LE Feature Extraction ({k}) Done: {sub_end_time}')
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k][f'train_LE'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_LE.npy')
    feature_set_dict[k][f'test_LE'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_LE.npy')
    feature_set_dict[k][f'LE'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_LE.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    logging.info('\tPrevious LE Output imported: {sub_end_time}')
  except:
    sub_start_time = dt.datetime.now()
    logging.info(f'\LE Started: {sub_start_time}')
    train_LE, test_LE, LE = LE_fs(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'], feature_set_dict[k]['test_x'].loc[:,feature_set_dict[k]['test_x'].columns != 'Subject'], n_jobs = 14)
    feature_set_dict[k]['train_LE'] = train_LE
    feature_set_dict[k]['test_LE'] = test_LE
    feature_set_dict[k]['LE'] = LE
    np.save(f'{fs_outpath}{k}/{run_uid}_train_LE.npy',feature_set_dict[k]['train_LE'])
    np.save(f'{fs_outpath}{k}/{run_uid}_test_LE.npy',feature_set_dict[k]['test_LE'])
    pk.dump(feature_set_dict[k]['LE'], open(f'{fs_outpath}{k}/{run_uid}_LE.pkl', "wb"))
    sub_end_time = dt.datetime.now()
    logging.info(f'\LE Done: {sub_end_time}')


# MDS 
# (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)
# Did not work, no transform function, only fit_transform
for k in target_keys:
  sub_end_time = dt.datetime.now()
  logging.info(f'\MDS Feature Extraction ({k}) Done: {sub_end_time}')
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k][f'train_MDS'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_MDS.npy')
    feature_set_dict[k][f'test_MDS'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_MDS.npy')
    feature_set_dict[k][f'MDS'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_MDS.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    logging.info('\tPrevious MDS Output imported: {sub_end_time}')
  except:
    sub_start_time = dt.datetime.now()
    logging.info(f'\MDS Started: {sub_start_time}')
    train_MDS, test_MDS, MDS = MDS_fs(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns != 'Subject'], feature_set_dict[k]['test_x'].loc[:,feature_set_dict[k]['test_x'].columns != 'Subject'], n_jobs = 14)
    feature_set_dict[k]['train_MDS'] = train_MDS
    feature_set_dict[k]['test_MDS'] = test_MDS
    feature_set_dict[k]['MDS'] = MDS
    np.save(f'{fs_outpath}{k}/{run_uid}_train_MDS.npy',feature_set_dict[k]['train_MDS'])
    np.save(f'{fs_outpath}{k}/{run_uid}_test_MDS.npy',feature_set_dict[k]['test_MDS'])
    pk.dump(feature_set_dict[k]['MDS'], open(f'{fs_outpath}{k}/{run_uid}_MDS.pkl', "wb"))
    sub_end_time = dt.datetime.now()
    logging.info(f'\MDS Done: {sub_end_time}')


# LDA 
# (https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
for k in target_keys:
  sub_end_time = dt.datetime.now()
  logging.info(f'\LDA Feature Extraction ({k}) Started: {sub_end_time}')
  try:
    sub_start_time = dt.datetime.now()
    feature_set_dict[k][f'train_LDA'] = np.load(f'{fs_outpath}{k}/{run_uid}_train_LDA.npy')
    feature_set_dict[k][f'test_LDA'] = np.load(f'{fs_outpath}{k}/{run_uid}_test_LDA.npy')
    feature_set_dict[k][f'LDA'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_LDA.pkl', 'rb'))
    sub_end_time = dt.datetime.now()
    logging.info('\tPrevious LDA Output imported: {sub_end_time}')
  except:
    sub_start_time = dt.datetime.now()
    logging.info(f'\LDA Started: {sub_start_time}')
    train_LDA, test_LDA, LDA = LDA_fs(feature_set_dict[k]['train_x'].loc[:,feature_set_dict[k]['train_x'].columns!='Subject'], feature_set_dict[k]['test_x'].loc[:,feature_set_dict[k]['test_x'].columns!='Subject'], feature_set_dict[k]['train_y'].values.ravel())
    feature_set_dict[k]['train_LDA'] = train_LDA
    feature_set_dict[k]['test_LDA'] = test_LDA
    feature_set_dict[k]['LDA'] = LDA
    np.save(f'{fs_outpath}{k}/{run_uid}_train_LDA.npy',feature_set_dict[k]['train_LDA'])
    np.save(f'{fs_outpath}{k}/{run_uid}_test_LDA.npy',feature_set_dict[k]['test_LDA'])
    pk.dump(feature_set_dict[k]['LDA'], open(f'{fs_outpath}{k}/{run_uid}_LDA.pkl', "wb"))
    sub_end_time = dt.datetime.now()
    logging.info(f'\LDA Done: {sub_end_time}')
=======
    logging.info(f'\tPermutation Importance on {n_estimators} estimators and {n_repeats} repeats from {k} Done: {now}')
>>>>>>> master
