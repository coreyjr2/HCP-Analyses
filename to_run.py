for k in feature_set_dict.keys():
  sub_start_time = dt.datetime.now()
  logging.info(f'\tPCA Started: {sub_start_time}')
  train_pca_auto, test_pca_auto, pca_auto = pca_fs(feature_set_dict[k]['train_x'], feature_set_dict[k]['test_x'], k_components=None)
  feature_set_dict[k]['train_pca_auto'] = train_pca_auto
  feature_set_dict[k]['test_pca_auto'] = test_pca_auto
  feature_set_dict[k]['pca_auto'] = pca_auto
  np.save(f'{fs_outpath}{k}/{run_uid}_train_pca-auto.npy',feature_set_dict[k]['train_pca_auto'])
  np.save(f'{fs_outpath}{k}/{run_uid}_test_pca-auto.npy',feature_set_dict[k]['test_pca_auto'])
  pk.dump(feature_set_dict[k]['pca_auto'], open(f'{fs_outpath}{k}/{run_uid}_pca-auto.pkl', "wb"))
  for x in feature_set_dict[k]['hierarchical_selected_features'].keys():
    if x>1:
      print(x, len(feature_set_dict[k]['hierarchical_selected_features'][x]))
      train_pca_auto, test_pca_auto, pca_auto = pca_fs(feature_set_dict[k]['train_x'], feature_set_dict[k]['test_x'], k_components=None)
      feature_set_dict[k][f'train_pca_{x}'] = train_pca_auto
      feature_set_dict[k][f'test_pca_{x}'] = test_pca_auto
      feature_set_dict[k][f'pca_{x}'] = pca_auto
      np.save(f'{fs_outpath}{k}/{run_uid}_train_pca-{x}.npy',feature_set_dict[k][f'train_pca_{x}'])
      np.save(f'{fs_outpath}{k}/{run_uid}_test_pca-{x}.npy',feature_set_dict[k][f'test_pca_{x}'])
      pk.dump(feature_set_dict[k][f'pca_{x}'], open(f'{fs_outpath}{k}/{run_uid}_pca-{x}.pkl', "wb"))
  sub_end_time = dt.datetime.now()
  logging.info(f'\tPCA Done: {sub_end_time}')








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
import platform
from joblib import Parallel, delayed

# Global Variables
sep = os.path.sep
source_path = 'C:/Users/kyle/repos/HCP-Analyses/'
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

def str_combine(x, y):
  return str(x) + str(y)
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
def generate_parcel_connection_features_V2(con_dict, labels): # Tested
  out_dict = {}
  out_df_dict = {}
  for session in con_dict.keys():
    parcel_dict = con_dict[session]
    out_dict[session] = {}
    def con_to_vetor(subject):
      cor_coef = parcel_dict[subject]
      vect = list(cor_coef[np.triu_indices_from(cor_coef, k=1)])
      vect.append(subject)
      out_dict[session][subject] = vect
    # for subject in parcel_dict.keys():
    #   cor_coef = parcel_dict[subject]
    #   out_dict[session][subject] = list(cor_coef[np.triu_indices_from(cor_coef, k=1)])
    #   out_dict[session][subject].append(subject)
    Parallel(n_jobs=6)(delayed(con_to_vetor)(SUBJECT) for SUBJECT in parcel_dict.keys())
    cor_coef_ex = parcel_dict[list(parcel_dict.keys())[0]]
    colnames = connection_names(cor_coef_ex, labels)
    colnames.append('Subject')
    out_df_dict[session] = pd.DataFrame.from_dict(out_dict[session], orient='index', columns = colnames)
    sub = out_df_dict[session]['Subject']
    out_df_dict[session].drop(labels=['Subject'], axis=1, inplace=True)
    out_df_dict[session].insert(0, 'Subject', sub)
    out_df_dict[session].insert(0, 'task',numeric_task_ref[session])
  parcels_connections_full = pd.DataFrame(pd.concat(list(out_df_dict.values()), axis = 0))
  return parcels_connections_full
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
def random_forest_fs(x, y, n_estimators, n_repeats=10, n_jobs=1):
  #Returns a list of columns to use as features
  sel = SelectFromModel(RandomForestClassifier(n_estimators = n_estimators, n_jobs=n_jobs, random_state=42), max_features=500)
  sel.fit(x,y)
  return list(sel.get_support())
def random_forest_fs_v2(x, y, n_estimators, n_repeats=10, n_jobs=1):
  #Returns a list of columns to use as features
  forest = RandomForestClassifier(random_state=42 ,n_estimators=n_estimators)
  forest.fit(x,y)
  result = permutation_importance(forest, x, y, n_repeats=10, random_state=42, n_jobs=n_jobs)
  forest_importances = pd.Series(result.importances_mean, index=x.columns)
  return forest_importances
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

class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)
args = {
  'source_path':source_path,
  'local_path':'S:\\HCP\\HCP_69354adf\\',
  'output':'C:\\Users\\kyle\\repos\\GTM-FeatureReduction\\' + f'Output{sep}',
  'n_jobs':4,
  'atlas_path':'S:\\HCP\\HCP_69354adf\\69354adf_parcellation-metadata.json',
  'confound_subset':[],
  'confounds':None,
  'movement_regressor':None,
  'ica_aroma':False,
  'smoothed':False,
  'no_concatenate':False
}
args['atlas_name'] = os.path.basename(args['atlas_path'])[:8] # Pull the atlas_name from the atlas_path variable (UID)

parcellation_dict = json.load(open(args['atlas_path']))

actual_atlas_name = parcellation_dict['atlas_name']

meta_dict = {
  'atlas_name' : args['atlas_name'],
  'smoothed' : args['smoothed'],
  'ICA-Aroma' : args['ica_aroma'],
  # 'confounds': args2.,
  'Random Forest Estimators': 1000,
  'Random State':42,
  'subtract parcel-wise mean': True, # Fixing this at true, no cli arg
  'concatenate':(not args['no_concatenate'])
}
args2 = Bunch(args)

if args2.confounds != None:
  demographics = pd.read_csv(args2.confounds)
  if args2.confound_subset != None:
    demographics = demographics[args2.confound_subset]
  if args2.movement_regressor is None:
    confounds = demographics


if args2.movement_regressor != None:
  relative_RMS_means_collapsed = pd.read_csv(args2.movement_regressor)
  if args2.confounds is None:
    confounds = relative_RMS_means_collapsed


if (args2.movement_regressor != None) and (args2.confounds != None):
  confounds = pd.merge(relative_RMS_means_collapsed, demographics, how='left', on='Subject')

try:
  meta_dict['confounds'] = list(confounds.columns)
except:
  meta_dict['confounds'] = None

def generate_uid(meta_dict, hash_len=6):
  dhash = hashlib.md5()
  encoded = json.dumps(meta_dict, sort_keys=True).encode()
  dhash.update(encoded)
  run_uid = dhash.hexdigest()[:hash_len]
  return run_uid
run_uid = generate_uid(meta_dict)
outpath = args2.output + run_uid + sep
try:
  os.makedirs(outpath)
except Exception as e:
  print(e, 'Output directory already created.')


with open(outpath + run_uid + 'metadata.json', 'w') as outfile:
  json.dump(meta_dict, outfile)

total_start_time = dt.datetime.now()
logging.basicConfig(filename=f'{run_uid}_DEBUG.log', level=logging.DEBUG) # Set level to DEBUG for detailed output
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

sessions = [
  "tfMRI_MOTOR",
  "tfMRI_WM",
  "tfMRI_EMOTION",
  "tfMRI_GAMBLING",
  "tfMRI_LANGUAGE",
  "tfMRI_RELATIONAL",
  "tfMRI_SOCIAL"
]
con_template = '{basepath}HCP{sep}HCP_1200{sep}{subject}{sep}MNINonLinear{sep}Results{sep}{atlas_name}_{session}_CorrlationMatrix.npy'
basepath = args2.local_path
HCP_1200 = f'{basepath}{sep}HCP{sep}HCP_1200{sep}'
parcel_labels, network_labels = fetch_labels(meta_dict, os.path.dirname(args2.atlas_path)+ sep)
atlas_name = run_uid
subjects = []
for f in os.listdir(HCP_1200):
  if len(f)==6:
    subjects.append(f)
con_dict = {}

from joblib import Parallel, delayed

for session in sessions:
  con_dict[session] = {}
  def read_con(subject):
    try:
      ar = np.load(con_template.format(basepath = basepath, sep = sep, subject = subject, atlas_name = atlas_name, session = session))
      con_dict[session][subject] = ar
    except:
      pass
  Parallel(n_jobs=6)(delayed(read_con)(SUBJECT) for SUBJECT in subjects)
  print(len(con_dict[session].keys()))