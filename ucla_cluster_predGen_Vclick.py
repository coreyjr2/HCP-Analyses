#!/usr/bin/env python3

# To run models for all the random feature sets on each machine in it's own ssh session

import numpy as np
import argparse
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import datetime as dt
import json
import hashlib
import time
import random
import platform
import os
import re
from joblib import Parallel, delayed

def generate_uid(metadata, length = 8):
  dhash = hashlib.md5()
  encoded = json.dumps(metadata, sort_keys=True).encode()
  dhash.update(encoded)
  # You can change the 8 value to change the number of characters in the unique id via truncation.
  run_uid = dhash.hexdigest()[:length]
  return run_uid

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

def run_model(classifier, train_x, train_y, test_x, test_y, dataset, subset, split_ind, method, C=1, n_estimators=500, max_depth = None):
  if classifier=='SVC':
    return run_svc( train_x, train_y, test_x, test_y, dataset, subset, split_ind, method, C=C)
  elif classifier=='Random Forest':
    return run_rfc( train_x, train_y, test_x, test_y, dataset, subset, split_ind, method, n_estimators=n_estimators, max_depth = max_depth)

k = 'parcel_connection'
run_uid = '89952a'
outpath = f'/data/hx-hx1/kbaacke/datasets/ucla_analysis_output/{run_uid}/'
input_dir = f'{outpath}inputs/'
meta_path = f'{outpath}metadata/'
confusion_path = f'{outpath}confusion/'
classification_path = f'{outpath}classification/'
weight_path = f'{outpath}weights/'
fs_outpath = f'/data/hx-hx1/kbaacke/datasets/ucla_analysis_output/{run_uid}/FeatureSelection/'
temp_accuracy_dir = f'{outpath}Accuracies_Temp/'

sep = '/'
random_state=42
meta_dict = json.load(open(f'{outpath}{run_uid}metadata.json'))
metadata = meta_dict
metadata['random_state'] = random_state
node = platform.node()


data = pd.DataFrame(np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_x.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_x_colnames.npy', allow_pickle=True))
outcome = pd.DataFrame(np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_y.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_y_colnames.npy', allow_pickle=True))

file_names = os.listdir(f'{fs_outpath}{k}')

target_file_names = []
for fname in file_names:
  if 'Random' in fname:
    target_file_names.append(fname)


target_file_names.sort()


#### Change Here ####
n_jobs = 14
for runtime_ind in [1,2,3,4,5,6,7,8,9,10]:
  # runtime_ind = 1 # Change this each time 1 2 3 4 5 6 7 8 9 10
  ########
  print(runtime_ind)
  key_inds = [
    0, 117, 234, 351, 468,
    585, 702, 819, 936, 1053, 1170
  ]
  start_ind = key_inds[runtime_ind-1]#0
  end_ind = key_inds[runtime_ind]#117
  subset_file_list = target_file_names[start_ind:end_ind]
  fs_dict = {}
  for fname in subset_file_list:
    sval = re.search(("[.]*_Random_(.+?)_v(.+?).npy"), fname)
    n_features = sval[1]
    version = sval[2]
    if n_features not in fs_dict.keys():
      fs_dict[n_features] = {}
    fs_dict[n_features][version] = np.load(f'{fs_outpath}{k}{sep}{fname}', allow_pickle=True)
  cv_split_dict = {}
  for ind in range(10):
    train_ind = np.load(f'{fs_outpath}{k}{sep}{run_uid}_split_{ind}_train.npy', allow_pickle=True)
    test_ind = np.load(f'{fs_outpath}{k}{sep}{run_uid}_split_{ind}_test.npy', allow_pickle=True)
    cv_split_dict[ind] = (train_ind, test_ind)
  for n_features in list(fs_dict.keys()):
    n_dict = fs_dict[n_features]
    for version in n_dict.keys():
      print(n_features, version)
      data_dict = {}
      # try:
      accuracy_df_node = pd.read_csv(f'{outpath}Prediction_Accuracies.csv')
      df_concat_list = [accuracy_df_node]
      # except:
      #   df_concat_list = []
      #   pass
      input_subset = data[list(n_dict[version])]
      for split_ind in cv_split_dict.keys():
        data_dict[split_ind] = {
          'train_x':input_subset.iloc[cv_split_dict[split_ind][0]],
          'train_y':outcome.iloc[cv_split_dict[split_ind][0]]['task'].values.ravel().astype('int'),
          'test_x':input_subset.iloc[cv_split_dict[split_ind][1]],
          'test_y':outcome.iloc[cv_split_dict[split_ind][1]]['task'].values.ravel().astype('int')
        }
      classifiers = ['SVC', 'Random Forest']
      results_list = Parallel(n_jobs = n_jobs)(
        delayed(run_model)(
          classifier=classifier,
          train_x = data_dict[split_ind]['train_x'],
          train_y = data_dict[split_ind]['train_y'],
          test_x = data_dict[split_ind]['test_x'],
          test_y = data_dict[split_ind]['test_y'],
          dataset = 'UCLA',
          subset = f'Random_{n_features}_v{version}',
          split_ind = split_ind,
          method='Random'
          ) for split_ind in cv_split_dict.keys() for classifier in classifiers
        )
      for res_dict in results_list:
        df_concat_list.append(pd.DataFrame(res_dict))
      accuracy_df_node = pd.concat(df_concat_list)
      print(accuracy_df_node)
      datetime_str = dt.datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
      # accuracy_df_node.drop_duplicates(keep='last', subset='metadata_ref', inplace=True)
      accuracy_df_node.to_csv(f'{outpath}Prediction_Accuracies.csv', index=False)
      # accuracy_df_node.to_csv(f'{outpath}Prediction_Accuracies_{node}_{datetime_str}.csv', index=False)

