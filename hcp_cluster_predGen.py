#!/usr/bin/env python3
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


print(sys.argv)

def parse_args(args):
  parser = argparse.ArgumentParser(
    description='Analysis Script for task decoding.'
  )
  parser.add_argument(
    "-n_features",
    help="intiger specifying how many features will be included",
    required=True,
    default=None
  )
  parser.add_argument(
    "-version",
    help="intiger specifying which version of the randomly selected features to use (0-9)",
    required=True,
    default=None
  )
  parser.add_argument(
    "-split",
    help="intiger specifying which split to use (0-9)",
    required=True,
    default=None
  )
  return parser.parse_known_args(args)

def generate_uid(metadata, length = 8):
  dhash = hashlib.md5()
  encoded = json.dumps(metadata, sort_keys=True).encode()
  dhash.update(encoded)
  # You can change the 8 value to change the number of characters in the unique id via truncation.
  run_uid = dhash.hexdigest()[:length]
  return run_uid

# args, leftovers = parse_args(sys.argv[1:])

args, leftovers = parse_args(['-n_features', '19900', '-version', '0', '-split', '0'])

# Set Global Variables
k = 'parcel_connection'
outpath = '/data/hx-hx1/kbaacke/datasets/hcp_analysis_output/8d2513/'
input_dir = f'{outpath}inputs/'
meta_path = f'{outpath}metadata/'
confusion_path = f'{outpath}confusion/'
classification_path = f'{outpath}classification/'
weight_path = f'{outpath}weights/'
fs_outpath = '/data/hx-hx1/kbaacke/datasets/hcp_analysis_output/8d2513/FeatureSelection/'
temp_accuracy_dir = f'{outpath}Accuracies_Temp/'
run_uid = '8d2513'
sep = '/'
random_state=42
meta_dict = json.load(open(f'{outpath}{run_uid}metadata.json'))
metadata = meta_dict
metadata['random_state'] = random_state

# Define predGen functions
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
    'N_Features':[len(train_x.columns)],
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
    'N_Features':[len(train_x.columns)],
    'metadata_ref':[pred_uid],
    'runtime':(end_time - start_time).total_seconds()
    # 'classification_report':classification_rep,
    # 'confusion_matrix':confusion_mat
  }
  forest_param_df.to_csv(f'{weight_path}{pred_uid}_weights.csv', index=False)
  np.savetxt(f'{confusion_path}{pred_uid}_confusion_matrix.csv', confusion_mat, delimiter=",")
  np.savetxt(f'{classification_path}{pred_uid}_classification_report.csv', confusion_mat, delimiter=",")
  return results_dict

# Read in full training set
# In try loop till 2 min have passed
# To solve possible error reading in two jobs at the same time
indlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
random.seed(int(args.n_features)*int(args.n_samples)*int(args.version))
random.shuffle(indlist)

ind = 0
tf = True
while tf:
  copy_number = indlist[ind]
  try:
    data = pd.DataFrame(np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_x_c{copy_number}.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_x_colnames_c{copy_number}.npy', allow_pickle=True))
    tf = False
  except Exception as e:
    try_count+=1
    if try_count>30:
      raise(e)

# Read in vector containing selected column names
subset = list(np.load(f'{fs_outpath}{k}{sep}{run_uid}_Random_{args.n_features}_v{args.version}.npy', allow_pickle=True))
# Read in split indices
ind = args.split
train_ind = np.load(f'{fs_outpath}{k}{sep}{run_uid}_split_{ind}_train.npy', allow_pickle=True)
test_ind = np.load(f'{fs_outpath}{k}{sep}{run_uid}_split_{ind}_test.npy', allow_pickle=True)
# Filter input data
data_train = data.iloc[train_ind][subset].copy()
data_test = data.iloc[test_ind][subset].copy()
del(data)
# Read in outcome vector
try_count = 1
tf = True
while tf:
  try:
    outcome = pd.DataFrame(np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_y_c{try_count}.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_y_colnames_c{try_count}.npy', allow_pickle=True))
    tf = False
  except Exception as e:
    try_count+=1
    if try_count>30:
      raise(e)

# Subset outcome vector by split indices
outcome_train = outcome.iloc[train_ind].values.ravel().astype('int')
outcome_test = outcome.iloc[test_ind].values.ravel().astype('int')
del(outcome)

df_concat_list = []
# Run Models
try:
  svc_out = run_svc(data_train, outcome_train, data_test, outcome_test, 'HCP', f'Random_f{args.n_features}_v{args.version}', ind, method='Random')
  df_concat_list.append(pd.DataFrame(svc_out))
except Exception as e:
  print(e)
  rfc_out = {
    'dataset':['HCP'],
    'subset':[f'Random_{args.n_features}_v{args.version}'],
    'split_ind':[ind],
    'Classifier':['Support Vector Machine'],
    'train_accuracy':['NA'],
    'test_accuracy':['NA'],
    'ICA_Aroma':[metadata['ICA-Aroma']],
    'metadata_ref':['NA'],
    'runtime':['NA']
  }
  df_concat_list.append(pd.DataFrame(rfc_out))

try:
  rfc_out = run_rfc(data_train, outcome_train, data_test, outcome_test, 'HCP', f'Random_{args.n_features}_v{args.version}', ind, method='Random')
  df_concat_list.append(pd.DataFrame(rfc_out))
except Exception as e:
  print(e)
  rfc_out = {
    'dataset':['HCP'],
    'subset':[f'Random_{args.n_features}_v{args.version}'],
    'split_ind':[ind],
    'Classifier':['Random Forest'],
    'train_accuracy':['NA'],
    'test_accuracy':['NA'],
    'ICA_Aroma':[metadata['ICA-Aroma']],
    'metadata_ref':['NA'],
    'runtime':['NA']
  }
  df_concat_list.append(pd.DataFrame(rfc_out))

if len(df_concat_list)>1:
  out_df = pd.concat(df_concat_list)
else:
  out_df = df_concat_list[0]

out_df.to_csv(f'{temp_accuracy_dir}f{args.n_features}_v{args.version}_s{args.split}.csv', index=False)