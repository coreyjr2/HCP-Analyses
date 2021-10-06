#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import platform
import json
import hashlib
import getpass
import pandas as pd
import datetime as dt
import nibabel as nib
import numpy as np
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, KFold, GridSearchCV

from sklearn import svm
from pathlib import Path

sep = os.path.sep
source_path = os.path.dirname(os.path.abspath(__file__)) + sep
sys_name = platform.system() 
parcel_ref = { #Parcels, unique values from cor matrix, N networks;   #Unused#
    'harvard_oxford':(96,4560), #(((1+p)p)/2)-p=.5p(p-1) lower diagonal
    'msdl':(39,741),
    'mni_glasser':(360,64620),
    'yeo_7_thin':(7,21),
    'yeo_7_thick':(7,21),
    'yeo_17_thin':(17,136),
    'yeo_17_thick':(17,136),
  }

numeric_task_ref = {
  "tfMRI_MOTOR":4,
  "tfMRI_WM":7,
  "tfMRI_EMOTION":1,
  "tfMRI_GAMBLING":2,
  "tfMRI_LANGUAGE":3,
  "tfMRI_RELATIONAL":5,
  "tfMRI_SOCIAL":6
}

def create_ordered_network_labels():
  gregions = pd.DataFrame(np.load(source_path + "glasser_regions.npy"), columns=['Label','network','unkown'])
  gregions = gregions[['Label','network']]
  glabels = pd.read_csv(source_path + 'Glasser_labels.csv')
  full_label_file = pd.merge(glabels, gregions, how='left',on='Label')
  full_label_file.to_csv(source_path + 'mni_glasser_info.csv', index=False)

def parcellate_timeseries(nifty_file, atlas_name, confounds=None): # Tested
  # Other atlases in MNI found here: https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
  raw_timeseries = nib.load(nifty_file)
  if atlas_name=='harvard_oxford':
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
    atlas_filename = atlas.maps
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
  elif atlas_name == 'msdl':
    atlas = datasets.fetch_atlas_msdl()
    atlas_filename = atlas.maps
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='nilearn_cache')
  elif 'yeo' in atlas_name:
    yeo = datasets.fetch_atlas_yeo_2011()
    if atlas_name == 'yeo_7_thin':
      masker = NiftiLabelsMasker(labels_img=yeo['thin_7'], standardize=True,memory='nilearn_cache')
    elif atlas_name == 'yeo_7_thick':
      masker = NiftiLabelsMasker(labels_img=yeo['thick_7'], standardize=True,memory='nilearn_cache')
    elif atlas_name == 'yeo_17_thin':
      masker = NiftiLabelsMasker(labels_img=yeo['thin_17'], standardize=True,memory='nilearn_cache')
    elif atlas_name == 'yeo_17_thick':
      masker = NiftiLabelsMasker(labels_img=yeo['thick_17'], standardize=True,memory='nilearn_cache')
  elif atlas_name == 'mni_glasser':
    atas_glasser_01_filename = source_path + 'MMP_in_MNI_corr.nii.gz' # Downlaoded from https://neurovault.org/collections/1549/
    masker = NiftiLabelsMasker(labels_img=atas_glasser_01_filename, standardize=True)
  elif 'Schaefer2018_' in atlas_name:
    atlas_filename = source_path + atlas_name + '.nii.gz'
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
  else:
    return NotImplementedError
  #Transform the motor task imaging data with the masker and check the shape
  masked_timeseries = []
  if confounds is not None:
    masked_timeseries = masker.fit_transform(raw_timeseries, counfounds = confounds)
  else:
    masked_timeseries = masker.fit_transform(raw_timeseries)
  return masked_timeseries

def load_parcellated_task_timeseries(meta_dict, nifty_template, subjects, session, npy_template = None, run_names = ['RL','LR'], confounds_path = None): # Tested
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
        if confounds_path is not None:
          confounds = confounds_path.format(subject = subject, run = run, session = session)
        else:
          confounds = None
        data_path_template = nifty_template.format(subject = subject, session = session, run = run)
        if npy_template is not None:
          try:
            # First try to load in numpy file
            masked_timeseries = np.load(npy_template.format(subject=subject, session=session, run=run, atlas_name=atlas_name))
          except:
            masked_timeseries = parcellate_timeseries(data_path_template, atlas_name, confounds)
            np.save(npy_template.format(subject=subject, session=session, run=run, atlas_name=atlas_name), masked_timeseries)
        else:
          masked_timeseries = parcellate_timeseries(data_path_template, atlas_name, confounds)
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

def tune_hyperparams():
  pass

def run_svc_new(train_x, train_y, test_x, test_y, random_state = 42, k=5):
  '''
  performs k-fold cross validation on the training data
  tests the winning model from cross-validation on the test data
  within k-folds, also iterates through values for C
  returns out_dict
  *new keys for out_dict:
    C
    k
    winning_fold
    fold_accuracy
  '''
  k_fold = KFold(n_splits=k)
  l_svc =  svm.SVC(kernel='linear')
  SVCs = []
  accuracies = []
  folds = []
  fold = 0
  C_s = np.logspace(-10, 0, 10)
  Cs = []
  for train_indices, test_indices in k_fold.split(train_x):
    fold+=1
    for C in C_s:
      l_svc_temp = svm.SVC(kernel='linear')
      l_svc_temp.C = C
      l_svc_temp.fit(train_x.iloc[train_indices], train_y.iloc[train_indices])
      accuracies.append(l_svc_temp.score(train_x.iloc[test_indices], train_y.iloc[test_indices]))
      SVCs.append(l_svc_temp)
      folds.append(fold)
      Cs.append(C) # Just to check that you have the right model and .copy() is working
  max_training_accuracy = max(accuracies)
  max_index = accuracies.index(max_training_accuracy)
  l_svc = SVCs[max_index]
  y_pred = l_svc.predict(test_x)
  training_accuracy = l_svc.score(train_x, train_y)
  test_accuracy = l_svc.score(test_x, test_y)
  classification_rep = classification_report(test_y, y_pred)
  confusion_mat = confusion_matrix(test_y, y_pred)
  out_dict = {
    'Training Accuracy':training_accuracy,
    'Test Accuracy':test_accuracy,
    'Classification Report':classification_rep,
    'Confusion Matrix': confusion_mat,
    'Training N':len(train_x),
    'Test N':len(test_x),
    'k-folds':k,
    'C':l_svc.C,
    'Winning Fold':folds[max_index],
    'Fold Accuracy':accuracies[max_index]
  }
  return out_dict

def run_rfc(train_x, train_y, test_x, test_y, n_estimators = 500, random_state = 42):
  forest = RandomForestClassifier(random_state=random_state ,n_estimators=n_estimators)
  forest.fit(train_x, train_y.values.ravel())
  pred_y = forest.predict(test_x)
  training_accuracy = forest.score(train_x, train_y)
  test_accuracy = forest.score(test_x, test_y)
  classification_rep = classification_report(test_y, pred_y)
  confusion_mat = confusion_matrix(test_y, pred_y)
  out_dict = {
    'Training Accuracy':training_accuracy,
    'Test Accuracy':test_accuracy,
    'Classification Report':classification_rep,
    'Confusion Matrix': confusion_mat,
    'Training N':len(train_x),
    'Test N':len(test_x)
  }
  return out_dict

def run_models(meta_dict, out_dict, training_label, test_label, train_x, train_y, test_x, test_y):
  dhash = hashlib.md5()
  encoded = json.dumps(meta_dict, sort_keys=True).encode()
  dhash.update(encoded)
  run_uid = dhash.hexdigest()
  svc_out = run_svc_new(train_x, train_y, test_x, test_y, random_state = meta_dict['Random State'])
  pd.DataFrame(svc_out['Confusion Matrix']).to_csv(source_path + 'Output' + sep + run_uid + sep + training_label + '_' + test_label + 'SCV Confusion Matrix.csv', index=False)
  rf_out = run_rfc(train_x, train_y, test_x, test_y, random_state = meta_dict['Random State'], n_estimators=meta_dict['Random Forest Estimators'])
  pd.DataFrame(rf_out['Confusion Matrix']).to_csv(source_path + 'Output' + sep + run_uid + sep + training_label + '_' + test_label + 'RFC Confusion Matrix.csv', index=False)
  # Store output in output_dictionary
  ind = len(out_dict.keys())
  out_dict[ind] = [
    training_label,               #'Training Data'
    test_label,                   #'Test Data'
    'SVM',                        #'Analysis Method'
    svc_out['Training Accuracy'], #'Training Accuracy'
    svc_out['Test Accuracy'],     #'Test Accuracy'
    svc_out['Training N'],        #'Training N'
    svc_out['Test N'],            #'Test N'
    ''                            #'Notes'
    ]
  ind+=1
  out_dict[ind] = [
    training_label,               #'Training Data'
    test_label,                   #'Test Data'
    'RFC',                        #'Analysis Method'
    rf_out['Training Accuracy'],  #'Training Accuracy'
    rf_out['Test Accuracy'],      #'Test Accuracy'
    rf_out['Training N'],         #'Training N'
    rf_out['Test N'],             #'Test N'
    'n_estimators='+str(meta_dict['Random Forest Estimators'])                 #'Notes'
    ]
  ind+=1
  return out_dict

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

def feature_reduction(x, y, type):
  if type == 'Logistic Regression':
    lreg1 = LogisticRegression(random_state=meta_dict['Random State']).fit(x, y)
    lreg1.get_params()

if __name__=='__main__':
  total_start_time = dt.datetime.now()
  meta_dict = {
    'atlas_name' : '69354adf',
    'smoothed' : False,
    'ICA-Aroma' : False,
    'confounds': [],
    'Random Forest Estimators': 1000,
    'Random State':42,
    'subtract parcel-wise mean': True,
    'concatenate':True
  }
  # Generate unique hash for metadata
  dhash = hashlib.md5()
  encoded = json.dumps(meta_dict, sort_keys=True).encode()
  dhash.update(encoded)
  run_uid = dhash.hexdigest()
  #Make folder to contain output
  try:
    os.mkdir(source_path + 'Output' + sep + run_uid)
    with open(source_path + 'Output' + sep + run_uid + sep + 'metadata.json', 'w') as outfile:
      json.dump(meta_dict, outfile)
  except:
    pass
  # if os.path.exists(source_path + 'Output' + sep + run_uid + sep + 'metadata.json'):
  #   print(f'An analysis with this same metadata dictionary has been run: {run_uid}')
  #   print('Would you like to re-run? (y/n)')
  #   if not 'y' in input().lower():
  #     raise Exception('Analyses halted.')
  
  if getpass.getuser() == 'kyle':
    HCP_DIR = "S:\\HCP\\"
    HCP_DIR_REST = f"{HCP_DIR}hcp_rest\\subjects\\"
    HCP_DIR_TASK = f"{HCP_DIR}hcp_task\\subjects\\"
    HCP_1200 = f"{HCP_DIR}HCP_1200\\"
    basepath = str("S:\\HCP\\HCP_1200\\{}\\MNINonLinear\\Results\\")
    subjects = pd.read_csv('C:\\Users\\kyle\\repos\\HCP-Analyses\\subject_list.csv')['ID']
    path_pattern = "S:\\HCP\\HCP_1200\\{}\\MNINonLinear\\Results\\{}\\{}.npy"
    nifty_template_hcp = 'S:\\HCP\\HCP_1200\\{subject}\\MNINonLinear\\Results\\{session}_{run}\\{session}_{run}.nii.gz'
    npy_template_hcp_alt = 'S:\\HCP\\HCP_1200\\{subject}\\MNINonLinear\\Results\\{session}_{run}\\{session}_{run}_{atlas_name}.npy'
    npy_template_hcp = 'S:\\HCP\\HCP_1200\\{subject}\\MNINonLinear\\Results\\{session}_{run}\\{atlas_name}_{session}_{run}.npy'
  else:
    HCP_DIR = "/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_Numpy/"
    basepath = str('/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_Numpy/{}/MNINonLinear/Results/')
    HCP_DIR_REST = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_rest/subjects/"
    HCP_DIR_TASK = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_task/subjects/"
    HCP_DIR_EVS = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_task/"
    HCP_DIR_BEHAVIOR = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_behavior/"
    subjects = pd.read_csv('/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/subject_list.csv')['ID']
    path_pattern ="/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_Numpy/{}/MNINonLinear/Results/{}/{}.npy"
    if not os.path.isdir(HCP_DIR): os.mkdir(HCP_DIR)
  
  parcel_labels, network_labels = fetch_labels(meta_dict, HCP_1200)
  #Use this line to subset the subject list to something shorter as needed
  subjects = subjects[:]
  sessions = [
    "tfMRI_MOTOR",
    "tfMRI_WM",
    "tfMRI_EMOTION",
    "tfMRI_GAMBLING",
    "tfMRI_LANGUAGE",
    "tfMRI_RELATIONAL",
    "tfMRI_SOCIAL"
  ]

  parcellated_data = {}
  for session in sessions:
    #Read in parcellated data, or parcellate data if meta-data conditions not met by available data
    parcellated_data[session] = load_parcellated_task_timeseries(meta_dict, nifty_template_hcp, subjects, session, npy_template = npy_template_hcp)

  parcels_sums = generate_parcel_input_features(parcellated_data, parcel_labels)

  network_sums = generate_network_input_features(parcels_sums, network_labels)

  parcel_connection_task_data = generate_parcel_connection_features(parcellated_data, parcel_labels)

  network_connection_features = generate_network_connection_features(parcellated_data, network_labels)

  demographics = pd.read_csv(source_path + 'demographics_with_dummy_vars.csv')
  demographics_dummy = demographics[[
    'Subject',
    'Age__22-25',
    'Age__26-30',
    'Age__31-35',
    'Age__36+',
    'Gender__F',
    'Gender__M',
    'Acquisition__Q01',
    'Acquisition__Q02',
    'Acquisition__Q03',
    'Acquisition__Q04',
    'Acquisition__Q05',
    'Acquisition__Q06',
    'Acquisition__Q07',
    'Acquisition__Q08',
    'Acquisition__Q09',
    'Acquisition__Q10',
    'Acquisition__Q11',
    'Acquisition__Q12',
    'Acquisition__Q13'
  ]]

  relative_RMS_means_collapsed = pd.read_csv(source_path + 'relative_RMS_means_collapsed.csv')

  confounds = pd.merge(relative_RMS_means_collapsed, demographics_dummy, how='left', on='Subject')

  parcel_sum_input = pd.merge(confounds, parcels_sums, on=['Subject','task'], how = 'right').dropna()
  network_sum_input = pd.merge(confounds, network_sums, on=['Subject','task'], how = 'right').dropna()
  parcel_connection_input = pd.merge(confounds, parcel_connection_task_data, on=['Subject','task'], how = 'right').dropna()
  network_connection_input = pd.merge(confounds, network_connection_features, on=['Subject','task'], how = 'right').dropna()

  parcel_sum_x, parcel_sum_y = XY_split(parcel_sum_input, 'task')
  network_sum_x, network_sum_y = XY_split(network_sum_input, 'task')
  parcel_connection_x, parcel_connection_y = XY_split(parcel_connection_input, 'task')
  network_connection_x, network_connection_y = XY_split(network_connection_input, 'task')

  parcel_sum_x_train, parcel_sum_x_test, parcel_sum_y_train, parcel_sum_y_test = train_test_split(parcel_sum_x, parcel_sum_y, test_size = 0.2)
  network_sum_x_train, network_sum_x_test, network_sum_y_train, network_sum_y_test = train_test_split(network_sum_x, network_sum_y, test_size = 0.2)
  parcel_connection_x_train, parcel_connection_x_test, parcel_connection_y_train, parcel_connection_y_test = train_test_split(parcel_connection_x, parcel_connection_y, test_size = 0.2)
  network_connection_x_train, network_connection_x_test, network_connection_y_train, network_connection_y_test = train_test_split(network_connection_x, network_connection_y, test_size = 0.2)

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

  # Feature Selection
  # lreg1 = LogisticRegression(random_state=meta_dict['Random State']).fit(parcel_sum_x_train, parcel_sum_y_train)
  # lreg1.get_params()

  # Run Models
  out_dict = {}
  confounds_complete = confounds.dropna(axis=0, how='any')
  confounds_y = confounds_complete[['task']]
  confounds_x = confounds_complete.loc[:,confounds_complete.columns != 'task']
  confounds_x_train, confounds_x_test, confounds_y_train, confounds_y_test = train_test_split(confounds_x, confounds_y, test_size = 0.2)
  test_label = 'confounds' 
  training_label = 'confounds'

  meta_dict = meta_dict
  out_dict = out_dict
  training_label = training_label
  test_label = test_label
  train_x = confounds_x_train
  train_y = confounds_y_train
  test_x = confounds_x_test
  test_y = confounds_y_test

  C_s = list(np.logspace(-10, 0, 4))
  svc =  svm.SVC()
  parameters = {'kernel':['linear'], 'C':C_s}
  n_folds=5
  clf = GridSearchCV(svc, parameters, cv=n_folds, refit=False, n_jobs=3)

  clf.fit(train_x, train_y)
  #out_dict = run_models(meta_dict, out_dict, training_label, test_label, confounds_x_train, confounds_y_train, confounds_x_test, confounds_y_test)

  test_label = 'parcel_sum' 
  training_label = 'parcel_sum' 
  #out_dict = run_models(meta_dict, out_dict, training_label, test_label, parcel_sum_x_train, parcel_sum_y_train, parcel_sum_x_test, parcel_sum_y_test)

  test_label = 'network_sum' 
  training_label = 'network_sum' 
  meta_dict = meta_dict
  out_dict = out_dict
  training_label = training_label
  test_label = test_label
  train_x = network_sum_x_train
  train_y = network_sum_y_train
  test_x = network_sum_x_test
  test_y = network_sum_y_test

  start_time = dt.datetime.now()
  C_s = list(np.logspace(0, -10, 4))
  svc =  svm.SVC()
  parameters = {'kernel':['linear'], 'C':C_s[0:1]}
  n_folds=3
  clf = GridSearchCV(svc, parameters, cv=n_folds, refit=True)

  clf.fit(train_x, train_y)

  print(clf.best_params_)
  print(clf.best_score_)
  results = pd.DataFrame(clf.cv_results_)

  clf.score(network_sum_x_test, network_sum_y_test)
  
  end_time = dt.datetime.now()
  print('Network Sum done. Runtime: ', end_time - start_time)
  
  #out_dict = run_models(meta_dict, out_dict, training_label, test_label, network_sum_x_train, network_sum_y_train, network_sum_x_test, network_sum_y_test)

  test_label = 'parcel_connection' 
  training_label = 'parcel_connection' 
  #out_dict = run_models(meta_dict, out_dict, training_label, test_label, parcel_connection_x_train, parcel_connection_y_train, parcel_connection_x_test, parcel_connection_y_test)

  test_label = 'network_connection' 
  training_label = 'network_connection' 
  training_label = training_label
  test_label = test_label
  train_x = network_connection_x_train
  train_y = network_connection_y_train
  test_x = network_connection_x_test
  test_y = network_connection_y_test

  start_time = dt.datetime.now()
  C_s = list(np.logspace(0, -10, 4))
  svc =  svm.SVC()
  parameters = {'kernel':['linear'], 'C':C_s[0:3]}
  n_folds=5
  clf = GridSearchCV(svc, parameters, cv=n_folds, refit=True, n_jobs=-1)

  clf.fit(train_x, train_y)

  print(clf.best_params_)
  print(clf.best_score_)
  results = pd.DataFrame(clf.cv_results_)

  print(clf.score(network_connection_x_test, network_connection_y_test))
  
  end_time = dt.datetime.now()
  print('Network Connection done. Runtime: ', end_time - start_time)
  #out_dict = run_models(meta_dict, out_dict, training_label, test_label, network_connection_x_train, network_connection_y_train, network_connection_x_test, network_connection_y_test)


  output_df = pd.DataFrame.from_dict(out_dict, orient='index', columns = ['Training Data','Test Data','Analysis Method','Training Accuracy','Test Accuracy','Training N','Test N','Notes'])
  output_df.to_csv(
    source_path + 'Output' + sep + run_uid + sep + 'HCP Task Decoding.csv',
    index=False
    )
  total_end_time = dt.datetime.now()
  print('Done. Runtime: ', total_end_time - total_start_time)

