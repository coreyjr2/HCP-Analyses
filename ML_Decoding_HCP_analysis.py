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
from nilearn.input_data import NiftiLabelsMasker
from nilearn.input_data import NiftiMapsMasker

sep = os.path.sep
source_path = os.path.abspath(os.getcwd()) + sep
sys_name = platform.system() 
parcel_dict = { #Parcels, unique values from cor matrix, N networks
    'harvard_oxford':(96,4560),
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


def parcellate_timeseries(nifty_file, atlas_name, confounds=None):
  # Other atlases in MNI found here: https://www.lead-dbs.org/helpsupport/knowledge-base/atlasesresources/cortical-atlas-parcellations-mni-space/
  raw_timeseries = nib.load(nifty_file)
  #raise Exception(nifty_file, ' not available.')
  if atlas_name=='harvard_oxford':
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm', symmetric_split=True)
    atlas_filename = atlas.maps
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
  elif atlas_name == 'msdl':
    atlas = datasets.fetch_atlas_msdl()
    atlas_filename = atlas.maps
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, memory='nilearn_cache')
  elif atlas_name == 'mni_glasser':
    atas_glasser_01_filename = source_path + 'MMP_in_MNI_corr.nii.gz' # Downlaoded from https://neurovault.org/collections/1549/
    masker = NiftiLabelsMasker(labels_img=atas_glasser_01_filename, standardize=True)
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
  #Transform the motor task imaging data with the masker and check the shape
  masked_timeseries = []
  if confounds is not None:
    masked_timeseries = masker.fit_transform(raw_timeseries, counfounds = confounds)
  else:
    masked_timeseries = masker.fit_transform(raw_timeseries)
  return masked_timeseries

def load_parcellated_task_timeseries(meta_dict, nifty_template, subjects, session, npy_template = None, run_names = ['RL','LR'], confounds_path = None):
  remove_mean = meta_dict['subtract parcel-wise mean']
  atlas_name = meta_dict['atlas_name']
  concatenate = meta_dict['concatenate']
  parcellated_dict = {}
  concat_dict = {}
  print('Loading in parcellated data for task: ', session)
  for subject in subjects:
    try:
      print('\t',subject)
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
      print(f'Subject {subject} is not available: {e}')
  if concatenate:
    return concat_dict
  else:
    return parcellated_dict

def generate_parcel_input_features(parcellated_data, labels):
  out_dict = {}
  out_df_dict = {}
  for session in parcellated_data.keys():
    parcel_dict = parcellated_data[session]
    out_dict[session] = np.zeros((len(parcel_dict), parcel_dict[parcel_dict.keys()[0]].shape[1]), dtype='float64')
    for subject, ts in enumerate(parcel_dict.values()):#(284, 78)
      out_dict[session][subject] = np.mean(ts.T, axis=1)
    out_df_dict[session] = pd.DataFrame(out_dict[session], columns= labels)
    out_df_dict[session]['task'] = numeric_task_ref[session]
  parcels_full = pd.DataFrame(np.concatenate(out_df_dict.values(), axis = 0))
  return parcels_full

def generate_network_input_features():
  pass

def generate_connection_features():
  pass

def training_test_split():
  pass

def feature_reduction():
  pass

def run_svc():
  pass

def run_rfc():
  pass

def fetch_labels(mea_dict):
  if 'glasser' in meta_dict['atlas_name']:
    regions_file = np.load(source_path + "glasser_regions.npy").T
    parcel_labels = regions_file[0]
    network_labels = regions_file[1]
  elif meta_dict['atlas_name'] == 'msdl':
    atlas_MSDL = datasets.fetch_atlas_msdl()
    parcel_labels = atlas_MSDL['labels']
    network_labels = atlas_MSDL['networks']
  else:
    raise NotImplementedError
  return parcel_labels, network_labels

if __name__=='__main__':
  total_start_time = dt.datetime.now()
  meta_dict = {
    'atlas_name' : 'mni_glasser',
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
  # Make folder to contain output
  # try:
  #   os.mkdir(source_path + 'Output' + sep + run_uid)
  #   with open(source_path + 'Output' + sep + run_uid + sep + 'metadata.json', 'w') as outfile:
  #     json.dump(meta_dict, outfile)
  # except:
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
    npy_template_hcp = 'S:\\HCP\\HCP_1200\\{subject}\\MNINonLinear\\Results\\{session}_{run}\\{session}_{run}_{atlas_name}.npy'
  
  parcel_labels, network_labels = fetch_labels(meta_dict)
  #Use this line to subset the subject list to something shorter as needed
  subjects = subjects[:]
  sessions = [
    #"tfMRI_MOTOR",
    # "tfMRI_WM",
    # "tfMRI_EMOTION",
    # "tfMRI_GAMBLING",
    "tfMRI_LANGUAGE",
    "tfMRI_RELATIONAL",
    "tfMRI_SOCIAL"
  ]
parcellated_data = {}
for session in sessions:
  parcellated_data[session] = load_parcellated_task_timeseries(meta_dict, nifty_template_hcp, subjects, session, npy_template = npy_template_hcp)

parcels_full = generate_parcel_input_features(parcellated_data, parcel_labels[0])
