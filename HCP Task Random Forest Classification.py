#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:14:43 2020

@author: cjrichier, kabaacke-psy
"""
glasser = False #Set to true to use the smaller dataset with the glasser parcelation applied
#################################################
########## HCP decoding project code ############
#################################################

###############################
## Load the needed libraries ##
###############################
if True:
  import os
  import time
  import nibabel as nib
  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd
  import sklearn as sk
  from sklearn.ensemble import RandomForestClassifier
  import nilearn 
  from matplotlib import cm # cm=colormap
  from nilearn.connectome import sym_matrix_to_vec
  from sklearn.preprocessing import StandardScaler
  from sklearn import svm
  from sklearn.model_selection import train_test_split
  import getpass
  import platform
  import hcp_utils as hcp
  import datetime as dt
  from nilearn import datasets
  import brainconn # pip install git+https://github.com/fiuneuro/brainconn@master
  import seaborn as sns
  from sklearn.preprocessing import scale 
  from sklearn import model_selection
  from sklearn.model_selection import RepeatedKFold
  from sklearn.model_selection import train_test_split
  from sklearn.cross_decomposition import PLSRegression
  from sklearn.metrics import mean_squared_error
  from sklearn.model_selection import cross_val_predict
  from sklearn.metrics import mean_squared_error, r2_score
  from collections import defaultdict
  from nilearn.input_data import NiftiLabelsMasker
  from nilearn.input_data import NiftiMapsMasker
  import matplotlib.pyplot as plt
  import numpy as np
  from scipy.stats import spearmanr
  from scipy.cluster import hierarchy

  from sklearn.datasets import load_breast_cancer
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.inspection import permutation_importance
  from sklearn.model_selection import train_test_split
#record the start time 
#start_time = time.time()
total_start_time = dt.datetime.now()
# Set relevant directories

sep = os.path.sep
source_path = os.path.dirname(os.path.abspath(__file__)) + sep
sys_name = platform.system() 
visualization = False
if getpass.getuser() == 'kyle':
  HCP_DIR = "S:\\HCP\\"
  HCP_DIR_REST = f"{HCP_DIR}hcp_rest\\subjects\\"
  HCP_DIR_TASK = f"{HCP_DIR}hcp_task\\subjects\\"
  HCP_1200 = f"{HCP_DIR}HCP_1200\\"
  basepath = str("S:\\HCP\\HCP_1200\\{}\\MNINonLinear\\Results\\")
  subjects = pd.read_csv('C:\\Users\\kyle\\repos\\HCP-Analyses\\subject_list.csv')['ID']
  path_pattern = "S:\\HCP\\HCP_1200\\{}\\MNINonLinear\\Results\\{}\\{}.npy"
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

class FileDirr(object):
    '''
    Object to contain directory structures 
    in order to operate on different host computers
    
    '''
    def __init__(self, downloads = "Downloads", data = 'S:\\'):
        self.uname = getpass.getuser()
        if sys_name =='Windows' or sys_name=='Darwin':
            self.home = str(sep + 'Users' + sep + self.uname+sep)
        elif sys_name =='Linux':
            self.home = str(sep + 'home' + sep)
        self.down_dirr = str(self.home + downloads + sep)
        self.data = str(self.home + data + sep)
        self.hcp = str(self.data + 'HCP' + sep)
        self.hcp_rest = str(self.hcp + 'hcp_rest' + sep) 
        self.hcp_rest_s = str(self.hcp_rest + 'subjects' + sep) 
        self.hcp_task = str(self.hcp + 'hcp_task' + sep) 
        self.hcp_task_s = str(self.hcp_task + 'subjects' + sep)
        self.hcp_behavior = str(self.hcp + 'hcp_behavior' + sep) #I don't have this one - doesn't matter
        self.hcp_output = self.hcp_rest = str(self.hcp + 'output' + sep) 
        self.hcp_full = str(self.hcp + 'HCP_1200' + sep)

# Analysis metadata
if True:
  # The data shared for NMA projects is a subset of the full HCP dataset
  if glasser:
    N_SUBJECTS = 339
  else:
    N_SUBJECTS = len(subjects)

  # The data have already been aggregated into ROIs from the Glasser parcellation
  if glasser:
    N_PARCELS = 360
  else:
    # The full dataset is stored using the MSDL Parcellation, doi: 10.1007/978-3-642-22092-0_46
    N_PARCELS = 39

  # How many networks?
  if glasser:
    N_NETWORKS = 12 
  else:
    N_NETWORKS = 17

  # The acquisition parameters for all tasks were identical
  TR = 0.72  # Time resolution, in sec

  # The parcels are matched across hemispheres with the same order
  HEMIS = ["Right", "Left"]

  # Each experiment was repeated multiple times in each subject
  N_RUNS_REST = 4
  N_RUNS_TASK = 2

  # Time series data are organized by experiment, with each experiment
  # having an LR and RL (phase-encode direction) acquistion
  BOLD_NAMES = [ "rfMRI_REST1_LR", 
                "rfMRI_REST1_RL", 
                "rfMRI_REST2_LR", 
                "rfMRI_REST2_RL", 
                "tfMRI_MOTOR_RL", 
                "tfMRI_MOTOR_LR",
                "tfMRI_WM_RL", 
                "tfMRI_WM_LR",
                "tfMRI_EMOTION_RL", 
                "tfMRI_EMOTION_LR",
                "tfMRI_GAMBLING_RL", 
                "tfMRI_GAMBLING_LR", 
                "tfMRI_LANGUAGE_RL", 
                "tfMRI_LANGUAGE_LR", 
                "tfMRI_RELATIONAL_RL", 
                "tfMRI_RELATIONAL_LR", 
                "tfMRI_SOCIAL_RL", 
                "tfMRI_SOCIAL_LR"]

  # This will use all subjects:
  if glasser:
    subjects = range(N_SUBJECTS) #This no longer works since the subjects are no longer incremental ints

  #You may want to limit the subjects used during code development. This will only load in 10 subjects if you use this list.
  #SUBJECT_SUBSET = 100
  subjects = subjects[:]
  N_SUBJECTS = len(subjects)

  ''' #import the demographics and bheavior data --MISSING IN FULL DATASET
    demographics = pd.read_csv('/Volumes/Byrgenwerth/Datasets/HCP/HCP_demographics/demographics_behavior.csv')

    #What is our gender breakdown?
    demographics['Gender'].value_counts()
    demographics['Age'].value_counts()
  '''

  # Pull information about the atlas
  if glasser:
    regions = np.load(f"{HCP_DIR}hcp_rest\\regions.npy").T
    region_info = dict(
        name=regions[0].tolist(),
        network=regions[1],
        myelin=regions[2].astype(np.float),
    )
    region_transpose = pd.DataFrame(regions.T, columns=['Region', 'Network', 'Myelination'])
    #print(region_info)
    regions = region_info['name']
    networks = region_info['network']
  else:
    atlas_MSDL = datasets.fetch_atlas_msdl()
    regions = atlas_MSDL['labels']
    networks = atlas_MSDL['networks']
    ''' region_coords = []
    # Not sure if it is this:
    region_coords_1 = [coord for tup in regions['region_coords'] for coord in tup]
    #or this:
    for x in range(3):
          for tup in regions['region_coords']:
              region_coords.append(tup[x]) 
    region_info = dict(
        name=regions['labels'],
        network=regions['networks'],
        myelin=regions['region_coords'],
    ) '''

######################################
###### Define useful functions #######
######################################
if True:
  def parcellate_timeseries(nifty_file, atlas_name, confounds=None):
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
    elif atlas_name == 'mni_glasser':
      atas_glasser_01_filename = source_path + 'MMP_in_MNI_corr.nii.gz'
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

  def get_image_ids(name):
    """Get the 1-based image indices for runs in a given experiment.

      Args:
        name (str) : Name of experiment ("rest" or name of task) to load
      Returns:
        run_ids (list of int) : Numeric ID for experiment image files

    """
    run_ids = [
      i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code
    ]
    if not run_ids:
      raise ValueError(f"Found no data for '{name}''")
    return run_ids

  def load_rest_timeseries(subject, name, runs=None, concat=True, remove_mean=True): #UNUSED
    """Load timeseries data for a single subject.
    
    Args:
      subject (int): 0-based subject ID to load
      name (str) : Name of experiment ("rest" or name of task) to load
      run (None or int or list of ints): 0-based run(s) of the task to load,
        or None to load all runs.
      concat (bool) : If True, concatenate multiple runs in time
      remove_mean (bool) : If True, subtract the parcel-wise mean

    Returns
      ts (n_parcel x n_tp array): Array of BOLD data values

    """
    # Get the list relative 0-based index of runs to use
    if runs is None:
      runs = range(N_RUNS_REST) if name == "rest" else range(N_RUNS_TASK)
    elif isinstance(runs, int):
      runs = [runs]

    # Get the first (1-based) run id for this experiment 
    offset = get_image_ids(name)[0]

    # Load each run's data
    bold_data = [
        load_single_rest_timeseries(subject, offset + run, remove_mean) for run in runs
    ]

    # Optionally concatenate in time
    if concat:
      bold_data = np.concatenate(bold_data, axis=-1)

    return bold_data

  def load_task_timeseries(subject, name, runs=None, concat=True, remove_mean=True, full_name_runs=None):
    """Load timeseries data for a single subject.
    
    Args:
      subject (int): 0-based subject ID to load
      name (str) : Name of experiment ("rest" or name of task) to load
      run (None or int or list of ints): 0-based run(s) of the task to load,
        or None to load all runs.
      concat (bool) : If True, concatenate multiple runs in time
      remove_mean (bool) : If True, subtract the parcel-wise mean

    Returns
      ts (n_parcel x n_tp array): Array of BOLD data values

    """
    # Get the list relative 0-based index of runs to use
    if full_name_runs is not None:
      bold_data = []
      for name in full_name_runs:
        try:
          bold_data.append(load_single_task_timeseries(subject, name, full_name=True))
        except:
          pass
      ''' bold_data = [
        load_single_task_timeseries for full_name in full_name_runs
      ] '''
    else:
      if runs is None:
        runs = range(N_RUNS_REST) if name == "rest" else range(N_RUNS_TASK)
      elif isinstance(runs, int):
        runs = [runs]

      # Get the first (1-based) run id for this experiment 
      offset = get_image_ids(name)[0]

      # Load each run's data
      bold_data = [
          load_single_task_timeseries(subject, offset + run, remove_mean) for run in runs
      ]

    # Optionally concatenate in time
    if concat and len(bold_data)>1:
      bold_data = np.concatenate(bold_data, axis=-1)

    return bold_data

  def load_single_rest_timeseries(subject, bold_run, remove_mean=True): #UNUSED
    """Load timeseries data for a single subject and single run.
    
    Args:
      subject (int): 0-based subject ID to load
      bold_run (int): 1-based run index, across all tasks
      remove_mean (bool): If True, subtract the parcel-wise mean

    Returns
      ts (n_parcel x n_timepoint array): Array of BOLD data values

    """
    bold_path = f"{HCP_DIR}/hcp_rest/subjects/{subject}/timeseries"
    bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
    ts = np.load(f"{bold_path}/{bold_file}")
    if remove_mean:
      ts -= ts.mean(axis=1, keepdims=True)
    return ts

  def load_single_task_timeseries(subject, bold_run, remove_mean=True, full_name=False):
    """Load timeseries data for a single subject and single run.
    
    Args:
      subject (int): 0-based subject ID to load
      bold_run (int): 1-based run index, across all tasks
      remove_mean (bool): If True, subtract the parcel-wise mean

    Returns
      ts (n_parcel x n_timepoint array): Array of BOLD data values

    """
    if not full_name:
      bold_path = f"{HCP_DIR}{sep}hcp_task{sep}subjects{sep}{subject}{sep}timeseries"
      bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
    else:
      bold_path = basepath.format(subject) + bold_run #f"{HCP_DIR}/hcp_task/subjects/{subject}/timeseries"
      bold_file = f"{bold_run}.npy"
    print(f"{bold_path}{sep}{bold_file}")
    ts = np.load(f"{bold_path}{sep}{bold_file}")
    if remove_mean:
      ts -= ts.mean(axis=1, keepdims=True)
    return ts

  def load_evs(subject, name, condition): #UNUSED
    """Load EV (explanatory variable) data for one task condition.

    Args:
      subject (int): 0-based subject ID to load
      name (str) : Name of task
      condition (str) : Name of condition

    Returns
      evs (list of dicts): A dictionary with the onset, duration, and amplitude
        of the condition for each run.

    """
    evs = []
    for id in get_image_ids(name):
      task_key = BOLD_NAMES[id - 1]
      ev_file = f"{HCP_DIR_EVS}subjects/{subject}/EVs/{task_key}/{condition}.txt"
      ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
      ev = dict(zip(["onset", "duration", "amplitude"], ev_array))
      evs.append(ev)
    return evs

  def condition_frames(run_evs, skip=0): # UNUSED
    """Identify timepoints corresponding to a given condition in each run.

    Args:
      run_evs (list of dicts) : Onset and duration of the event, per run
      skip (int) : Ignore this many frames at the start of each trial, to account
        for hemodynamic lag

    Returns:
      frames_list (list of 1D arrays): Flat arrays of frame indices, per run

    """
    frames_list = []
    for ev in run_evs:

      # Determine when trial starts, rounded down
      start = np.floor(ev["onset"] / TR).astype(int)

      # Use trial duration to determine how many frames to include for trial
      duration = np.ceil(ev["duration"] / TR).astype(int)

      # Take the range of frames that correspond to this specific trial
      frames = [s + np.arange(skip, d) for s, d in zip(start, duration)]

      frames_list.append(np.concatenate(frames))

    return frames_list

  def selective_average(timeseries_data, ev, skip=0): # UNUSED
    """Take the temporal mean across frames for a given condition.

    Args:
      timeseries_data (array or list of arrays): n_parcel x n_tp arrays
      ev (dict or list of dicts): Condition timing information
      skip (int) : Ignore this many frames at the start of each trial, to account
        for hemodynamic lag

    Returns:
      avg_data (1D array): Data averagted across selected image frames based
      on condition timing

    """
    # Ensure that we have lists of the same length
    if not isinstance(timeseries_data, list):
      timeseries_data = [timeseries_data]
    if not isinstance(ev, list):
      ev = [ev]
    if len(timeseries_data) != len(ev):
      raise ValueError("Length of `timeseries_data` and `ev` must match.")

    # Identify the indices of relevant frames
    frames = condition_frames(ev, skip)

    # Select the frames from each image
    selected_data = []
    for run_data, run_frames in zip(timeseries_data, frames):
      run_frames = run_frames[run_frames < run_data.shape[1]]
      selected_data.append(run_data[:, run_frames])

    # Take the average in each parcel
    avg_data = np.concatenate(selected_data, axis=-1).mean(axis=-1)

    return avg_data

  def calculate_FD_P(in_file): 
    # Function taken from  FCP-INDI/C-PAC/CPAC/generate_motion_statistics/generate_motion_statistics.py
    """
    Method to calculate Framewise Displacement (FD)  as per Power et al., 2012
    Parameters
    ----------
    in_file : string
        movement parameters vector file path
    Returns
    -------
    out_file : string
        Frame-wise displacement mat
        file path


    # Modified to return a vector and using the order of the columns provided in HCP_1200
    """

    motion_params = np.genfromtxt(in_file).T

    rotations = np.transpose(np.abs(np.diff(motion_params[3:6, :])))
    translations = np.transpose(np.abs(np.diff(motion_params[0:3, :])))

    fd = np.sum(translations, axis=1) + \
      (50 * np.pi / 180) * np.sum(rotations, axis=1)

    fd = np.insert(fd, 0, 0)

    return fd

  def brainconn_arrays(corr, p_thresh=0.2):
    '''
      Takes a correlational matrix
      retruns a tuple of a correlational matrix with a zeroed diagonal and a binary ajacency matrix using a p theshold of p_thresh
    '''
    adj_wei = corr - np.eye(corr.shape[0])
    adj_bin = brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei, p_thresh))
    return (adj_wei, p_thresh)

# Import Demographic and behavioral data
multi_dummy = True # Set True to create dummy columns per unique value in the categorical columns of Age, Gender, Acquisition, Release
if False:
  behavior_full = pd.read_csv(HCP_DIR + 'Behavior' + sep + 'behavior.csv')
  demographics = behavior_full[['Subject','Release','Acquisition','Gender','Age']]
  if multi_dummy:
    demographics = demographics.join(pd.get_dummies(demographics['Age'], prefix='Age_'), how='outer')
    demographics = demographics.join(pd.get_dummies(demographics['Gender'], prefix='Gender_'), how='outer')
    demographics = demographics.join(pd.get_dummies(demographics['Acquisition'], prefix='Acquisition_'), how='outer')
    demographics = demographics.join(pd.get_dummies(demographics['Release'], prefix='Release_'), how='outer')
  demographics.to_csv(source_path + sep + 'demographics_with_dummy_vars.csv', index=False)
else:
  demographics = pd.read_csv(source_path + sep + 'demographics_with_dummy_vars.csv')

# Import movement information
if False:
  movement = {}
  for s in subjects:
    movement[s] = {}
    for t in BOLD_NAMES:
      movement[s][t] = {}
      try:
        movement[s][t]['Physio_log'] = pd.read_csv( # See page 38 of HCP_1200 release manual. 400HZ approx 288 samples per frame
          path_pattern[:-6].format(s, t) + t + '_Physio_log.txt',
          sep='\t', 
          header=None, 
          names=[
            'trigger_pulse','respiration','pulse_oximeter'
          ]
        )
        #We can shift the timescale on this 
      except:
        movement[s][t]['Physio_log'] = None
      try:
        movement[s][t]['Movement_RelativeRMS_mean'] = pd.read_csv( # mean amount of motion between neighboring timepoints
          path_pattern[:-6].format(s, t) + 'Movement_RelativeRMS_mean.txt',
          sep='\t',
          header=None,
          names=['0']
        )['0'][0]
      except:
        movement[s][t]['Movement_RelativeRMS_mean'] = None
      try:
        movement[s][t]['Movement_RelativeRMS'] = pd.read_csv( # amount of motion from the previous time point, alternative to FD see https://www.mail-archive.com/hcp-users@humanconnectome.org/msg04444.html
          path_pattern[:-6].format(s, t) +'Movement_RelativeRMS.txt',
          sep='\t',
          header=None, 
          names = ['Movement_RelativeRMS']
        )
      except:
        movement[s][t]['Movement_RelativeRMS'] = None
      try:
        movement[s][t]['Movement_Regressors'] = pd.read_csv( # See HCP1200 release manual page 96, also see https://www.mail-archive.com/hcp-users@humanconnectome.org/msg02961.html
          path_pattern[:-6].format(s, t) + 'Movement_Regressors.txt',
          sep='\t',
          header=None,
          names = [
            'trans_x','trans_y','trans_z',
            'rot_x','rot_y','rot_z',
            'trans_dx','trans_dy','trans_dz',
            'rot_dx','rot_dy','rot_dz'
          ]
        )
      except:
        movement[s][t]['Movement_Regressors'] = None
      try:
          movement[s][t]['Movement_Regressors_dt'] = pd.read_csv( # Made from removing the mean and linear trend from each variable in Movement_Regressors.txt
          path_pattern[:-6].format(s, t) + 'Movement_Regressors_dt.txt',
          sep='\t',
          header=None,
          names = [
            'trans_x_dt','trans_y_dt','trans_z_dt',
            'rot_x_dt','rot_y_dt','rot_z_dt',
            'trans_dx_dt','trans_dy_dt','trans_dz_dt',
            'rot_dx_dt','rot_dy_dt','rot_dz_dt'
          ]
        )
      except:
        movement[s][t]['Movement_Regressors_dt'] = None

  # Create a dictionary to store Movement_RelativeRMS_mean values per scan
  task_number_dict = { "rfMRI_REST1_LR": 0, 
                "rfMRI_REST1_RL": 0, 
                "rfMRI_REST2_LR": 0, 
                "rfMRI_REST2_RL": 0, 
                "tfMRI_MOTOR_RL": 4, 
                "tfMRI_MOTOR_LR": 4,
                "tfMRI_WM_RL": 7, 
                "tfMRI_WM_LR": 7,
                "tfMRI_EMOTION_RL": 1, 
                "tfMRI_EMOTION_LR": 1,
                "tfMRI_GAMBLING_RL": 2, 
                "tfMRI_GAMBLING_LR": 2, 
                "tfMRI_LANGUAGE_RL": 3, 
                "tfMRI_LANGUAGE_LR": 3, 
                "tfMRI_RELATIONAL_RL": 5, 
                "tfMRI_RELATIONAL_LR": 5, 
                "tfMRI_SOCIAL_RL": 6, 
                "tfMRI_SOCIAL_LR": 6}

  relative_RMS_mean_dict = {}
  for s in subjects:
    for t in BOLD_NAMES:
      relative_RMS_mean_dict[str(s)+t] = [s, t, task_number_dict[t], movement[s][t]['Movement_RelativeRMS_mean']]
  # Create a dataframe to merge the demographics data onto
  relative_RMS_means = pd.DataFrame.from_dict(relative_RMS_mean_dict, orient='index', columns = ['Subject','Run','task','Movement_RelativeRMS_mean'])
  relative_RMS_means.to_csv(source_path + sep + 'relative_RMS_means.csv', index=False)
  relative_RMS_means_g = relative_RMS_means.groupby(['Subject','task'])
  # Collapsed aross subject, task to accomidate concatenation of timeseries
  relative_RMS_means_collapsed = pd.DataFrame(relative_RMS_means_g.agg(np.mean))
  relative_RMS_means_collapsed.reset_index(inplace=True)
  relative_RMS_means_collapsed.to_csv(source_path + sep + 'relative_RMS_means_collapsed.csv', index=False)

  # Merge demographics and motion data, duplicating dmeographics information to fill in all scans per subject
  regressors = pd.merge(relative_RMS_means, demographics, how='left', on='Subject')
  regressors.to_csv(source_path + sep + 'regressors.csv', index=False)
else:
  relative_RMS_means = pd.read_csv(source_path + sep + 'relative_RMS_means.csv')
  relative_RMS_means_collapsed = pd.read_csv(source_path + sep + 'relative_RMS_means_collapsed.csv')
  regressors = pd.read_csv(source_path + sep + 'regressors.csv')
  
  regressors_matrix = regressors[['Subject', 'Run', 'task', 'Age__22-25', 'Age__26-30',
       'Age__31-35', 'Age__36+', 'Gender__F', 'Gender__M', 'Acquisition__Q01',
       'Acquisition__Q02', 'Acquisition__Q03', 'Acquisition__Q04',
       'Acquisition__Q05', 'Acquisition__Q06', 'Acquisition__Q07',
       'Acquisition__Q08', 'Acquisition__Q09', 'Acquisition__Q10',
       'Acquisition__Q11', 'Acquisition__Q12', 'Acquisition__Q13',
       'Release__MEG2', 'Release__Q1', 'Release__Q2', 'Release__Q3',
       'Release__S1200', 'Release__S500', 'Release__S900']]

################################
#### Making the input data #####
################################
if True:
  if glasser: # Glasser Version with subset
    # Make a list of the task names
    tasks_names = ["motor", "wm", "gambling", "emotion", "language", "relational", "social"]

    #Load in all of the timeseries for each subject for each task

    timeseries_motor = []
    for subject in subjects:
        timeseries_motor.append(load_task_timeseries(subject, "motor", concat=True))
    timeseries_wm = []
    for subject in subjects:
      timeseries_wm.append(load_task_timeseries(subject, "wm", concat=True))
    timeseries_gambling = []
    for subject in subjects:
      timeseries_gambling.append(load_task_timeseries(subject, "gambling", concat=True))
    timeseries_emotion = []
    for subject in subjects:
      timeseries_emotion.append(load_task_timeseries(subject, "emotion", concat=True))
    timeseries_language = []
    for subject in subjects:
      timeseries_language.append(load_task_timeseries(subject, "language", concat=True))
    timeseries_relational = []
    for subject in subjects:
      timeseries_relational.append(load_task_timeseries(subject, "relational", concat=True))
    timeseries_social = []
    for subject in subjects:
      timeseries_social.append(load_task_timeseries(subject, "social", concat=True))
  else: # Full Dataset with MSDL
    # Make a list of the task names
    full_task_names = [
        "tfMRI_MOTOR_RL", 
        "tfMRI_MOTOR_LR",
        "rfMRI_REST1_LR", 
        "rfMRI_REST1_RL", 
        "rfMRI_REST2_LR", 
        "rfMRI_REST2_RL", 
        "tfMRI_WM_RL", 
        "tfMRI_WM_LR",
        "tfMRI_EMOTION_RL", 
        "tfMRI_EMOTION_LR",
        "tfMRI_GAMBLING_RL", 
        "tfMRI_GAMBLING_LR", 
        "tfMRI_LANGUAGE_RL", 
        "tfMRI_LANGUAGE_LR", 
        "tfMRI_RELATIONAL_RL", 
        "tfMRI_RELATIONAL_LR", 
        "tfMRI_SOCIAL_RL", 
        "tfMRI_SOCIAL_LR"
        ]
    motor_data_list_LR = {}
    wm_data_list_LR = {}
    emotion_data_list_LR = {}
    gambling_data_list_LR = {}
    language_data_list_LR = {}
    relational_data_list_LR = {}
    social_data_list_LR = {}
    motor_data_list_RL = {}
    wm_data_list_RL = {}
    emotion_data_list_RL = {}
    gambling_data_list_RL = {}
    language_data_list_RL = {}
    relational_data_list_RL = {}
    social_data_list_RL = {}
    rest_data_list_LR = {}
    rest_data_list_RL = {}
    tasks_missing = []
    
    for subject in subjects:
      for task in full_task_names:
        try:
          if 'MOTOR' in task:
            if 'LR' in task:
              motor_data_list_LR[subject] = np.load(path_pattern.format(subject, task, task))
              
            else:
              motor_data_list_RL[subject] = np.load(path_pattern.format(subject, task, task))
          elif 'WM' in task:
            if 'LR' in task:
              wm_data_list_LR[subject] = np.load(path_pattern.format(subject, task, task))
            else:
              wm_data_list_RL[subject] = np.load(path_pattern.format(subject, task, task))
          elif 'EMOTION' in task:
            if 'LR' in task:
              emotion_data_list_LR[subject] = np.load(path_pattern.format(subject, task, task))
            else:
              emotion_data_list_RL[subject] = np.load(path_pattern.format(subject, task, task))
          elif 'GAMBLING' in task:
            if 'LR' in task:
              gambling_data_list_LR[subject] = np.load(path_pattern.format(subject, task, task))
            else:
              gambling_data_list_RL[subject] = np.load(path_pattern.format(subject, task, task))
          elif 'LANGUAGE' in task:
            if 'LR' in task:
              language_data_list_LR[subject] = np.load(path_pattern.format(subject, task, task))
            else:
              language_data_list_RL[subject] = np.load(path_pattern.format(subject, task, task))
          elif 'RELATIONAL' in task:
            if 'LR' in task:
              relational_data_list_LR[subject] = np.load(path_pattern.format(subject, task, task))
            else:
              relational_data_list_RL[subject] = np.load(path_pattern.format(subject, task, task))
          elif 'SOCIAL' in task:
            if 'LR' in task:
              social_data_list_LR[subject] = np.load(path_pattern.format(subject, task, task))
            else:
              social_data_list_RL[subject] = np.load(path_pattern.format(subject, task, task))
          ''' elif 'REST' in task:
            if 'LR' in task:
              rest_data_list_LR[subject] = np.load(path_pattern.format(subject, task, task))
            else:
              rest_data_list_RL[subject] = np.load(path_pattern.format(subject, task, task)) '''
        except:
          tasks_missing.append(f"{subject}: {task}")
   
##################################
#### Concatenating timeseries ####
##################################
if True:
  motor_data_dict = {}
  wm_data_dict = {}
  emotion_data_dict = {}
  gambling_data_dict = {}
  language_data_dict = {}
  relational_data_dict = {}
  social_data_dict = {}
  tasks_missing_2 = []

  for subject in subjects:
    try:
      motor_data_dict[subject] = np.vstack((motor_data_list_LR[subject], motor_data_list_RL[subject]))
    except:
      try:
        motor_data_dict[subject] = motor_data_list_RL[subject]
        #print(f"Subject {subject} only has RL motor data.")
        tasks_missing_2.append(f"{subject}: tfMRI_MOTOR_LR")
      except:
        try:
          motor_data_dict[subject] = motor_data_list_LR[subject]
          #print(f"Subject {subject} only has LR motor data.")
          tasks_missing_2.append(f"{subject}: tfMRI_MOTOR_RL")
        except:
          #print(f"Subject {subject} is missing motor data.")
          tasks_missing_2.append(f"{subject}: tfMRI_MOTOR_RL")
          tasks_missing_2.append(f"{subject}: tfMRI_MOTOR_LR")
  for subject in subjects:
    try:
      wm_data_dict[subject] = np.vstack((wm_data_list_LR[subject], wm_data_list_RL[subject]))
    except:
      try:
        wm_data_dict[subject] = wm_data_list_RL[subject]
        #print(f"Subject {subject} only has RL wm data.")
        tasks_missing_2.append(f"{subject}: tfMRI_WM_LR")
      except:
        try:
          wm_data_dict[subject] = wm_data_list_LR[subject]
          #print(f"Subject {subject} only has LR wm data.")
          tasks_missing_2.append(f"{subject}: tfMRI_WM_RL")
        except:
          #print(f"Subject {subject} is missing wm data.")
          tasks_missing_2.append(f"{subject}: tfMRI_WM_RL")
          tasks_missing_2.append(f"{subject}: tfMRI_WM_LR")
  for subject in subjects:
    try:
      emotion_data_dict[subject] = np.vstack((emotion_data_list_LR[subject], emotion_data_list_RL[subject]))
    except:
      try:
        emotion_data_dict[subject] = emotion_data_list_RL[subject]
        #print(f"Subject {subject} only has RL emotion data.")
        tasks_missing_2.append(f"{subject}: tfMRI_EMOTION_LR")
      except:
        try:
          emotion_data_dict[subject] = emotion_data_list_LR[subject]
          #print(f"Subject {subject} only has LR emotion data.")
          tasks_missing_2.append(f"{subject}: tfMRI_EMOTION_RL")
        except:
          #print(f"Subject {subject} is missing emotion data.")
          tasks_missing_2.append(f"{subject}: tfMRI_EMOTION_RL")
          tasks_missing_2.append(f"{subject}: tfMRI_EMOTION_LR")
  for subject in subjects:
    try:
      gambling_data_dict[subject] = np.vstack((gambling_data_list_LR[subject], gambling_data_list_RL[subject]))
    except:
      try:
        gambling_data_dict[subject] = gambling_data_list_RL[subject]
        #print(f"Subject {subject} only has RL gambling data.")
        tasks_missing_2.append(f"{subject}: tfMRI_GAMBLING_LR")
      except:
        try:
          gambling_data_dict[subject] = gambling_data_list_LR[subject]
          #print(f"Subject {subject} only has LR gambling data.")
          tasks_missing_2.append(f"{subject}: tfMRI_GAMBLING_RL")
        except:
          #print(f"Subject {subject} is missing gambling data.")
          tasks_missing_2.append(f"{subject}: tfMRI_GAMBLING_RL")
          tasks_missing_2.append(f"{subject}: tfMRI_GAMBLING_LR")
  for subject in subjects:
    try:
      language_data_dict[subject] = np.vstack((language_data_list_LR[subject], language_data_list_RL[subject]))
    except:
      try:
        language_data_dict[subject] = language_data_list_RL[subject]
        #print(f"Subject {subject} only has RL language data.")
        tasks_missing_2.append(f"{subject}: tfMRI_LANGUAGE_LR")
      except:
        try:
          language_data_dict[subject] = language_data_list_LR[subject]
          #print(f"Subject {subject} only has LR language data.")
          tasks_missing_2.append(f"{subject}: tfMRI_LANGUAGE_RL")
        except:
          #print(f"Subject {subject} is missing language data.")
          tasks_missing_2.append(f"{subject}: tfMRI_LANGUAGE_RL")
          tasks_missing_2.append(f"{subject}: tfMRI_LANGUAGE_LR")
  for subject in subjects:
    try:
      relational_data_dict[subject] = np.vstack((relational_data_list_LR[subject], relational_data_list_RL[subject]))
    except:
      try:
        relational_data_dict[subject] = relational_data_list_RL[subject]
        #print(f"Subject {subject} only has RL relational data.")
        tasks_missing_2.append(f"{subject}: tfMRI_RELATIONAL_LR")
      except:
        try:
          relational_data_dict[subject] = relational_data_list_LR[subject]
          #print(f"Subject {subject} only has LR relational data.")
          tasks_missing_2.append(f"{subject}: tfMRI_RELATIONAL_RL")
        except:
          #print(f"Subject {subject} is missing relational data.")
          tasks_missing_2.append(f"{subject}: tfMRI_RELATIONAL_RL")
          tasks_missing_2.append(f"{subject}: tfMRI_RELATIONAL_LR")
  for subject in subjects:
    try:
      social_data_dict[subject] = np.vstack((social_data_list_LR[subject], social_data_list_RL[subject]))
    except:
      try:
        social_data_dict[subject] = social_data_list_RL[subject]
        #print(f"Subject {subject} only has RL social data.")
        tasks_missing_2.append(f"{subject}: tfMRI_SOCIAL_LR")
      except:
        try:
          social_data_dict[subject] = social_data_list_LR[subject]
          #print(f"Subject {subject} only has LR social data.")
          tasks_missing_2.append(f"{subject}: tfMRI_SOCIAL_RL")
        except:
          #print(f"Subject {subject} is missing social data.")
          tasks_missing_2.append(f"{subject}: tfMRI_SOCIAL_RL")
          tasks_missing_2.append(f"{subject}: tfMRI_SOCIAL_LR")
  RL_ONLY = False
else:
  RL_ONLY = True

##################################
#### Parcel-based input data #####
##################################
if True:
  #Initialize dataframes
  parcel_average_motor = np.zeros((len(motor_data_dict), N_PARCELS), dtype='float64')
  parcel_average_wm = np.zeros((len(wm_data_dict), N_PARCELS), dtype='float64')
  parcel_average_gambling = np.zeros((len(gambling_data_dict), N_PARCELS), dtype='float64')
  parcel_average_emotion = np.zeros((len(emotion_data_dict), N_PARCELS), dtype='float64')
  parcel_average_language = np.zeros((len(language_data_dict), N_PARCELS), dtype='float64')
  parcel_average_relational = np.zeros((len(relational_data_dict), N_PARCELS), dtype='float64')
  parcel_average_social = np.zeros((len(social_data_dict), N_PARCELS), dtype='float64')

  if glasser:
    #calculate average for each parcel in each task
    for subject, ts in enumerate(timeseries_motor):#(284, 78)
      parcel_average_motor[subject] = np.mean(ts, axis=1)
    for subject, ts in enumerate(timeseries_wm):#(405,78)
      parcel_average_wm[subject] = np.mean(ts, axis=1)
    for subject, ts in enumerate(timeseries_gambling):#(253,78)
      parcel_average_gambling[subject] = np.mean(ts, axis=1)
    for subject, ts in enumerate(timeseries_emotion):#(176,78)
      parcel_average_emotion[subject] = np.mean(ts, axis=1)
    for subject, ts in enumerate(timeseries_language):#(316,78)
      parcel_average_language[subject] = np.mean(ts, axis=1)
    for subject, ts in enumerate(timeseries_relational):#(232,78)
      parcel_average_relational[subject] = np.mean(ts, axis=1)
    for subject, ts in enumerate(timeseries_social):#(274,78)
      parcel_average_social[subject] = np.mean(ts, axis=1)  
  else:
    if RL_ONLY:
      #calculate average for each parcel in each task
      for subject, ts in enumerate(motor_data_list_RL.values()):#(284, 78)
        parcel_average_motor[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(wm_data_list_RL.values()):#(405,78)
        parcel_average_wm[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(gambling_data_list_RL.values()):#(253,78)
        parcel_average_gambling[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(emotion_data_list_RL.values()):#(176,78)
        parcel_average_emotion[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(language_data_list_RL.values()):#(316,78)
        parcel_average_language[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(relational_data_list_RL.values()):#(232,78)
        parcel_average_relational[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(social_data_list_RL.values()):#(274,78)
        parcel_average_social[subject] = np.mean(ts.T, axis=1)    
    else: #Concatenated version
      #calculate average for each parcel in each task
      for subject, ts in enumerate(motor_data_dict.values()):#(284, 78)
        parcel_average_motor[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(wm_data_dict.values()):#(405,78)
        parcel_average_wm[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(gambling_data_dict.values()):#(253,78)
        parcel_average_gambling[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(emotion_data_dict.values()):#(176,78)
        parcel_average_emotion[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(language_data_dict.values()):#(316,78)
        parcel_average_language[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(relational_data_dict.values()):#(232,78)
        parcel_average_relational[subject] = np.mean(ts.T, axis=1)
      for subject, ts in enumerate(social_data_dict.values()):#(274,78)
        parcel_average_social[subject] = np.mean(ts.T, axis=1)    
  #Make parcel dataframes
  if glasser:
    motor_parcels = pd.DataFrame(parcel_average_motor, columns= region_transpose['Network'])
    wm_parcels = pd.DataFrame(parcel_average_wm, columns= region_transpose['Network'])
    gambling_parcels = pd.DataFrame(parcel_average_gambling, columns= region_transpose['Network'])
    emotion_parcels = pd.DataFrame(parcel_average_emotion, columns= region_transpose['Network'])
    language_parcels = pd.DataFrame(parcel_average_language, columns= region_transpose['Network'])
    relational_parcels = pd.DataFrame(parcel_average_relational, columns= region_transpose['Network'])
    social_parcels = pd.DataFrame(parcel_average_social, columns= region_transpose['Network'])
  else:
    motor_parcels = pd.DataFrame(parcel_average_motor, columns= regions)
    wm_parcels = pd.DataFrame(parcel_average_wm, columns= regions)
    gambling_parcels = pd.DataFrame(parcel_average_gambling, columns= regions)
    emotion_parcels = pd.DataFrame(parcel_average_emotion, columns= regions)
    language_parcels = pd.DataFrame(parcel_average_language, columns= regions)
    relational_parcels = pd.DataFrame(parcel_average_relational, columns= regions)
    social_parcels = pd.DataFrame(parcel_average_social, columns= regions)

  if visualization:
    emotion_parcel_ex = emotion_parcels.iloc[0]
    gambling_parcel_ex = gambling_parcels.iloc[0]
    language_parcel_ex = language_parcels.iloc[0]
    motor_parcel_ex = motor_parcels.iloc[0]
    relational_parcel_ex = relational_parcels.iloc[0]
    social_parcel_ex = social_parcels.iloc[0]
    wm_parcel_ex = wm_parcels.iloc[0]
    parcel_ex_full = np.array([emotion_parcel_ex, gambling_parcel_ex, language_parcel_ex, motor_parcel_ex, relational_parcel_ex, social_parcel_ex, wm_parcel_ex])
    ax_parcel = sns.heatmap(parcel_ex_full, cbar=False, xticklabels=False, yticklabels=False, cmap = 'mako')#, linewidths=1, linecolor='white')
    # ax_parcel.set_yticklabels(['Emotion Processing','Gambling','Language','Motor','Relational Processing','Social','Working Memory'], rotation=0)
    # plt.xlabel('Parcels', color='white')
    # plt.ylabel('Tasks')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Parcel Visualization no_label mako.png', transparent=True, dpi = 1000, bbox_inches='tight')

    network_ex_full = pd.DataFrame(parcel_ex_full, columns= networks)
    scaler = StandardScaler() 
    network_ex_full = network_ex_full.groupby(lambda x:x, axis=1).sum()
    ax_network = sns.heatmap(network_ex_full, cbar=False, xticklabels=False, yticklabels=False, cmap = 'mako')#, linewidths=1, linecolor='white')
    # ax_network.set_yticklabels(['Emotion Processing','Gambling','Language','Motor','Relational Processing','Social','Working Memory'], rotation=0)
    # ax_network.set_xticklabels([
    #   'Ant IPS',
    #   'Aud',
    #   'Basal',
    #   'Cereb',
    #   'Cing-Ins',
    #   'D Att',
    #   'DMN',
    #   'Dors PCC',
    #   'L V Att',
    #   'Language',
    #   'Motor',
    #   'Occ post',
    #   'R V Att',
    #   'Salience',
    #   'Striate',
    #   'Temporal',
    #   'Vis Sec'
    # ])
    # plt.xlabel('Networks')
    # plt.ylabel('Tasks')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Network Visualization no_label mako.png', transparent=True, dpi = 1000, bbox_inches='tight')

            

  # Add the categorical label to each dataframe
  emotion_parcels['task'] = 1
  gambling_parcels['task'] = 2
  language_parcels['task'] = 3
  motor_parcels['task'] = 4
  relational_parcels['task'] = 5
  social_parcels['task'] = 6
  wm_parcels['task'] = 7

  # Stack all of the parcel dataframes together
  parcels_full = pd.DataFrame(np.concatenate((emotion_parcels, gambling_parcels, language_parcels,
                                              motor_parcels, relational_parcels, social_parcels, wm_parcels), axis = 0))


  # Make model input data
  scaler = StandardScaler()
  X_parcels = parcels_full.iloc[:, :-1]
  y_parcels = parcels_full.iloc[:,-1]

  # Delete unused preprocessing variables
  del parcel_average_motor
  del parcel_average_wm
  del parcel_average_gambling
  del parcel_average_emotion
  del parcel_average_language
  del parcel_average_relational
  del parcel_average_social

  # del motor_parcels
  # del wm_parcels
  # del gambling_parcels 
  # del emotion_parcels
  # del language_parcels
  # del relational_parcels
  # del social_parcels

#############################################
#### Parcel Connection-based input data #####
#############################################
if True:
  #Make FC matrices for each subject for each task
  fc_matrix_motor = np.zeros((len(motor_data_dict), N_PARCELS, N_PARCELS))
  fc_matrix_wm = np.zeros((len(wm_data_dict), N_PARCELS, N_PARCELS))
  fc_matrix_gambling = np.zeros((len(gambling_data_dict), N_PARCELS, N_PARCELS))
  fc_matrix_emotion = np.zeros((len(emotion_data_dict), N_PARCELS, N_PARCELS))
  fc_matrix_language = np.zeros((len(language_data_dict), N_PARCELS, N_PARCELS))
  fc_matrix_relational = np.zeros((len(relational_data_dict), N_PARCELS, N_PARCELS))
  fc_matrix_social = np.zeros((len(social_data_dict), N_PARCELS, N_PARCELS))

  # Calculate the correlations (FC) for each task
  if glasser:
    for subject, ts in enumerate(timeseries_motor):
      fc_matrix_motor[subject] = np.corrcoef(ts)
    for subject, ts in enumerate(timeseries_wm):
      fc_matrix_wm[subject] = np.corrcoef(ts)
    for subject, ts in enumerate(timeseries_gambling):
      fc_matrix_gambling[subject] = np.corrcoef(ts)
    for subject, ts in enumerate(timeseries_emotion):
      fc_matrix_emotion[subject] = np.corrcoef(ts)
    for subject, ts in enumerate(timeseries_language):
      fc_matrix_language[subject] = np.corrcoef(ts)
    for subject, ts in enumerate(timeseries_relational):
      fc_matrix_relational[subject] = np.corrcoef(ts)
    for subject, ts in enumerate(timeseries_social):
      fc_matrix_social[subject] = np.corrcoef(ts)
  else:
    if RL_ONLY:
      for subject, ts in enumerate(motor_data_list_RL.values()):
        fc_matrix_motor[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(wm_data_list_RL.values()):
        fc_matrix_wm[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(gambling_data_list_RL.values()):
        fc_matrix_gambling[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(emotion_data_list_RL.values()):
        fc_matrix_emotion[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(language_data_list_RL.values()):
        fc_matrix_language[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(relational_data_list_RL.values()):
        fc_matrix_relational[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(social_data_list_RL.values()):
        fc_matrix_social[subject] = np.corrcoef(ts.T)
    else:
      for subject, ts in enumerate(motor_data_dict.values()):
        fc_matrix_motor[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(wm_data_dict.values()):
        fc_matrix_wm[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(gambling_data_dict.values()):
        fc_matrix_gambling[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(emotion_data_dict.values()):
        fc_matrix_emotion[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(language_data_dict.values()):
        fc_matrix_language[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(relational_data_dict.values()):
        fc_matrix_relational[subject] = np.corrcoef(ts.T)
      for subject, ts in enumerate(social_data_dict.values()):
        fc_matrix_social[subject] = np.corrcoef(ts.T)
  

  if visualization:
    emotion_parcel_con_ex = fc_matrix_emotion[0]
    mask = np.zeros_like(emotion_parcel_con_ex)
    mask[np.triu_indices_from(mask)] = True

    ax_parcel_con_emotion = sns.heatmap(emotion_parcel_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False, cmap = 'mako')
    #ax_parcel_con_emotion.set_title('Emotion Processing')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Parcel Connection Emotion mako.png', transparent=True, dpi = 1000, bbox_inches='tight')

    gambling_parcel_con_ex = fc_matrix_gambling[0]
    ax_parcel_con_gambling = sns.heatmap(gambling_parcel_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_parcel_con_gambling.set_title('Gambling')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Parcel Connection Gambling.png', transparent=True, dpi = 1000, bbox_inches='tight')

    language_parcel_con_ex = fc_matrix_language[0]
    ax_parcel_con_language = sns.heatmap(language_parcel_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_parcel_con_language.set_title('Language')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Parcel Connection Language.png', transparent=True, dpi = 1000, bbox_inches='tight')

    motor_parcel_con_ex = fc_matrix_motor[0]
    ax_parcel_con_motor = sns.heatmap(motor_parcel_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_parcel_con_motor.set_title('Motor')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Parcel Connection Motor.png', transparent=True, dpi = 1000, bbox_inches='tight')

    relational_parcel_con_ex = fc_matrix_relational[0]
    ax_parcel_con_relational = sns.heatmap(relational_parcel_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_parcel_con_relational.set_title('Relational Processing')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Parcel Connection Relational Processing.png', transparent=True, dpi = 1000, bbox_inches='tight')

    social_parcel_con_ex = fc_matrix_social[0]
    ax_parcel_con_social = sns.heatmap(social_parcel_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_parcel_con_social.set_title('Social')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Parcel Connection Social.png', transparent=True, dpi = 1000, bbox_inches='tight')

    wm_parcel_con_ex = fc_matrix_wm[0]
    ax_parcel_con_wm = sns.heatmap(wm_parcel_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_parcel_con_wm.set_title('Working Memory')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Parcel Connection Working Memory.png', transparent=True, dpi = 1000, bbox_inches='tight')

  # Initialize the vector form of each task, where each row is a participant and each column is a connection
  if glasser:
    vector_motor = np.zeros((N_SUBJECTS, 64620))
    vector_wm = np.zeros((N_SUBJECTS, 64620))
    vector_gambling = np.zeros((N_SUBJECTS, 64620))
    vector_emotion = np.zeros((N_SUBJECTS, 64620))
    vector_language = np.zeros((N_SUBJECTS, 64620))
    vector_relational = np.zeros((N_SUBJECTS, 64620))
    vector_social = np.zeros((N_SUBJECTS, 64620))
  else:
    vector_motor = np.zeros((len(motor_data_dict), 741))
    vector_wm = np.zeros((len(wm_data_dict), 741))
    vector_gambling = np.zeros((len(gambling_data_dict), 741))
    vector_emotion = np.zeros((len(emotion_data_dict), 741))
    vector_language = np.zeros((len(language_data_dict), 741))
    vector_relational = np.zeros((len(relational_data_dict), 741))
    vector_social = np.zeros((len(social_data_dict), 741))

  # Extract the diagonal of the FC matrix for each subject for each task
  subject_list = np.array(np.unique(range(len(subjects))))
  for subject in range(len(motor_data_dict.keys())):
      vector_motor[subject,:] = sym_matrix_to_vec(fc_matrix_motor[subject,:,:], discard_diagonal=True)
      if glasser:
        vector_motor[subject,:] = fc_matrix_motor[subject][np.triu_indices_from(fc_matrix_motor[subject], k=1)]
  for subject in range(len(wm_data_dict.keys())):
      vector_wm[subject,:] = sym_matrix_to_vec(fc_matrix_wm[subject,:,:], discard_diagonal=True)
      if glasser:
        vector_wm[subject,:] = fc_matrix_wm[subject][np.triu_indices_from(fc_matrix_wm[subject], k=1)]
  for subject in range(len(gambling_data_dict.keys())):
      vector_gambling[subject,:] = sym_matrix_to_vec(fc_matrix_gambling[subject,:,:], discard_diagonal=True)
      if glasser:
        vector_gambling[subject,:] = fc_matrix_gambling[subject][np.triu_indices_from(fc_matrix_gambling[subject], k=1)]
  for subject in range(len(emotion_data_dict.keys())):
      vector_emotion[subject,:] = sym_matrix_to_vec(fc_matrix_emotion[subject,:,:], discard_diagonal=True)
      if glasser:
        vector_emotion[subject,:] = fc_matrix_emotion[subject][np.triu_indices_from(fc_matrix_emotion[subject], k=1)]
  for subject in range(len(language_data_dict.keys())):
      vector_language[subject,:] = sym_matrix_to_vec(fc_matrix_language[subject,:,:], discard_diagonal=True)
      if glasser:
        vector_language[subject,:] = fc_matrix_language[subject][np.triu_indices_from(fc_matrix_language[subject], k=1)]
  for subject in range(len(relational_data_dict.keys())):
      vector_relational[subject,:] = sym_matrix_to_vec(fc_matrix_relational[subject,:,:], discard_diagonal=True)
      if glasser:
        vector_relational[subject,:] = fc_matrix_relational[subject][np.triu_indices_from(fc_matrix_relational[subject], k=1)]
  for subject in range(len(social_data_dict.keys())):
      vector_social[subject,:] = sym_matrix_to_vec(fc_matrix_social[subject,:,:], discard_diagonal=True)
      if glasser:
        vector_social[subject,:] = fc_matrix_social[subject][np.triu_indices_from(fc_matrix_social[subject], k=1)]

  # Make everything pandas dataframes
  input_data_parcel_connections_emotion = pd.DataFrame(vector_emotion)
  input_data_parcel_connections_gambling = pd.DataFrame(vector_gambling)
  input_data_parcel_connections_language = pd.DataFrame(vector_language)
  input_data_parcel_connections_motor = pd.DataFrame(vector_motor)
  input_data_parcel_connections_relational = pd.DataFrame(vector_relational)
  input_data_parcel_connections_social = pd.DataFrame(vector_social)
  input_data_parcel_connections_wm = pd.DataFrame(vector_wm)

  # Add column with task identifier for classifier
  input_data_parcel_connections_emotion['task'] = 1
  input_data_parcel_connections_gambling['task'] = 2
  input_data_parcel_connections_language['task'] = 3
  input_data_parcel_connections_motor['task'] = 4
  input_data_parcel_connections_relational['task'] = 5
  input_data_parcel_connections_social['task'] = 6
  input_data_parcel_connections_wm['task'] = 7

  # make large dataframe
  parcel_connections_task_data = pd.DataFrame(np.concatenate((input_data_parcel_connections_emotion, 
                                                              input_data_parcel_connections_gambling,  
                                                              input_data_parcel_connections_language, 
                                                              input_data_parcel_connections_motor, 
                                                              input_data_parcel_connections_relational, 
                                                              input_data_parcel_connections_social, 
                                                              input_data_parcel_connections_wm), axis = 0))

  # Make input data
  X_parcel_connections = parcel_connections_task_data.iloc[:, :-1]
  y_parcel_connections = parcel_connections_task_data.iloc[:,-1]
  

  # Free up memory
  if False:
    # Delete unused preprocessing variables
    del fc_matrix_motor 
    del fc_matrix_wm 
    del fc_matrix_gambling 
    del fc_matrix_emotion
    del fc_matrix_language
    del fc_matrix_relational
    del fc_matrix_social

    del vector_motor 
    del vector_wm 
    del vector_gambling 
    del vector_emotion 
    del vector_language 
    del vector_relational 
    del vector_social 

#########################################
## Graph Theory Metrics with brainconn ##
#########################################
if False:
  # Create dictionary to contain objects
  brainconn_dict = {
    'motor':[],
    'wm':[],
    'gambling':[],
    'emotion':[],
    'language':[],
    'relational':[],
    'social':[],
  }
  # Iterate through functional connectivity matrices and add to the network analysis dictionary
  for corr in fc_matrix_motor:
    adj_wei = corr - np.eye(corr.shape[0])  # Weighted matrix
    adj_bin = brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei, 0.2)) # Binary matrix with a theshold of .2
    brainconn_dict['motor'].append((adj_wei, adj_bin))
  for corr in fc_matrix_wm:
    adj_wei = corr - np.eye(corr.shape[0])  # Weighted matrix
    adj_bin = brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei, 0.2)) # Binary matrix with a theshold of .2
    brainconn_dict['wm'].append((adj_wei, adj_bin))
  for corr in fc_matrix_gambling:
    adj_wei = corr - np.eye(corr.shape[0])  # Weighted matrix
    adj_bin = brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei, 0.2)) # Binary matrix with a theshold of .2
    brainconn_dict['gambling'].append((adj_wei, adj_bin))
  for corr in fc_matrix_emotion:
    adj_wei = corr - np.eye(corr.shape[0])  # Weighted matrix
    adj_bin = brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei, 0.2)) # Binary matrix with a theshold of .2
    brainconn_dict['emotion'].append((adj_wei, adj_bin))
  for corr in fc_matrix_language:
    adj_wei = corr - np.eye(corr.shape[0])  # Weighted matrix
    adj_bin = brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei, 0.2)) # Binary matrix with a theshold of .2
    brainconn_dict['language'].append((adj_wei, adj_bin))
  for corr in fc_matrix_relational:
    adj_wei = corr - np.eye(corr.shape[0])  # Weighted matrix
    adj_bin = brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei, 0.2)) # Binary matrix with a theshold of .2
    brainconn_dict['relational'].append((adj_wei, adj_bin))
  for corr in fc_matrix_social:
    adj_wei = corr - np.eye(corr.shape[0])  # Weighted matrix
    adj_bin = brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei, 0.2)) # Binary matrix with a theshold of .2
    brainconn_dict['social'].append((adj_wei, adj_bin))

###############################################
#### Network Graph Creation with networkx #####
###############################################
# This bit takes a very long time to run
if False:
  import networkx as nx
  threshold = .2
  motor_graphs = {}
  for subject, ts in enumerate(timeseries_motor):
    motor_graphs[subject] = nx.Graph()
    for x in range(ts.shape[0]):
      motor_graphs[subject].add_node(region_info['name'][x], network = region_info['network'][x])
    for x in range(ts.shape[0]):
      for y in range(x, ts.shape[0]):
        cooef = np.corrcoef(ts[x],ts[y])[0][1]
        if abs(cooef) >= threshold:
          motor_graphs[subject].add_edge(
            region_info['name'][x],
            region_info['name'][y],
            weight=cooef
          )
  wm_graphs = {}
  for subject, ts in enumerate(timeseries_wm):
    wm_graphs[subject] = nx.Graph()
    for x in range(ts.shape[0]):
      wm_graphs[subject].add_node(region_info['name'][x], network = region_info['network'][x])
    for x in range(ts.shape[0]):
      for y in range(x, ts.shape[0]):
        cooef = np.corrcoef(ts[x],ts[y])[0][1]
        if abs(cooef) >= threshold:
          wm_graphs[subject].add_edge(
            region_info['name'][x],
            region_info['name'][y],
            weight=cooef
          )
  gambling_graphs = {}
  for subject, ts in enumerate(timeseries_gambling):
    gambling_graphs[subject] = nx.Graph()
    for x in range(ts.shape[0]):
      gambling_graphs[subject].add_node(region_info['name'][x], network = region_info['network'][x])
    for x in range(ts.shape[0]):
      for y in range(x, ts.shape[0]):
        cooef = np.corrcoef(ts[x],ts[y])[0][1]
        if abs(cooef) >= threshold:
          gambling_graphs[subject].add_edge(
            region_info['name'][x],
            region_info['name'][y],
            weight=cooef
          )
  emotion_graphs = {}
  for subject, ts in enumerate(timeseries_emotion):
    emotion_graphs[subject] = nx.Graph()
    for x in range(ts.shape[0]):
      emotion_graphs[subject].add_node(region_info['name'][x], network = region_info['network'][x])
    for x in range(ts.shape[0]):
      for y in range(x, ts.shape[0]):
        cooef = np.corrcoef(ts[x],ts[y])[0][1]
        if abs(cooef) >= threshold:
          emotion_graphs[subject].add_edge(
            region_info['name'][x],
            region_info['name'][y],
            weight=cooef
          )
  language_graphs = {}
  for subject, ts in enumerate(timeseries_language):
    language_graphs[subject] = nx.Graph()
    for x in range(ts.shape[0]):
      language_graphs[subject].add_node(region_info['name'][x], network = region_info['network'][x])
    for x in range(ts.shape[0]):
      for y in range(x, ts.shape[0]):
        cooef = np.corrcoef(ts[x],ts[y])[0][1]
        if abs(cooef) >= threshold:
          language_graphs[subject].add_edge(
            region_info['name'][x],
            region_info['name'][y],
            weight=cooef
          )
  relational_graphs = {}
  for subject, ts in enumerate(timeseries_relational):
    relational_graphs[subject] = nx.Graph()
    for x in range(ts.shape[0]):
      relational_graphs[subject].add_node(region_info['name'][x], network = region_info['network'][x])
    for x in range(ts.shape[0]):
      for y in range(x, ts.shape[0]):
        cooef = np.corrcoef(ts[x],ts[y])[0][1]
        if abs(cooef) >= threshold:
          relational_graphs[subject].add_edge(
            region_info['name'][x],
            region_info['name'][y],
            weight=cooef
          )
  social_graphs = {}
  for subject, ts in enumerate(timeseries_social):
    social_graphs[subject] = nx.Graph()
    for x in range(ts.shape[0]):
      social_graphs[subject].add_node(region_info['name'][x], network = region_info['network'][x])
    for x in range(ts.shape[0]):
      for y in range(x, ts.shape[0]):
        cooef = np.corrcoef(ts[x],ts[y])[0][1]
        if abs(cooef) >= threshold:
          social_graphs[subject].add_edge(
            region_info['name'][x],
            region_info['name'][y],
            weight=cooef
          )
  g1 = motor_graphs[0]
  # Example of how to pull nodes that belong to a certain network:
  print([
      node
      for node, attr in motor_graphs[0].nodes(data=True)
      if ((attr.get('network') == 'Visual2'))
  ])

####################################
#### Network-based input data  #####
####################################
if True:
  #Attach the labels to the parcels 
  #region_transpose = pd.DataFrame(regions.T, columns=['Region', 'Network', 'Myelination'])
  X_network = pd.DataFrame(X_parcels, columns= networks)

  #Add the columns of the same network together and then scale them normally
  scaler = StandardScaler() 
  X_network = X_network.groupby(lambda x:x, axis=1).sum()
  #X_network = scaler.fit_transform(X_network) 

  #Make y vector
  y_network = parcels_full.iloc[:,-1]

###############################################
#### Network-connection based input data  #####
###############################################
if True:
  #Get the number of time points for each task
  if glasser:
    TIMEPOINTS_MOTOR = timeseries_motor[0][0].shape[0]
    TIMEPOINTS_WM = timeseries_wm[0][0].shape[0]
    TIMEPOINTS_GAMBLING = timeseries_gambling[0][0].shape[0]
    TIMEPOINTS_EMOTION = timeseries_emotion[0][0].shape[0]
    TIMEPOINTS_LANGUAGE = timeseries_language[0][0].shape[0]
    TIMEPOINTS_RELATIONAL = timeseries_relational[0][0].shape[0]
    TIMEPOINTS_SOCIAL = timeseries_social[0][0].shape[0]
  else:
    TIMEPOINTS_MOTOR = next(iter(motor_data_dict.values())).shape[0]
    TIMEPOINTS_WM = next(iter(wm_data_dict.values())).shape[0]
    TIMEPOINTS_GAMBLING = next(iter(gambling_data_dict.values())).shape[0]
    TIMEPOINTS_EMOTION = next(iter(emotion_data_dict.values())).shape[0]
    TIMEPOINTS_LANGUAGE = next(iter(language_data_dict.values())).shape[0]
    TIMEPOINTS_RELATIONAL = next(iter(relational_data_dict.values())).shape[0]
    TIMEPOINTS_SOCIAL = next(iter(social_data_dict.values())).shape[0]

  #Initialize data matrices
  network_task = []
  parcel_transpose_motor = np.zeros((N_SUBJECTS, TIMEPOINTS_MOTOR, N_PARCELS))
  parcel_transpose_wm = np.zeros((N_SUBJECTS, TIMEPOINTS_WM, N_PARCELS))
  parcel_transpose_gambling = np.zeros((N_SUBJECTS, TIMEPOINTS_GAMBLING, N_PARCELS))
  parcel_transpose_emotion = np.zeros((N_SUBJECTS, TIMEPOINTS_EMOTION , N_PARCELS))
  parcel_transpose_language = np.zeros((N_SUBJECTS, TIMEPOINTS_LANGUAGE, N_PARCELS))
  parcel_transpose_relational = np.zeros((N_SUBJECTS, TIMEPOINTS_RELATIONAL, N_PARCELS))
  parcel_transpose_social = np.zeros((N_SUBJECTS, TIMEPOINTS_SOCIAL , N_PARCELS))

  excluded = {
    'motor':0,
    'wm':0,
    'gambling':0,
    'emotion':0,
    'language':0,
    'relational':0,
    'social':0,
  }

  #transponse dimensions so that we can add the labels for each network
  if glasser:
    for subject, ts in enumerate(timeseries_motor):
      parcel_transpose_motor[subject] = ts.T
    for subject, ts in enumerate(timeseries_wm):
      parcel_transpose_wm[subject] = ts.T
    for subject, ts in enumerate(timeseries_gambling):
      parcel_transpose_gambling[subject] = ts.T 
    for subject, ts in enumerate(timeseries_emotion):
      parcel_transpose_emotion[subject] = ts.T
    for subject, ts in enumerate(timeseries_language):
      parcel_transpose_language[subject] = ts.T
    for subject, ts in enumerate(timeseries_relational):
      parcel_transpose_relational[subject] = ts.T
    for subject, ts in enumerate(timeseries_social):
      parcel_transpose_social[subject] = ts.T
  else:
    if RL_ONLY:
      for subject, ts in enumerate(motor_data_list_LR.values()):
        parcel_transpose_motor[subject] = ts
      for subject, ts in enumerate(wm_data_list_LR.values()):
        try:
          parcel_transpose_wm[subject] = ts
        except:
          print(f"Subject {subject} WM timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_WM}, {N_PARCELS})")
      for subject, ts in enumerate(gambling_data_list_LR.values()):
        try:
          parcel_transpose_gambling[subject] = ts
        except:
          print(f"Subject {subject} Gambling timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_GAMBLING}, {N_PARCELS})")
      for subject, ts in enumerate(emotion_data_list_LR.values()):
        try:
          parcel_transpose_emotion[subject] = ts
        except:
          print(f"Subject {subject} Gambling timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_EMOTION}, {N_PARCELS})")
      for subject, ts in enumerate(language_data_list_LR.values()):
        try:
          parcel_transpose_language[subject] = ts
        except:
          print(f"Subject {subject} Gambling timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_LANGUAGE}, {N_PARCELS})")
      for subject, ts in enumerate(relational_data_list_LR.values()):
        try:
          parcel_transpose_relational[subject] = ts
        except:
          print(f"Subject {subject} Gambling timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_RELATIONAL}, {N_PARCELS})")
      for subject, ts in enumerate(social_data_list_LR.values()):
        try:
          parcel_transpose_social[subject] = ts
        except:
          print(f"Subject {subject} Gambling timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_SOCIAL}, {N_PARCELS})")
    else:
      for subject, ts in enumerate(motor_data_dict.values()):
        try:
          parcel_transpose_motor[subject] = ts
        except:
          #print(f"Subject {subject} Motor timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_MOTOR}, {N_PARCELS})")
          excluded['motor'] = excluded['motor'] + 1
      for subject, ts in enumerate(wm_data_dict.values()):
        try:
          parcel_transpose_wm[subject] = ts
        except:
          #print(f"Subject {subject} WM timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_WM}, {N_PARCELS})")
          excluded['wm'] = excluded['wm'] + 1
      for subject, ts in enumerate(gambling_data_dict.values()):
        try:
          parcel_transpose_gambling[subject] = ts
        except:
          #print(f"Subject {subject} Gambling timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_GAMBLING}, {N_PARCELS})")
          excluded['gambling'] = excluded['gambling'] + 1
      for subject, ts in enumerate(emotion_data_dict.values()):
        try:
          parcel_transpose_emotion[subject] = ts
        except:
          #print(f"Subject {subject} Emotion timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_EMOTION}, {N_PARCELS})")
          excluded['emotion'] = excluded['emotion'] + 1
      for subject, ts in enumerate(language_data_dict.values()):
        try:
          parcel_transpose_language[subject] = ts
        except:
          #print(f"Subject {subject} Language timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_LANGUAGE}, {N_PARCELS})")
          excluded['language'] = excluded['language'] + 1
      for subject, ts in enumerate(relational_data_dict.values()):
        try:
          parcel_transpose_relational[subject] = ts
        except:
          #print(f"Subject {subject} Relational timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_RELATIONAL}, {N_PARCELS})")
          excluded['relational'] = excluded['relational'] + 1
      for subject, ts in enumerate(social_data_dict.values()):
        try:
          parcel_transpose_social[subject] = ts
        except:
          #print(f"Subject {subject} Social timeseries is of the wrong dimenstions: {ts.shape} instead of ({TIMEPOINTS_SOCIAL}, {N_PARCELS})")
          excluded['social'] = excluded['social'] + 1
      
    
  #Make the dataframes
  parcel_transpose_motor_dfs = []
  parcel_transpose_motor = list(parcel_transpose_motor)
  parcel_transpose_wm_dfs = []
  parcel_transpose_wm = list(parcel_transpose_wm)
  parcel_transpose_gambling_dfs = []
  parcel_transpose_gambling = list(parcel_transpose_gambling)
  parcel_transpose_emotion_dfs = []
  parcel_transpose_emotion = list(parcel_transpose_emotion)
  parcel_transpose_language_dfs = []
  parcel_transpose_language = list(parcel_transpose_language)
  parcel_transpose_relational_dfs = []
  parcel_transpose_relational = list(parcel_transpose_relational)
  parcel_transpose_social_dfs = []
  parcel_transpose_social = list(parcel_transpose_social)

  #Rotate each dataframe so that the network names are the column names
  if glasser:
    for array in parcel_transpose_motor:
        parcel_transpose_motor_dfs.append(pd.DataFrame(array, columns = region_info['network']))
    for array in parcel_transpose_wm:
        parcel_transpose_wm_dfs.append(pd.DataFrame(array, columns = region_info['network']))
    for array in parcel_transpose_gambling:
        parcel_transpose_gambling_dfs.append(pd.DataFrame(array, columns = region_info['network']))
    for array in parcel_transpose_emotion:
        parcel_transpose_emotion_dfs.append(pd.DataFrame(array, columns = region_info['network']))
    for array in parcel_transpose_language:
        parcel_transpose_language_dfs.append(pd.DataFrame(array, columns = region_info['network']))
    for array in parcel_transpose_relational:
        parcel_transpose_relational_dfs.append(pd.DataFrame(array, columns = region_info['network']))
    for array in parcel_transpose_social:
        parcel_transpose_social_dfs.append(pd.DataFrame(array, columns = region_info['network']))
  else:
    for array in parcel_transpose_motor:
        parcel_transpose_motor_dfs.append(pd.DataFrame(array, columns = networks))
    for array in parcel_transpose_wm:
        parcel_transpose_wm_dfs.append(pd.DataFrame(array, columns = networks))
    for array in parcel_transpose_gambling:
        parcel_transpose_gambling_dfs.append(pd.DataFrame(array, columns = networks))
    for array in parcel_transpose_emotion:
        parcel_transpose_emotion_dfs.append(pd.DataFrame(array, columns = networks))
    for array in parcel_transpose_language:
        parcel_transpose_language_dfs.append(pd.DataFrame(array, columns = networks))
    for array in parcel_transpose_relational:
        parcel_transpose_relational_dfs.append(pd.DataFrame(array, columns = networks))
    for array in parcel_transpose_social:
        parcel_transpose_social_dfs.append(pd.DataFrame(array, columns = networks))

  # Create a new dataframe where we standardize each network in a new object  
  scaler = StandardScaler() 
  network_columns_motor = [] 
  for dataframe in parcel_transpose_motor_dfs:
      network_columns_motor.append(scaler.fit_transform(dataframe.groupby(lambda x:x, axis=1).sum()).T)
  network_columns_wm = [] 
  for dataframe in parcel_transpose_wm_dfs:
      network_columns_wm.append(scaler.fit_transform(dataframe.groupby(lambda x:x, axis=1).sum()).T)
  network_columns_gambling = [] 
  for dataframe in parcel_transpose_gambling_dfs:
      network_columns_gambling.append(scaler.fit_transform(dataframe.groupby(lambda x:x, axis=1).sum()).T)
  network_columns_emotion = [] 
  for dataframe in parcel_transpose_emotion_dfs:
      network_columns_emotion.append(scaler.fit_transform(dataframe.groupby(lambda x:x, axis=1).sum()).T)
  network_columns_language = [] 
  for dataframe in parcel_transpose_language_dfs:
      network_columns_language.append(scaler.fit_transform(dataframe.groupby(lambda x:x, axis=1).sum()).T)
  network_columns_relational = [] 
  for dataframe in parcel_transpose_relational_dfs:
      network_columns_relational.append(scaler.fit_transform(dataframe.groupby(lambda x:x, axis=1).sum()).T)
  network_columns_social = [] 
  for dataframe in parcel_transpose_social_dfs:
      network_columns_social.append(scaler.fit_transform(dataframe.groupby(lambda x:x, axis=1).sum()).T)

  #initialize the network dictionary
  fc_matrix_motor_networks = {}
  fc_matrix_wm_networks = {}
  fc_matrix_gambling_networks = {}
  fc_matrix_emotion_networks = {}
  fc_matrix_language_networks = {}
  fc_matrix_relational_networks = {}
  fc_matrix_social_networks = {}

  #Calcualte functional connectivity of each additive network
  for subject, ts in enumerate(network_columns_motor):
    fc_matrix_motor_networks[subject] = np.corrcoef(ts)
  for subject, ts in enumerate(network_columns_wm):
    fc_matrix_wm_networks[subject] = np.corrcoef(ts)
  for subject, ts in enumerate(network_columns_gambling):
    fc_matrix_gambling_networks[subject] = np.corrcoef(ts)
  for subject, ts in enumerate(network_columns_emotion):
    fc_matrix_emotion_networks[subject] = np.corrcoef(ts)
  for subject, ts in enumerate(network_columns_language):
    fc_matrix_language_networks[subject] = np.corrcoef(ts)
  for subject, ts in enumerate(network_columns_relational):
    fc_matrix_relational_networks[subject] = np.corrcoef(ts)
  for subject, ts in enumerate(network_columns_social):
    fc_matrix_social_networks[subject] = np.corrcoef(ts)

  if visualization:
    emotion_network_con_ex = fc_matrix_emotion_networks[0]
    mask = np.zeros_like(emotion_network_con_ex)
    mask[np.triu_indices_from(mask)] = True

    ax_network_con_emotion = sns.heatmap(emotion_network_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False, cmap = 'mako')
    #ax_network_con_emotion.set_title('Emotion Processing')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Network Connection Emotion no_label mako.png', transparent=True, dpi = 1000, bbox_inches='tight')

    gambling_network_con_ex = fc_matrix_gambling_networks[0]
    ax_network_con_gambling = sns.heatmap(gambling_network_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_network_con_gambling.set_title('Gambling')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Network Connection Gambling.png', transparent=True, dpi = 1000, bbox_inches='tight')

    language_network_con_ex = fc_matrix_language_networks[0]
    ax_network_con_language = sns.heatmap(language_network_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_network_con_language.set_title('Language')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Network Connection Language.png', transparent=True, dpi = 1000, bbox_inches='tight')

    motor_network_con_ex = fc_matrix_motor_networks[0]
    ax_network_con_motor = sns.heatmap(motor_network_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_network_con_motor.set_title('Motor')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Network Connection Motor.png', transparent=True, dpi = 1000, bbox_inches='tight')

    relational_network_con_ex = fc_matrix_relational_networks[0]
    ax_network_con_relational = sns.heatmap(relational_network_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_network_con_relational.set_title('Relational Processing')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Network Connection Relational Processing.png', transparent=True, dpi = 1000, bbox_inches='tight')

    social_network_con_ex = fc_matrix_social_networks[0]
    ax_network_con_social = sns.heatmap(social_network_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_network_con_social.set_title('Social')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Network Connection Social.png', transparent=True, dpi = 1000, bbox_inches='tight')

    wm_network_con_ex = fc_matrix_wm_networks[0]
    ax_network_con_wm = sns.heatmap(wm_network_con_ex, mask=mask, cbar=False, xticklabels=False, yticklabels=False)
    ax_network_con_wm.set_title('Working Memory')
    plt.savefig(source_path + sep + 'Visuals' + sep + 'Network Connection Working Memory.png', transparent=True, dpi = 1000, bbox_inches='tight')

  #Make a vectorized form of the connections (unique FC matrix values)
  if glasser:
    n_net = 66
  else:
    n_net = 136
  input_data_motor_network_connections = np.zeros((N_SUBJECTS, n_net))
  input_data_wm_network_connections = np.zeros((N_SUBJECTS, n_net))
  input_data_gambling_network_connections = np.zeros((N_SUBJECTS, n_net))
  input_data_emotion_network_connections = np.zeros((N_SUBJECTS, n_net))
  input_data_language_network_connections = np.zeros((N_SUBJECTS, n_net))
  input_data_relational_network_connections = np.zeros((N_SUBJECTS, n_net))
  input_data_social_network_connections = np.zeros((N_SUBJECTS, n_net))

  #Fill in the empty vectors with unique matrix values 
  for subject in fc_matrix_motor_networks.keys():
      input_data_motor_network_connections[subject, :] = sym_matrix_to_vec(fc_matrix_motor_networks[subject], 
                                                                    discard_diagonal=True)
  for subject in fc_matrix_wm_networks.keys():
      input_data_wm_network_connections[subject, :] = sym_matrix_to_vec(fc_matrix_wm_networks[subject], 
                                                                    discard_diagonal=True)
  for subject in fc_matrix_gambling_networks.keys():
      input_data_gambling_network_connections[subject, :] = sym_matrix_to_vec(fc_matrix_gambling_networks[subject], 
                                                                    discard_diagonal=True)
  for subject in fc_matrix_emotion_networks.keys():
      input_data_emotion_network_connections[subject, :] = sym_matrix_to_vec(fc_matrix_emotion_networks[subject], 
                                                                    discard_diagonal=True)
  for subject in fc_matrix_language_networks.keys():
      input_data_language_network_connections[subject, :] = sym_matrix_to_vec(fc_matrix_language_networks[subject], 
                                                                    discard_diagonal=True)
  for subject in fc_matrix_relational_networks.keys():
      input_data_relational_network_connections[subject, :] = sym_matrix_to_vec(fc_matrix_relational_networks[subject], 
                                                                    discard_diagonal=True)
  for subject in fc_matrix_social_networks.keys():
      input_data_social_network_connections[subject, :] = sym_matrix_to_vec(fc_matrix_social_networks[subject], 
                                                                    discard_diagonal=True)
  #Make objects dataframes
  input_data_emotion_network_connections = pd.DataFrame(input_data_emotion_network_connections).dropna(axis='index')
  input_data_gambling_network_connections = pd.DataFrame(input_data_gambling_network_connections).dropna(axis='index')
  input_data_language_network_connections = pd.DataFrame(input_data_language_network_connections).dropna(axis='index')
  input_data_motor_network_connections = pd.DataFrame(input_data_motor_network_connections).dropna(axis='index')
  input_data_relational_network_connections = pd.DataFrame(input_data_relational_network_connections).dropna(axis='index')
  input_data_social_network_connections = pd.DataFrame(input_data_social_network_connections).dropna(axis='index')
  input_data_wm_network_connections = pd.DataFrame(input_data_wm_network_connections).dropna(axis='index')

  # Add the labels for each task
  input_data_emotion_network_connections['task'] = 1
  input_data_gambling_network_connections['task'] = 2
  input_data_language_network_connections['task'] = 3
  input_data_motor_network_connections['task'] = 4
  input_data_relational_network_connections['task'] = 5
  input_data_social_network_connections['task'] = 6
  input_data_wm_network_connections['task'] = 7

  # Make the dataframes
  network_connections_task_data = pd.DataFrame(np.concatenate((input_data_emotion_network_connections, 
                                                              input_data_gambling_network_connections,  
                                                              input_data_language_network_connections, 
                                                              input_data_motor_network_connections, 
                                                              input_data_relational_network_connections, 
                                                              input_data_social_network_connections, 
                                                              input_data_wm_network_connections), axis = 0))
  X_network_connections = network_connections_task_data.iloc[:, :-1]
  y_network_connections = network_connections_task_data.iloc[:,-1]
  # Centering network connections to prevent multicolinearity
  X_network_connections_centered = X_network_connections.copy().apply(lambda x: x-x.mean())
  # Delete variables to save memory
  if False:
    del timeseries_motor
    del timeseries_wm
    del timeseries_gambling
    del timeseries_emotion
    del timeseries_language
    del timeseries_relational
    del timeseries_social

    del parcel_transpose_motor
    del parcel_transpose_wm
    del parcel_transpose_gambling
    del parcel_transpose_emotion
    del parcel_transpose_language
    del parcel_transpose_relational
    del parcel_transpose_social

    del parcel_transpose_motor_dfs
    del parcel_transpose_wm_dfs
    del parcel_transpose_gambling_dfs 
    del parcel_transpose_emotion_dfs 
    del parcel_transpose_language_dfs 
    del parcel_transpose_relational_dfs 
    del parcel_transpose_social_dfs 

    del fc_matrix_motor_networks
    del fc_matrix_wm_networks
    del fc_matrix_gambling_networks
    del fc_matrix_emotion_networks
    del fc_matrix_language_networks
    del fc_matrix_relational_networks
    del fc_matrix_social_networks

    del network_columns_motor
    del network_columns_wm
    del network_columns_gambling
    del network_columns_emotion
    del network_columns_language
    del network_columns_relational
    del network_columns_social

    del input_data_emotion_network_connections
    del input_data_gambling_network_connections
    del input_data_language_network_connections
    del input_data_motor_network_connections
    del input_data_relational_network_connections
    del input_data_social_network_connections
    del input_data_wm_network_connections

  elapsed_time = dt.datetime.now() - total_start_time
  print(f"Elapsed time to preprocess input data: ", elapsed_time)


##################################
#### making test-train splits ####
##################################
if True:
  #Parcel data partitioning and transforming
  scaler = StandardScaler()
  train_X_parcels, test_X_parcels, train_y_parcels, test_y_parcels = train_test_split(X_parcels, y_parcels, test_size = 0.2)
  train_X_parcels = scaler.fit_transform(train_X_parcels)
  test_X_parcels = scaler.transform(test_X_parcels)

  # Parcel connection data
  scaler = StandardScaler()
  train_X_parcon, test_X_parcon, train_y_parcon, test_y_parcon = train_test_split(X_parcel_connections, y_parcel_connections, test_size = 0.2)
  train_X_parcon = scaler.fit_transform(train_X_parcon)
  test_X_parcon = scaler.transform(test_X_parcon)

  # Network summation data
  scaler = StandardScaler()
  train_X_network, test_X_network, train_y_network, test_y_network = train_test_split(X_network, y_network, test_size = 0.2)
  train_X_network = scaler.fit_transform(train_X_network)
  test_X_network = scaler.transform(test_X_network)
  
  # Network connection data
  scaler = StandardScaler()
  train_X_netcon, test_X_netcon, train_y_netcon, test_y_netcon = train_test_split(X_network_connections, y_network_connections, test_size = 0.2)

#######################################
#### Checking for multicolinearity ####
#######################################
if False: 
  import seaborn as sns
  import matplotlib.pyplot as plt

  import statsmodels.api as sm
  from statsmodels.stats.outliers_influence import variance_inflation_factor

  #Check for multicolinearity in the parcel connections
  vif_info = pd.DataFrame()
  vif_info['VIF'] = [variance_inflation_factor(pd.DataFrame(train_X_parcon).values, i) for i in range(pd.DataFrame(train_X_parcon).shape[1])]
  vif_info['Column'] = train_X_parcon.columns
  vif_info.sort_values('VIF', ascending=False, inplace=True)

  vif_info_centered = pd.DataFrame()
  vif_info_centered['VIF'] = [variance_inflation_factor(X_network_connections_centered.values, i) for i in range(X_network_connections_centered.shape[1])]
  vif_info_centered['Column'] = X_network_connections_centered.columns
  vif_info_centered.sort_values('VIF', ascending=False, inplace=True)

##############################################
#### parcel connections feature selection ####
##############################################
if True:
    
  #Define function to retrive names of connections
  def vector_names(names, output_list):
    cur = names[0]
    for n in names[1:]:
      output_list.append((str(cur) + ' | ' + str(n)))
    if len(names)>2:
      output_list = vector_names(names[1:], output_list)
    return output_list
      
  #Retrive the list of connections and netowrks for the connection data
  if glasser:
    list_of_connections = np.array(vector_names(region_info['name'], []))
    list_of_networks = np.array(vector_names(region_info['network'], []))
  else:
    list_of_connections = np.array(vector_names(regions, []))
    list_of_networks = np.array(vector_names(networks, []))

  parcel_connection_corr = np.corrcoef(parcel_connections_task_data.T)

  feature_correlation_with_outcome = pd.DataFrame(parcel_connection_corr[:741, 741])

  list_of_connections = pd.DataFrame(list_of_connections)
  list_of_networks = pd.DataFrame(list_of_networks)


  feat_corr_net_connections = pd.DataFrame(pd.concat([feature_correlation_with_outcome,  list_of_connections,  list_of_networks, vif_info], axis=1))
  feat_corr_net_connections.columns = ['Correlation to outcome', 'Connections', 'Networks', "VIF"]

  #Keep only features that have a correlation to the task greater than .1
  top_feat_corr_net_connections = feat_corr_net_connections[feat_corr_net_connections['Correlation to outcome'] > .1]

  top_feat_corr_net_connections_indices = list(top_feat_corr_net_connections.index)
  data_top_feat_corr_net_connections = X_parcel_connections.iloc[:, top_feat_corr_net_connections_indices]


feat_corr_net_connections = pd.DataFrame(pd.concat([feature_correlation_with_outcome,  list_of_connections,  list_of_networks, vif_info], axis=1))
#feat_corr_net_connections.columns = ['Correlation to outcome', 'Connections', 'Networks', "VIF"]
feat_corr_net_connections.columns = ['Correlation to outcome', 'Connections', 'Networks']


  #Check for multicolinearity in the parcel connections top features
  vif_info_top = pd.DataFrame()
  vif_info_top['VIF'] = [variance_inflation_factor(pd.DataFrame(train_X_parcon_top).values, i) for i in range(pd.DataFrame(train_X_parcon_top).shape[1])]
  vif_info_top['Column'] = train_X_parcon_top.columns
  vif_info_top.sort_values('VIF', ascending=False, inplace=True)


  # Parcel connection data
  scaler = StandardScaler()
  train_X_parcon_top, test_X_parcon_top, train_y_parcon_top, test_y_parcon_top = train_test_split(data_top_feat_corr_net_connections, y_parcel_connections, test_size = 0.2)
  train_X_parcon_top = scaler.fit_transform(train_X_parcon_top)
  test_X_parcon_top = scaler.transform(test_X_parcon_top)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(train_X_parcon).correlation

corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(
    corr_linkage, ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro['ivl']))

ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()
plt.show()

n = []
accuracy = []
features = []
for x in range(10):
  n_sub = x+1
  n.append(n)

  cluster_ids = hierarchy.fcluster(corr_linkage, n_sub, criterion='distance')
  cluster_id_to_feature_ids = defaultdict(list)
  for idx, cluster_id in enumerate(cluster_ids):
      cluster_id_to_feature_ids[cluster_id].append(idx)
  selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

  X_train_sel = train_X_parcon[:, selected_features]
  X_test_sel = test_X_parcon[:, selected_features]

  clf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
  clf_sel.fit(X_train_sel, train_y_parcon)
  print("Accuracy on test data with {} features: {:.2f}".format(
        X_train_sel.shape[1],
        clf_sel.score(X_test_sel, test_y_parcon)))
  features.append(X_train_sel.shape[1])
  accuracy.append(clf_sel.score(X_test_sel, test_y_parcon))






# Parcel connections top features only SVC
lin_clf_parcel_connections = svm.LinearSVC(C=.1)
lin_clf_parcel_connections.fit(train_X_parcon_top, train_y_parcon_top)
print('SVC Parcel Connection Training accuracy: ', lin_clf_parcel_connections.score(train_X_parcon_top, train_y_parcon_top))
print('SVC Parcel Connection Test accuracy: ', lin_clf_parcel_connections.score(test_X_parcon_top, test_y_parcon_top))
svm_coef_parcel_connections = pd.DataFrame(lin_clf_parcel_connections.coef_.T)
  
##### Parcel connections #####if True:
   
   
#Tune RFC hyperparameters

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
  # Parcel connections top features only SVC
  lin_clf_parcel_connections = svm.LinearSVC(C=.1)
  lin_clf_parcel_connections.fit(train_X_parcon_top, train_y_parcon_top)
  print('SVC Parcel Connection Training accuracy: ', lin_clf_parcel_connections.score(train_X_parcon_top, train_y_parcon_top))
  print('SVC Parcel Connection Test accuracy: ', lin_clf_parcel_connections.score(test_X_parcon_top, test_y_parcon_top))
  svm_coef_parcel_connections = pd.DataFrame(lin_clf_parcel_connections.coef_.T)

##### Parcel connections #####
if True:
  #Tune RFC hyperparameters
  
  # Number of trees in random forest
  n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt']
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  max_depth.append(None)
  # Minimum number of samples required to split a node
  min_samples_split = [2, 5, 10]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 2, 4]
  # Method of selecting samples for training each tree
  bootstrap = [True, False]
  # Create the random grid
  random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}


  ####################################
  ## Search for RFC hyperparameters ##
  ####################################

  # # Use the random grid to search for best hyperparameters
  # # First create the base model to tune
  # # Random search of parameters, using 3 fold cross validation, 
  # # search across 100 different combinations, and use all available cores
  # rf = RandomForestClassifier()
  # from sklearn.model_selection import RandomizedSearchCV
  # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
  # # Fit the random search model
  # rf_random.fit(train_X_parcon_top, train_y_parcon_top)

  # rf_random.best_params_

  # def evaluate(model, test_features, test_labels):
  # predictions = model.predict(test_features)
  # errors = abs(predictions - test_labels)
  # mape = 100 * np.mean(errors / test_labels)
  # accuracy = 100 - mape
  # print('Model Performance')
  # print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
  # print('Accuracy = {:0.2f}%.'.format(accuracy))

  # return accuracy
  # base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
  # base_model.fit(train_X_parcon_top, train_y_parcon_top)
  # base_accuracy = evaluate(base_model, test_X_parcon_top, test_y_parcon_top)

  # best_random = rf_random.best_estimator_
  # random_accuracy = evaluate(best_random, test_X_parcon_top, test_y_parcon_top)

  # print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


  forest = RandomForestClassifier(random_state=1, n_estimators=1000)
  forest.fit(train_X_parcon_top, train_y_parcon_top)
  pred_y_parcon_top = np.array(forest.predict(test_X_parcon_top).astype(int))
  # How does it perform?
  print('RFC Parcel Connection Training accuracy: ', forest.score(train_X_parcon_top, train_y_parcon_top))
  print('RFC Parcel Connection Test accuracy: ', forest.score(test_X_parcon_top, test_y_parcon_top))

  # Visualize the confusion matrix
  print(classification_report(test_y_parcon_top,  pred_y_parcon_top))
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(test_y_parcon_top, pred_y_parcon_top)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print(cm)
  # let's see the cross validated score 
  # score = cross_val_score(forest,X,y, cv = 10, scoring = 'accuracy')
  # print(score)
  #predictions = pd.Series(forest.predict(test_X_parcels))
  predictions = pd.Series(pred_y_parcon_top)
  ground_truth_test_y_parcon_top = pd.Series(test_y_parcon_top)
  ground_truth_test_y_parcon_top = ground_truth_test_y_parcon_top.reset_index(drop = True)
  predictions = predictions.rename("Task")
  ground_truth_test_y_parcon_top = ground_truth_test_y_parcon_top.rename("Task")
  predict_vs_true = pd.concat([ground_truth_test_y_parcon_top, predictions],axis =1)
  predict_vs_true.columns = ["Actual", "Prediction"]
  accuracy = predict_vs_true.duplicated()
  accuracy.value_counts()


########################################
###### Support Vector Classifier #######
########################################
if True:
  # Parcels
  lin_clf_parcel = svm.LinearSVC(C=1e-5)
  lin_clf_parcel.fit(train_X_parcels, train_y_parcels)
  print('SVC Parcel Training accuracy: ', lin_clf_parcel.score(train_X_parcels, train_y_parcels))
  print('SVC Parcel Test accuracy: ', lin_clf_parcel.score(test_X_parcels, test_y_parcels))
  svm_coef_parcel = pd.DataFrame(lin_clf_parcel.coef_.T)

  # Parcel connections
  lin_clf_parcel_connections = svm.LinearSVC(C=.001)
  lin_clf_parcel_connections.fit(train_X_parcon, train_y_parcon)
  print('SVC Parcel Connection Training accuracy: ', lin_clf_parcel_connections.score(train_X_parcon, train_y_parcon))
  print('SVC Parcel Connection Test accuracy: ', lin_clf_parcel_connections.score(test_X_parcon, test_y_parcon))
  svm_coef_parcel_connections = pd.DataFrame(lin_clf_parcel_connections.coef_.T)

  # Network summations
  lin_clf_network_sum = svm.LinearSVC(C=1e-1)
  lin_clf_network_sum.fit(train_X_network, train_y_network)
  print('SVC Network Summation Training accuracy: ', lin_clf_network_sum.score(train_X_network, train_y_network))
  print('SVC Network Summation Test accuracy: ', lin_clf_network_sum.score(test_X_network, test_y_network))
  svm_coef_network_sum = pd.DataFrame(lin_clf_network_sum.coef_.T)

  # Network connections
  lin_clf_network_connection = svm.LinearSVC()
  lin_clf_network_connection.fit(train_X_netcon, train_y_netcon)
  print('SVC Network Connection Training accuracy: ', lin_clf_network_connection.score(train_X_netcon, train_y_netcon))
  print('SVC Network Connection Test accuracy: ', lin_clf_network_connection.score(test_X_netcon, test_y_netcon))
  svm_coef_network_connection = pd.DataFrame(lin_clf_network_connection.coef_.T)


###########################################
###### Principal Component Analysis #######
###########################################

from sklearn.decomposition import PCA
 
pca = PCA().fit(train_X_parcon)

plt.rcParams["figure.figsize"] = (30,20)

fig, ax = plt.subplots()
xi = np.arange(1, 742, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
# plt.xticks(np.arange(0, 742, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()

pca = PCA(300).fit(train_X_parcon)


train_pca = pca.transform(train_X_parcon)
test_pca = pca.transform(test_X_parcon)


pca.components_.shape

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

loading_matrix = pd.DataFrame(loadings)




# Parcel connections
lin_clf_parcel_connections = svm.LinearSVC(C=.001)
lin_clf_parcel_connections.fit(train_pca, train_y_parcon)
print('SVC Parcel Connection Training accuracy: ', lin_clf_parcel_connections.score(train_pca, train_y_parcon))
print('SVC Parcel Connection Test accuracy: ', lin_clf_parcel_connections.score(test_pca, test_y_parcon))
svm_coef_parcel_connections = pd.DataFrame(lin_clf_parcel_connections.coef_.T)


########################################
###### Partial Least Squares LDA #######
########################################
# import PLSRegression from scikitlearn
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

#https://stackoverflow.com/questions/53944247/performing-multiclass-pls-da-with-mlr-package-in-r

# fit the pls regression
my_plsr = PLSRegression(n_components=2, scale=False)
my_plsr.fit(train_X_parcon, train_y_parcon)
y_pred_pls= my_plsr.predict(test_X_parcon)
plt.plot(y_pred_pls)
# Cross-validation
y_cv = cross_val_predict(my_plsr, train_X_parcon, train_y_parcon, cv=10)
 
predictions = my_plsr.predict(test_X_parcon)
round(predictions)

# Calculate scores
score = r2_score(train_y_parcon, y_cv)
mse = mean_squared_error(train_y_parcon, y_cv)



# extract scores (one score per individual per component)
scores_df = pd.DataFrame(my_plsr.x_scores_)

# standardize scores between -1 and 1 so they fit on the plot
std_scores_dim1 = 2 * ( (scores_df[0] - min(scores_df[0])) / (max(scores_df[0]) - min(scores_df[0])) ) -1
std_scores_dim2 = 2 * ( (scores_df[1] - min(scores_df[1])) / (max(scores_df[1]) - min(scores_df[1])) ) -1

#extract loadings (one loading per variable per component)
loadings_df = pd.DataFrame(my_plsr.x_loadings_)


from scipy.signal import savgol_filter
 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score

def optimise_pls_cv(X, y, n_comp):
    # Define PLS object
    pls = PLSRegression(n_components=n_comp)

    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)

    # Calculate scores
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv)
    rpd = y.std()/np.sqrt(mse)
    
    return (y_cv, r2, mse, rpd)


r2s = []
mses = []
rpds = []
xticks = np.arange(1, 40)
for n_comp in xticks:
    y_cv, r2, mse, rpd = optimise_pls_cv(train_X_parcon, train_y_parcon, n_comp)
    r2s.append(r2)
    mses.append(mse)
    rpds.append(rpd)

# Plot the mses
def plot_metrics(vals, ylabel, objective):
    with plt.style.context('ggplot'):
        plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Number of PLS components')
        plt.xticks = xticks
        plt.ylabel(ylabel)
        plt.title('PLS')

    plt.show()

plot_metrics(mses, 'MSE', 'min')
plot_metrics(rpds, 'RPD', 'max')
plot_metrics(r2s, 'R2', 'max')


##################################
######## SVC Importances #########
##################################

if True:
  #Make a dataframe with task coefficients and labels for SVC
  svm_coef = svm_coef_parcel #svm_coef_parcel svm_coef_parcel_connections svm_coef_network_sum svm_coef_network_connection
  list_of_connections_series = pd.Series(list_of_connections)
  list_of_networks_series = pd.Series(list_of_networks)
  svm_important_features = pd.concat([svm_coef, list_of_connections_series, list_of_networks_series], axis=1)
  svm_important_features = np.array(svm_important_features)
  svm_important_features = pd.DataFrame(svm_important_features)

  #Create objects for each task and their coeffecients
  emotion_important_coef_svm = pd.DataFrame([svm_coef.iloc[:,0], list_of_connections_series, list_of_networks_series]).T
  emotion_important_coef_svm.columns = ['Coeffecient', 'Regions', 'Network Connections']
  gambling_important_coef_svm = pd.DataFrame([svm_coef.iloc[:,1], list_of_connections_series, list_of_networks_series]).T
  gambling_important_coef_svm.columns = ['Coeffecient', 'Regions', 'Network Connections']
  language_important_coef_svm = pd.DataFrame([svm_coef.iloc[:,2], list_of_connections_series, list_of_networks_series]).T
  language_important_coef_svm.columns = ['Coeffecient', 'Regions', 'Network Connections']
  motor_important_coef_svm = pd.DataFrame([svm_coef.iloc[:,3], list_of_connections_series, list_of_networks_series]).T
  motor_important_coef_svm.columns = ['Coeffecient', 'Regions', 'Network Connections']
  relational_important_coef_svm = pd.DataFrame([svm_coef.iloc[:,4], list_of_connections_series, list_of_networks_series]).T
  relational_important_coef_svm.columns = ['Coeffecient', 'Regions', 'Network Connections']
  social_important_coef_svm = pd.DataFrame([svm_coef.iloc[:,5], list_of_connections_series, list_of_networks_series]).T
  social_important_coef_svm.columns = ['Coeffecient', 'Regions', 'Network Connections']
  wm_important_coef_svm = pd.DataFrame([svm_coef.iloc[:,6], list_of_connections_series, list_of_networks_series]).T
  wm_important_coef_svm.columns = ['Coeffecient', 'Regions', 'Network Connections']

#######################################
###### Random Forest Classifier #######
#######################################
if True:
  ##### Parcels #####
  if True:
    forest = RandomForestClassifier(random_state=1, n_estimators=5000)
    forest.fit(train_X_parcels, train_y_parcels)
    pred_y_parcels = forest.predict(test_X_parcels)
    # How does it perform?
    print('RFC Parcel Training accuracy: ', forest.score(train_X_parcels, train_y_parcels))
    print('RFC Parcel Test accuracy: ', forest.score(test_X_parcels, test_y_parcels))

    # Visualize the confusion matrix
    from sklearn.metrics import classification_report
    #print(classification_report(test_X_parcels, test_y_parcels))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_y_parcels, pred_y_parcels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)
    # let's see the cross validated score 
    # score = cross_val_score(forest,X,y, cv = 10, scoring = 'accuracy')
    # print(score)
    #predictions = pd.Series(forest.predict(test_X_parcels))
    predictions = pd.Series(pred_y_parcels)
    ground_truth_test_y_parcels = pd.Series(test_y_parcels)
    ground_truth_test_y_parcels = ground_truth_test_y_parcels.reset_index(drop = True)
    predictions = predictions.rename("Task")
    ground_truth_test_y_parcels = ground_truth_test_y_parcels.rename("Task")
    predict_vs_true = pd.concat([ground_truth_test_y_parcels, predictions],axis =1)
    predict_vs_true.columns = ["Actual", "Prediction"]
    accuracy = predict_vs_true.duplicated()
    accuracy.value_counts()

  ##### Parcel connections #####
  if True:
    forest = RandomForestClassifier(random_state=1, n_estimators=5000)
    forest.fit(train_X_parcon, train_y_parcon)
    pred_y_parcon = np.array(forest.predict(test_X_parcon).astype(int))
    # How does it perform?
    print('RFC Parcel Connection Training accuracy: ', forest.score(train_X_parcon, train_y_parcon))
    print('RFC Parcel Connection Test accuracy: ', forest.score(test_X_parcon, test_y_parcon))

    # Visualize the confusion matrix
    print(classification_report(test_y_parcon,  pred_y_parcon))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_y_parcon, pred_y_parcon)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    # let's see the cross validated score 
    # score = cross_val_score(forest,X,y, cv = 10, scoring = 'accuracy')
    # print(score)
    #predictions = pd.Series(forest.predict(test_X_parcels))
    predictions = pd.Series(pred_y_parcon)
    ground_truth_test_y_parcon = pd.Series(test_y_parcon)
    ground_truth_test_y_parcon = ground_truth_test_y_parcon.reset_index(drop = True)
    predictions = predictions.rename("Task")
    ground_truth_test_y_parcon = ground_truth_test_y_parcon.rename("Task")
    predict_vs_true = pd.concat([ground_truth_test_y_parcon, predictions],axis =1)
    predict_vs_true.columns = ["Actual", "Prediction"]
    accuracy = predict_vs_true.duplicated()
    accuracy.value_counts()

  ##### Network summations #####
  if True:
    forest = RandomForestClassifier(random_state=1, n_estimators=5000)
    forest.fit(train_X_network, train_y_network)
    pred_y_network = forest.predict(test_X_network)
    # How does it perform?
    print('RFC Network Summation Training accuracy: ', forest.score(train_X_network, train_y_network))
    print('RFC Network Connection Test accuracy: ', forest.score(test_X_network, test_y_network))

    # Visualize the confusion matrix
    #print(classification_report(test_X_network, test_y_network))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_y_network, pred_y_network)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)
    # let's see the cross validated score 
    # score = cross_val_score(forest,X,y, cv = 10, scoring = 'accuracy')
    # print(score)
    #predictions = pd.Series(forest.predict(test_X_parcels))
    predictions = pd.Series(pred_y_network)
    ground_truth_test_y_network = pd.Series(test_y_network)
    ground_truth_test_y_network = ground_truth_test_y_network.reset_index(drop = True)
    predictions = predictions.rename("Task")
    ground_truth_test_y_network = ground_truth_test_y_network.rename("Task")
    predict_vs_true = pd.concat([ground_truth_test_y_network, predictions],axis =1)
    predict_vs_true.columns = ["Actual", "Prediction"]
    accuracy = predict_vs_true.duplicated()
    accuracy.value_counts()

  ##### Network Connections #####
  if True:
    forest = RandomForestClassifier(random_state=1, n_estimators=5000)
    forest.fit(train_X_netcon, train_y_netcon)
    pred_y_netcon = forest.predict(test_X_netcon)
    # How does it perform?
    print('RFC Network Connection Training accuracy: ', forest.score(train_X_netcon, train_y_netcon))
    print('RFC Network Connection Test accuracy: ', forest.score(test_X_netcon, test_y_netcon))

    # Visualize the confusion matrix
    #print(classification_report(test_X_netcon, test_y_netcon))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_y_netcon, pred_y_netcon)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)
    # let's see the cross validated score 
    # score = cross_val_score(forest,X,y, cv = 10, scoring = 'accuracy')
    # print(score)
    #predictions = pd.Series(forest.predict(test_X_parcels))
    predictions = pd.Series(pred_y_netcon)
    ground_truth_test_y_netcon = pd.Series(test_y_netcon)
    ground_truth_test_y_netcon = ground_truth_test_y_netcon.reset_index(drop = True)
    predictions = predictions.rename("Task")
    ground_truth_test_y_netcon = ground_truth_test_y_netcon.rename("Task")
    predict_vs_true = pd.concat([ground_truth_test_y_netcon, predictions],axis =1)
    predict_vs_true.columns = ["Actual", "Prediction"]
    accuracy = predict_vs_true.duplicated()
    accuracy.value_counts()

##################################
###### Feature Importances #######
##################################
if True:
  #Define function to retrive names of connections
  def vector_names(names, output_list):
    cur = names[0]
    for n in names[1:]:
      output_list.append((str(cur) + ' | ' + str(n)))
    if len(names)>2:
      output_list = vector_names(names[1:], output_list)
    return output_list
      

  #Retrive the list of connections and netowrks for the connection data
  if glasser:
    list_of_connections = np.array(vector_names(region_info['name'], []))
    list_of_networks = np.array(vector_names(region_info['network'], []))
  else:
    list_of_connections = np.array(vector_names(regions, []))
    list_of_networks = np.array(vector_names(networks, []))
















#################################################################################
##### Here is where it stops working for me, var declaration out of order? ######
#################################################################################
if False:
  #calculate the feature importances for RFC
  feature_names = [f'feature {i}' for i in range(X_parcel_connections.shape[1])]
  #start_time = time.time()
  #importances = forest.feature_importances_
  #std = np.std([
  #    tree.feature_importances_ for tree in forest.estimators_], axis=0)
  #elapsed_time = time.time() - start_time
  #print(f"Elapsed time to compute the importances: "
  #      f"{elapsed_time:.3f} seconds")
  #forest_importances = pd.Series(importances, index=feature_names)

  #Don't run this unless you want to wait a long time... 
  #from sklearn.inspection import permutation_importance
  #start_time = time.time()
  #result = permutation_importance(forest, test_X_netcon, test_y_netcon, n_repeats=5, random_state=1, n_jobs=-1)
  #elapsed_time = time.time() - start_time
  #print(f"Elapsed time to compute the importances: "
  #      f"{elapsed_time:.3f} seconds")


  #Permutation_forest_importances = pd.DataFrame(result.importances_mean, feature_names)
  #from numpy import savetxt
  #forest_importances.to_csv(f'C:\\Users\\kyle\\repos\\HCP-Analyses\\forest_importances_netcon.csv')


  #forest_importances_series = np.squeeze(np.array(forest_importances))

  #Now that we have the feature importances, let's organize them all into a separate dataframe
  #Permutation_features_full = pd.DataFrame(np.array((forest_importances_series,list_of_connections, list_of_networks)).T)
  #from numpy import savetxt
  #forest_importances.to_csv('C:\\Users\\kyle\\repos\\HCP-Analyses\\forest_importances_netcon.csv')

  Permutation_features_full.to_csv('C:\\Users\\kyle\\repos\\HCP-Analyses\\Permutation_features_full.csv')

  name_map = pd.read_csv("C:\\Users\\kyle\\repos\\HCP-Analyses\\network_name_map.csv")
  Permutation_features_full = pd.read_csv('C:\\Users\\kyle\\repos\\HCP-Analyses\\Permutation_features_full.csv')


  Permutation_features_full.columns = ['Importance Value', 'Regions', 'Network Connection']
  Permutation_features_full_sorted = Permutation_features_full.sort_values(by='Importance Value', ascending=False)

  Rank_order_features = Permutation_features_full_sorted.index
  Rank_order_features = list(Rank_order_features)
  Non_zero_features = Rank_order_features[:561]
  Non_zero_features = str(Non_zero_features)

  Permutation_features_full_sorted = Permutation_features_full_sorted.reset_index()
  Permutation_features_full_sorted.drop('index', axis=1, inplace=True)



  list_of_connection_counts = Permutation_features_full_sorted.iloc[:,2].value_counts()

  #The proportion of each network's total connections in the full data
  for network in pd.Series(region_info['network']).unique():
      subframe = Permutation_features_full[Permutation_features_full['Network Connection'].str.contains(network)]
      print(network, len(subframe['Network Connection'])/len(Permutation_features_full['Network Connection']))

  for network in pd.Series(region_info['network']).unique():
      subframe = Permutation_features_full[Permutation_features_full['Network Connection'].str.contains(network)]
      for network2 in pd.Series(region_info['network']).unique():
          if network2 not in network:
              subframe2 = subframe[subframe['Network Connection'].str.contains(network2)]
          else:
              subframe2 = subframe[subframe['Network Connection'].str.contains(str(network + ' | '+ network2))]
          print(network, ' | ', network2, len(subframe2['Network Connection'])/len(Permutation_features_full['Network Connection']))
          
  Permutation_features_full['Network Connection2'] = ''
  print(Permutation_features_full['Network Connection2'])
  for network in pd.Series(region_info['network']).unique():
      #Permutation_features_full[Permutation_features_full['Network Connection'].str.contains(network)]
      for network2 in pd.Series(region_info['network']).unique():
          print(str(network + ' | ' + network2))
          str1 = network + ' | ' + network2
          if network2 not in network:
              #print(len(Permutation_features_full.loc[(Permutation_features_full['Network Connection'].str.contains(network)) & (Permutation_features_full['Network Connection'].str.contains(network2)), 'Network Connection2']))
              Permutation_features_full.loc[((Permutation_features_full['Network Connection'].str.contains(network) & Permutation_features_full['Network Connection'].str.contains(network2))), 'Network Connection2'] = str1
          else:
              print(network + ' | ' + network2)
              Permutation_features_full.loc[Permutation_features_full['Network Connection'].str.contains(str(network + ' | '+network2)), 'Network Connection2'] = str1
          print(Permutation_features_full.loc[((Permutation_features_full['Network Connection'].str.contains(network) & Permutation_features_full['Network Connection'].str.contains(network2))), 'Network Connection2'])
          
  print(Permutation_features_full['Network Connection2'].unique())
      
  def network_parsing(df, names, output_list):
      cur = names[0]
      #for n in names[1:]:
          
  name_map = pd.read_csv("/Users/cjrichier/Documents/Github/HCP-Analyses/network_name_map.csv")
  Permutation_features_full = pd.read_csv(f'/Users/cjrichier/Documents/Github/HCP-Analyses/Permutation_features_full.csv')


  def name_merge(target_df, target_col, name_map):
      '''
      target_df as the dataframe to change
      target_col is the column to change on
      name_map is a 2 column dataframe with colnames: 
      ['Orignial Name','Universal Name']
      '''
      target_df.rename(columns={target_col:'Original Name'}, inplace=True)
      name_map.rename(columns={'Universal Name':target_col}, inplace=True)
      target_df = pd.merge(target_df, name_map, how='left', on= "Original Name")
      return(target_df)


  Permutation_features_full = name_merge(Permutation_features_full, 'Network Connection', name_map)
  Permutation_features_full_sorted = Permutation_features_full.sort_values(by='Importance Value', ascending=False)


  number_of_connections = Permutation_features_full['Network connection'].value_counts()

  plt.hist()

  top_features_only = Permutation_features_full_sorted.iloc[:562, :]
  top_features_indices = list(top_features_only.index)
  number_of_connections_top_features = top_features_only['Network connection'].value_counts()
 

  proportion_of_top_connections = number_of_connections_top_features / number_of_connections

  top_features_indices[-4]

  #Now let's try a model where we only use the importance features that are greater than 0
  data_only_important_features = task_data.iloc[:, top_features_indices]
  X = data_only_important_features.iloc[:, :-1]
  y = task_data.iloc[:,-1]
  '''make test-train split'''
  from sklearn.model_selection import train_test_split
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)

  #fit the model
  forest = RandomForestClassifier(random_state=1 ,n_estimators=1000)
  forest.fit(train_X, train_y)
  pred_y = forest.predict(test_X)
  #How does it perform?
  print(forest.score(train_X, train_y))
  print(forest.score(test_X, test_y))


  '''visualize the confusion matrix'''
  from sklearn.metrics import classification_report
  print(classification_report(test_y, pred_y))
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(test_y, pred_y)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print(cm)

  '''let's see the cross validated score'''
  score = cross_val_score(forest,X,y, cv = 10, scoring = 'accuracy')
  print(score)

  '''Let's visualize the difference between 
  the predicted and actual tasks'''
  predictions = pd.Series(forest.predict(test_X))
  ground_truth_test_y = pd.Series(test_y)
  ground_truth_test_y = ground_truth_test_y.reset_index(drop = True)
  predictions = predictions.rename("Task")
  ground_truth_test_y = ground_truth_test_y.rename("Task")
  predict_vs_true = pd.concat([ground_truth_test_y, predictions],axis =1)
  predict_vs_true.columns = ["Actual", "Prediction"]
  accuracy = predict_vs_true.duplicated()
  accuracy.value_counts()

  #Now let's try to plot what happens when you pull out connections progressively from the model, and see how that affects accuracy 
  list_of_model_train_accuracies = []
  list_of_model_test_accuracies = []

  temp = top_features_indices.copy()
  while len(temp)>1:
      print(len(temp))
      important_features_progressively_removed = data_only_important_features.iloc[:, temp]
      X = important_features_progressively_removed.iloc[:,:-1]
      y = task_data.iloc[:,-1]
      '''make test-train split'''
      train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)
      #fit the model
      forest = RandomForestClassifier(random_state=1 ,n_estimators=10)
      forest.fit(train_X, train_y)
      pred_y = forest.predict(test_X)
      #How does it perform?
      list_of_model_train_accuracies.append(forest.score(train_X, train_y))
      list_of_model_test_accuracies.append(forest.score(test_X, test_y))
      temp = temp[:len(temp)-1]
      
  plt.plot(list_of_model_test_accuracies)
  # One network connection at a time
  map_of_model_train_accuracies = {}
  map_of_model_test_accuracies = {}
  for network in Permutation_features_full_sorted['Network connection'].unique():
      print(network)
      subset = Permutation_features_full_sorted[Permutation_features_full_sorted['Network connection'].str.contains(network)]
      data_only_important_features = task_data.iloc[:, subset.index.intersection(top_features_indices)]
      print(len(data_only_important_features.columns))
      X = data_only_important_features.iloc[:,:-1]
      y = task_data.iloc[:,-1]
      '''make test-train split'''
      train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)
      #fit the model
      forest = RandomForestClassifier(random_state=1 ,n_estimators=10)
      forest.fit(train_X, train_y)
      pred_y = forest.predict(test_X)
      #How does it perform?
      map_of_model_train_accuracies[network] = [network, len(data_only_important_features.columns), forest.score(train_X, train_y)]
      map_of_model_test_accuracies[network] = [network, len(data_only_important_features.columns), forest.score(test_X, test_y)]
      print(forest.score(test_X, test_y))
      
  map_of_model_test_accuracies = pd.DataFrame.from_dict(map_of_model_test_accuracies, orient='index', columns = ['Network','Number of Features','Accuracy'])
      
  # Without one network connection at a time
  map_of_model_train_accuracies_rm = {}
  map_of_model_test_accuracies_rm = {}
  for network in Permutation_features_full_sorted['Network connection'].unique():
      print(network)
      subset = Permutation_features_full_sorted[~(Permutation_features_full_sorted['Network connection'].str.contains(network))]
      data_only_important_features = task_data.iloc[:, subset.index]
      X = data_only_important_features.iloc[:,:-1]
      y = task_data.iloc[:,-1]
      '''make test-train split'''
      train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)
      #fit the model
      forest = RandomForestClassifier(random_state=1 ,n_estimators=10)
      forest.fit(train_X, train_y)
      pred_y = forest.predict(test_X)
      #How does it perform?
      map_of_model_train_accuracies_rm[network] = forest.score(train_X, train_y)
      map_of_model_test_accuracies_rm[network] = forest.score(test_X, test_y)
      print(forest.score(test_X, test_y))

  # One Network at a time
  map_of_model_train_accuracies1 = {}
  map_of_model_test_accuracies1 = {}
  for network in set(region_info['network']):
      print(network)
      subset = Permutation_features_full_sorted[Permutation_features_full_sorted['Network connection'].str.contains(network)]
      data_only_important_features = task_data.iloc[:, subset.index]
      X = data_only_important_features.iloc[:,:-1]
      y = task_data.iloc[:,-1]
      '''make test-train split'''
      train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)
      #fit the model
      forest = RandomForestClassifier(random_state=1 ,n_estimators=10)
      forest.fit(train_X, train_y)
      pred_y = forest.predict(test_X)
      #How does it perform?
      map_of_model_train_accuracies1[network] = forest.score(train_X, train_y)
      map_of_model_test_accuracies1[network] = forest.score(test_X, test_y)
      print(forest.score(test_X, test_y))


  # WITHOUT one network at a time
  map_of_model_train_accuracies2 = {}
  map_of_model_test_accuracies2 = {}
  for network in set(region_info['network']):
      print(network)
      subset = Permutation_features_full_sorted[~(Permutation_features_full_sorted['Network connection'].str.contains(network))]
      data_only_important_features = task_data.iloc[:, subset.index]
      X = data_only_important_features.iloc[:,:-1]
      y = task_data.iloc[:,-1]
      '''make test-train split'''
      train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)
      #fit the model
      forest = RandomForestClassifier(random_state=1 ,n_estimators=10)
      forest.fit(train_X, train_y)
      pred_y = forest.predict(test_X)
      #How does it perform?
      map_of_model_train_accuracies2[network] = forest.score(train_X, train_y)
      map_of_model_test_accuracies2[network] = forest.score(test_X, test_y)
      print(forest.score(test_X, test_y))
      
      
      
      
      
  plt.scatter(map_of_model_test_accuracies['Number of Features'], map_of_model_test_accuracies['Accuracy'])
      
      
      
      
  ####################################
  ###### Parcel-based analysis #######
  ####################################   
      

  from sklearn.ensemble import RandomForestClassifier

  #fit the model
  forest = RandomForestClassifier(random_state=1 ,n_estimators=1000)
  forest.fit(train_X, train_y)
  pred_y = forest.predict(test_X)
  #How does it perform?
  print(forest.score(train_X, train_y))
  print(forest.score(test_X, test_y))


  '''visualize the confusion matrix'''
  from sklearn.metrics import classification_report
  print(classification_report(test_y, pred_y))
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(test_y, pred_y)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print(cm)

  '''let's see the cross validated score'''
  score = cross_val_score(forest,X,y, cv = 10, scoring = 'accuracy')
  print(score)

  '''Let's visualize the difference between 
  the predicted and actual tasks'''
  predictions = pd.Series(forest.predict(test_X))
  ground_truth_test_y = pd.Series(test_y)
  ground_truth_test_y = ground_truth_test_y.reset_index(drop = True)
  predictions = predictions.rename("Task")
  ground_truth_test_y = ground_truth_test_y.rename("Task")
  predict_vs_true = pd.concat([ground_truth_test_y, predictions],axis =1)
  predict_vs_true.columns = ["Actual", "Prediction"]
  accuracy = predict_vs_true.duplicated()
  accuracy.value_counts()

##################################
###### Graph Theory Analyses #####
##################################
if True:
  print('Yay')
