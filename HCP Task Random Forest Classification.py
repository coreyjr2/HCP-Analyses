#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:14:43 2020

@author: cjrichier
"""

#################################################
########## HCP decoding project code ############
#################################################

###############################
## Load the needed libraries ##
###############################
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

#record the start time 
#start_time = time.time()
total_start_time = dt.datetime.now()
# Set relevant directories

sep = os.path.sep
sys_name = platform.system() 

if getpass.getuser() == 'kyle':
  HCP_DIR = "S:\\HCP\\"
  HCP_DIR_REST = f"{HCP_DIR}hcp_rest\\subjects\\"
  HCP_DIR_TASK = f"{HCP_DIR}hcp_task\\subjects\\"
  HCP_1200 = f"{HCP_DIR}HCP_1200\\"
  basepath = str("S:\\HCP\\HCP_1200\\{}\\MNINonLinear\\Results\\")
else:
  HCP_DIR = "/Volumes/Byrgenwerth/Datasets/HCP/"
  HCP_DIR_REST = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_rest/subjects/"
  HCP_DIR_TASK = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_task/subjects/"
  HCP_DIR_EVS = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_task/"
  HCP_DIR_BEHAVIOR = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_behavior/"

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


subjects = pd.read_csv('C:\\Users\\kyle\\repos\\HCP-Task-Classification-01\\subject_list.csv')['ID']

# The data shared for NMA projects is a subset of the full HCP dataset
N_SUBJECTS = 339#len(subjects)

# The data have already been aggregated into ROIs from the Glasesr parcellation
N_PARCELS = 360
# The full dataset is stored using the MSDL Parcellation, doi: 10.1007/978-3-642-22092-0_46
#N_PARCELS = 284

# How many networks?
N_NETWORKS = 12 #17

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
subjects = range(N_SUBJECTS) #This no longer works since the subjects are no longer incremental ints

#You may want to limit the subjects used during code development. This will only load in 10 subjects if you use this list.
SUBJECT_SUBSET = 100
subjects = subjects[:SUBJECT_SUBSET]
N_SUBJECTS = SUBJECT_SUBSET


#import the demographics and bheavior data --MISSING IN FULL DATASET
''' 
demographics = pd.read_csv('/Volumes/Byrgenwerth/Datasets/HCP/HCP_demographics/demographics_behavior.csv')

#What is our gender breakdown?
demographics['Gender'].value_counts()
demographics['Age'].value_counts()
'''
#For visualization - COULD NOT GET THIS TO WORK WITH msdl ATLAS
''' 
with np.load(f"{HCP_DIR}hcp_atlas.npz") as dobj:
  atlas = dict(**dobj)
'''


#let's generate some information about the regions using the rest data

regions = np.load(f"{HCP_DIR}hcp_rest\\regions.npy").T
region_info = dict(
    name=regions[0].tolist(),
    network=regions[1],
    myelin=regions[2].astype(np.float),
)
region_transpose = pd.DataFrame(regions.T, columns=['Region', 'Network', 'Myelination'])
#print(region_info)

''' 
regions = datasets.fetch_atlas_msdl()
region_coords = []
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
)

dtype1 = np.dtype(str,str,(int,(3)))
region_array = np.array([regions['labels'],regions['networks'],regions['region_coords']], dtype = dtype1)
region_array = np.asarray(regions[['labels','networks','region_coords']])
region_transpose = pd.DataFrame(regions.T, columns=['Region', 'Network', 'Myelination'])
print(region_info)

 '''

######################################
###### Define useful functions #######
######################################
if True:
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

################################
#### Making the input data #####
################################
if True:
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
  '''
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
      
  #Load in all of the timeseries for each subject for each task
  timeseries_motor = []
  for subject in subjects:
      timeseries_motor.append(
        load_task_timeseries(subject, name='', 
        full_name_runs=["tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR"], concat=True))
  timeseries_wm = []
  for subject in subjects:
    timeseries_wm.append(
      load_task_timeseries(subject, name='', 
      full_name_runs=["tfMRI_WM_RL", "tfMRI_WM_LR"], concat=True))
  timeseries_gambling = []
  for subject in subjects:
    timeseries_gambling.append(
      load_task_timeseries(subject, name='', 
      full_name_runs=["tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR"],  concat=True))
  timeseries_emotion = []
  for subject in subjects:
    timeseries_emotion.append(
      load_task_timeseries(subject, name='', 
      full_name_runs=["tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR"], concat=True))
  timeseries_language = []
  for subject in subjects:
    timeseries_language.append(
      load_task_timeseries(subject, name='',
      full_name_runs=["tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR"],  concat=True))
  timeseries_relational = []
  for subject in subjects:
    timeseries_relational.append(
      load_task_timeseries(subject, name='', 
      full_name_runs=["tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR"], concat=True))
  timeseries_social = []
  for subject in subjects:
    timeseries_social.append(
      load_task_timeseries(subject, name='', 
      full_name_runs=["tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"], concat=True))
  timeseries_rest = []
  for subject in subjects:
    timeseries_rest.append(
      load_task_timeseries(subject, name='', 
      full_name_runs=["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"],  concat=True))
  '''

##################################
#### Parcel-based input data #####
##################################
if True:
  #Initialize dataframes
  parcel_average_motor = np.zeros((N_SUBJECTS, N_PARCELS), dtype='float64')
  parcel_average_wm = np.zeros((N_SUBJECTS, N_PARCELS), dtype='float64')
  parcel_average_gambling = np.zeros((N_SUBJECTS, N_PARCELS), dtype='float64')
  parcel_average_emotion = np.zeros((N_SUBJECTS, N_PARCELS), dtype='float64')
  parcel_average_language = np.zeros((N_SUBJECTS, N_PARCELS), dtype='float64')
  parcel_average_relational = np.zeros((N_SUBJECTS, N_PARCELS), dtype='float64')
  parcel_average_social = np.zeros((N_SUBJECTS, N_PARCELS), dtype='float64')

  #calculate average for each parcel in each task
  for subject, ts in enumerate(timeseries_motor):#(284, 78)
    parcel_average_motor[subject] = np.mean(ts, axis=1)
    print(ts.shape)
  for subject, ts in enumerate(timeseries_wm):#(405,78)
    parcel_average_wm[subject] = np.mean(ts, axis=1)
    print(ts.shape)
  for subject, ts in enumerate(timeseries_gambling):#(253,78)
    print(ts.shape)
    parcel_average_gambling[subject] = np.mean(ts, axis=1)
  for subject, ts in enumerate(timeseries_emotion):#(176,78)
    print(ts.shape)
    parcel_average_emotion[subject] = np.mean(ts, axis=1)
  for subject, ts in enumerate(timeseries_language):#(316,78)
    print(ts.shape)
    parcel_average_language[subject] = np.mean(ts, axis=1)
  for subject, ts in enumerate(timeseries_relational):#(232,78)
    print(ts.shape)
    parcel_average_relational[subject] = np.mean(ts, axis=1)
  for subject, ts in enumerate(timeseries_social):#(274,78)
    print(ts.shape)
    parcel_average_social[subject] = np.mean(ts, axis=1)    

  #Make parcel dataframes
  motor_parcels = pd.DataFrame(parcel_average_motor, columns= region_transpose['Network'])
  wm_parcels = pd.DataFrame(parcel_average_wm, columns= region_transpose['Network'])
  gambling_parcels = pd.DataFrame(parcel_average_gambling, columns= region_transpose['Network'])
  emotion_parcels = pd.DataFrame(parcel_average_emotion, columns= region_transpose['Network'])
  language_parcels = pd.DataFrame(parcel_average_language, columns= region_transpose['Network'])
  relational_parcels = pd.DataFrame(parcel_average_relational, columns= region_transpose['Network'])
  social_parcels = pd.DataFrame(parcel_average_social, columns= region_transpose['Network'])

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

  del motor_parcels
  del wm_parcels
  del gambling_parcels 
  del emotion_parcels
  del language_parcels
  del relational_parcels
  del social_parcels

#############################################
#### Parcel Connection-based input data #####
#############################################
if True:
  #Make FC matrices for each subject for each task
  fc_matrix_motor = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
  fc_matrix_wm = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
  fc_matrix_gambling = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
  fc_matrix_emotion = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
  fc_matrix_language = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
  fc_matrix_relational = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
  fc_matrix_social = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))

  # Calculate the correlations (FC) for each task
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


  # Initialize the vector form of each task, where each row is a participant and each column is a connection
  vector_motor = np.zeros((N_SUBJECTS, 64620))
  vector_wm = np.zeros((N_SUBJECTS, 64620))
  vector_gambling = np.zeros((N_SUBJECTS, 64620))
  vector_emotion = np.zeros((N_SUBJECTS, 64620))
  vector_language = np.zeros((N_SUBJECTS, 64620))
  vector_relational = np.zeros((N_SUBJECTS, 64620))
  vector_social = np.zeros((N_SUBJECTS, 64620))

  # Extract the diagonal of the FC matrix for each subject for each task
  subject_list = np.array(np.unique(range(N_SUBJECTS)))
  for subject in range(subject_list.shape[0]):
      vector_motor[subject,:] = sym_matrix_to_vec(fc_matrix_motor[subject,:,:], discard_diagonal=True)
      vector_motor[subject,:] = fc_matrix_motor[subject][np.triu_indices_from(fc_matrix_motor[subject], k=1)]
  for subject in range(subject_list.shape[0]):
      vector_wm[subject,:] = sym_matrix_to_vec(fc_matrix_wm[subject,:,:], discard_diagonal=True)
      vector_wm[subject,:] = fc_matrix_wm[subject][np.triu_indices_from(fc_matrix_wm[subject], k=1)]
  for subject in range(subject_list.shape[0]):
      vector_gambling[subject,:] = sym_matrix_to_vec(fc_matrix_gambling[subject,:,:], discard_diagonal=True)
      vector_gambling[subject,:] = fc_matrix_gambling[subject][np.triu_indices_from(fc_matrix_gambling[subject], k=1)]
  for subject in range(subject_list.shape[0]):
      vector_emotion[subject,:] = sym_matrix_to_vec(fc_matrix_emotion[subject,:,:], discard_diagonal=True)
      vector_emotion[subject,:] = fc_matrix_emotion[subject][np.triu_indices_from(fc_matrix_emotion[subject], k=1)]
  for subject in range(subject_list.shape[0]):
      vector_language[subject,:] = sym_matrix_to_vec(fc_matrix_language[subject,:,:], discard_diagonal=True)
      vector_language[subject,:] = fc_matrix_language[subject][np.triu_indices_from(fc_matrix_language[subject], k=1)]
  for subject in range(subject_list.shape[0]):
      vector_relational[subject,:] = sym_matrix_to_vec(fc_matrix_relational[subject,:,:], discard_diagonal=True)
      vector_relational[subject,:] = fc_matrix_relational[subject][np.triu_indices_from(fc_matrix_relational[subject], k=1)]
  for subject in range(subject_list.shape[0]):
      vector_social[subject,:] = sym_matrix_to_vec(fc_matrix_social[subject,:,:], discard_diagonal=True)
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

##################################
#### Network Graph Creation #####
##################################
threshold = .2
if True:
  import networkx as nx
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
  g1.
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
  region_transpose = pd.DataFrame(regions.T, columns=['Region', 'Network', 'Myelination'])
  X_network = pd.DataFrame(X_parcels, columns= region_transpose['Network'])

  #Add the columns of the same network together and then scale them normally
  scaler = StandardScaler() 
  X_network = X_network.groupby(lambda x:x, axis=1).sum()
  X_network = scaler.fit_transform(X_network) 

  #Make y vector
  y_network = parcels_full.iloc[:,-1]

###############################################
#### Network-connection based input data  #####
###############################################
if True:
  #Get the number of time points for each task
  TIMEPOINTS_MOTOR = timeseries_motor[0][0].shape[0]
  TIMEPOINTS_WM = timeseries_wm[0][0].shape[0]
  TIMEPOINTS_GAMBLING = timeseries_gambling[0][0].shape[0]
  TIMEPOINTS_EMOTION = timeseries_emotion[0][0].shape[0]
  TIMEPOINTS_LANGUAGE = timeseries_language[0][0].shape[0]
  TIMEPOINTS_RELATIONAL = timeseries_relational[0][0].shape[0]
  TIMEPOINTS_SOCIAL = timeseries_social[0][0].shape[0]

  #Initialize data matrices
  network_task = []
  parcel_transpose_motor = np.zeros((N_SUBJECTS, TIMEPOINTS_MOTOR, N_PARCELS))
  parcel_transpose_wm = np.zeros((N_SUBJECTS, TIMEPOINTS_WM, N_PARCELS))
  parcel_transpose_gambling = np.zeros((N_SUBJECTS, TIMEPOINTS_GAMBLING, N_PARCELS))
  parcel_transpose_emotion = np.zeros((N_SUBJECTS, TIMEPOINTS_EMOTION , N_PARCELS))
  parcel_transpose_language = np.zeros((N_SUBJECTS, TIMEPOINTS_LANGUAGE, N_PARCELS))
  parcel_transpose_relational = np.zeros((N_SUBJECTS, TIMEPOINTS_RELATIONAL, N_PARCELS))
  parcel_transpose_social = np.zeros((N_SUBJECTS, TIMEPOINTS_SOCIAL , N_PARCELS))

  #transponse dimensions so that we can add the labels for each network
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

  #Make a vectorized form of the connections (unique FC matrix values)
  input_data_motor_network_connections = np.zeros((N_SUBJECTS, 66))
  input_data_wm_network_connections = np.zeros((N_SUBJECTS, 66))
  input_data_gambling_network_connections = np.zeros((N_SUBJECTS, 66))
  input_data_emotion_network_connections = np.zeros((N_SUBJECTS, 66))
  input_data_language_network_connections = np.zeros((N_SUBJECTS, 66))
  input_data_relational_network_connections = np.zeros((N_SUBJECTS, 66))
  input_data_social_network_connections = np.zeros((N_SUBJECTS, 66))

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
  input_data_emotion_network_connections = pd.DataFrame(input_data_emotion_network_connections)
  input_data_gambling_network_connections = pd.DataFrame(input_data_gambling_network_connections)
  input_data_language_network_connections = pd.DataFrame(input_data_language_network_connections)
  input_data_motor_network_connections = pd.DataFrame(input_data_motor_network_connections)
  input_data_relational_network_connections = pd.DataFrame(input_data_relational_network_connections)
  input_data_social_network_connections = pd.DataFrame(input_data_social_network_connections)
  input_data_wm_network_connections = pd.DataFrame(input_data_wm_network_connections)

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
  if True:
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

#######################################
#### Checking for multicolinearity ####
#######################################
if True:
  import seaborn as sns
  import matplotlib.pyplot as plt

  import statsmodels.api as sm
  from statsmodels.stats.outliers_influence import variance_inflation_factor

  #Check for multicolinearity in the network connections
  vif_info = pd.DataFrame()
  vif_info['VIF'] = [variance_inflation_factor(X_network_connections.values, i) for i in range(X_network_connections.shape[1])]
  vif_info['Column'] = X_network_connections.columns
  vif_info.sort_values('VIF', ascending=False, inplace=True)

  vif_info_centered = pd.DataFrame()
  vif_info_centered['VIF'] = [variance_inflation_factor(X_network_connections_centered.values, i) for i in range(X_network_connections_centered.shape[1])]
  vif_info_centered['Column'] = X_network_connections_centered.columns
  vif_info_centered.sort_values('VIF', ascending=False, inplace=True)

  ##################################
  #### making test-train splits ####
  ##################################

  #Parcel data partitioning and transforming
  train_X_parcels, test_X_parcels, train_y_parcels, test_y_parcels = train_test_split(X_parcels, y_parcels, test_size = 0.2)
  train_X_parcels = scaler.fit_transform(train_X_parcels)
  test_X_parcels = scaler.transform(test_X_parcels)

  # Parcel connection data
  train_X_parcon, test_X_parcon, train_y_parcon, test_y_parcon = train_test_split(X_parcel_connections, y_parcel_connections, test_size = 0.2)

  # Network summation data
  train_X_network, test_X_network, train_y_network, test_y_network = train_test_split(X_network, y_network, test_size = 0.2)
  train_X_network = scaler.fit_transform(train_X_network)
  test_X_network = scaler.transform(test_X_network)

  train_X_netcon, test_X_netcon, train_y_netcon, test_y_netcon = train_test_split(X_network_connections, y_network_connections, test_size = 0.2)

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
  lin_clf_parcel_connections = svm.LinearSVC()
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

##################################
######## SVC Importances #########
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
  list_of_connections = np.array(vector_names(region_info['name'], []))
  list_of_networks = np.array(vector_names(region_info['network'], []))

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
    forest = RandomForestClassifier(random_state=1, n_estimators=1000)
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
    forest = RandomForestClassifier(random_state=1, n_estimators=1000)
    forest.fit(train_X_parcon, train_y_parcon)
    pred_y_parcon = np.array(forest.predict(test_X_parcon).astype(int))
    # How does it perform?
    print('RFC Parcel Connection Training accuracy: ', forest.score(train_X_parcon, train_y_parcon))
    print('RFC Parcel Connection Test accuracy: ', forest.score(test_X_parcon, test_y_parcon))

    # Visualize the confusion matrix
    from sklearn.metrics import classification_report
    #print(classification_report(np.array(test_X_parcon), np.array(test_y_parcon).astype(int)))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_y_parcon, pred_y_parcon)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)
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
    forest = RandomForestClassifier(random_state=1, n_estimators=1000)
    forest.fit(train_X_network, train_y_network)
    pred_y_network = forest.predict(test_X_network)
    # How does it perform?
    print('RFC Network Summation Training accuracy: ', forest.score(train_X_network, train_y_network))
    print('RFC Network Connection Test accuracy: ', forest.score(test_X_network, test_y_network))

    # Visualize the confusion matrix
    from sklearn.metrics import classification_report
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
    forest = RandomForestClassifier(random_state=1, n_estimators=1000)
    forest.fit(train_X_netcon, train_y_netcon)
    pred_y_netcon = forest.predict(test_X_netcon)
    # How does it perform?
    print('RFC Network Connection Training accuracy: ', forest.score(train_X_netcon, train_y_netcon))
    print('RFC Network Connection Test accuracy: ', forest.score(test_X_netcon, test_y_netcon))

    # Visualize the confusion matrix
    from sklearn.metrics import classification_report
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
  list_of_connections = np.array(vector_names(region_info['name'], []))
  list_of_networks = np.array(vector_names(region_info['network'], []))

#################################################################################
##### Here is where it stops working for me, var declaration out of order? ######
#################################################################################
if False:
  #calculate the feature importances for RFC
  feature_names = [f'feature {i}' for i in range(X.shape[1])]
  start_time = time.time()
  importances = forest.feature_importances_
  std = np.std([
      tree.feature_importances_ for tree in forest.estimators_], axis=0)
  elapsed_time = time.time() - start_time
  print(f"Elapsed time to compute the importances: "
        f"{elapsed_time:.3f} seconds")
  forest_importances = pd.Series(importances, index=feature_names)

  #Don't run this unless you want to wait a long time... 
  from sklearn.inspection import permutation_importance
  start_time = time.time()
  result = permutation_importance(forest, test_X_netcon, test_y_netcon, n_repeats=5, random_state=1, n_jobs=1)
  elapsed_time = time.time() - start_time
  print(f"Elapsed time to compute the importances: "
        f"{elapsed_time:.3f} seconds")


  Permutation_forest_importances = pd.DataFrame(result.importances_mean, feature_names)
  from numpy import savetxt
  forest_importances.to_csv(f'C:\\Users\\kyle\\repos\\HCP-Analyses\\forest_importances_netcon.csv')

  names1 = ['0','1','2','3','4','5','6']


  out1 = vector_names(names1, [])
  print(out1)

  forest_importances_series = np.squeeze(np.array(forest_importances))

  #Now that we have the feature importances, let's organize them all into a separate dataframe
  Permutation_features_full = pd.DataFrame(np.array((forest_importances_series,list_of_connections, list_of_networks)).T)
  from numpy import savetxt
  forest_importances.to_csv('C:\\Users\\kyle\\repos\\HCP-Analyses\\forest_importances_netcon.csv')

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
