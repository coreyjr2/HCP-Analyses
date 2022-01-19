#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:04:52 2020

@author: cjrichier
"""

###HCP task data analysis###

#Load the needed libraries
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
# Necessary for visualization
from nilearn import plotting, datasets

import csv
import urllib.request as urllib2
# matplotlib
import matplotlib.pyplot as plt # For changing the color maps
from matplotlib import cm # cm=colormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



'''Set the path, and create some variables'''
# The download cells will store the data in nested directories starting here:
HCP_DIR = "/Volumes/Byrgenwerth/Datasets/HCP/"
HCP_DIR_REST = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_rest/subjects/"
HCP_DIR_TASK = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_task/subjects/"
HCP_DIR_EVS = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_task/"
HCP_DIR_BEHAVIOR = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_behavior/"
if not os.path.isdir(HCP_DIR): os.mkdir(HCP_DIR)
# The data shared for NMA projects is a subset of the full HCP dataset
N_SUBJECTS = 339
# The data have already been aggregated into ROIs from the Glasesr parcellation
N_PARCELS = 360
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
subjects = range(N_SUBJECTS)


'''You may want to limit the subjects used during code development. This 
will only load in 10 subjects if you use this list.'''
SUBJECT_SUBSET = 10
test_subjects = range(SUBJECT_SUBSET)




'''this is useful for visualizing:'''
with np.load(f"{HCP_DIR}hcp_atlas.npz") as dobj:
  atlas = dict(**dobj)

'''let's generate some information about the regions using the rest data'''
regions = np.load("/Volumes/Byrgenwerth/Datasets/HCP/hcp_rest/regions.npy").T
region_info = dict(
    name=regions[0].tolist(),
    network=regions[1],
    myelin=regions[2].astype(np.float),
)

print(region_info)

'''Now let's define a few helper functions'''
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

def load_rest_timeseries(subject, name, runs=None, concat=True, remove_mean=True):
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


def load_task_timeseries(subject, name, runs=None, concat=True, remove_mean=True):
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
      load_single_task_timeseries(subject, offset + run, remove_mean) for run in runs
  ]

  # Optionally concatenate in time
  if concat:
    bold_data = np.concatenate(bold_data, axis=-1)

  return bold_data



def load_single_rest_timeseries(subject, bold_run, remove_mean=True):
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

def load_single_task_timeseries(subject, bold_run, remove_mean=True):
  """Load timeseries data for a single subject and single run.
  
  Args:
    subject (int): 0-based subject ID to load
    bold_run (int): 1-based run index, across all tasks
    remove_mean (bool): If True, subtract the parcel-wise mean

  Returns
    ts (n_parcel x n_timepoint array): Array of BOLD data values

  """
  bold_path = f"{HCP_DIR}/hcp_task/subjects/{subject}/timeseries"
  bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
  ts = np.load(f"{bold_path}/{bold_file}")
  if remove_mean:
    ts -= ts.mean(axis=1, keepdims=True)
  return ts

def load_evs(subject, name, condition):
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


####################################
 ####### Taks Data Analysis #######
####################################

'''Make a list of the task names. This will be helpful in the future'''
tasks_names = ["motor", "wm", "gambling", "emotion", "language", "relational", "social"]

'''Now let's switch to doing some task-based 
analysis. Here are some helper functions for that.'''
def condition_frames(run_evs, skip=0):
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


def selective_average(timeseries_data, ev, skip=0):
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

##########################################
 ####### Behavioral Data Analysis #######
##########################################



'''         # Working memory #         '''

'''Now let's load the working memory behavioral data'''
behavior_wm = np.genfromtxt(f"{HCP_DIR_BEHAVIOR}/behavior/wm.csv",
                            delimiter=",",
                            names=True,
                            dtype=None,
                            encoding="utf")


'''create a list of the subject ID's'''
subj_list = np.array(np.unique(behavior_wm['Subject']))


print(behavior_wm)
print(behavior_wm[:5])
print(behavior_wm.dtype.names)
'''Looks good. Let's make it a pandas object, because that is easier to work with.'''
behavior_wm = pd.DataFrame(behavior_wm)
print(behavior_wm)
'''Again, looks good. What are the variables we are working with here? We can do
this by iterating a command to print all the names of the variables (in this case
they are columns)'''
for variable in behavior_wm.columns: 
    print(variable) 

'''Let's do some more data exploration. What are the names of the conditions 
in the working memory task?'''
print(np.unique(behavior_wm["ConditionName"]))

'''let's now find each subject's total accuracy 
for each condition for each subject'''
wm_accuracy = behavior_wm.groupby(['Subject', 'ConditionName'])['ACC'].mean().reset_index()
print(wm_accuracy)


'''doing some more data wrangling. Let's make objects for each condition.'''
wm_0bk_body = wm_accuracy.loc[wm_accuracy['ConditionName'] == '0BK_BODY']
wm_0bk_face = wm_accuracy.loc[wm_accuracy['ConditionName'] == '0BK_FACE']
wm_0bk_place = wm_accuracy.loc[wm_accuracy['ConditionName'] == '0BK_PLACE']
wm_0bk_tool = wm_accuracy.loc[wm_accuracy['ConditionName'] == '0BK_TOOL']
wm_2bk_body = wm_accuracy.loc[wm_accuracy['ConditionName'] == '2BK_BODY']
wm_2bk_face = wm_accuracy.loc[wm_accuracy['ConditionName'] == '2BK_FACE']
wm_2bk_place = wm_accuracy.loc[wm_accuracy['ConditionName'] == '2BK_PLACE']
wm_2bk_tool = wm_accuracy.loc[wm_accuracy['ConditionName'] == '2BK_TOOL']


'''         # Emotion #         '''
'''Now let's load the emotion behavioral data'''
behavior_emotion = np.genfromtxt(f"{HCP_DIR_BEHAVIOR}behavior/emotion.csv",
                            delimiter=",",
                            names=True,
                            dtype=None,
                            encoding="utf")
print(behavior_emotion)
print(behavior_emotion[:5])
print(behavior_emotion.dtype.names)
'''Looks good. Let's make it a pandas object, because that is easier to work with.'''
behavior_emotion = pd.DataFrame(behavior_emotion)
print(behavior_emotion)
'''Again, looks good. What are the variables we are working with here? We can do
this by iterating a command to print all the names of the variables (in this case
they are columns)'''
for variable in behavior_emotion.columns: 
    print(variable) 

'''Let's do some more data exploration. What are the names of the conditions 
in the working memory task?'''
print(np.unique(behavior_emotion["ConditionName"]))

'''let's now find each subject's total accuracy 
for each condition for each subject'''
emotion_accuracy = behavior_emotion.groupby(['Subject', 'ConditionName'])['ACC'].mean().reset_index()
print(emotion_accuracy)

'''doing some more data wrangling. Let's make objects for each condition.'''
emotion_face = emotion_accuracy.loc[emotion_accuracy['ConditionName'] == 'FACE']
emotion_shape = emotion_accuracy.loc[emotion_accuracy['ConditionName'] == 'SHAPE']

'''         # Language #         '''
'''Now let's load the languagebehavioral data'''
behavior_language = np.genfromtxt(f"{HCP_DIR_BEHAVIOR}/behavior/language.csv",
                            delimiter=",",
                            names=True,
                            dtype=None,
                            encoding="utf")
print(behavior_language)
print(behavior_language[:5])
print(behavior_language.dtype.names)
'''Looks good. Let's make it a pandas object, because that is easier to work with.'''
behavior_language = pd.DataFrame(behavior_language)
print(behavior_language)
'''Again, looks good. What are the variables we are working with here? We can do
this by iterating a command to print all the names of the variables (in this case
they are columns)'''
for variable in behavior_language.columns: 
    print(variable) 

'''Let's do some more data exploration. What are the names of the conditions 
in the working memory task?'''
print(np.unique(behavior_language["ConditionName"]))

'''let's now find each subject's total accuracy 
for each condition for each subject'''
language_accuracy = behavior_language.groupby(['Subject', 'ConditionName'])['ACC'].mean().reset_index()
print(language_accuracy)

'''doing some more data wrangling. Let's make objects for each condition.'''
language_math = language_accuracy.loc[language_accuracy['ConditionName'] == 'MATH']
language_story = language_accuracy.loc[language_accuracy['ConditionName'] == 'STORY']

######################################################
'''load in subsets of each task timeseries data'''
######################################################

timeseries_motor_subset = []
for subject in test_subjects:
    timeseries_motor_subset.append(load_task_timeseries(subject, "motor", concat=True))
print(timeseries_motor_subset)
timeseries_wm_subset = []
for subject in test_subjects:
  timeseries_wm_subset.append(load_task_timeseries(subject, "wm", concat=True))
print(timeseries_wm_subset)
timeseries_gambling_subset = []
for subject in test_subjects:
  timeseries_gambling_subset.append(load_task_timeseries(subject, "gambling", concat=True))
print(timeseries_gambling_subset)
timeseries_emotion_subset = []
for subject in test_subjects:
  timeseries_emotion_subset.append(load_task_timeseries(subject, "emotion", concat=True))
print(timeseries_emotion_subset)
timeseries_language_subset = []
for subject in test_subjects:
  timeseries_language_subset.append(load_task_timeseries(subject, "language", concat=True))
print(timeseries_language_subset)
timeseries_relational_subset = []
for subject in test_subjects:
  timeseries_relational_subset.append(load_task_timeseries(subject, "relational", concat=True))
print(timeseries_relational_subset)
timeseries_social_subset = []
for subject in test_subjects:
  timeseries_social_subset.append(load_task_timeseries(subject, "social", concat=True))
print(timeseries_social_subset)


'''now let's make FC matrices for each task'''

'''Initialize the matrices'''
fc_matrix_task_subsets = []
fc_matrix_motor_subset = np.zeros((SUBJECT_SUBSET, N_PARCELS, N_PARCELS))
fc_matrix_wm_subset = np.zeros((SUBJECT_SUBSET, N_PARCELS, N_PARCELS))
fc_matrix_gambling_subset = np.zeros((SUBJECT_SUBSET, N_PARCELS, N_PARCELS))
fc_matrix_emotion_subset = np.zeros((SUBJECT_SUBSET, N_PARCELS, N_PARCELS))
fc_matrix_language_subset = np.zeros((SUBJECT_SUBSET, N_PARCELS, N_PARCELS))
fc_matrix_relational_subset = np.zeros((SUBJECT_SUBSET, N_PARCELS, N_PARCELS))
fc_matrix_social_subset = np.zeros((SUBJECT_SUBSET, N_PARCELS, N_PARCELS))

fc_matrix_task_subsets = np.append(fc_matrix_motor_subset, fc_matrix_wm_subset, 
                                   axis = 0)


'''calculate the correlations (FC) for each task'''
for sub, ts in enumerate(timeseries_motor_subset):
  fc_matrix_motor_subset[sub] = np.corrcoef(ts)
for sub, ts in enumerate(timeseries_wm_subset):
  fc_matrix_wm_subset[sub] = np.corrcoef(ts)
for sub, ts in enumerate(timeseries_gambling_subset):
  fc_matrix_gambling_subset[sub] = np.corrcoef(ts)
for sub, ts in enumerate(timeseries_emotion_subset):
  fc_matrix_emotion_subset[sub] = np.corrcoef(ts)
for sub, ts in enumerate(timeseries_language_subset):
  fc_matrix_language_subset[sub] = np.corrcoef(ts)
for sub, ts in enumerate(timeseries_relational_subset):
  fc_matrix_relational_subset[sub] = np.corrcoef(ts)
for sub, ts in enumerate(timeseries_social_subset):
  fc_matrix_social_subset[sub] = np.corrcoef(ts)

'''Initialize the vector form of each task, 
where each row is a participant and each column is a connection'''
vector_motor_subset = np.zeros((SUBJECT_SUBSET, 64620))
vector_wm_subset = np.zeros((SUBJECT_SUBSET, 64620))
vector_gambling_subset = np.zeros((SUBJECT_SUBSET, 64620))
vector_emotion_subset = np.zeros((SUBJECT_SUBSET, 64620))
vector_language_subset = np.zeros((SUBJECT_SUBSET, 64620))
vector_relational_subset = np.zeros((SUBJECT_SUBSET, 64620))
vector_social_subset = np.zeros((SUBJECT_SUBSET, 64620))

'''import a package to extract the diagonal of the correlation matrix, as well as
initializing a list of the subset of subjects. It is a neccesary step in appending the list 
of subjects to the connection data'''
from nilearn.connectome import sym_matrix_to_vec
subject_subset_list = np.array(np.unique(range(10)))


for subject in range(subject_subset_list.shape[0]):
    vector_motor_subset[subject,:] = sym_matrix_to_vec(fc_matrix_motor_subset[subject,:,:], discard_diagonal=True)
    vector_motor_subset[subject,:] = fc_matrix_motor_subset[subject][np.triu_indices_from(fc_matrix_motor_subset[subject], k=1)]
for subject in range(subject_subset_list.shape[0]):
    vector_wm_subset[subject,:] = sym_matrix_to_vec(fc_matrix_wm_subset[subject,:,:], discard_diagonal=True)
    vector_wm_subset[subject,:] = fc_matrix_wm_subset[subject][np.triu_indices_from(fc_matrix_wm_subset[subject], k=1)]
for subject in range(subject_subset_list.shape[0]):
    vector_gambling_subset[subject,:] = sym_matrix_to_vec(fc_matrix_gambling_subset[subject,:,:], discard_diagonal=True)
    vector_gambling_subset[subject,:] = fc_matrix_gambling_subset[subject][np.triu_indices_from(fc_matrix_gambling_subset[subject], k=1)]
for subject in range(subject_subset_list.shape[0]):
    vector_emotion_subset[subject,:] = sym_matrix_to_vec(fc_matrix_emotion_subset[subject,:,:], discard_diagonal=True)
    vector_emotion_subset[subject,:] = fc_matrix_emotion_subset[subject][np.triu_indices_from(fc_matrix_emotion_subset[subject], k=1)]
for subject in range(subject_subset_list.shape[0]):
    vector_language_subset[subject,:] = sym_matrix_to_vec(fc_matrix_language_subset[subject,:,:], discard_diagonal=True)
    vector_language_subset[subject,:] = fc_matrix_language_subset[subject][np.triu_indices_from(fc_matrix_language_subset[subject], k=1)]
for subject in range(subject_subset_list.shape[0]):
    vector_relational_subset[subject,:] = sym_matrix_to_vec(fc_matrix_relational_subset[subject,:,:], discard_diagonal=True)
    vector_relational_subset[subject,:] = fc_matrix_relational_subset[subject][np.triu_indices_from(fc_matrix_relational_subset[subject], k=1)]
for subject in range(subject_subset_list.shape[0]):
    vector_social_subset[subject,:] = sym_matrix_to_vec(fc_matrix_social_subset[subject,:,:], discard_diagonal=True)
    vector_social_subset[subject,:] = fc_matrix_social_subset[subject][np.triu_indices_from(fc_matrix_social_subset[subject], k=1)]


'''now add it all into one vector, that contains each subject in the first dimension, FC in the second, and task in the third'''
subject_connection_task_array = np.zeros((SUBJECT_SUBSET, 64620, 7))
subject_connection_task_array = np.dstack((vector_motor_subset, vector_wm_subset, vector_gambling_subset,
                                          vector_emotion_subset, vector_language_subset, vector_relational_subset,
                                          vector_social_subset))



'''Here is a much shorter, but less interpretable way to do what we did above. The nice thing about the above way is
that we have objects for each task, as opposed to just looping straight to the result here.'''
test_FC_subjects = np.zeros((SUBJECT_SUBSET, 7, 360, 360))
test_flatten = np.zeros((SUBJECT_SUBSET, 7, 360*360))
subject_connection_task_array_alternate = np.zeros((SUBJECT_SUBSET, 7, 64620))

for subject in test_subjects:
  '''loop over the subjects, in this case the subset'''
  ti = 0;
  for task in tasks_names:
    '''loop over all seven tasks'''
    tmp = load_task_timeseries(subject,task, concat = True)
    '''make an object that concatenates the time series'''
    test_FC_subjects[subject, ti, :, :] = np.corrcoef(tmp)
    '''take the correlation coefficient over the time series for each task'''
    subject_connection_task_array_alternate[subject, ti, :] = sym_matrix_to_vec(test_FC_subjects[subject, ti, :, :], discard_diagonal=True)
    '''finally arrange them all into one object'''
    ti = ti+1

'''arrange the dimensions such that tasks are the third dimension and print to make sure it works as we wanted'''
subject_connection_task_array_alternate = subject_connection_task_array_alternate.transpose(0,2,1)
subject_connection_task_array_alternate.shape

##############################################
 ####### Building a predictive model #######
##############################################


'''Let's build a practice model to predict the accuracy score of the 0-back WM task using the activity 
from the resting state scans. We'll test out a basic GLM, ridge, and lasso regression.'''
#Make it so the indicies match              
wm_0bk_body.set_index('Subject', inplace=True)


'''Building 0-back model'''
'''Lets take a subset of each dataset. If we get too crazy the computer will 
probably crash given all this data. Let's just try ten subjects.'''
wm_0bk_body_subset = wm_0bk_body.iloc[0:10,1]

#Start building our model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

'''Let's start with regular linear regression'''
lin_reg = LinearRegression()
sorted(sk.metrics.SCORERS.keys())
MSEs = cross_val_score(lin_reg, subject_connection_task_array[:,:,0], 
                       wm_0bk_body_subset,
                       scoring = 'neg_mean_squared_error',
                       cv=2)
mean_MSE = np.mean(MSEs)
print(mean_MSE)
'''Now let's try ridge regression'''
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()
#Make a dictionary of alphas
ridge_parameters = {'alpha': [1e-15, 1e-10, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_regressor = GridSearchCV(ridge, ridge_parameters, scoring = 'neg_mean_squared_error',cv=2)
ridge_regressor.fit(subject_connection_task_array[:,:,0], 
                       wm_0bk_body_subset)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
'''Straightforward enough! Let's try to use a lasso regression'''
from sklearn.linear_model import Lasso
lasso = Lasso()
#Make a dictionary of alphas
lasso_parameters = {'alpha': [1e-15, 1e-10, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regressor = GridSearchCV(lasso, lasso_parameters, scoring = 'neg_mean_squared_error',cv=2)
lasso_regressor.fit(subject_connection_task_array[:,:,0], 
                       wm_0bk_body_subset)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

'''Building 2-back model'''
'''Lets take a subset of each dataset. If we get too crazy the computer will 
probably crash given all this data. Let's just try ten subjects.'''
wm_2bk_body_subset = wm_2bk_body.iloc[0:10,1]

#Start building our model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

'''Let's start with regular linear regression'''
lin_reg = LinearRegression()
sorted(sk.metrics.SCORERS.keys())
MSEs = cross_val_score(lin_reg, subject_connection_task_array[:,:,0], 
                       wm_2bk_body_subset,
                       scoring = 'neg_mean_squared_error',
                       cv=2)
mean_MSE = np.mean(MSEs)
print(mean_MSE)
'''Now let's try ridge regression'''
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()
#Make a dictionary of alphas
ridge_parameters = {'alpha': [1e-15, 1e-10, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridge_regressor = GridSearchCV(ridge, ridge_parameters, scoring = 'neg_mean_squared_error',cv=2)
ridge_regressor.fit(subject_connection_task_array[:,:,0], 
                       wm_2bk_body_subset)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
'''Straightforward enough! Let's try to use a lasso regression'''
from sklearn.linear_model import Lasso
lasso = Lasso()
#Make a dictionary of alphas
lasso_parameters = {'alpha': [1e-15, 1e-10, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso_regressor = GridSearchCV(lasso, lasso_parameters, scoring = 'neg_mean_squared_error',cv=2)
lasso_regressor.fit(subject_connection_task_array[:,:,0], 
                       wm_2bk_body_subset)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)




##################################################################################

##################################################################################

##################################################################################

##################################################################################

##################################################################################




#defining a list with a loop?
#flattened = [element for j, row in enumerate(matrix) for i, element in enumerate(row) if i < j]

#defines a list with numbers 0-99, called list comprehension
#simple = [i for i in range(100)]

#img = image.load_image(path)
#confounds = image.high_variance_counfounds(image)
#time_series = masker.fit_transform(image_apth, confounds=confounds)

#use the harvar-oxford d atlas, discrete (thresholded atlas?)








timeseries_motor = []
for subject in subjects:
    timeseries_motor.append(load_task_timeseries(subject, "motor", concat=True))
print(timeseries_motor)
timeseries_wm = []
for subject in subjects:
  timeseries_wm.append(load_task_timeseries(subject, "wm", concat=True))
print(timeseries_wm)
timeseries_gambling = []
for subject in subjects:
  timeseries_gambling.append(load_task_timeseries(subject, "gambling", concat=True))
print(timeseries_gambling)
timeseries_emotion = []
for subject in subjects:
  timeseries_emotion.append(load_task_timeseries(subject, "emotion", concat=True))
print(timeseries_emotion)
timeseries_language = []
for subject in subjects:
  timeseries_language.append(load_task_timeseries(subject, "language", concat=True))
print(timeseries_language)
timeseries_relational = []
for subject in subjects:
  timeseries_relational.append(load_task_timeseries(subject, "relational", concat=True))
print(timeseries_relational)
timeseries_social = []
for subject in subjects:
  timeseries_social.append(load_task_timeseries(subject, "social", concat=True))
print(timeseries_social)


'''now let's make FC matrices for each task'''

'''Initialize the matrices'''
fc_matrix_task = []
fc_matrix_motor = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_wm = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_gambling = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_emotion = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_language = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_relational = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_social = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))


'''calculate the correlations (FC) for each task'''
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

'''Initialize the vector form of each task, 
where each row is a participant and each column is a connection'''
vector_motor = np.zeros((N_SUBJECTS, 64620))
vector_wm = np.zeros((N_SUBJECTS, 64620))
vector_gambling = np.zeros((N_SUBJECTS, 64620))
vector_emotion = np.zeros((N_SUBJECTS, 64620))
vector_language = np.zeros((N_SUBJECTS, 64620))
vector_relational = np.zeros((N_SUBJECTS, 64620))
vector_social = np.zeros((N_SUBJECTS, 64620))

'''import a package to extract the diagonal of the correlation matrix, as well as
initializing a list of the subset of subjects. It is a neccesary step in appending the list 
of subjects to the connection data'''
from nilearn.connectome import sym_matrix_to_vec
subject_list = np.array(np.unique(range(339)))


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



'''remove stuff we don't need to save memory'''
del timeseries_motor
del timeseries_wm
del timeseries_gambling
del timeseries_emotion
del timeseries_language
del timeseries_relational
del timeseries_social

del fc_matrix_motor 
del fc_matrix_wm 
del fc_matrix_gambling 
del fc_matrix_emotion
del fc_matrix_language
del fc_matrix_relational
del fc_matrix_social


'''make everything pandas dataframes'''
emotion_brain = pd.DataFrame(vector_emotion)
gambling_brain = pd.DataFrame(vector_gambling)
language_brain = pd.DataFrame(vector_language)
motor_brain = pd.DataFrame(vector_motor)
relational_brain= pd.DataFrame(vector_relational)
social_brain = pd.DataFrame(vector_social)
wm_brain = pd.DataFrame(vector_wm)


'''Delete the old vectors to save space'''
del vector_motor 
del vector_wm 
del vector_gambling 
del vector_emotion 
del vector_language 
del vector_relational 
del vector_social 






'''make our prediction dataset'''


emotion_brain['task'] =1
gambling_brain['task'] =2
language_brain['task'] =3
motor_brain['task'] =4
relational_brain['task'] =5
social_brain['task'] =6
wm_brain['task'] =7


task_data = pd.DataFrame(np.concatenate((emotion_brain, gambling_brain,  language_brain,
          motor_brain, relational_brain, social_brain, wm_brain), axis = 0))
X = task_data
y = np.array(X.iloc[:,-1])
#X.drop(X.columns[len(X.columns)-1], axis=1, inplace=True)


'''make more space'''
del emotion_brain
del gambling_brain
del language_brain
del motor_brain
del relational_brain
del social_brain
del wm_brain                 
        

'''Now let's try the decoding analysis'''


'''decoding task in HCP'''

'''Analysis time'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.utils import shuffle


'''make test-train split'''
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3)

forest = RandomForestClassifier(random_state=1 ,n_estimators=10)
forest.fit(train_X, train_y)

print(forest.score(train_X, train_y))
print(forest.score(test_X, test_y))



score = cross_val_score(forest,X,y, cv = 10)
score.mean()







