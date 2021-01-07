#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 09:04:29 2020

@author: cjrichier
"""

#Load the needed libraries
import os
import nibabel as nib
import numpy as np
import matplotlib as plt

import pandas as pd
import sklearn as sk
# Necessary for visualization
import nilearn
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

















'''         # Working memory #         '''

'''Now let's load the working memory behavioral data'''
behavior_wm = np.genfromtxt(f"{HCP_DIR_BEHAVIOR}/behavior/wm.csv",
                            delimiter=",",
                            names=True,
                            dtype=None,
                            encoding="utf")


'''create a list of the subject ID's'''
subj_list = np.array(np.unique(behavior_wm['Subject']))

'''Make it Pandas data frame'''
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

'''now get the total accuracy'''
wm_total = behavior_wm.groupby(['Subject'])['ACC'].mean().reset_index()
print(wm_total)















'''load in the wm time series data'''
from nilearn.connectome import sym_matrix_to_vec
timeseries_wm = []
for subject in subjects:
  timeseries_wm.append(load_task_timeseries(subject, "wm", concat=True))
fc_matrix_wm = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
for subject, timeseries in enumerate(timeseries_wm):
  fc_matrix_wm[sub] = np.corrcoef(ts)
vector_wm = np.zeros((N_SUBJECTS, 64620))
for subject in subjects:
    vector_wm[subject,:] = sym_matrix_to_vec(fc_matrix_wm[subject,:,:], discard_diagonal=True)
    vector_wm[subject,:] = fc_matrix_wm[subject][np.triu_indices_from(fc_matrix_wm[subject], k=1)]
wm_activity = vector_wm

timeseries_wm= None
fc_matrix_wm= None
vector_wm = None

'''load in the rest time series data'''
'''load the time series for all participants'''

timeseries_rest = []
for subject in subjects:
  timeseries_rest.append(load_rest_timeseries(subject, "rest", concat=True))
fc_matrix_rest = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
for subject, timeseries in enumerate(timeseries_rest):
  fc_matrix_rest[subject] = np.corrcoef(timeseries)
vector_rest = np.zeros((N_SUBJECTS, 64620))
for subject in subjects:
    vector_rest[subject,:] = sym_matrix_to_vec(fc_matrix_rest[subject,:,:], discard_diagonal=True)
    vector_rest[subject,:] = fc_matrix_rest[subject][np.triu_indices_from(fc_matrix_rest[subject], k=1)]
rest_activity = vector_rest
timeseries_rest= None
fc_matrix_rest= None
vector_rest = None
timeseries = None
ts= None


'''start building models'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


'''Predicting WM with ridge regression on task activity'''

ridge = Ridge()
ridge_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
ridge_regressor = GridSearchCV(ridge, ridge_parameters, scoring = 'neg_root_mean_squared_error',cv=10)
ridge_regressor.fit(wm_activity, wm_total['ACC'])
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

task_ridge_alpha = ridge_regressor.best_params_
task_ridge_rmse = ridge_regressor.best_score_

ridge = Ridge()
ridge_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
ridge_regressor = GridSearchCV(ridge, ridge_parameters, scoring = 'r2',cv=10)
ridge_regressor.fit(wm_activity, wm_total['ACC'])
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

task_ridge_r2 = ridge_regressor.best_score_



'''Predicting WM with ridge regression on rest activity'''

ridge = Ridge()
ridge_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
ridge_regressor = GridSearchCV(ridge, ridge_parameters, scoring = 'neg_root_mean_squared_error',cv=10)
ridge_regressor.fit(rest_activity, wm_total['ACC'])
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

rest_ridge_alpha = ridge_regressor.best_params_
rest_ridge_rmse = ridge_regressor.best_score_

ridge = Ridge()
ridge_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
ridge_regressor = GridSearchCV(ridge, ridge_parameters, scoring = 'r2',cv=10)
ridge_regressor.fit(rest_activity, wm_total['ACC'])
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

rest_ridge_r2 = ridge_regressor.best_score_

'''Predicting WM with lasso regression on task activity'''
from sklearn.linear_model import Lasso
lasso = Lasso()
#Make a dictionary of alphas
lasso_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
lasso_regressor = GridSearchCV(lasso, lasso_parameters, scoring = 'neg_mean_squared_error',cv=10)
lasso_regressor.fit(wm_activity, wm_total['ACC'])
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

task_lasso_alpha = lasso_regressor.best_params_
task_lasso_rmse = lasso_regressor.best_score_

lasso = Lasso()
lasso_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
lasso_regressor = GridSearchCV(lasso, lasso_parameters, scoring = 'r2',cv=10)
lasso_regressor.fit(wm_activity, wm_total['ACC'])
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

task_lasso_r2 = lasso_regressor.best_score_

'''Predicting WM with lasso regression on rest activity'''

lasso = Lasso()
#Make a dictionary of alphas
lasso_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
lasso_regressor = GridSearchCV(lasso, lasso_parameters, scoring = 'neg_mean_squared_error',cv=10)
lasso_regressor.fit(rest_activity, wm_total['ACC'])
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

rest_lasso_alpha = lasso_regressor.best_params_
rest_lasso_rmse = lasso_regressor.best_score_

lasso = Lasso()
lasso_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
lasso_regressor = GridSearchCV(lasso, lasso_parameters, scoring = 'r2',cv=10)
lasso_regressor.fit(rest_activity, wm_total['ACC'])
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

rest_lasso_r2 = lasso_regressor.best_score_

'''Predicting WM with elastic net regression on task activity'''
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
#Make a dictionary of alphas
elasticnet_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
elasticnet_regressor = GridSearchCV(elasticnet, elasticnet_parameters, scoring = 'neg_mean_squared_error',cv=10)
elasticnet_regressor.fit(wm_activity, wm_total['ACC'])
print(elasticnet_regressor.best_params_)
print(elasticnet_regressor.best_score_)

task_elasticnet_alpha = elasticnet_regressor.best_params_
task_elasticnet_rmse = elasticnet_regressor.best_score_

elasticnet = ElasticNet()
#Make a dictionary of alphas
elasticnet_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
elasticnet_regressor = GridSearchCV(elasticnet, elasticnet_parameters, scoring = 'r2',cv=10)
elasticnet_regressor.fit(wm_activity, wm_total['ACC'])
print(elasticnet_regressor.best_params_)
print(elasticnet_regressor.best_score_)

task_elasticnet_r2 = elasticnet_regressor.best_score_

'''Predicting WM with elastic net regression on rest activity'''
elasticnet = ElasticNet()
#Make a dictionary of alphas
elasticnet_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
elasticnet_regressor = GridSearchCV(elasticnet, elasticnet_parameters, scoring = 'neg_mean_squared_error',cv=10)
elasticnet_regressor.fit(rest_activity, wm_total['ACC'])
print(elasticnet_regressor.best_params_)
print(elasticnet_regressor.best_score_)

rest_elasticnet_alpha = elasticnet_regressor.best_params_
rest_elasticnet_rmse = elasticnet_regressor.best_score_


elasticnet = ElasticNet()
#Make a dictionary of alphas
elasticnet_parameters = {'alpha': [1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 100]}
elasticnet_regressor = GridSearchCV(elasticnet, elasticnet_parameters, scoring = 'r2',cv=10)
elasticnet_regressor.fit(rest_activity, wm_total['ACC'])
print(elasticnet_regressor.best_params_)
print(elasticnet_regressor.best_score_)

rest_elasticnet_r2 = elasticnet_regressor.best_score_

from sklearn.model_selection import train_test_split
task_train, task_test, task_accuracy_train, task_accuracy_test = train_test_split(
    wm_activity, wm_total['ACC'], test_size=0.3, random_state=0)
prediction_ridge = ridge_regressor.predict(task_test)


plt.pyplot.scatter(prediction_ridge, task_accuracy_test)
np.corrcoef(prediction_ridge, task_accuracy_test)

rest_train, rest_test, rest_accuracy_train, rest_accuracy_test = train_test_split(
    rest_activity, wm_total['ACC'], test_size=0.3, random_state=0)
prediction_ridge = ridge_regressor.predict(task_test)



import seaborn as sns
sns.distplot(task_accuracy_test-prediction_ridge)
plt.pyplot.scatter(prediction_ridge, task_accuracy_test)

    
    
    

