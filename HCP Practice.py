#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:20:56 2020

@author: cjrichier
"""

'''This is a practice analysis with the Human Connectome Project'''

#Load the needed libraries
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Necessary for visualization
from nilearn import plotting, datasets
# Surface plot with BrainSpace
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.utils.parcellation import map_to_labels
from brainspace.mesh import mesh_cluster
from brainspace.gradient import GradientMaps
import panel as pn
pn.extension('vtk')
import csv
import urllib.request as urllib2
# matplotlib
import matplotlib.pyplot as plt # For changing the color maps
from matplotlib import cm # cm=colormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



'''Set the path, and create some variables'''
# The download cells will store the data in nested directories starting here:
HCP_DIR = "/Volumes/Byrgenwerth/Datasets/HCP/"
HCP_DIR_REST = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_rest/"
HCP_DIR_TASK = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_task/"
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
BOLD_NAMES = [ "rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL", "tfMRI_MOTOR_RL", "tfMRI_MOTOR_LR","tfMRI_WM_RL", "tfMRI_WM_LR", "tfMRI_EMOTION_RL", "tfMRI_EMOTION_LR", "tfMRI_GAMBLING_RL", "tfMRI_GAMBLING_LR", "tfMRI_LANGUAGE_RL", "tfMRI_LANGUAGE_LR", "tfMRI_RELATIONAL_RL", "tfMRI_RELATIONAL_LR", "tfMRI_SOCIAL_RL", "tfMRI_SOCIAL_LR"]
# This will use all subjects:
subjects = range(N_SUBJECTS)
# You may want to limit the subjects used during code development.
test_subjects = 5


'''create a list of the subject ID's'''
subj_list = np.array(np.unique(wm_behavior['Subject']))


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

def load_timeseries(subject, name, runs=None, concat=True, remove_mean=True):
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
      load_single_timeseries(subject, offset + run, remove_mean) for run in runs
  ]

  # Optionally concatenate in time
  if concat:
    bold_data = np.concatenate(bold_data, axis=-1)

  return bold_data


def load_single_timeseries(subject, bold_run, remove_mean=True):
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
    ev_file = f"{HCP_DIR}/subjects/{subject}/EVs/{task_key}/{condition}.txt"
    ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
    ev = dict(zip(["onset", "duration", "amplitude"], ev_array))
    evs.append(ev)
  return evs

'''load the time series for all participants'''
timeseries_rest = []
for subject in subjects:
  ts_concat = load_timeseries(subject, "rest")
  timeseries_rest.append(ts_concat)

'''calculate the functional connecitvity matrix across all 
all participants'''
fc = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))

'''What this code will iterate over the subject dimension (the first) 
and the time series dimension (the second) and then calculate the correlation coefficient
for every subject's time series'''
for sub, ts in enumerate(timeseries_rest):
  fc[sub] = np.corrcoef(ts)
'''as a result, we get an object called fc, which is dimensions 339x360x360. 
This represents the fc matrix for all 339 subjects'''

'''Now with that object, we can plot group level FC.
We will take the mean across subjects in fc, which is a 3 dimensional object.
doing this in the first dimension takes the average across participants'''
group_fc = fc.mean(axis=0)
#Now let's plot the average FC across all participants
plt.imshow(group_fc, interpolation="none", cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.show()



'''We need to vectorize the matrices so they play nicer with data in later analyses'''
from nilearn.connectome import sym_matrix_to_vec
#all_fc_data = {}
all_fc_data = np.zeros((339, 64620))

prova = sym_matrix_to_vec(fc[0], discard_diagonal=True)
for subject in range(subj_list.shape[0]):
    #all_fc_data[subject,:] = sym_matrix_to_vec(fc[subject,:,:], discard_diagonal=True)
    all_fc_data[subject,:] = fc[subject][np.triu_indices_from(fc[subject], k=1)]

all_fc_data = pd.DataFrame(data=all_fc_data, index=subj_list)
all_fc_data.head()

s = 1
plt.imshow(sp.spatial.distance.squareform(all_fc_data.iloc[s,:]), interpolation="none", cmap="bwr", vmin=-1, vmax=1)
plt.colorbar()
plt.show()








'''Now let's try something a little different. Why don't we see
how one particular parcel (of the 360) relates to the total cconnectivity
in the rest of the brain, split by right and left hemispheres.'''

'''Let's take a look at some of the parcel's names first. Let's pick 
a random one from the list to use here.'''
print(region_info)
seed_roi = "R_V1"  # name of seed parcel
ind = region_info["name"].index(seed_roi)
hemi_fc = np.split(group_fc, 2)
# Plot the FC profile across the right and left hemisphere target regions
for i, hemi_fc in enumerate(hemi_fc):
  plt.plot(hemi_fc[:, ind], label=f"{HEMIS[i]} hemisphere")
plt.title(f"FC for region {seed_roi}")
plt.xlabel("Target region")
plt.ylabel("Correlation (FC)")
plt.legend()
plt.show()

'''Now let's extract the time series for one subject'''
ts_sub0 = load_timeseries(0, "rest")

'''let's look at the functional connectivity for just this participant'''
ts_sub0_fc = np.zeros((N_PARCELS, N_PARCELS))
for parcel in enumerate(timeseries_rest):
  ts_sub0_fc = np.corrcoef(ts)
plt.imshow(ts_sub0_fc, interpolation="none", cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.show()

'''Now let's extract the time series for one parcel in one subject'''
seed_ts_df = pd.DataFrame(ts_sub0 [0,:])
seed_ts_df.plot()
plt.title('Time series for a the first subject')
plt.xlabel('Time')
plt.ylabel('Normalized signal')
plt.tight_layout()
plt.show()
'''You can see how this is a lot of data for all 
participants at all timepoints!'''









####################################
 ####### Taks Data Analysis #######
####################################

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

'''Now let's load some wm task data'''
'''I cannot figure out why this is happening, but in order for task analyses to
run we have to cahnge the directory to be in the task section, or else it will
get mad at us.'''

HCP_DIR = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_task/"
'''REMEMBER TO CHANGE THIS!!!'''

'''Let's load the working memory data'''
timeseries_wm = []
for subject in subjects:
  timeseries_wm.append(load_timeseries(subject, "wm", concat=True))
print(timeseries_wm)

'''Let's calculate FC for the WM activity'''
fc_wm = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))

'''What this code will iterate over the subject dimension (the first) 
and the time series dimension (the second) and then calculate the correlation coefficient
for every subject's time series'''
for sub, ts in enumerate(timeseries_wm):
  fc_wm[sub] = np.corrcoef(ts)
'''as a result, we get an object called fc, which is dimensions 339x360x360. 
This represents the working memory fc matrix for all 339 subjects'''

'''Now let's average this across all participants'''
group_fc_wm = fc_wm.mean(axis=0)
#Now let's plot the average FC across all participants
plt.imshow(group_fc_wm, interpolation="none", cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.show()




##########################################
 ####### Behavioral Data Analysis #######
##########################################

'''Now let's load the behavioral data'''
wm_behavior = np.genfromtxt(f"{HCP_DIR_BEHAVIOR}/behavior/wm.csv",
                            delimiter=",",
                            names=True,
                            dtype=None,
                            encoding="utf")
print(wm_behavior[:5])
print(wm_behavior.dtype.names)
'''Looks good. Let's make it a pandas object, because that is easier to work with.'''
wm_behavior = pd.DataFrame(wm_behavior)
print(wm_behavior)
'''Again, looks good. What are the variables we are working with here? We can do
this by iterating a command to print all the names of the variables (in this case
they are columns)'''
for variable in wm_behavior.columns: 
    print(variable) 

'''Let's do some more data exploration. What are the names of the conditions 
in the working memory task?'''
print(np.unique(wm_behavior["ConditionName"]))

'''let's now find each subject's total accuracy 
for each condition for each subject'''
wm_organized = wm_behavior.groupby(['Subject', 'ConditionName'])['ACC'].mean().reset_index()
print(wm_organized)


'''doing some more data wrangling. Let's make objects for each condition.'''
wm_0bk_body = wm_organized.loc[wm_organized['ConditionName'] == '0BK_BODY']
wm_0bk_face = wm_organized.loc[wm_organized['ConditionName'] == '0BK_FACE']
wm_0bk_place = wm_organized.loc[wm_organized['ConditionName'] == '0BK_PLACE']
wm_0bk_tool = wm_organized.loc[wm_organized['ConditionName'] == '0BK_TOOL']
wm_2bk_body = wm_organized.loc[wm_organized['ConditionName'] == '2BK_BODY']
wm_2bk_face = wm_organized.loc[wm_organized['ConditionName'] == '2BK_FACE']
wm_2bk_place = wm_organized.loc[wm_organized['ConditionName'] == '2BK_PLACE']
wm_2bk_tool = wm_organized.loc[wm_organized['ConditionName'] == '2BK_TOOL']

'''Now that 


'''How correlated are the timeseries of the rest and the wm activity?'''
'''first we have to flatten the matricies into vectors'''
group_fc_vector = group_fc.flatten()
group_fc_wm_vector = group_fc_wm.flatten()
np.corrcoef(group_fc_vector, group_fc_wm_vector)
'''we get an r of 0.86063865, so the FC matrices of these are pretty related.'''


# Get unique network labels
network_names = np.unique(region_info["network"])
print(network_names)



