#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:14:43 2020

@author: cjrichier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 13:04:52 2020

@author: cjrichier
"""

###HCP task data analysis###

#Load the needed libraries
import os
import time
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import nilearn 
# Necessary for visualization
# matplotlib
from matplotlib import cm # cm=colormap

start_time = time.time()

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


'''import the demographics and bheavior data'''
demographics = pd.read_csv('/Volumes/Byrgenwerth/Datasets/HCP/HCP_demographics/demographics_behavior.csv')
#What is our gender breakdown?
demographics['Gender'].value_counts()
demographics['Age'].value_counts()



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


################################
#### Making the input data #####
################################


'''load in the timeseries for each task'''
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

'''calculate average for each parcel in each task'''
parcel_average_motor = np.zeros((N_SUBJECTS, N_PARCELS), dtype='float64')
parcel_average_wm = np.zeros((N_SUBJECTS, N_PARCELS))
parcel_average_gambling = np.zeros((N_SUBJECTS, N_PARCELS))
parcel_average_emotion = np.zeros((N_SUBJECTS, N_PARCELS))
parcel_average_language = np.zeros((N_SUBJECTS, N_PARCELS))
parcel_average_relational = np.zeros((N_SUBJECTS, N_PARCELS))
parcel_average_social = np.zeros((N_SUBJECTS, N_PARCELS))

for subject, ts in enumerate(timeseries_motor):
    parcel_average_motor[subject] = np.mean(ts, axis=1)
for subject, ts in enumerate(timeseries_wm):
    parcel_average_wm[subject] = np.mean(ts, axis=1)
for subject, ts in enumerate(timeseries_gambling):
    parcel_average_gambling[subject] = np.mean(ts, axis=1)
for subject, ts in enumerate(timeseries_emotion):
    parcel_average_emotion[subject] = np.mean(ts, axis=1)
for subject, ts in enumerate(timeseries_language):
    parcel_average_language[subject] = np.mean(ts, axis=1)
for subject, ts in enumerate(timeseries_relational):
    parcel_average_relational[subject] = np.mean(ts, axis=1)
for subject, ts in enumerate(timeseries_social):
    parcel_average_social[subject] = np.mean(ts, axis=1)    

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

motor_parcels = pd.DataFrame(parcel_average_motor)
wm_parcels = pd.DataFrame(parcel_average_wm)
gambling_parcels = pd.DataFrame(parcel_average_gambling)
emotion_parcels = pd.DataFrame(parcel_average_emotion)
language_parcels = pd.DataFrame(parcel_average_language)
relational_parcels = pd.DataFrame(parcel_average_relational)
social_parcels = pd.DataFrame(parcel_average_social)

'''Delete the old vectors to save space'''
del vector_motor 
del vector_wm 
del vector_gambling 
del vector_emotion 
del vector_language 
del vector_relational 
del vector_social 


'''make our prediction dataset'''
emotion_brain['task'] = 1
gambling_brain['task'] = 2
language_brain['task'] = 3
motor_brain['task'] = 4
relational_brain['task'] = 5
social_brain['task'] = 6
wm_brain['task'] = 7

emotion_parcels['task'] = 1
gambling_parcels['task'] = 2
language_parcels['task'] = 3
motor_parcels['task'] = 4
relational_parcels['task'] = 5
social_parcels['task'] = 6
wm_parcels['task'] = 7

#make the data frames

    
task_data = pd.DataFrame(np.concatenate((emotion_brain, gambling_brain,  language_brain,
          motor_brain, relational_brain, social_brain, wm_brain), axis = 0))
X = task_data.iloc[:, :-1]
y = task_data.iloc[:,-1]


'''make more space'''
del emotion_brain
del gambling_brain
del language_brain
del motor_brain
del relational_brain
del social_brain
del wm_brain                 
        






########################################
###### Support Vector Classifier #######
########################################


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

'''make test-train split'''
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)


lin_clf = svm.LinearSVC()
lin_clf.fit(train_X, train_y)
print(lin_clf.score(train_X, train_y))
print(lin_clf.score(test_X, test_y))
svm_coef = pd.DataFrame(lin_clf.coef_.T)


####################################
##### Random Forest Classifier #####
####################################

'''Now let's try the decoding analysis'''
'''Analysis time'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.utils import shuffle



#fit the model
forest = RandomForestClassifier(random_state=1 ,n_estimators=1000, n_jobs=4)
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
###### Feature Importances #######
##################################

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


######## SVC Importances #############
#Make a dataframe with task coefficients and labels for SVC

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
result = permutation_importance(
    forest, test_X, test_y, n_repeats=5, random_state=1, n_jobs=6)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: "
      f"{elapsed_time:.3f} seconds")


Permutation_forest_importances = pd.DataFrame(result.importances_mean, feature_names)
from numpy import savetxt
forest_importances.to_csv(f'/Users/cjrichier/Documents/Github/Deep-Learning-Brain-Age/forest_importances.csv')

names1 = ['0','1','2','3','4','5','6']


out1 = vector_names(names1, [])
print(out1)

forest_importances_series = np.squeeze(np.array(forest_importances))



#Now that we have the feature importances, let's organize them all into a separate dataframe
Permutation_features_full = pd.DataFrame(np.array((forest_importances_series,list_of_connections, list_of_networks)).T)
from numpy import savetxt
forest_importances.to_csv(f'/Users/cjrichier/Documents/Github/HCP-Analyses/forest_importances.csv')

Permutation_features_full.to_csv(f'/Users/cjrichier/Documents/Github/HCP-Analyses/Permutation_features_full.csv')
Permutation_features_full.columns = ['Importance Value', 'Regions', 'Network Connection']
Permutation_features_full_sorted = Permutation_features_full.sort_values(by='Importance Value', ascending=False)

Rank_order_features = Permutation_features_full_sorted.index
Rank_order_features = list(Rank_order_features)
Non_zero_features = Rank_order_features[:561]
Non_zero_features = str(Non_zero_features)

Permutation_features_full_sorted = Permutation_features_full_sorted.reset_index()
Permutation_features_full_sorted.drop('index', axis=1, inplace=True)



list_of_connection_counts = features_full.iloc[:,2].value_counts()

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
    for n in names[1:]:
        







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
forest = RandomForestClassifier(random_state=1 ,n_estimators=10)
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
    
parcels_full =  pd.DataFrame(np.concatenate((emotion_parcels, gambling_parcels, language_parcels,
                                             motor_parcels, relational_parcels, social_parcels, wm_parcels), axis = 0))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X = parcels_full.iloc[:, :-1]
X = scaler.fit_transform(X) 
y = parcels_full.iloc[:,-1]


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










