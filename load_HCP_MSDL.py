#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:39:30 2021

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

#####################################
#### Setting starting variables #####
#####################################

# Import the total list of subjects to loop through
subject_list = os.listdir('/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/')
# on Mac this command returns a ds.store file, this will delete
del subject_list[0]

# Store the phase direction objects so that the files can loop over and append later
PHASE_DIRECTIONS = ['LR', 'RL']

# Store the names of the task for iteration
TASK_NAMES =  ["tfMRI_MOTOR", 
              "tfMRI_WM", 
              "tfMRI_EMOTION",
              "tfMRI_GAMBLING", 
              "tfMRI_LANGUAGE", 
              "tfMRI_RELATIONAL", 
              "tfMRI_SOCIAL"]

# Import the MSDL atlas 
atlas_MSDL = nilearn.datasets.fetch_atlas_msdl()
regions = atlas_MSDL['labels']
networks = atlas_MSDL['networks']

################################
#### Making the input data #####
################################

motor_data_list_LR = {}
wm_data_list_LR = {}
emotion_data_list_LR = {}
gambling_data_list_LR = {}
language_data_list_LR = {}
relational_data_list_LR = {}
social_data_list_LR = {}
tasks_missing = {}
for subject in subject_list:
    for task in TASK_NAMES:
        if task == "tfMRI_MOTOR":
            try:
                motor_data_list_LR[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_LR/{task}_LR.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_WM":
            try:
                wm_data_list_LR[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_LR/{task}_LR.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_EMOTION":
            try:
                emotion_data_list_LR[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_LR/{task}_LR.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_GAMBLING":
            try:
                gambling_data_list_LR[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_LR/{task}_LR.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_LANGUAGE":
            try:
                language_data_list_LR[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_LR/{task}_LR.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_RELATIONAL":
            try:
                relational_data_list_LR[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_LR/{task}_LR.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_SOCIAL":
            try:
                social_data_list_LR[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_LR/{task}_LR.npy')
            except:
                tasks_missing[subject] = str(task)       


motor_data_list_RL = {}
wm_data_list_RL = {}
emotion_data_list_RL = {}
gambling_data_list_RL = {}
language_data_list_RL = {}
relational_data_list_RL = {}
social_data_list_RL = {}
tasks_missing = {}
for subject in subject_list:
    for task in TASK_NAMES:
        if task == "tfMRI_MOTOR":
            try:
                motor_data_list_RL[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_RL/{task}_RL.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_WM":
            try:
                wm_data_list_RL[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_RL/{task}_RL.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_EMOTION":
            try:
                emotion_data_list_RL[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_RL/{task}_RL.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_GAMBLING":
            try:
                gambling_data_list_RL[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_RL/{task}_RL.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_LANGUAGE":
            try:
                language_data_list_RL[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_RL/{task}_RL.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_RELATIONAL":
            try:
                relational_data_list_RL[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_RL/{task}_RL.npy')
            except:
                tasks_missing[subject] = str(task)
        if task == "tfMRI_SOCIAL":
            try:
                social_data_list_RL[subject] = np.load(f'/Volumes/Byrgenwerth/Datasets/HCP 1200 MSDL Numpy/HCP_1200_NumPy/{subject}/MNINonLinear/Results/{task}_RL/{task}_RL.npy')
            except:
                tasks_missing[subject] = str(task)  

# The number of subjects in this data release
N_SUBJECTS = len(subject_list)

# The data have already been aggregated into ROIs from the Glasesr parcellation
N_PARCELS = 39

# How many networks?
N_NETWORKS = 12

##################################
#### Parcel-based input data #####
##################################

#Initialize dataframes
parcel_average_motor = np.zeros((N_SUBJECTS, N_PARCELS), dtype='float64')
parcel_average_wm = np.zeros((N_SUBJECTS, N_PARCELS))
parcel_average_gambling = np.zeros((N_SUBJECTS, N_PARCELS))
parcel_average_emotion = np.zeros((N_SUBJECTS, N_PARCELS))
parcel_average_language = np.zeros((N_SUBJECTS, N_PARCELS))
parcel_average_relational = np.zeros((N_SUBJECTS, N_PARCELS))
parcel_average_social = np.zeros((N_SUBJECTS, N_PARCELS))

#calculate average for each parcel in each task
for subject, ts in enumerate(motor_data_list_RL.values()):
    parcel_average_motor[subject] = np.mean(ts.T, axis = 1)
for subject, ts in enumerate(wm_data_list_RL.values()):
    parcel_average_wm[subject] = np.mean(ts.T, axis = 1)
for subject, ts in enumerate(gambling_data_list_RL.values()):
    parcel_average_gambling[subject] = np.mean(ts.T, axis = 1)  
for subject, ts in enumerate(emotion_data_list_RL.values()):
    parcel_average_emotion[subject] = np.mean(ts.T, axis = 1)  
for subject, ts in enumerate(language_data_list_RL.values()):
    parcel_average_language[subject] = np.mean(ts.T, axis = 1)  
for subject, ts in enumerate(relational_data_list_RL.values()):
    parcel_average_relational[subject] = np.mean(ts.T, axis = 1)  
for subject, ts in enumerate(social_data_list_RL.values()):
    parcel_average_social[subject] = np.mean(ts.T, axis = 1)  
    
#Make parcel dataframes
motor_parcels = pd.DataFrame(parcel_average_motor, columns = networks)
wm_parcels = pd.DataFrame(parcel_average_wm, columns = networks)
gambling_parcels = pd.DataFrame(parcel_average_gambling, columns = networks)
emotion_parcels = pd.DataFrame(parcel_average_emotion, columns = networks)
language_parcels = pd.DataFrame(parcel_average_language, columns = networks)
relational_parcels = pd.DataFrame(parcel_average_relational, columns = networks)
social_parcels = pd.DataFrame(parcel_average_social, columns = networks)

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

#Make FC matrices for each subject for each task
fc_matrix_motor = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_wm = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_gambling = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_emotion = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_language = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_relational = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_social = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))

# Calculate the correlations (FC) for each task
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


# Initialize the vector form of each task, where each row is a participant and each column is a connection
vector_motor = np.zeros((N_SUBJECTS, 741))
vector_wm = np.zeros((N_SUBJECTS, 741))
vector_gambling = np.zeros((N_SUBJECTS, 741))
vector_emotion = np.zeros((N_SUBJECTS, 741))
vector_language = np.zeros((N_SUBJECTS, 741))
vector_relational = np.zeros((N_SUBJECTS, 741))
vector_social = np.zeros((N_SUBJECTS, 741))

# Extract the diagonal of the FC matrix for each subject for each task
subject_list = np.array(np.unique(range(len(subject_list))))
for subject in range(subject_list.shape[0]):
    vector_motor[subject,:] = sym_matrix_to_vec(fc_matrix_motor[subject,:,:], discard_diagonal=True)
    #vector_motor[subject,:] = fc_matrix_motor[subject][np.triu_indices_from(fc_matrix_motor[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_wm[subject,:] = sym_matrix_to_vec(fc_matrix_wm[subject,:,:], discard_diagonal=True)
    #vector_wm[subject,:] = fc_matrix_wm[subject][np.triu_indices_from(fc_matrix_wm[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_gambling[subject,:] = sym_matrix_to_vec(fc_matrix_gambling[subject,:,:], discard_diagonal=True)
    #vector_gambling[subject,:] = fc_matrix_gambling[subject][np.triu_indices_from(fc_matrix_gambling[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_emotion[subject,:] = sym_matrix_to_vec(fc_matrix_emotion[subject,:,:], discard_diagonal=True)
    #vector_emotion[subject,:] = fc_matrix_emotion[subject][np.triu_indices_from(fc_matrix_emotion[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_language[subject,:] = sym_matrix_to_vec(fc_matrix_language[subject,:,:], discard_diagonal=True)
    #vector_language[subject,:] = fc_matrix_language[subject][np.triu_indices_from(fc_matrix_language[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_relational[subject,:] = sym_matrix_to_vec(fc_matrix_relational[subject,:,:], discard_diagonal=True)
    #vector_relational[subject,:] = fc_matrix_relational[subject][np.triu_indices_from(fc_matrix_relational[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_social[subject,:] = sym_matrix_to_vec(fc_matrix_social[subject,:,:], discard_diagonal=True)
    #vector_social[subject,:] = fc_matrix_social[subject][np.triu_indices_from(fc_matrix_social[subject], k=1)]

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

####################################
#### Network-based input data  #####
####################################

#Attach the labels to the parcels 
X_network = pd.DataFrame(X_parcels, columns = networks)

#Add the columns of the same network together and then scale them normally
X_network = X_network.groupby(lambda x:x, axis=1).sum()

#Make y vector
y_network = parcels_full.iloc[:,-1]


###############################################
#### Network-connection based input data  #####
###############################################

#Get the number of time points for each task
TIMEPOINTS_MOTOR = next(iter(motor_data_list_LR.values())).shape[0]
TIMEPOINTS_WM = next(iter(wm_data_list_LR.values())).shape[0]
TIMEPOINTS_GAMBLING = next(iter(gambling_data_list_LR.values())).shape[0]
TIMEPOINTS_EMOTION = next(iter(emotion_data_list_LR.values())).shape[0]
TIMEPOINTS_LANGUAGE = next(iter(language_data_list_LR.values())).shape[0]
TIMEPOINTS_RELATIONAL = next(iter(relational_data_list_LR.values())).shape[0]
TIMEPOINTS_SOCIAL = next(iter(social_data_list_LR.values())).shape[0]

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
for subject, ts in enumerate(motor_data_list_RL.values()):
    parcel_transpose_motor[subject] = ts
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

# Delete variables to save memory
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


elapsed_time = time.time() - start_time
print(f"Elapsed time to preprocess input data: "
      f"{elapsed_time:.3f} seconds")

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

#Network connection data
train_X_netcon, test_X_netcon, train_y_netcon, test_y_netcon = train_test_split(X_network_connections, y_network_connections, test_size = 0.2)

########################################
###### Support Vector Classifier #######
########################################

# Parcels
lin_clf = svm.LinearSVC(C=1e-5)
lin_clf.fit(train_X_parcels, train_y_parcels)
print(lin_clf.score(train_X_parcels, train_y_parcels))
print(lin_clf.score(test_X_parcels, test_y_parcels))
#svm_coef = pd.DataFrame(lin_clf.coef_.T)

# Parcel connections
lin_clf = svm.LinearSVC()
lin_clf.fit(train_X_parcon, train_y_parcon)
print(lin_clf.score(train_X_parcon, train_y_parcon))
print(lin_clf.score(test_X_parcon, test_y_parcon))
#svm_coef = pd.DataFrame(lin_clf.coef_.T)

# Network summations
lin_clf = svm.LinearSVC(C=1e-1)
lin_clf.fit(train_X_network, train_y_network)
print(lin_clf.score(train_X_network, train_y_network))
print(lin_clf.score(test_X_network, test_y_network))
#svm_coef = pd.DataFrame(lin_clf.coef_.T)

# Network connections
lin_clf = svm.LinearSVC()
lin_clf.fit(train_X_netcon, train_y_netcon)
print(lin_clf.score(train_X_netcon, train_y_netcon))
print(lin_clf.score(test_X_netcon, test_y_netcon))
#svm_coef = pd.DataFrame(lin_clf.coef_.T)


#######################################
###### Random Forest Classifier #######
#######################################

##### Parcels #####
forest = RandomForestClassifier(random_state=1, n_estimators=1000)
forest.fit(train_X_parcels, train_y_parcels)
pred_y_parcels = forest.predict(test_X_parcels)
# How does it perform?
print(forest.score(train_X_parcels, train_y_parcels))
print(forest.score(test_X_parcels, test_y_parcels))

# Visualize the confusion matrix
from sklearn.metrics import classification_report
#print(classification_report(test_X_parcels, test_y_parcels))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y_parcels, pred_y_parcels)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)
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
forest = RandomForestClassifier(random_state=1, n_estimators=5000)
forest.fit(train_X_parcon, train_y_parcon)
pred_y_parcon = np.array(forest.predict(test_X_parcon).astype(int))
# How does it perform?
print(forest.score(train_X_parcon, train_y_parcon))
print(forest.score(test_X_parcon, test_y_parcon))

# Visualize the confusion matrix
from sklearn.metrics import classification_report
#print(classification_report(np.array(test_X_parcon), np.array(test_y_parcon).astype(int)))
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
forest = RandomForestClassifier(random_state=1, n_estimators=5000)
forest.fit(train_X_network, train_y_network)
pred_y_network = forest.predict(test_X_network)
# How does it perform?
print(forest.score(train_X_network, train_y_network))
print(forest.score(test_X_network, test_y_network))

# Visualize the confusion matrix
from sklearn.metrics import classification_report
#print(classification_report(test_X_network, test_y_network))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y_network, pred_y_network)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)
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

##### Network summations #####
forest = RandomForestClassifier(random_state=1, n_estimators=1000)
forest.fit(train_X_netcon, train_y_netcon)
pred_y_network = forest.predict(test_X_netcon)
# How does it perform?
print(forest.score(train_X_netcon, train_y_netcon))
print(forest.score(test_X_netcon, test_y_netcon))

# Visualize the confusion matrix
from sklearn.metrics import classification_report
#print(classification_report(test_X_netcon, test_y_netcon))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y_netcon, pred_y_netcon)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)
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