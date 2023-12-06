import pandas as pd
import json
import os 


# Declare paths
output_path = '/data/hx-hx1/kbaacke/datasets/UCLA_Decoding/'
comparison_path = f'{output_path}comparison/' # Sheets with subject ID, predicted y, actual y
confusion_path = f'{output_path}confusion/' # Sheets with confusion matrices
metadata_path = f'{output_path}metadata/' # metadata json files with information about the models
probs_path = f'{output_path}probs/' # probabilities of each of the predictions? not totally sure

# Read in accuracies 
accuracy_df = pd.read_csv(f'{output_path}Full_Diag_Accuracy_V2.csv')
accuracy_df.drop_duplicates(subset='runuid', inplace=True, keep='last')

# Read in confusion matrices and convert them to column format
confusion_dfs = []
# iterate through all model runs


# Read in meta-info about the models
metadata_dfs = []
# iterate through all model runs
for runuid in accuracy_df['runuid'].unique():
  # Read in the json files as dictionaries
  run_metadata_dict = json.load(open(f'{metadata_path}{runuid}_metadict.json'))
  # Convert the dictionaries to dataframes
  run_metadata_df = pd.DataFrame(run_metadata_dict, index=[0])
  # add a column to indicate the runuid
  run_metadata_df['runuid'] = runuid
  # add the run_dataframe to the list of dataframes
  metadata_dfs.append(run_metadata_df)

# Concatenate the list of metadata dataframes
metadata_df = pd.concat(metadata_dfs, ignore_index=True)
# Create a new column to indicate whether control subjects were included in the analysis
metadata_df['subjects_used'] = ''

accuracy_df['Task'].fillna('all', inplace=True)
accuracy_df['sessions_used'].fillna(accuracy_df['Task'], inplace=True)
accuracy_df['subjects_used'].fillna('all', inplace=True)

# Read in record of all test splits
test_split_dict = {
  'No_Control':{},
  'All_Subjects':{}
}
for k in test_split_dict:
  for task in accuracy_df['sessions_used'].unique():
    test_split_dict[k][task] = {}

for task in accuracy_df['sessions_used'].unique():
  for split_ind in range(10):
    if task=='all':
      test_split_dict['No_Control'][task][split_ind] = pd.read_csv(f'{output_path}NC_SplitInfo_Test_{split_ind}.csv')
      test_split_dict['All_Subjects'][task][split_ind] = pd.read_csv(f'{output_path}SplitInfo_Test_{split_ind}.csv')
    else:
      test_split_dict['No_Control'][task][split_ind] = pd.read_csv(f'{output_path}{task}_NC_SplitInfo_Test_{split_ind}.csv')
      test_split_dict['All_Subjects'][task][split_ind] = pd.read_csv(f'{output_path}{task}_SplitInfo_Test_{split_ind}.csv')

# Read in predictions on the subject level
diagnoses = ['CONTROL', 'SCHZ', 'BIPOLAR', 'ADHD']
prob_dfs = []
# iterate through all model runs
for runuid in accuracy_df['runuid']:
  # Read in prediction files
  run_probs = pd.read_csv(f'{probs_path}{runuid}_probs.csv', header=None)
  if len(run_probs.columns)==3:
    run_probs.columns = ['ADHD', 'BIPOLAR', 'SCHZ']
  elif len(run_probs.columns)==4:
    run_probs.columns = ['ADHD', 'BIPOLAR', 'CONTROL', 'SCHZ']
  else:
    print('Something has gone terribly wrong!')
  # add a column to indicate the runuid
  run_probs['runuid'] = runuid
  # Identify the split number from the metadata_df
  split_number = list(metadata_df.loc[metadata_df['runuid']==runuid]['split'])[0]
  subjects_used = list(accuracy_df.loc[accuracy_df['runuid']==runuid]['subjects_used'])[0]
  sessions_used = list(accuracy_df.loc[accuracy_df['runuid']==runuid]['sessions_used'])[0]
  if subjects_used=='all':
    subjects_used_key = 'All_Subjects'
  else:
    subjects_used_key = 'No_Control'
  target_split_df = test_split_dict[subjects_used_key][sessions_used][split_number]
  metadata_df.loc[metadata_df['runuid']==runuid, 'subjects_used'] = subjects_used
  metadata_df.loc[metadata_df['runuid']==runuid, 'sessions_used'] = sessions_used
  # print('rp1',len(run_predictions))
  # print('sp_df',len(target_split_df))
  run_probs['Task'] = target_split_df['Task']
  run_probs['Participant_ID'] = target_split_df['participant_id']
  # print('rp2',len(run_predictions))
  prob_dfs.append(run_probs)

probs_df = pd.concat(prob_dfs, ignore_index=True)
probs_df.to_csv(f'{output_path}probabilities.csv',index=False)