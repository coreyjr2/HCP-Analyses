import pandas as pd
import boto3
import os
import datetime as dt
sep = os.path.sep
# Read user credentials if not the default in AWS credential manager
cred = pd.read_csv(''+sep+'Users'+sep+'kyle'+sep+'repos'+sep+'HCP-Task-Classification-01'+sep+'misc'+sep+'kp.csv') 
# Import the full list of HCP cubjects
subjects = pd.read_csv(''+sep+'Users'+sep+'kyle'+sep+'repos'+sep+'HCP-Task-Classification-01'+sep+'subject_list.csv')['ID']

test_subjects = subjects[0:]
base_dir = 'S:'+sep+'HCP'+sep+''

files_to_download = [ #Specify target files

    # 'HCP_1200/{subject}/MNINonLinear/Results/{task}_LR/Movement_AbsoluteRMS.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_LR/Movement_AbsoluteRMS_mean.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_LR/Movement_Regressors.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_LR/Movement_Regressors_dt.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_LR/Movement_RelativeRMS.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_LR/Movement_RelativeRMS_mean.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_LR/{task}_LR_Physio_log.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_RL/Movement_AbsoluteRMS.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_RL/Movement_AbsoluteRMS_mean.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_RL/Movement_Regressors.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_RL/Movement_Regressors_dt.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_RL/Movement_RelativeRMS.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_RL/Movement_RelativeRMS_mean.txt'
    # ,'HCP_1200/{subject}/MNINonLinear/Results/{task}_RL/{task}_RL_Physio_log.txt'
    ]

necessary_directories = [

]

task_list = [
  'tfMRI_MOTOR',
  'rfMRI_REST1',
  'rfMRI_REST2',
  'tfMRI_WM',
  'tfMRI_EMOTION',
  'tfMRI_GAMBLING',
  'tfMRI_LANGUAGE',
  'tfMRI_RELATIONAL',
  'tfMRI_SOCIAL'
]

if __name__ == '__main__':
  missing_files = []
  start_time = dt.datetime.now()
  # create s3 resource object with HCP credentials
  hcp_s3_r = boto3.resource( 
    's3',
    region_name='us-east-2',
    aws_access_key_id=cred['Access key ID'][1],
    aws_secret_access_key=cred['Secret access key'][1])
  # create s3 client object with HCP credentials to search within s3
  hcp_s3_c = boto3.client(
    's3',
    region_name='us-east-2',
    aws_access_key_id=cred['Access key ID'][1],
    aws_secret_access_key=cred['Secret access key'][1])
  # Iterate through subjects 
  for s in test_subjects:
    #Create a subject specific filepath
    s_dir = base_dir +sep+'HCP_1200'+sep+ str(s)
    for dirr in necessary_directories:
      try:
        # Attempt to make subject directory
        os.mkdir(dirr.format(base_dir, s)) 
      except:
        print('subject dirr already made: ', dirr.format(base_dir, s))
    for file in files_to_download:
      for t in task_list:
        file_key = file.format(subject=s, task = t) # Format full filepaths to include the subject ID
        if file_key[-1] == '*': # Download all files in the directory
          # List all files in the directory specified
          subject_response = hcp_s3_c.list_objects_v2(
            Bucket='hcp-openaccess',
            Prefix = file_key[:-1]
            )
          keys = []
          try:
            # Format contnts as a iterable list
            for obj in subject_response['Contents']:
              keys.append(obj['Key'])
            for k in keys:
              try:
                #attempt ro doanload the file
                hcp_s3_r.meta.client.download_file('hcp-openaccess', k, str(base_dir + k)) #download file from s3
              except Exception as e:
                print(e, '/n', k) # print any excpetions 
                missing_files.append(k)
          except Exception as e:
            print(e, ': ', file_key)
        else: # When a full filepath is listed, there is no need to pull contents and iterate
          try:
            hcp_s3_r.meta.client.download_file('hcp-openaccess', file_key, str(base_dir + file_key)) #download file from s3
          except Exception as e:
            print(e, '/n', file_key) # print any excpetions 
            missing_files.append(file_key)
  end_time = dt.datetime.now()
  print('Missing files:')
  for f in missing_files:
    print(' ', f)
  print('Done')
  print('Runtime: ', str(end_time-start_time))


