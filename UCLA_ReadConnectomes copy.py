# Script to read in connectomes from \\...\\UCLA/69354adf_FunctionalConnectomes
import os
import shutil
import datetime as dt
import paramiko
from scp import SCPClient
import zipfile
import getpass
import pandas as pd
import numpy as np

hostname = 'r2.psych.uiuc.edu'


subject_list = [
  '10159','10171','10189','10193','10206','10217','10225','10227','10228',
  '10235','10249','10269','10271','10273','10274','10280','10290','10292',
  '10304','10316','10321','10325','10329','10339','10340','10345','10347',
  '10356','10361','10365','10376','10377','10388','10429','10438','10440',
  '10448','10455','10460','10471','10478','10487','10492','10506','10517',
  '10523','10524','10525','10527','10530','10557','10565','10570','10575',
  '10624','10629','10631','10638','10668','10672','10674','10678','10680',
  '10686','10692','10696','10697','10704','10707','10708','10719','10724',
  '10746','10762','10779','10785','10788','10844','10855','10871','10877',
  '10882','10891','10893','10912','10934','10940','10948','10949','10958',
  '10963','10968','10975','10977','10987','10998','11019','11030','11044',
  '11050','11052','11059','11061','11062','11066','11067','11068','11077',
  '11082','11088','11090','11097','11098','11104','11105','11106','11108',
  '11112','11122','11128','11131','11142','11143','11149','11156','50004',
  '50005','50006','50007','50008','50010','50013','50014','50015','50016',
  '50020','50021','50022','50023','50025','50027','50029','50032','50033',
  '50034','50035','50036','50038','50043','50047','50048','50049','50050',
  '50051','50052','50053','50056','50058','50059','50060','50061','50064',
  '50066','50067','50069','50073','50075','50076','50077','50080','50081',
  '50083','50085','60001','60005','60006','60008','60010','60011','60012',
  '60014','60015','60017','60020','60021','60022','60028','60030','60033',
  '60036','60037','60038','60042','60043','60045','60046','60048','60049',
  '60051','60052','60053','60055','60056','60057','60060','60062','60065',
  '60066','60068','60070','60072','60073','60074','60076','60077','60078',
  '60079','60080','60084','60087','60089','70001','70002','70004','70007',
  '70010','70015','70017','70020','70021','70022','70026','70029','70034',
  '70037','70040','70046','70048','70049','70051','70052','70055','70057',
  '70058','70060','70074','70075','70076','70077','70079','70080','70081',
  '70083','70086'
  ]
session_list = [
  'scap','bart','pamret','pamenc',
  'taskswitch','stopsignal','rest','bht'
  ]
output_template = '{OUTPATH}sub-{SUBJECT}_task-{SESSION}.npy'
ucla_path = '/data/hx-hx1/kbaacke/datasets/UCLA/'
parcellation_path = '/data/hx-hx1/kbaacke/Code/Parcellation/'

uid_dict = {
  '4023aba1':200,
  '5384cc6b':1000,
  '69354adf':200,
  '8a52e148':1000
}
def connection_names(corr_matrix, labels):
  name_idx = np.triu_indices_from(corr_matrix, k=1)
  out_list = []
  for i in range(len(name_idx[0])):
    out_list.append(str(labels[name_idx[0][i]]) + '|' + str(labels[name_idx[1][i]]))
  return out_list


participant_data = pd.read_csv(f'{ucla_path}UCLA-ds000030/participants.tsv', sep = "\t")

for uid in uid_dict.keys():
  print(uid)
  parcel_info = pd.read_csv(f'{parcellation_path}Schaefer2018_{uid_dict[uid]}Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
  node_names = list(parcel_info['ROI Name'])
  outpath = f'{ucla_path}{uid}_FunctionalConnectomes/'
  data_dict = {}
  # Iterate through session labels
  for session in session_list:
    # Create a new dictionary within the data dictionary to contain data for each session
    data_dict[session] = {}
    for subject in subject_list:
      # Read in data in try block to prevent errors from stopping the script due to missing data
      try:
        data_dict[session][subject] = np.load(output_template.format(OUTPATH=outpath, SUBJECT=subject, SESSION=session))
      except Exception as e:
        # This may print a lot. Don't worry, that is expected
        print(f'Error loading data from sub-{subject}_task-{session}: {e}')
  dummy_array = np.zeros((uid_dict[uid],uid_dict[uid]))
  colnames = connection_names(dummy_array, node_names)
  colnames.append('Subject')
  colnames.append('Task')
  vector_dict = {}
  for task in data_dict.keys():
    vector_dict[task] = {}
    # print(task)
    for subject in data_dict[task].keys():
      # print(subject)
      connectome = data_dict[task][subject]
      vector_dict[task][subject] = list(connectome[np.triu_indices_from(connectome, k=1)])
      vector_dict[task][subject].append(subject)
      vector_dict[task][subject].append(task)
  dfs_to_concat = []
  for task in vector_dict.keys():
    task_df = pd.DataFrame.from_dict(vector_dict[task], orient = 'index', columns=colnames)
    dfs_to_concat.append(task_df)
  full_data = pd.concat(dfs_to_concat)
  full_data['temp'] = 'sub-'
  full_data['participant_id'] = full_data['temp'] + full_data['Subject'].astype(str)
  full_data.drop(columns=['temp'], inplace=True)
  full_data = pd.merge(participant_data, full_data, how='right', on='participant_id')
  full_data.to_csv(f'{ucla_path}{uid}_FunctionalConnectomes.csv',index=False)
  np.save(f'{ucla_path}{uid}_FunctionalConnectomes.npy', np.array(full_data))
  np.save(f'{ucla_path}{uid}_FunctionalConnectomes_colnames.npy', np.array(full_data.columns))
  for session in session_list:
    print(session)
    full_data[full_data['Task']==session].to_csv(f'{ucla_path}{uid}_{session}_FunctionalConnectomes.csv',index=False)
    np.save(f'{ucla_path}{uid}_{session}_FunctionalConnectomes.npy', np.array(full_data[full_data['Task']==session]))
    np.save(f'{ucla_path}{uid}_{session}_FunctionalConnectomes_colnames.npy', np.array(full_data[full_data['Task']==session].columns))



import os
import shutil
import datetime as dt
import paramiko
from scp import SCPClient
import zipfile
import getpass
import pandas as pd
import numpy as np
session_list = [
  'scap','bart','pamret','pamenc',
  'taskswitch','stopsignal','rest','bht'
]

uid_dict = {
  '4023aba1':200,
  '69354adf':200,
  '8a52e148':1000,
  '5384cc6b':1000,
}
hostname = 'r2.psych.uiuc.edu'

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

sep = os.path.sep
remote_sep = '/'
source_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + sep
base = f'S:\\'
remote_base = '/data/hx-hx1/kbaacke/datasets/'
local_ucla_path = f'{base}UCLA{sep}'
ucla_path = f'{remote_base}UCLA/'
files_to_transfer = []
for uid in uid_dict.keys():
  files_to_transfer.append('{ucla_path}' + f'{uid}_FunctionalConnectomes.csv')
  files_to_transfer.append('{ucla_path}' + f'{uid}_FunctionalConnectomes.npy')
  files_to_transfer.append('{ucla_path}' + f'{uid}_FunctionalConnectomes_colnames.npy')
  for session in session_list:
    files_to_transfer.append('{ucla_path}' + f'{uid}_{session}_FunctionalConnectomes.csv')
    files_to_transfer.append('{ucla_path}' + f'{uid}_{session}_FunctionalConnectomes.npy')
    files_to_transfer.append('{ucla_path}' + f'{uid}_{session}_FunctionalConnectomes_colnames.npy')

metadata_folder = '{base}parcellation_metadata'

# psswd = getpass.getpass(f'Password for {username}@{hostname}:')
username = 'kbaacke'
ssh = createSSHClient(f'{hostname}', 22, f'{username}', getpass.getpass(f'Password for {username}@{hostname}:'))
scp = SCPClient(ssh.get_transport())
start_time = dt.datetime.now()
scp.get(metadata_folder.format(base=remote_base), base, recursive=True)
for f in files_to_transfer:
  scp.get(f.format(ucla_path = ucla_path), f.format(ucla_path = local_ucla_path))



end_time = dt.datetime.now()
print('Done. Runtime: ', end_time - start_time)