import os
import shutil
import datetime as dt
import pandas as pd
import paramiko
from scp import SCPClient
import zipfile
import getpass
base = f'S:\\' # Change this to where you wan the UCLA_Decoding folder to be 
username = 'kbaacke' #Change this to be your username on r2
hostname = 'hx.psych.uiuc.edu'

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client

sep = os.path.sep
remote_sep = '/'
source_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + sep

remote_base = '/data/hx-hx1/kbaacke/datasets/'

output_path = '{base}UCLA_Decoding{sep}'

new_accuracy_path = output_path + 'Full_Diag_Accuracy_V2.csv'
prediction_path = output_path + 'predictions.csv'
confusion_path = output_path + 'confusion.csv'
accuracy_metadata_path = output_path + 'accuracy_metadata.csv'
accuracy_metadata_confusion_path = output_path + 'accuracy_metadata_confusion.csv'
prediction_metadata_path = output_path + 'predictions_metadat_confusion.csv'
confusion_pivot_path = output_path + 'confusion_pivot_01.csv'

files_paths_to_download = [
  new_accuracy_path,
  prediction_path,
  confusion_path,
  accuracy_metadata_path,
  accuracy_metadata_confusion_path,
  prediction_metadata_path,
  confusion_pivot_path
]

# metapath = output_path + 'metadata{sep}'
# confpath =  output_path +  'confusion{sep}'
# cmoppath = output_path + 'comparison{sep}'
# probspath = output_path + 'probs{sep}'

start_time = dt.datetime.now()
ssh = createSSHClient(f'{hostname}', 22, f'{username}', getpass.getpass(f'Password for {username}@{hostname}:'))
scp = SCPClient(ssh.get_transport())

# scp.get(output_path.format(base = remote_base, sep=remote_sep), output_path.format(base = base, sep=sep), recursive=True)

for filepath in files_paths_to_download:
  scp.get(filepath.format(base = remote_base, sep=remote_sep), output_path.format(base = base, sep=sep))

end_time = dt.datetime.now()
print('Done. Runtime: ', end_time - start_time)