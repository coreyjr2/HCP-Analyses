import os
import shutil
import datetime as dt
import pandas as pd
import paramiko
from scp import SCPClient
import zipfile
import getpass
username = 'kbaacke' # Change this to be your username
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
base = f'S:\\' # Change this to the path where you want the files to go
remote_base = '/data/hx-hx1/kbaacke/datasets/'
run_uid = '89952a'
output_paths = [
  '{base}ucla_analysis_output{sep}{run_uid}{sep}',
  '{base}hcp_analysis_output{sep}{run_uid}{sep}'
]

ssh = createSSHClient(f'{hostname}', 22, f'{username}', getpass.getpass(f'Password for {username}@{hostname}:'))
ftp = ssh.open_sftp()
scp = SCPClient(ssh.get_transport())

start_time = dt.datetime.now()
for output_path in output_paths:
  inputpath = output_path + 'FeatureSelection{sep}parcel_connection{sep}'
  try:
    os.makedirs(inputpath.format(base = base, sep=sep, run_uid = run_uid))
  except:
    pass
  input_files = ftp.listdir(inputpath.format(
    base=remote_base,
    sep = remote_sep,
    run_uid = run_uid
  ))
  target_files = []
  for f in input_files:
    if '.npy' in f and 'Random' not in f:
      target_files.append(f)
  for t in target_files:
    try:
      scp.get(inputpath.format(base = remote_base, sep=remote_sep, run_uid = run_uid) + t, inputpath.format(base = base, sep=sep, run_uid = run_uid) + t)
      print('Downloaded ', t)
    except:
      print(t, " failed to download.")


