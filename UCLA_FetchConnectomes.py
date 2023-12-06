# Script to pull data from the cluster; specifically the folder containing UCLA connectomes
# I ran this to save the files locally before zipping them to send over. This can be easily modified for you to pull directly if you are interested

import os
import shutil
import datetime as dt
import pandas as pd
import paramiko
from scp import SCPClient
import zipfile
import getpass
username = 'kbaacke'
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
# remote_base = '/mnt/usb1/'
# output_path = '{base}hcp_analysis_output{sep}8d2513{sep}'
output_path = ('{base}UCLA{sep}69354adf_FunctionalConnectomes{sep}','{base}UCLA{sep}')

# Make the local path
try:
  os.makedirs(output_path.format(base = base, sep=sep))
except:
  pass

start_time = dt.datetime.now()
ssh = createSSHClient(f'{hostname}', 22, f'{username}', getpass.getpass(f'Password for {username}@{hostname}:'))
scp = SCPClient(ssh.get_transport())
scp.get(output_path[0].format(base = remote_base, sep=remote_sep), output_path[1].format(base = base, sep=sep), recursive=True)
end_time = dt.datetime.now()
print('Done. Runtime: ', end_time - start_time)