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
dest_sep = '/'
source_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) + sep
base = f'C:\\Users\\kyle\\repos\\'
dest_base = '/mnt/usb1/Code/'

shutil.make_archive(f'{base}HCP_Analyses', 'zip', f'{base}HCP-Analyses')

archive_pattern = '{base}HCP_Analyses.zip'
psswd = getpass.getpass(f'Password for {username}@{hostname}:')
start_time = dt.datetime.now()
ssh = createSSHClient(f'{hostname}', 22, f'{username}', f'{psswd}')
scp = SCPClient(ssh.get_transport())

scp.put(archive_pattern.format(base = base), archive_pattern.format(base = dest_base))

end_time = dt.datetime.now()
print('Done. Runtime: ', end_time - start_time)