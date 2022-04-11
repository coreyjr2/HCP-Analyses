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
dest_base = '/data/hx-hx1/kbaacke/Code/'

# shutil.make_archive(f'{base}HCP-Analyses', 'zip', f'{base}HCP-Analyses')

archive_pattern = '{base}HCP-Analyses.zip'
psswd = getpass.getpass(f'Password for {username}@{hostname}:')
start_time = dt.datetime.now()
ssh = createSSHClient(f'{hostname}', 22, f'{username}', f'{psswd}')
scp = SCPClient(ssh.get_transport())
job_pattern = '{base}HCP-Analyses{sep}hcp_predgen.sh'
runtime_pattern = '{base}HCP-Analyses{sep}hcp_cluster_predGen.py'

# scp.put(archive_pattern.format(base = base, sep=sep), archive_pattern.format(base = dest_base, sep=dest_sep))
# scp.put(job_pattern.format(base = base, sep=sep), job_pattern.format(base = dest_base, sep=dest_sep))
scp.put(runtime_pattern.format(base = base, sep=sep), runtime_pattern.format(base = dest_base, sep=dest_sep))

end_time = dt.datetime.now()
print('Done. Runtime: ', end_time - start_time)
