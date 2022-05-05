# Python script to read in all available parcellated fMRI data from the UCLA dataset (ds000030) and save as functional connectomes
# * I already ran this on the cluster, but if you want to re-run to play with it, everything should work as-is on a cluster ssh session on any machine
# If you don't want to overwrite any of the output files, just comment out the lines that include 'np.save()'

# Imports 
import pandas as pd
import numpy as np
import os


# Global Variables

# run_uid = '69354adf' # 200, No ICA
# run_uid = '5384cc6b' # 1000, No ICA
# run_uid = '4023aba1' # 
run_uid = '8a52e148' # 
# numpy_template = '{DATAPATH}/UCLA-ds000030_out/sub-{SUBJECT}/func/{RUN_UID}_sub-{SUBJECT}_task-{SESSION}_space-MNI152NLin2009cAsym_desc-preproc_bold.npy'
numpy_template = '{DATAPATH}/UCLA-ds000030_out/sub-{SUBJECT}/func/{RUN_UID}_sub-{SUBJECT}_task-{SESSION}_space-MNI152NLin6Asym_desc-smoothAROMAnonaggr_bold.npy'


datapath = '/data/hx-hx1/kbaacke/datasets/UCLA/'
outpath = f'{datapath}{run_uid}_FunctionalConnectomes/'
# the f formats the string, replacing the words in the {} with their variable values (same as .format() used later where you can specify the variable values in the ())
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

# Make sure the output folder exists. If not, make one
try:
  os.mkdir(outpath)
except Exception as e:
  print(e)

# Iterate through scans to generate functional connectomes
for subject in subject_list:
  for session in session_list:
    # Format the string template with the variables created in the for loops 
    input_file_name = numpy_template.format(DATAPATH=datapath, SUBJECT=subject, SESSION=session, RUN_UID=run_uid)
    output_file_name = output_template.format(OUTPATH=outpath, SUBJECT=subject, SESSION=session, RUN_UID=run_uid)
    # Some sessions for some subjects are missing. TO prevent errors from halting the script, we will try read the file in a try-except block.
    try:
      # Read the numpy file into memory
      timeseries = np.load(input_file_name)
      # Generate a correlation matrix from the timeseries
      cor_coef = np.corrcoef(timeseries.T) # The T rotates the array into the correct orientation for this function
      # Save the matrix to an output numpy array
      np.save(output_file_name, cor_coef)
    except Exception as e:
      # This allows us to print the error rather than raising it and halting the script
      print(f'Error processing data for sub-{subject}_task-{session}: {e}')
      

# If you would like to inspect the data at any point, you can always set the values in the for loop manually to inspect them as you go
subject = '10189'
session = 'rest'
input_file_name = numpy_template.format(DATAPATH=datapath, SUBJECT=subject, SESSION=session, RUN_UID=run_uid)
print(numpy_template, '->', input_file_name)
output_file_name = output_template.format(OUTPATH=outpath, SUBJECT=subject, SESSION=session, RUN_UID=run_uid)
print(output_template, '->', output_file_name)
timeseries = np.load(input_file_name)
print(timeseries.shape)
print(timeseries.T.shape)
cor_coef = np.corrcoef(timeseries.T)
print(cor_coef.shape)
print(cor_coef)
