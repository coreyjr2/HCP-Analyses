import pandas as pd

subjects = pd.read_csv('C:\\Users\\kyle\\repos\\HCP-Analyses\\subject_list.csv')['ID']
subjects = subjects
path_pattern = "S:\\HCP\\HCP_1200\\{}\\MNINonLinear\\Results\\{}\\{}.npy"
BOLD_NAMES = [
  "rfMRI_REST1_LR", 
  "rfMRI_REST1_RL", 
  "rfMRI_REST2_LR", 
  "rfMRI_REST2_RL", 
  "tfMRI_MOTOR_RL", 
  "tfMRI_MOTOR_LR",
  "tfMRI_WM_RL", 
  "tfMRI_WM_LR",
  "tfMRI_EMOTION_RL", 
  "tfMRI_EMOTION_LR",
  "tfMRI_GAMBLING_RL", 
  "tfMRI_GAMBLING_LR", 
  "tfMRI_LANGUAGE_RL", 
  "tfMRI_LANGUAGE_LR", 
  "tfMRI_RELATIONAL_RL", 
  "tfMRI_RELATIONAL_LR", 
  "tfMRI_SOCIAL_RL", 
  "tfMRI_SOCIAL_LR"
]

for s in subjects:
  for t in BOLD_NAMES:
    confounds = []
    print(s, t)
    try:
      confounds.append(pd.read_csv( # See HCP1200 release manual page 96, also see https://www.mail-archive.com/hcp-users@humanconnectome.org/msg02961.html
        path_pattern[:-6].format(s, t) + 'Movement_Regressors.txt',
        sep='  ',
        header=None,
        names = [
          'trans_x','trans_y','trans_z',
          'rot_x','rot_y','rot_z',
          'trans_dx','trans_dy','trans_dz',
          'rot_dx','rot_dy','rot_dz'
        ]
      ))
    except:
      print('Movement_Regressors.txt not available.')
      pass
    try:
      confounds.append(pd.read_csv( # Made from removing the mean and linear trend from each variable in Movement_Regressors.txt
        path_pattern[:-6].format(s, t) + 'Movement_Regressors_dt.txt',
        sep='  ',
        header=None,
        names = [
          'trans_x_dt','trans_y_dt','trans_z_dt',
          'rot_x_dt','rot_y_dt','rot_z_dt',
          'trans_dx_dt','trans_dy_dt','trans_dz_dt',
          'rot_dx_dt','rot_dy_dt','rot_dz_dt'
        ]
      ))
    except:
      print('Movement_Regressors_dt.txt not available.')
      pass
    try:
      confounds.append(pd.read_csv( # amount of motion from the previous time point, alternative to FD see https://www.mail-archive.com/hcp-users@humanconnectome.org/msg04444.html
        path_pattern[:-6].format(s, t) +'Movement_RelativeRMS.txt',
        sep='  ',
        header=None, 
        names = ['Movement_RelativeRMS']
      ))
    except:
      print('Movement_RelativeRMS.txt not available.')
      pass

    # try:
    #    confounds.apend(pd.read_csv( # See page 38 of HCP_1200 release manual. 400HZ approx 288 samples per frame
    #     path_pattern[:-6].format(s, t) + t + '_Physio_log.txt',
    #     sep='\t', 
    #     header=None, 
    #     names=[
    #       'trigger_pulse','respiration','pulse_oximeter'
    #     ]
    #   ))
    #   #We can shift the timescale on this 
    # except:
    #   movement[s][t]['Physio_log'] = None
    try:
      full_confounds = pd.concat(confounds, axis=1)
      full_confounds.to_csv(path_pattern[:-6].format(s, t) + '{}_{}_full-confounds.csv'.format(s,t), index=False)
    except:
      print(f'No movement data available for {s} {t}')
