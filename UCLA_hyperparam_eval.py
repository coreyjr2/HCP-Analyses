import numpy as np
import pandas as pd

output_path = 'S:\\UCLA_Decoding\\'

new_accuracy_path = output_path + 'Full_Diag_Accuracy_V2.csv'
prediction_path = output_path + 'predictions.csv'
confusion_path = output_path + 'confusion.csv'
accuracy_metadata_path = output_path + 'accuracy_metadata.csv'
accuracy_metadata_confusion_path = output_path + 'accuracy_metadata_confusion.csv'
prediction_metadata_path = output_path + 'predictions_metadat_confusion.csv'
confusion_pivot_path = output_path + 'confusion_pivot_01.csv'

confusions_df = pd.read_csv(confusion_path)

acc_meta_confusion = pd.read_csv(accuracy_metadata_confusion_path)

acc_meta_confusion_nc = acc_meta_confusion[acc_meta_confusion['subjects_used_x']!='No CONTROL']

acc_meta_confusion_nc_all = acc_meta_confusion_nc[acc_meta_confusion_nc['sessions_used_x']=='all']

target_columns = ['testacc']
for col in confusions_df.columns:
  if 'CONTROL' not in col:
    if 'percent' in col:
      target_columns.append(col)

acc_meta_confusion_nc_all['class_weight'].fillna('None', inplace=True)

hyperparam_df_pivot_1 = acc_meta_confusion_nc_all.pivot_table(
  index=['C','class_weight','kernel'],
  values=['testacc'],
  aggfunc=[np.mean,np.std]
)

hyperparam_df_pivot_1.style.format("{:.2f}")

hyperparam_df_pivot_2 = acc_meta_confusion_nc_all.pivot_table(
  index=['C','class_weight','kernel'],
  values=target_columns,
  aggfunc=np.mean
)

hyperparam_df_pivot_2.style.format("{:.2f}")

target_columns_2 = [
  'testacc',
  'BIPOLAR_BIPOLAR_percent','BIPOLAR_SCHZ_percent',
  'SCHZ_SCHZ_percent','SCHZ_BIPOLAR_percent'
  ]

hyperparam_df_pivot_3 = acc_meta_confusion_nc_all.pivot_table(
  index=['C','class_weight','kernel'],
  values=target_columns_2,
  aggfunc=np.mean
)

hyperparam_df_pivot_3.style.format("{:.2f}")