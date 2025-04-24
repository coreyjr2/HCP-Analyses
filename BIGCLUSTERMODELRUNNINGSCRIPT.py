import os
import sys
import io
import pandas as pd
import math
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import json
import hashlib
import logging
import platform

# Global Variables
sep = os.path.sep
source_path = os.path.dirname(os.path.abspath(__file__)) + sep
outpath = "/raid-18/LS/medinak/kbaacke/dr-fs/" ####
local_path = "/raid-18/LS/medinak/kbaacke/dr-fs/" ###
input_path = "" ###
run_uid = '89952a'

total_start_time = dt.datetime.now()
# logging.basicConfig(filename=f'{outpath}{run_uid}_DEBUG.log', level=logging.DEBUG)
# arch = str(platform.architecture()[0])
# logging.debug(f'Architecture: {arch}')
# machine = platform.machine()
# logging.debug(f'Processor: {machine}')
# node = platform.node()
# logging.debug(f'Node Name: {node}')
# logging.info(f'Started; {total_start_time}') #Adds a line to the logfile to be exported after analysis

meta_dict = json.load(open(local_path + run_uid + sep + run_uid + 'metadata.json'))

feature_set_dict = {
  'parcel_connection':{
  }
}
fs_outpath = f'{local_path}{run_uid}{sep}FeatureSelection{sep}'
sub_start_time = dt.datetime.now()
# logging.info(f'Attempting to read data from {fs_outpath}: {sub_start_time}')


# Setting which parameters to run

## Slurm version
sys.stdout = sys.__stdout__
sys.stdout = io.StringIO()

job_id = os.getenv('SLURM_ARRAY_TASK_ID') #This will only work in an actual job.
array_index = int(job_id)

# TODO: Write script to assign method, split and model to a csv file to designate indices of models runs
runtime_df = pd.read_csv(f'{local_path}array_assignment.csv')

method = runtime_df.iloc[array_index]['model']
split_index = runtime_df.iloc[array_index]['split']

# Read in the training-test split indices
train_index = np.load(f'{input_path}{run_uid}_split_{split_index}_train.npy')
test_index = np.load(f'{input_path}{run_uid}_split_{split_index}_test.npy')

random_state = 42

def train_models_with_gridsearch(train_data, test_data, y_train, y_test, label='fc'):
    results = {}

    # Define model hyperparameter grids
    model_configs = {
        'svm': {
            'model': SVC(),
            'params': {
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 1, 10],
                'epsilon': [0.1, 0.2, 0.5]
            }
        },
        'rf': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'rr': {
            'model': Ridge(),
            'params': {
                'alpha': [1e-3, 1e-2, 1e-1, 1, 10]
            }
        },
        'gbc':{
            'model':GradientBoostingClassifier(),
            'params':{
                'max_depth':[None, 2, 10, 25],
                'max_features':['sqrt', 'log2', None],
                'n_estimators':[500, 1000],
                'learning_rate':[.01, .1, 1, 10, 100]
            }
        }#,
        # 'lasso': {
        #     'model': Lasso(max_iter=10000),
        #     'params': {
        #         'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1]
        #     }
        # },
        # 'en': {
        #     'model': ElasticNet(max_iter=10000),
        #     'params': {
        #         'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1],
        #         'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]
        #     }
        # }
    }

    for name, config in model_configs.items():
        print(f"Training {name.upper()} model for {label.upper()} data...")
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1
        )
        grid.fit(train_data, y_train['Age'])
        best_model = grid.best_estimator_
        predictions = best_model.predict(test_data)

        # Save results with descriptive keys
        # results[f'{name}_{label}_model'] = best_model
        results[f'{name}_{label}_predictions'] = predictions
        results[f'{name}_{label}_best_params'] = grid.best_params_

    return results
# Method
## Split
### Model
#### Hyperparameters