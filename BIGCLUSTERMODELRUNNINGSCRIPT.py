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
input_path = "/raid-18/LS/medinak/kbaacke/dr-fs/" ###
run_uid = '89952a'

sys.stdout = sys.__stdout__
sys.stdout = io.StringIO()
job_id = os.getenv('SLURM_ARRAY_TASK_ID') #This will only work in an actual job.
array_index = int(job_id)
runtime_df = pd.read_csv(f'{local_path}array_assignment.csv')
method = runtime_df.iloc[array_index]['model']
split_index = runtime_df.iloc[array_index]['split']
dataset = runtime_df.iloc[array_index]['dataset']
total_start_time = dt.datetime.now()

# Read in the training-test split indices
train_index = np.load(f'{input_path}{run_uid}_split_{split_index}_train.npy')
test_index = np.load(f'{input_path}{run_uid}_split_{split_index}_test.npy')

fs_outpath = f'{local_path}{dataset}{run_uid}{sep}FeatureSelection{sep}parcel_connection{sep}'
sub_start_time = dt.datetime.now()
# logging.info(f'Attempting to read data from {fs_outpath}: {sub_start_time}')
meta_dict = json.load(open(local_path + run_uid + sep + run_uid + 'metadata.json'))


if method =='All':
    train = np.load(f'{fs_outpath}{run_uid}_{target_df}.npy')

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