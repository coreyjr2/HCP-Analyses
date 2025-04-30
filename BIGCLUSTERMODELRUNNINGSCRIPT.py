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
import pickle as pk
import re

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

feature_set_dict = {}
k = 'parcel_connection'
for target_df in ['train_x','test_x','train_y','test_y']:
    feature_set_dict[target_df] = pd.DataFrame(np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}_colnames.npy', allow_pickle=True))

if 'Hierarchical' in method:
    n = int(method.split('-')[1])
    try:
        feature_index = np.load(f'{fs_outpath}{k}{sep}{run_uid}_hierarchical-{n}.npy')
        train_x = feature_set_dict['train_x'][feature_set_dict['train_x'].columns[feature_index]]
        train_y = feature_set_dict['train_y'].values.ravel()
        test_x = feature_set_dict['test_x'][feature_set_dict['train_x'].columns[feature_index]]
        test_y = feature_set_dict['test_y'].values.ravel()
    except Exception as e:
        print(f'Error reading Hierarchical Features, n = {n}, Error: {e}')
elif 'PCA' in method:
    train_pca = np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_pca.npy')
    test_pca = np.load(f'{fs_outpath}{k}{sep}{run_uid}_test_pca.npy')
    pca_obj = pk.load(open(f'{fs_outpath}{k}{sep}{run_uid}_pca.pkl', 'rb'))
    if method == 'PCA Full':
        train_x = train_pca
        train_y = feature_set_dict['train_y'].values.ravel()
        test_x = test_pca
        test_y = feature_set_dict['test_y'].values.ravel()
    else:
        n_str = method.split('_')[1]
        train_x = train_pca
        train_y = feature_set_dict['train_y'].values.ravel()
        test_x = test_pca
        test_y = feature_set_dict['test_y'].values.ravel()
elif 'rf_selected' in method:
    x_len = int(method.split('_n')[1])
    try:
        feature_set = np.load(f'{fs_outpath}{k}{sep}{run_uid}_rf_selected_n{x_len}.npy')
        train_x = feature_set_dict['train_x'][feature_set]
        train_y = feature_set_dict['train_y'].values.ravel()
        test_x = feature_set_dict['test_x'][feature_set]
        test_y = feature_set_dict['test_y'].values.ravel()
    except Exception as e:
        sub_end_time = dt.datetime.now()
        print(f'Error reading SelectFromModel on RFC for {x_len} max features read from previous run: {e}, {sub_end_time}')
elif 'Permutation-Importance' in method:
    n_estimators = 500
    n_repeats = 50
    ranked_features = np.load(f'{fs_outpath}{k}{sep}{run_uid}_feature_importances_est-{n_estimators}.npy')
    feature_names = feature_set_dict['train_x'].columns
    rank_df = pd.DataFrame({
        'feature':list(feature_names)[1:],
        'importance':ranked_features
    })
    rank_df.sort_values(by='importance', ascending=False,inplace=True)
    ordered_features = list(rank_df['feature'])
    length = method.split('_')[1]
    train_x = feature_set_dict['train_x'][ordered_features[:length]]
    train_y = feature_set_dict['train_y'].values.ravel()
    test_x = feature_set_dict['test_x'][ordered_features[:length]]
    test_y = feature_set_dict['test_y'].values.ravel()
elif 'kPCA' in method:
    kernel = method.split('_')[1].split('-')[0]
    n_components = method.split('-')[1]
    train_kpca = np.load(f'{fs_outpath}{k}/{run_uid}_train_kpca-{kernel}.npy')
    test_kpca = np.load(f'{fs_outpath}{k}/{run_uid}_test_kpca-{kernel}.npy')
    # feature_set_dict[f'kpca-{kernel}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_kpca-{kernel}.pkl', 'rb'))
    train_x = train_kpca[:,0:n_components]
    train_y = feature_set_dict['train_y'].values.ravel()
    test_x = test_kpca[:,0:n_components]
    test_y = feature_set_dict['test_y'].values.ravel()
elif 'TruncatedSVD' in method:
    component_size = int(method.split('_')[1])
    train_x = np.load(f'{fs_outpath}{k}/{run_uid}_train_tSVD-{component_size}.npy')
    train_y = feature_set_dict['train_y'].values.ravel()
    test_x = np.load(f'{fs_outpath}{k}/{run_uid}_test_tSVD-{component_size}.npy')
    test_y = feature_set_dict['test_y'].values.ravel()
    # feature_set_dict[f'tSVD-{component_size}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_tSVD-{component_size}.pkl', 'rb'))
elif 'LDA' in method:
    train_x = np.load(f'{fs_outpath}{k}/{run_uid}_train_LDA.npy')
    train_y = feature_set_dict['train_y'].values.ravel()
    test_x = np.load(f'{fs_outpath}{k}/{run_uid}_test_LDA.npy')
    test_y = feature_set_dict['test_y'].values.ravel()
    # feature_set_dict[f'LDA'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_LDA.pkl', 'rb'))
elif 'Random' in method:
    sval = re.search(("[.]*_Random_(.+?)_v(.+?).npy"), method)
    n_features = sval[1]
    version = sval[2]
    selected_vars = np.load(f'{fs_outpath}{k}{sep}{method}.npy', allow_pickle=True)
    train_x = feature_set_dict['train_x'][selected_vars]
    train_y = feature_set_dict['train_y'].values.ravel()
    test_x = feature_set_dict['test_x'][selected_vars]
    test_y = feature_set_dict['test_y'].values.ravel()

random_state = 42

def generate_uid(metadata, length = 8):
  dhash = hashlib.md5()
  encoded = json.dumps(metadata, sort_keys=True).encode()
  dhash.update(encoded)
  # You can change the 8 value to change the number of characters in the unique id via truncation.
  run_uid = dhash.hexdigest()[:length]
  return f'_{run_uid}_'

def train_models_with_gridsearch(train_data, test_data, y_train, y_test, cv, label='fc'):
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
        meta_dict = {
            'data_uid':run_uid,
            'Classifier':'SVC',
            'C':c,
            'kernal':kernel,
            'class_weight':class_weight,
            'random_state':random_state,
            'feature_selection': selection_method,
            'features':list(train_x.columns),
            'split_index':split_index
        }
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            scoring='neg_mean_absolute_error',
            cv=5,
            n_jobs=-1
        )
        grid.fit(train_data, y_train)
        best_model = grid.best_estimator_
        predictions = best_model.predict(test_data)
        training_accuracy = best_model.score(train_data, y_train)
        test_accuracy = best_model.score(test_data, y_test)
        classification_rep = classification_report(test_data, predictions, output_dict=True)
        confusion_mat = confusion_matrix(test_data, predictions)
        results_dict = {
            'pred_uid':[pred_uid],
            'data_uid':[run_uid],
            'Classifier':['SVC'],
            'C':[c],
            'kernal':[kernel],
            'class_weight':[class_weight],
            'random_state':[random_state],
            'feature_selection': [selection_method],
            'n_features':[len(train_x.columns)],
            'training_accuracy':[training_accuracy],
            'test_accuracy':[test_accuracy],
            'split_index':[split_index],
            'runtime':[(end_time - start_time).total_seconds()]
        }
        results[f'{name}_{label}_predictions'] = predictions
        results[f'{name}_{label}_best_params'] = grid.best_params_
    return results


# Method
## Split
### Model
#### Hyperparameters