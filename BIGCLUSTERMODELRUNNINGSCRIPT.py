import os, sys, io, math, json, hashlib, logging, platform, re, psutil
import pandas as pd
import numpy as np
import datetime as dt
import pickle as pk
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, GroupKFold
from pathlib import Path
import pickle as pk


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
runtime_df = pd.read_csv(f'{local_path}scripts{sep}array_assignment.csv')
method = runtime_df.iloc[array_index]['method']
# split_index = runtime_df.iloc[array_index]['split']
dataset = runtime_df.iloc[array_index]['dataset']
logging.basicConfig(filename=f'{outpath}{dataset}{sep}{dataset}-{method}_DEBUG.log', level=logging.DEBUG)
logging.info(f'Method: {method}')
logging.info(f'Dataset: {dataset}')
# logging.debug(f'Total Avaiable Memory (GB): {psutil.virtual_memory()[1]/1000000000}')
# logging.debug(f'Available CPUs: {psutil.cpu_count()}')
logging.debug(f'Architecture: {platform.architecture()[0]}')

total_start_time = dt.datetime.now()

fs_outpath = f'{local_path}{dataset}{sep}FeatureSelection{sep}'
pred_path = f'{local_path}{dataset}{sep}predictions{sep}'
acc_path = f'{local_path}{dataset}{sep}accuracies{sep}'
machine = platform.machine()
logging.debug(f'Processor: {machine}')
node = platform.node()
logging.debug(f'Node Name: {node}')
logging.info(f'Started; {total_start_time}')
sub_start_time = dt.datetime.now()
# logging.info(f'Attempting to read data from {fs_outpath}: {sub_start_time}')
meta_dict = json.load(open(local_path + run_uid + 'metadata.json'))

feature_set_dict = {}
k = 'parcel_connection'
for target_df in ['train_x','test_x','train_y','test_y']:
    feature_set_dict[target_df] = pd.DataFrame(np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}_colnames.npy', allow_pickle=True))

if method == 'All':
    train_x = feature_set_dict['train_x']
    train_y = feature_set_dict['train_y'].values.ravel()
    test_x = feature_set_dict['test_x']
    test_y = feature_set_dict['test_y'].values.ravel()
elif 'Hierarchical' in method:
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

def train_models_with_gridsearch(train_data, test_data, y_train, y_test, dataset, cv=5):
    results = []
    # Define model hyperparameter grids
    model_configs = {
        'svm': {
            'model': SVC(),
            'params': {
                'kernel': ['rbf', 'linear'],
                'C': [0.1, 1, 10]#,
                # 'epsilon': [0.1, 0.2, 0.5]
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
        }
    }
    for name, config in model_configs.items():
        sub_start_time = dt.datetime.now()
        logging.info(f"Training {name.upper()} model for {dataset.upper()} data...")
        meta_dict = {
            'data_uid':run_uid,
            'Classifier':name,
            'random_state':random_state,
            'feature_selection': method,
            'features':list(train_data.columns),
            'dataset':dataset
        }
        meta_uid = generate_uid(meta_dict)
        grid = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            scoring='accuracy',
            cv=cv,
            n_jobs=-1
        )
        grid.fit(train_data, y_train, groups=train_data['Subject'])
        best_model = grid.best_estimator_
        predictions = best_model.predict(test_data)
        prediction_df = pd.DataFrame({
            'Subject':test_x['Subject'],
            'Task':test_y,
            'prediction':predictions
        })
        prediction_df.to_csv(f'{pred_path}{meta_uid}_predictions.csv', index=False)
        training_accuracy = best_model.score(train_data, y_train)
        test_accuracy = best_model.score(test_data, y_test)
        classification_rep = classification_report(test_data, predictions, output_dict=True)
        confusion_mat = confusion_matrix(test_data, predictions)
        sub_end_time = dt.datetime.now()
        results_dict = {
            'data_uid':[run_uid],
            'dataset':[dataset],
            'Classifier':[name],
            'random_state':[random_state],
            'feature_selection': [method],
            'n_features':[len(train_x.columns)],
            'training_accuracy':[training_accuracy],
            'test_accuracy':[test_accuracy],
            'runtime':[(end_time - start_time).total_seconds()],
            'uid': [meta_uid]
        }
        for param, value in grid.best_params_:
            results_dict[param] = value
        for level in ['macro avg',  'weighted avg']:
            for k in classification_rep[level]:
                results_dict[f'{level}|{k}'] = [classification_rep[level][k]]
        for group in set(test_y):
            results_dict[f'Class{group}|N'] = [np.sum(confusion_mat[group,])]
            results_dict[f'Class{group}|accuracy'] = [confusion_mat[group,group]/np.sum(confusion_mat[group,])]
            for k in classification_rep[str(group)].keys():
                results_dict[f'Class{group}|{k}'] = [classification_rep[str(group)][k]]
            for group2 in set(test_y):
                results_dict[f'Class{group}|Predicted{group2}'] = [confusion_mat[group,group2]]
            for group2 in set(test_y):
                results_dict[f'Class{group}|Predicted{group2}_percent'] = [confusion_mat[group,group2]/np.sum(confusion_mat[group,])]
        res_df = pd.DataFrame(results_dict)
        results.append(res_df)      
        logging.info(f'\tDone. Runtime: {sub_end_time - sub_start_time}')
    return pd.concat(results)


# define CV here
rng = np.random.RandomState(42)
gkfv = GroupKFold(n_splits=5)

# #TESTING
# train_data = train_x
# test_data = test_x
# y_train = train_y
# y_test = test_y
# cv=gkfv
# start_time = dt.datetime.now()
# results = []
# # Define model hyperparameter grids
# model_configs = {
#     'svm': {
#         'model': SVC(),
#         'params': {
#             'kernel': ['rbf', 'linear'],
#             'C': [0.1, 1, 10]
#         }
#     },
#     'rf': {
#         'model': RandomForestClassifier(random_state=42),
#         'params': {
#             'n_estimators': [100, 200],
#             'max_depth': [5, 10, None],
#             'min_samples_split': [2, 5],
#             'min_samples_leaf': [1, 2]
#         }
#     },
#     'rr': {
#         'model': Ridge(),
#         'params': {
#             'alpha': [1e-3, 1e-2, 1e-1, 1, 10]
#         }
#     },
#     'gbc':{
#         'model':GradientBoostingClassifier(),
#         'params':{
#             'max_depth':[None, 2, 10, 25],
#             'max_features':['sqrt', 'log2', None],
#             'n_estimators':[500, 1000],
#             'learning_rate':[.01, .1, 1, 10, 100]
#         }
#     }
# }
# name = 'svm'
# config = {
#     'model': SVC(),
#     'params': {
#         'kernel': ['rbf', 'linear'],
#         'C': [0.1, 1, 10]
#     }
# }
# meta_uid = generate_uid(meta_dict)
# grid = GridSearchCV(
#     estimator=config['model'],
#     param_grid=config['params'],
#     scoring='accuracy',
#     cv=cv,
#     n_jobs=-1
# )
# grid.fit(train_data, y_train, groups=train_data['Subject'])
# #TESTING END

cv_results = train_models_with_gridsearch(train_x, test_x, train_y, test_y, dataset, cv = gkfv)
cv_results.to_csv(f'{acc_path}{dataset}-{method}.csv', index=False)