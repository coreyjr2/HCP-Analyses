import os, sys, io, math, json, hashlib, logging, platform, re, psutil
import pandas as pd
import numpy as np
import datetime as dt
import pickle as pk
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import RidgeClassifier, Lasso, ElasticNet
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, GroupKFold, HalvingGridSearchCV
from pathlib import Path
import pickle as pk


# Global Variables
sep = os.path.sep
source_path = os.path.dirname(os.path.abspath(__file__)) + sep
outpath = "/raid-18/LS/medinak/kbaacke/dr-fs/" 
local_path = "/raid-18/LS/medinak/kbaacke/dr-fs/" 
input_path = "/raid-18/LS/medinak/kbaacke/dr-fs/" 
run_uid = '89952a'
dataset = 'ucla'


sys.stdout = sys.__stdout__
sys.stdout = io.StringIO()
job_id = os.getenv('SLURM_ARRAY_TASK_ID') #This will only work in an actual job.
array_index = int(job_id)


runtime_df = pd.read_csv(f'{local_path}scripts{sep}array_assignment.csv')
n_per_job = 2
start_index = array_index*n_per_job
end_index = start_index+n_per_job
if end_index>len(runtime_df):
    end_index-=1

subset_df = runtime_df.iloc[start_index:end_index]
# logging.basicConfig(filename=f'{outpath}{dataset}{sep}{dataset}-{array_index}_DEBUG.log', level=logging.DEBUG)
logging.basicConfig(filename=f'{outpath}{dataset}{sep}{dataset}-test_linear_HalvingGridSearchCV_DEBUG.log', level=logging.DEBUG)
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
logging.info(f'Started: {total_start_time}')
sub_start_time = dt.datetime.now()
# logging.info(f'Attempting to read data from {fs_outpath}: {sub_start_time}')
meta_dict = json.load(open(local_path + run_uid + 'metadata.json'))

feature_set_dict = {}
k = 'parcel_connection'
for target_df in ['train_x','test_x','train_y','test_y']:
    feature_set_dict[target_df] = pd.DataFrame(np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}.npy', allow_pickle=True), columns = np.load(f'{fs_outpath}{k}{sep}{run_uid}_{target_df}_colnames.npy', allow_pickle=True))

random_state = 42

def mean_norm(df_input):
  return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

def scale_subset(df, cols_to_exclude):
  df_excluded = df[cols_to_exclude]
  df_temp = df.drop(cols_to_exclude, axis=1, inplace=False)
  df_temp = mean_norm(df_temp)
  df_ret = pd.concat([df_excluded, df_temp], axis=1, join='inner')
  return df_ret

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
                'kernel': ['linear'], #'rbf', 
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
            'model': RidgeClassifier(),
            'params': {
                'alpha': [1e-3, 1e-2, 1e-1, 1, 10]
            }
        }#,
        # 'hgbc':{
        #     'model':HistGradientBoostingClassifier(),
        #     'params':{
        #         'learning_rate': [0.01, 0.1],
        #         'max_iter': [100, 200],
        #         'max_depth': [3, 5, 10, 20, None],
        #         'max_features': [0.8, 1.0],
        #         'l2_regularization': [0.0, 0.01, 0.1, 1.0]
        #     }
        # }
    }
    for name, config in model_configs.items():
        sub_start_time = dt.datetime.now()
        logging.info(f"Training {name.upper()} model for {dataset.upper()} data...")
        meta_dict = {
            'data_uid':run_uid,
            'Classifier':name,
            'random_state':random_state,
            'feature_selection': method,
            # 'features':list(train_data.columns),
            'dataset':dataset
        }
        meta_uid = generate_uid(meta_dict)
        grid = HalvingGridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            scoring='accuracy',
            cv=cv,
            n_jobs=-1,
            verbose=4
        )
        grid.fit(train_data, y_train, groups=train_data['Subject'])
        best_model = grid.best_estimator_
        predictions = best_model.predict(test_data)
        prediction_df = pd.DataFrame({
            'Subject':test_data['Subject'],
            'Task':test_y,
            'prediction':predictions
        })
        prediction_df.to_csv(f'{pred_path}{meta_uid}_predictions.csv', index=False)
        training_accuracy = best_model.score(train_data, y_train)
        test_accuracy = best_model.score(test_data, y_test)
        classification_rep = classification_report(y_test, predictions, output_dict=True)
        confusion_mat = confusion_matrix(y_test, predictions)
        sub_end_time = dt.datetime.now()
        try:
            feature_len = len(train_x.columns)
        except:
            feature_len = train_x.shape[1]
        results_dict = {
            'data_uid':[run_uid],
            'dataset':[dataset],
            'Classifier':[name],
            'random_state':[random_state],
            'feature_selection': [method],
            'n_features':[feature_len],
            'training_accuracy':[training_accuracy],
            'test_accuracy':[test_accuracy],
            'runtime':[(sub_end_time - sub_start_time).total_seconds()],
            'uid': [meta_uid]
        }
        for param, value in grid.best_params_.items():
            results_dict[param] = value
        for level in ['macro avg',  'weighted avg']:
            for k in classification_rep[level]:
                results_dict[f'{level}|{k}'] = [classification_rep[level][k]]
        for group in set(test_y):
            results_dict[f'Class{group}|N'] = [np.sum(confusion_mat[group-1,])]
            results_dict[f'Class{group}|accuracy'] = [confusion_mat[group-1,group-1]/np.sum(confusion_mat[group-1,])]
            for k in classification_rep[str(group)].keys():
                results_dict[f'Class{group}|{k}'] = [classification_rep[str(group)][k]]
            for group2 in set(test_y):
                results_dict[f'Class{group}|Predicted{group2}'] = [confusion_mat[group-1,group2-1]]
            for group2 in set(test_y):
                results_dict[f'Class{group}|Predicted{group2}_percent'] = [confusion_mat[group-1,group2-1]/np.sum(confusion_mat[group-1,])]
        res_df = pd.DataFrame(results_dict)
        results.append(res_df)      
        logging.info(f'\tDone. Runtime: {sub_end_time - sub_start_time}')
        if len(results)==1:
            res_df.to_csv(f'{acc_path}{dataset}-{method}.csv', index=False)
        else:
            cv_results = pd.concat(results)
            cv_results.to_csv(f'{acc_path}{dataset}-{method}.csv', index=False)
    return pd.concat(results)


method_test_list = [
    'kPCA_rbf-25',
    'PCA_34',
    'TruncatedSVD_100',
    'Permutation-Importance_361',
    'rf_selected_n142',
    'Hierarchical-33',
    'LDA',
    'All'
]
for method in method_test_list:#subset_df['method']:
    # try:
    logging.info(f'Method: {method}')
    logging.info(f'Dataset: {dataset}')
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
        length = int(method.split('_')[1])
        train_x = feature_set_dict['train_x'][ordered_features[:length]]
        train_y = feature_set_dict['train_y'].values.ravel()
        test_x = feature_set_dict['test_x'][ordered_features[:length]]
        test_y = feature_set_dict['test_y'].values.ravel()
    elif 'kPCA' in method:
        kernel = method.split('_')[1].split('-')[0]
        n_components = int(method.split('-')[1])
        train_kpca = np.load(f'{fs_outpath}{k}/{run_uid}_train_kpca-{kernel}.npy')
        test_kpca = np.load(f'{fs_outpath}{k}/{run_uid}_test_kpca-{kernel}.npy')
        # feature_set_dict[f'kpca-{kernel}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_kpca-{kernel}.pkl', 'rb'))
        train_x = pd.DataFrame(train_kpca[:,0:n_components])
        train_x.columns = train_x.columns.astype(str)
        train_x['Subject'] = feature_set_dict['train_x']['Subject']
        train_y = feature_set_dict['train_y'].values.ravel()
        test_x = pd.DataFrame(test_kpca[:,0:n_components])
        test_x.columns = test_x.columns.astype(str)
        test_x['Subject'] = feature_set_dict['test_x']['Subject']
        test_y = feature_set_dict['test_y'].values.ravel()
        train_x = scale_subset(train_x, ['Subject'])
        test_x = scale_subset(test_x, ['Subject'])
    elif 'PCA' in method:
        train_pca = np.load(f'{fs_outpath}{k}{sep}{run_uid}_train_pca.npy')
        test_pca = np.load(f'{fs_outpath}{k}{sep}{run_uid}_test_pca.npy')
        pca_obj = pk.load(open(f'{fs_outpath}{k}{sep}{run_uid}_pca.pkl', 'rb'))
        if method == 'PCA Full':
            train_x = pd.DataFrame(train_pca)
            train_x.columns = train_x.columns.astype(str)
            train_x['Subject'] = feature_set_dict['train_x']['Subject']
            train_y = feature_set_dict['train_y'].values.ravel()
            test_x = pd.DataFrame(test_pca)
            test_x.columns = test_x.columns.astype(str)
            test_x['Subject'] = feature_set_dict['test_x']['Subject']
            test_y = feature_set_dict['test_y'].values.ravel()
            train_x = scale_subset(train_x, ['Subject'])
            test_x = scale_subset(test_x, ['Subject'])
        else:
            n_components = int(method.split('_')[1])
            train_x = pd.DataFrame(train_pca[:,0:n_components])
            train_x.columns = train_x.columns.astype(str)
            train_x['Subject'] = feature_set_dict['train_x']['Subject']
            train_y = feature_set_dict['train_y'].values.ravel()
            test_x = pd.DataFrame(test_pca[:,0:n_components])
            test_x.columns = test_x.columns.astype(str)
            test_x['Subject'] = feature_set_dict['test_x']['Subject']
            test_y = feature_set_dict['test_y'].values.ravel()
            train_x = scale_subset(train_x, ['Subject'])
            test_x = scale_subset(test_x, ['Subject'])
    elif 'TruncatedSVD' in method:
        component_size = int(method.split('_')[1])
        train_x = pd.DataFrame(np.load(f'{fs_outpath}{k}/{run_uid}_train_tSVD-{component_size}.npy'))
        train_x.columns = train_x.columns.astype(str)
        train_x['Subject'] = feature_set_dict['train_x']['Subject']
        train_y = feature_set_dict['train_y'].values.ravel()
        test_x =  pd.DataFrame(np.load(f'{fs_outpath}{k}/{run_uid}_test_tSVD-{component_size}.npy'))
        test_x.columns = test_x.columns.astype(str)
        test_x['Subject'] = feature_set_dict['test_x']['Subject']
        test_y = feature_set_dict['test_y'].values.ravel()
        train_x = scale_subset(train_x, ['Subject'])
        test_x = scale_subset(test_x, ['Subject'])
        # feature_set_dict[f'tSVD-{component_size}'] = pk.load(open(f'{fs_outpath}{k}/{run_uid}_tSVD-{component_size}.pkl', 'rb'))
    elif 'LDA' in method:
        train_x = pd.DataFrame(np.load(f'{fs_outpath}{k}/{run_uid}_train_LDA.npy'))
        train_x['Subject'] = feature_set_dict['train_x']['Subject']
        train_y = feature_set_dict['train_y'].values.ravel()
        test_x = pd.DataFrame(np.load(f'{fs_outpath}{k}/{run_uid}_test_LDA.npy'))
        test_x['Subject'] = feature_set_dict['test_x']['Subject']
        test_y = feature_set_dict['test_y'].values.ravel()
        train_x = scale_subset(train_x, ['Subject'])
        test_x = scale_subset(test_x, ['Subject'])
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
    rng = np.random.RandomState(42)
    gkfv = GroupKFold(n_splits=5)
    cv_results = train_models_with_gridsearch(train_x, test_x, train_y, test_y, dataset, cv = gkfv)
    cv_results.to_csv(f'{acc_path}{dataset}-{method}.csv', index=False)
    # except Exception as e:
    #     logging.info(f'\tError: {e}')

total_end_time = dt.datetime.now()
logging.info(f'Finished: {total_end_time}')