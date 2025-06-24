import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from resnet import ResNet20
import ast
import json
import os
import sys
from utils import load_fold_data
from pathlib import Path
from sklearn.metrics import r2_score, f1_score, roc_auc_score, accuracy_score, mean_squared_error
from rdkit import Chem
import rdkit.Chem.Descriptors as descriptors
import joblib


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', help = 'path to experiment folder, i.e. trained model')
parser.add_argument('--model', choices = ['ResNet', 'MLP', 'LR'])
args = parser.parse_args()

print(args)
experiment = args.experiment
model_name = args.model
Path(f"{experiment}/predictions_{model_name}").mkdir(parents=True, exist_ok=True)

with open(os.path.join(experiment, 'config.json')) as f:
    params = json.load(f)

# TensorFlow 2.x compatibility - GPU setup handled automatically

data_path = params['data']
print(data_path)
print('Feature: ', params['feature'])

data_df = pd.read_csv(data_path)
data_df[params['feature']] = data_df[params['feature']].apply(ast.literal_eval)

print(data_df.shape)
input_len = len(data_df[params['feature']].iloc[0])
  
if params['property'].lower() == 'logs' or params['property'].lower() == 'logd':
    r2_list = []
    rmse_list = []

    if params['property'].lower() == 'logs':
        label_col = 'new_logS'
        resnet_model_name = 'model-2000'
    else:
        label_col = 'new_logD'
        resnet_model_name = 'model-1500'

    print('Predicting...')

    for fold in range(1, 11):
        test_index = np.load(os.path.join(params['fold_indices_dir'], f'fold{fold}/test_index.npy'))

        X_test = np.array(list(data_df.iloc[test_index][params['feature']].values), dtype=float)
        y_test = np.array(data_df.iloc[test_index][label_col].values)
        test_smiles = data_df.iloc[test_index]['smiles'].values
        preds_df = pd.DataFrame({'SMILES': test_smiles, f'true_{params["property"]}': y_test})

        if args.model.lower() == 'resnet':
            # Note: ResNet evaluation requires TensorFlow 2.x compatibility updates
            print(f'ResNet evaluation not yet implemented for TensorFlow 2.x. Fold: {fold}')
            predictions = np.zeros(len(X_test))  # Placeholder
                
        elif args.model.lower() == 'mlp':
            trained_model = joblib.load(os.path.join(experiment, f'models/mlp/mlp_fold{fold}.sav'))
            predictions = trained_model.predict(X_test)

        elif args.model.lower() == 'lr':
            trained_model = joblib.load(os.path.join(experiment, f'models/lr/lr_fold{fold}.sav'))
            predictions = trained_model.predict(X_test)

        else:
            raise ValueError('Invalid property name!')


        preds_df[f'predicted_{params["property"]}_by_{model_name}'] = predictions

        r2_list.append(r2_score(y_true = y_test, y_pred = predictions))
        rmse_list.append(np.sqrt(mean_squared_error(y_true = y_test, y_pred = predictions)))

        preds_df.to_csv(f'{experiment}/predictions_{model_name}/prediction_fold{fold}.csv', index = False)

    
    print('Mean R^2: ', np.mean(r2_list))
    print('Mean RMSE: ', np.mean(rmse_list))


elif params['property'].lower() == 'logbb':
    acc_list = []
    f1_list = []
    auc_list = []
    print('Predicting...')

    for fold in range(1, 11):
        test_index = np.load(os.path.join(params['fold_indices_dir'], f'fold{fold}/test_index.npy'))

        X_test = np.array(list(data_df.iloc[test_index][params['feature']].values), dtype=float)
        y_test = np.array(data_df.iloc[test_index].new_BBclass.values)
        test_smiles = data_df.iloc[test_index]['smiles'].values
        preds_df = pd.DataFrame({'SMILES': test_smiles, 'true_logBB': y_test})

        if args.model.lower() == 'resnet':
            # Note: ResNet evaluation requires TensorFlow 2.x compatibility updates
            print(f'ResNet evaluation not yet implemented for TensorFlow 2.x. Fold: {fold}')
            predictions = np.zeros(len(X_test))  # Placeholder

        elif args.model.lower() == 'mlp':
            trained_model = joblib.load(os.path.join(experiment, f'models/mlp/mlp_fold{fold}.sav'))
            predictions = trained_model.predict(X_test)

        elif args.model.lower() == 'lr':
            trained_model = joblib.load(os.path.join(experiment, f'models/lr/lr_fold{fold}.sav'))
            predictions = trained_model.predict(X_test)

        else:
            raise ValueError('Invalid property name!')
        

        preds_df[f'predicted_{params["property"]}_by_{model_name}'] = predictions
        acc_list.append(accuracy_score(y_true = y_test, y_pred = predictions))
        f1_list.append(f1_score(y_true = y_test, y_pred = predictions))
        auc_list.append(roc_auc_score(y_true = y_test, y_score = predictions))

        preds_df.to_csv(f'{experiment}/predictions_{model_name}/prediction_fold{fold}.csv', index = False)

    print('Mean acc: ', np.mean(acc_list), np.std(acc_list))
    print('Mean f1: ', np.mean(f1_list), np.std(f1_list))
    print('Mean auc: ', np.mean(auc_list), np.std(auc_list))