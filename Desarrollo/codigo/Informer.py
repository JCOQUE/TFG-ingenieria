import sys

# Needed to use the Informer2020 modules
#sys.path.append('C:/Users/jcoqu/OneDrive/Documents/U-tad/Curso5/TFG/TFG_ingenieria/Desarrollo/codigo/Informer2020')

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from prefect import flow, task
import dagshub
import pickle
import sys
import io
from datetime import datetime

from Informer2020.model import Informer

from tfg_module import my_get_time_series as mgts
from tfg_module import my_process_data as mpd
from tfg_module import my_future as mf
from tfg_module import my_get_directories as mgd

# IMPORTANT!! These lines are needed for the following error when connecting to dagshub:
#UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-2: character maps to <undefined>
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

ABS_PATH_CSV = mgd.get_csv_directory()
ABS_PATH_PLOT = mgd.get_pred_plots_directory()
ABS_PATH_PICKLE_MODELS = mgd.get_pickle_models_directory()

'''
NOTE: In this same folder you have a .ipynb notebook where you can follow in an easier way the core of this code.
'''

class InformerWrapper(BaseEstimator, RegressorMixin):
    '''
    Wrapper Class for Informer needed to apply GridSearchCV from the sklearn library with an algorithm 
    (Informer in this case) implemented using the Pytorch library.
    '''
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, n_heads=8, dropout=0.05, 
                 freq='M', learning_rate=0.0001, num_epochs=3):
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len
        self.n_heads = n_heads
        self.dropout = dropout
        self.freq = freq
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model = None

    def create_model(self):
        '''
        Initializes the Informer model with its respective parameters.
        '''
        self.model = Informer(
            enc_in=self.enc_in,
            dec_in=self.dec_in,
            c_out=self.c_out,
            seq_len=self.seq_len,
            label_len=self.label_len,
            out_len=self.out_len,          
            n_heads=self.n_heads,
            dropout=self.dropout,
            freq=self.freq,
        )

    def get_dataset(self, X, y):
        '''
        Returns a Tensor dataset needed for the batch_x batch_y split
        in the training.
        '''
        dataset = TensorDataset(X, y)
        dataset_batch = DataLoader(dataset, batch_size=12, shuffle=False)
        return dataset_batch

    def fit(self, X, y):
        '''
        Trains the Informer model.
        '''
        self.create_model()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        dataset_batch = self.get_dataset(X, y)
        for _ in range(self.num_epochs): 
            for batch_x, batch_y in dataset_batch:
                batch_x = torch.unsqueeze(batch_x, dim=0) # Adding batch dimension
                batch_y = torch.unsqueeze(batch_y, dim=0) # Adding batch dimension
                optimizer.zero_grad()
                outputs = self.model(batch_x, batch_x, batch_x, batch_x) # model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self
  
    def predict(self, X):
        '''
        Predicts values 12 months ahead.
        '''
        predictions = []
        with torch.no_grad():
            X = torch.unsqueeze(X, dim=0)  # Adding batch dimension
            outputs = self.model(X, X, X, X)
            predictions.append(outputs.squeeze().numpy())

        return np.concatenate(predictions)
    
    
class MyInformer:
    def __init__(self, target):
        self.target = target
        self.model_name = 'Informer'
        self.ts = None
        self.dataset = None
        self.X = None       
        self.y = None
        self.seq_len = None
        self.label_len = None
        self.pred_len = None
        self.enc_in = None
        self.dec_in = None
        self.c_out = None
        self.freq = None
        self.metrics = None
        self.opposite_metric = {'neg_mean_absolute_error':'neg_root_mean_squared_error',
                                'neg_root_mean_squared_error':'neg_mean_absolute_error'}
        self.best_results = None

    
    def setting_attributes(self):
        '''
        Initializes the correct attributes for the model. These are parameters
        that should not be changed.
        '''
        self.target = mgts.target_cleaned(self.target)
        self.train_ts = mgts.get_ts(self.target, type = 'train')
        self.X, self.y = mpd.create_features(self.train_ts.copy(), target=self.target, informer=True)
        self.seq_len = 12  # since it is montly data, use the past 12 values for prediction
        self.label_len = 12  # steps introduced in the decoder to start generating forecast. Usually seq_len = label_len
        self.pred_len = 12  # We want one year prediction,  and our data is monthly data
        self.enc_in = self.X.shape[1]  # X features (= columns)
        self.dec_in = self.X.shape[1]  # X features (= columns)
        self.c_out = 1  # We want for each seq_len, one output, one target prediction
        self.freq = 'M'
    
    def get_param_grid(self):
        '''
        Returns all possible parameters values for GridSearchCV method. 
        It will try all possible combinations.
        '''
        param_grid = {
            'n_heads': [2, 4, 6, 8],
            'dropout': [0.05, 0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'num_epochs': [50, 100, 300, 500]
        }

        return param_grid
    
    def set_metrics(self):
        '''
        Sets the metrics that will be tracked during the training. In this case, 
        negative MAE, and negative RMSE. Later on, in save_best_results functions, 
        these metrics are converted to positive (i.e. how they should be).
        '''
        self.metrics = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']

    def define_model(self, param_grid):
        '''
        Creates the GridSearchCV object and the model, parameters and
        metrics to track are passed. Also the number of cross validations (cv) that we 
        want to split our data into for training. If an error arises, try other number of
        cv.
        '''
        model = GridSearchCV(estimator=InformerWrapper(enc_in=self.enc_in, dec_in=self.dec_in, c_out=self.c_out, 
                                                                     seq_len=self.seq_len, label_len=self.label_len, 
                                                                     out_len=self.pred_len), 
                                           param_grid=param_grid,
                                           scoring = self.metrics,
                                           refit='neg_mean_absolute_error', cv=7)
        return model

    def train(self):
        '''
        Once the model, its metrics to track and its possible parameters
        to try, this function executes the training.
        '''
        param_grid = self.get_param_grid()
        self.set_metrics()
        best_model_informer = self.define_model(param_grid)
        best_model_informer.fit(self.X, self.y)
        print('Training finished')
        
        return best_model_informer
        
    def get_results(self, model):
        '''
        Returns the GridSearchCV results. It is a dictionary.
        '''
        return model.cv_results_

    def save_best_results(self, results):
        '''
        For all the results obtained in the training with GridSearch, this function
        saves the model with best MAE metric, with its metrics, its parameters and its 
        RMSE (other_metric_score). The same happens with the model with the best RMSE. 
        This is saved in a dictionary called best_results. 
        '''
        best_results = {}
        for metric in self.metrics:
            best_index = results[f'rank_test_{metric}'].argmin()
            best_score = round(float(results[f'mean_test_{metric}'][best_index]*-1),2)
            other_metric_score = round(float(results[f'mean_test_{self.opposite_metric[metric]}'][best_index]*-1),2)
            best_params = results['params'][best_index]
            best_model = InformerWrapper(enc_in=self.enc_in, dec_in=self.dec_in, c_out=self.c_out, 
                                         seq_len=self.seq_len, label_len=self.label_len, 
                                         out_len=self.pred_len, **best_params)
            best_model.fit(self.X, self.y)
            if metric == 'neg_mean_absolute_error':
                best_results[metric] = (best_model, best_params, best_score, other_metric_score)
            else:
                best_results[metric] = (best_model, best_params, other_metric_score, best_score)

        self.best_results_to_df(best_results)

    def best_results_to_df(self, best_results):
        '''
        Converts the results obtained in the function save_best_results
        from a dictionary into a pandas DataFrame. Its columns are best_MAE and best RMSE.
        Its rows, their associated model, parameters, their score and the other metric score.
        '''
        self.best_results = pd.DataFrame(best_results)
        self.best_results.rename(columns={'neg_mean_absolute_error': 'best_MAE', 
                                          'neg_root_mean_squared_error': 'best_RMSE'},
                                 inplace=True)
        self.best_results.index = ['model', 'parameters', 'mae', 'rmse']

    def make_predictions(self, metric):
        '''
        With the best models saved in the function best_results_to_df, this function
        is in charge of calling get_pred_df that makes the predictions and passes them to 
        a pandas DataFrame. These predictions are 12 months ahead.
        '''
        best_metric_model = self.best_results.loc['model', metric]
        predictions = mf.get_pred_df(self.train_ts, best_metric_model, informer=True)
        return predictions

    def save_predictions_to_csv(self, predictions, metric):
        '''
        Saves the predictions into a .csv that will be useful in 
        the save_mlflow function to save them as an artifact in mflow.
        '''
        if metric == 'best_MAE':
            predictions.to_csv(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_best_mae.csv')
        else:
            predictions.to_csv(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_best_rmse.csv')
        mf.save_pred_plot(ABS_PATH_PLOT, self.model_name, self.train_ts, predictions, metric) # it does not show the pred because plt.show() is commented.

    def get_current_time(self):
        '''
        Returns the current time to save this information 
        along with the model in mlflow.
        '''
        return datetime.now().strftime('%H:%M:%S %d/%m/%Y')
    
    def init_mlflow_repository(self):
        '''
        Since I created a dagshub repository to run mlflow experiments in a non-local way,
        this line of code connects or initializes this repository.
        '''
        dagshub.init(repo_owner='JCOQUE', repo_name='TFG-ingenieria', mlflow=True)
    
    def mlflow_connect(self):
        '''
        Sets where the experiments info (i.e. mlflow.<whatever> in the next function)
        should be saved (in the dagshub repository initialized in the previous function). 
        It also sets the experiment name.
        '''
        mlflow.set_tracking_uri(uri='https://dagshub.com/JCOQUE/TFG-ingenieria.mlflow')
        mlflow.set_experiment(f' {self.target} Informer')

    def save_mlflow(self):
        '''
        Logs all the important information about the best models obtained
        (for both MAE metric and RMSE metric) in mlflow.
        '''
        current_time = self.get_current_time()
        for metric in self.best_results.columns:
            with mlflow.start_run(run_name =f'{metric}'):
                mlflow.set_tag('model_name', f'{self.model_name}_{metric}')
                mlflow.set_tag('Time', f'{current_time}')
                #self.save_model_to_pickle(metric)
                #mlflow.log_artifact(f"{ABS_PATH_PICKLE_MODELS}/{self.model_name}_{self.target}_{metric}.pkl",
                                    #artifact_path="model")
                mlflow.log_params(self.best_results.loc['parameters', metric])
                mlflow.log_metric('MAE', self.best_results.loc['mae', metric])
                mlflow.log_metric('RMSE', self.best_results.loc['rmse', metric])
                mlflow.log_artifact(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_{metric.lower()}.csv', artifact_path = 'predictions')
                mlflow.log_artifact(f'{ABS_PATH_PLOT}/{self.model_name} {self.target} Prediction {metric.upper()}.png',
                                    artifact_path="plots")

                
    def save_model_to_pickle(self, metric):
        '''
        Since the Informer model is not a pytorch model, nor tensorflow, sklearn and other
        predefined models that mlflow supports, it is needed to save is as a pickle.
        '''
        with open(f"{ABS_PATH_PICKLE_MODELS}/{self.model_name}_{self.target}_{metric}.pkl", "wb") as save_model:
            pickle.dump(self.best_results.loc['model', metric], save_model)


# PREFECT CODE
# Prefect, at the moment, does not allow to use tasks in a class method. 
# That's why this redundant code needs to be made.
@task(task_run_name = 'Setting attributes', log_prints = True, retries = 2)
def set_attributes(informer):
    print('Setting attributes...')
    informer.setting_attributes()

@task(task_run_name = 'Train', log_prints = True, retries = 2)
def train(informer):
    print('Training...')
    return informer.train() 

@task(task_run_name = 'Get results', log_prints = True, retries = 2)
def get_results(informer, best_model_lgbm):
    return informer.get_results(best_model_lgbm)  

@task(task_run_name = 'Save best results', log_prints = True, retries = 2)
def save_best_results(informer, results):
    print('Saving best results...')
    informer.save_best_results(results)

@task(task_run_name = 'Make predictions {model}', log_prints = True, retries = 2)
def make_predictions(informer, model):
    print(f'Making {model} predictions...')
    return informer.make_predictions(model)

@task(task_run_name = 'Save predictions {model}', log_prints = True, retries = 2)
def save_predictions_to_csv(informer, predictions, model):
    print(f'Saving {model} predictions...')
    informer.save_predictions_to_csv(predictions, model)

@task(task_run_name = 'Init mlflow repository', log_prints = True, retries = 2)
def init_mlflow_repository(informer):
    informer.init_mlflow_repository()

@task(task_run_name = 'Connect to mlflow', log_prints = True, retries = 2)
def mlflow_connect(informer):
    print('Connecting to mlflow...')
    informer.mlflow_connect()

@task(task_run_name = 'Save results to mlflow', log_prints = True, retries = 2)
def save_mlflow(informer):
    print('Saving to mlflow...')
    informer.save_mlflow()
 
@flow(flow_run_name='Informer {target}', retries = 2)
def run(target):
        my_informer = MyInformer(target=target)
        set_attributes(my_informer)

        best_model_infomer = train(my_informer)
        results = get_results(my_informer, best_model_infomer)
        save_best_results(my_informer, results)

        predictions_mae = make_predictions(my_informer, 'best_MAE')
        save_predictions_to_csv(my_informer, predictions_mae, 'best_MAE')
        
        predictions_rmse = make_predictions(my_informer, 'best_RMSE')
        save_predictions_to_csv(my_informer, predictions_rmse, 'best_RMSE')

        init_mlflow_repository(my_informer)
        mlflow_connect(my_informer)
        save_mlflow(my_informer)

        return None

if __name__ == '__main__':
    run('Compras')


    