import pandas as pd
import torch.nn
from darts.models import TCNModel
from darts import TimeSeries
from darts.metrics.metrics import mae, rmse
from darts.utils.callbacks import TFMProgressBar
import mlflow
from prefect import flow, task
import dagshub
import pickle
import sys
import io
from datetime import datetime

import logging
import warnings

from tfg_module import my_get_time_series as mgts
from tfg_module import my_future as mf


logging.getLogger('pytorch_lightning').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')
prog_bar = TFMProgressBar(enable_train_bar = False, 
                          enable_prediction_bar = False,
                          enable_validation_bar = False,
                          enable_sanity_check_bar = False)

# IMPORTANT!! These lines are needed for the following error when connecting to dagshub:
#UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-2: character maps to <undefined>
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

ABS_PATH_CSV = 'C:/Users/jcoqu/OneDrive/Documents/U-tad/Curso5/TFG/TFG_ingenieria/Desarrollo/codigo/csv_predictions'
ABS_PATH_PLOT = 'C:/Users/jcoqu/OneDrive/Documents/U-tad/Curso5/TFG/TFG_ingenieria/Desarrollo/codigo/pred_plots'
ABS_PATH_PICKLE_MODELS = 'C:/Users/jcoqu/OneDrive/Documents/U-tad/Curso5/TFG/TFG_ingenieria/Desarrollo/codigo/pickle_models'

'''
NOTE: In this same folder you have a .ipynb notebook where you can follow in an easier way the core of this code.
'''

class MyTCN:
    def __init__(self, target):
        self.target = target
        self.model_name = 'TCN'
        self.ts_df = None
        self.ts = None
        self.best_results = None
        self.mae_pred_df = None
        self.mae_pred_df = None

    def setting_attributes(self):
        '''
        Gets the time series to work with and passes it to a 
        TimeSeries object to work with the TCN Darts library.
        '''
        self.ts_df = mgts.get_ts(self.target)
        self.ts = TimeSeries.from_dataframe(self.ts_df.copy(), time_col='date', value_cols=[self.target])
    
    def get_train_test_index(self, ts):
        '''
        Returns the index by which the time series is splitted in train 
        and test.
        '''
        return int(len(ts) * 0.8)

    def get_train_test(self):
        '''
        Since the library Darts has its own GridSearch method (it does not
        use the GridSearchCV method from the library sklearn), instead of
        cross validations, train test split is performed.
        '''
        split_index = self.get_train_test_index(self.ts)
        train = self.ts[:split_index]
        test = self.ts[split_index:]
        return train, test
    
    
    def create_model(self):
        '''
        It creates a TCN model from the Darts library. The parameters
        will be overwritten in the train function.
        '''
        tcn_model = TCNModel(
            input_chunk_length=7,
            output_chunk_length=3,
        )

        return tcn_model
    
    
    def get_param_grid(self):
        '''
        Returns all possible parameters values that the GridSearchCV method. 
        It will try all possible combinations.
        '''
        param_grid = dict(input_chunk_length = [4,8,12], 
                           output_chunk_length = [1, 2, 3],
                            n_epochs = [10],
                            dropout = [0.2],
                            dilation_base = [2],
                            weight_norm = [True],
                            kernel_size = [3],
                            num_filters = [6],
                            batch_size = [32], 
                            loss_fn = [torch.nn.MSELoss()],
                            optimizer_cls = [torch.optim.Adam],
                            optimizer_kwargs = [{'lr': 0.1}, {'lr': 0.01}],
                            random_state = [None],
                            pl_trainer_kwargs=[{"callbacks": [prog_bar]}])
        
        return param_grid
    
    
    def train(self, train, test):
        '''
        Since the the .gridsearch method from Darts library cannot
        track two metrics when training (like GridSearchCV from sklearn does),
        two models are needed to train: one for MAE and another one for RMSE.
        '''
        tcn_model = self.create_model()
        param_grid = self.get_param_grid()
        
        mae_model = tcn_model.gridsearch(parameters=param_grid, 
                                            series= train, 
                                            val_series= test,  metric = mae)
        rmse_model = tcn_model.gridsearch(parameters=param_grid, 
                                            series= train, 
                                            val_series= test,  metric = rmse)
        print('Training completed')
        
        return mae_model, rmse_model
    

    def save_best_results(self, train, test, mae_model, rmse_model):
        '''
        For all the results obtained in the training with GridSearch, this function
        saves the model with best MAE metric, with its metrics, its parameters and its 
        RMSE (other_metric_score). The same happens with the model with the best RMSE. 
        This is saved in a dictionary called best_results. 
        '''
        modelos = {'best_MAE':mae_model, 'best_RMSE': rmse_model}
        best_results = {}
        for metrica, MODELO in modelos.items():
            best_score = round(float(MODELO[2]),2)
            best_model = MODELO[0]
            best_model.fit(train, val_series = test, verbose = False)
            best_params = MODELO[1]
            if metrica == 'best_MAE':
                rmse_in_mae_model = round(float(rmse(mae_model[0].predict(n =len(test), series = train), test)),2)
                best_results[metrica] = (best_model, best_params, best_score, rmse_in_mae_model)
            else:
                mae_in_rmse_model = round(float(mae(rmse_model[0].predict(n =len(test), series = train), test)),2)
                best_results[metrica] = (best_model, best_params, mae_in_rmse_model, best_score)

        self.best_results_to_df(best_results)

    
    def best_results_to_df(self, best_results):
        '''
        Converts the results obtained in the function save_best_results
        from a dictionary into a pandas DataFrame. Its columns are best_MAE and best RMSE.
        Its rows, their associated model, parameters, their score and the other metric score.
        '''
        self.best_results = pd.DataFrame(best_results)
        self.best_results.index = ['model', 'parameters', 'mae', 'rmse']

    def make_predictions(self, metric):
        '''
        With the best models saved in the function best_results_to_df, this function
        is in charge of calling get_pred_df that makes the predictions and passes them to 
        a pandas DataFrame. These predictions are 12 months ahead.
        This function, unlike the others, does not use mf.get_pred_df function because
        the library Darts is a bit different.
        '''
        best_metric_model = self.best_results.loc['model',metric]
        predictions = best_metric_model.predict(n= 12, series = self.ts[:-1])
        return predictions
    
    def predictions_to_df(self, predictions):
        '''
        Since Darts is a bit different library from the other ones used for
        the other models, an additional functions is needed to convert a 
        Darts TimeSeries object into a pandas DataFrame.
        '''
        pred_df = predictions.pd_dataframe()
        pred_df.columns.name = None # Needed because darts sets this name to component by default
        pred_df.rename(columns = {self.target:'pred'}, inplace = True)
        pred_df['pred'] = pred_df['pred'].round(2)
        pred_df.reset_index(inplace = True)
        pred_df.loc[pred_df.index[0], 'pred'] = self.ts_df[self.ts_df.columns[1]].iloc[-1]
        return pred_df
    
    def save_predictions_to_csv(self, predictions, metric):
        '''
        Saves the predictions into a .csv that will be useful in 
        the save_mlflow function to save them as an artifact in mflow.
        '''
        if metric == 'best_MAE':
            predictions.to_csv(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_best_mae.csv')
        else:
            predictions.to_csv(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_best_rmse.csv')
        mf.save_pred_plot(self.model_name, self.ts_df, predictions, metric) # it does not show the pred because plt.show() is commented.

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
        mlflow.set_experiment(f' {self.target} TCN')
        
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
                mlflow_dataset = mlflow.data.from_pandas(self.ts_df.head(1)) # since I log the schema with a row is enough
                mlflow.log_input(mlflow_dataset, context = 'Input')
                self.save_model_to_pickle(metric)
                mlflow.log_artifact(f"{ABS_PATH_PICKLE_MODELS}/{self.model_name}_{self.target}_{metric}.pkl",
                                    artifact_path="model")
                mlflow.log_params(self.best_results.loc['parameters', metric])
                mlflow.log_metric('MAE', self.best_results.loc['mae', metric])
                mlflow.log_metric('RMSE', self.best_results.loc['rmse', metric])
                mlflow.log_artifact(f'{ABS_PATH_PLOT}/{self.model_name} {self.target} Prediction {metric.upper()}.png',
                                    artifact_path="plots")
                mlflow.log_artifact(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_{metric.lower()}.csv',
                                    artifact_path = 'predictions')

    def save_model_to_pickle(self, metric):
        '''
        Since the TCN model from the Darts library is not a pytorch model, nor tensorflow,
        sklearn and other predefined models that mlflow supports, it is needed to save is as a pickle.
        '''
        with open(f"{ABS_PATH_PICKLE_MODELS}/{self.model_name}_{self.target}_{metric}.pkl", "wb") as save_model:
            pickle.dump(self.best_results.loc['model', metric], save_model)


# PREFECT CODE
# Prefect, at the moment, does not allow to use tasks in a class method. 
# That's why this redundant code needs to be made.
@task(task_run_name = 'Setting attributes', log_prints = True, retries = 2)
def set_attributes(tcn):
    print('Setting attributes...')
    tcn.setting_attributes()

@task(task_run_name = 'Get train test', log_prints = True, retries = 2)
def get_train_test(tcn):
    return tcn.get_train_test()

@task(task_run_name = 'Train', log_prints = True, retries = 2)
def fit(tcn, train, test):
    print('Training...')
    return tcn.train(train, test)  

@task(task_run_name = 'Save best results', log_prints = True, retries = 2)
def save_best_results(tcn, train, test, best_mae_model, best_rmse_model):
    print('Saving best results...')
    tcn.save_best_results(train, test, best_mae_model, best_rmse_model)

@task(task_run_name = 'Make predictions {model}', log_prints = True, retries = 2)
def make_predictions(tcn, model):
    print(f'Making {model} predictions...')
    return tcn.make_predictions(model)

def predictions_to_df(tcn, predictions):
    return tcn.predictions_to_df(predictions)

@task(task_run_name = 'Save predictions {model}', log_prints = True, retries = 2)
def save_predictions_to_csv(tcn, predictions, model):
    print(f'Saving {model} predictions...')
    tcn.save_predictions_to_csv(predictions, model)

@task(task_run_name = 'Init mlflow repository', log_prints = True, retries = 2)
def init_mlflow_repository(tcn):
    tcn.init_mlflow_repository()

@task(task_run_name = 'Connect to mlflow', log_prints = True, retries = 2)
def mlflow_connect(tcn):
    print('Connecting to mlflow...')
    tcn.mlflow_connect()

@task(task_run_name = 'Save results to mlflow', log_prints = True, retries = 2)
def save_mlflow(tcn):
    print('Saving to mlflow...')
    tcn.save_mlflow()
 
@flow(flow_run_name='TCN {target}')
def run(target):
        my_tcn = MyTCN(target=target) 
        set_attributes(my_tcn)

        train, test = get_train_test(my_tcn)
        mae_model, rmse_model = fit(my_tcn, train, test)
        save_best_results(my_tcn, train, test, mae_model, rmse_model)

        mae_predictions = make_predictions(my_tcn, 'best_MAE')
        predictions_df = predictions_to_df(my_tcn, mae_predictions)
        save_predictions_to_csv(my_tcn, predictions_df, 'best_MAE')

        rmse_predictions = make_predictions(my_tcn, 'best_RMSE')
        predictions_df = predictions_to_df(my_tcn, rmse_predictions)
        save_predictions_to_csv(my_tcn, predictions_df, 'best_RMSE')

        init_mlflow_repository(my_tcn)
        mlflow_connect(my_tcn)
        save_mlflow(my_tcn)

        return None


if __name__ == '__main__':
    run('Compras')