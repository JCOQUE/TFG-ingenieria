import pandas as pd
from darts.models import TCNModel
from darts import TimeSeries
from darts.metrics.metrics import mae, rmse
from darts.utils.callbacks import TFMProgressBar
import mlflow
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
        print('Setting attributes...')
        self.ts_df = mgts.get_ts(self.target)
        self.ts = TimeSeries.from_dataframe(self.ts_df.copy(), time_col='date', value_cols=[self.target])
    
    def get_train_test_index(self, ts):
        return int(len(ts) * 0.8)

    def get_train_test(self):
        split_index = self.get_train_test_index(self.ts)
        train = self.ts[:split_index]
        test = self.ts[split_index:]
        return train, test
    
    
    def create_model(self):
        #params will be overwritten by params_grid
        tcn_model = TCNModel(
            input_chunk_length=7,
            output_chunk_length=3,
        )

        return tcn_model
    
    
    def get_param_grid(self):
        param_grid = dict(input_chunk_length = [4,8,12], 
                           output_chunk_length = [1, 2, 3],
                            n_epochs = [10],
                            dropout = [0.2],
                            dilation_base = [2],
                            weight_norm = [True],
                            kernel_size = [3],
                            num_filters = [6],
                            random_state = [0],
                            batch_size = [32], 
                            pl_trainer_kwargs=[{"callbacks": [prog_bar]}])
        
        return param_grid
    
    
    def train(self, train, test):
        print('Training...')
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
        print('Saving best results...')
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
        self.best_results = pd.DataFrame(best_results)
        self.best_results.index = ['model', 'parameters', 'mae', 'rmse']

    def make_predictions(self, metric):
        print(f'Making {metric} predictions...')
        best_metric_model = self.best_results.loc['model',metric]
        predictions = best_metric_model.predict(n= 12, series = self.ts[:-1])
        return predictions
    
    def predictions_to_df(self, predictions):
        pred_df = predictions.pd_dataframe()
        pred_df.columns.name = None # Needed because darts sets this name to component by default
        pred_df.rename(columns = {self.target:'pred'}, inplace = True)
        pred_df['pred'] = pred_df['pred'].round(2)
        pred_df.reset_index(inplace = True)
        pred_df.loc[pred_df.index[0], 'pred'] = self.ts_df[self.ts_df.columns[1]].iloc[-1]
        return pred_df
    
    def save_predictions_to_csv(self, predictions, metric):
        print(f'Saving {metric} predictions...')
        if metric == 'best_MAE':
            predictions.to_csv(f'csv_predictions/{self.model_name}_{self.target}_best_mae.csv')
        else:
            predictions.to_csv(f'csv_predictions/{self.model_name}_{self.target}_best_rmse.csv')
        mf.save_pred_plot(self.model_name, self.ts_df, predictions, metric) # it does not show the pred because plt.show() is commented.

    def get_current_time(self):
        return datetime.now().strftime('%H:%M:%S %d/%m/%Y')
    
    def mlflow_connect(self):
        print('Connecting to mlflow...')
        mlflow.set_tracking_uri(uri='https://dagshub.com/JCOQUE/TFG-ingenieria.mlflow')
        mlflow.set_experiment(f'{self.target} TCN')
        
    def save_mlflow(self, input, output):
        print('Saving to mlflow...')
        current_time = self.get_current_time()
        for metric in self.best_results.columns:
            with mlflow.start_run(run_name =f'{metric}'):
                mlflow.set_tag('model_name', f'{self.model_name}_{metric}')
                mlflow.set_tag('Time', f'{current_time}')
                mlflow_dataset = mlflow.data.from_pandas(input.head(1)) # since I log the schema with a row is enough
                mlflow.log_input(mlflow_dataset, context = 'Input')
                self.save_model_to_pickle(metric)
                mlflow.log_artifact(f"pickle_models/{self.model_name}_{self.target}_{metric}.pkl",
                                    artifact_path="model")
                mlflow.log_params(self.best_results.loc['parameters', metric])
                mlflow.log_metric('MAE', self.best_results.loc['mae', metric])
                mlflow.log_metric('RMSE', self.best_results.loc['rmse', metric])
                mlflow.log_artifact(f'pred_plots/{self.model_name} {self.target} Prediction {metric.upper()}.png',
                                    artifact_path="plots")
                mlflow.log_artifact(f'csv_predictions/{self.model_name}_{self.target}_{metric.lower()}.csv', artifact_path = 'predictions')

    def save_model_to_pickle(self, metric):
        with open(f"pickle_models/{self.model_name}_{self.target}_{metric}.pkl", "wb") as save_model:
            pickle.dump(self.best_results.loc['model', metric], save_model)


    def run(self):
        self.setting_attributes()
        train, test = self.get_train_test()
        mae_model, rmse_model = self.train(train, test)
        self.save_best_results(train,test, mae_model, rmse_model)

        mae_predictions = self.make_predictions('best_MAE')
        predictions_df = self.predictions_to_df(mae_predictions)
        self.save_predictions_to_csv(predictions_df, 'best_MAE')

        rmse_predictions = self.make_predictions('best_RMSE')
        predictions_df = self.predictions_to_df(rmse_predictions)
        self.save_predictions_to_csv(predictions_df, 'best_RMSE')

        self.mlflow_connect()
        self.save_mlflow(self.ts_df, predictions_df)

        return None

dagshub.init(repo_owner='JCOQUE', repo_name='TFG-ingenieria', mlflow=True)
my_tcn = MyTCN(target='Compras')
my_tcn.run()