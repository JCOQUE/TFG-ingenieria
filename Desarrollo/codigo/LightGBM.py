import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import lightgbm as lgbm
from datetime import datetime
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.models.signature import infer_signature
from prefect import flow
import dagshub
import sys
import io
import warnings

from tfg_module import my_get_time_series as mgts
from tfg_module import my_process_data as mpd
from tfg_module import my_future as mf

warnings.filterwarnings("ignore", category=UserWarning, module='mlflow.types.utils')

# IMPORTANT!! These lines are needed for the following error when connecting to dagshub:
#UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-2: character maps to <undefined>
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class MyLightGBM:

    def __init__(self, target):
        self.target = target
        self.model_name = 'LightGBM'
        self.ts = None
        self.X = None
        self.y = None
        self.metric = None
        self.opposite_metric = {'neg_mean_absolute_error':'neg_root_mean_squared_error',
                                'neg_root_mean_squared_error':'neg_mean_absolute_error'}
        self.best_results = None

    def setting_attributes(self):
        print('Setting attributes...')
        self.ts = mgts.get_ts(self.target)
        self.X, self.y = mpd.create_features(self.ts.copy(), target = self.target, informer = False)
    
    def create_model(self):
        return lgbm.LGBMRegressor(verbosity = -1)
    
    def get_param_grid(self):
        param_grid = {
            'max_depth': [3,5,10],
            'num_leaves': [10, 20, 30]#,
            # 'learning_rate': [0.1, 0.01],
            # 'n_estimators': [50, 500, 1000],
            # 'colsample_bytree': [0.3,  0.7],
            # 'objective': ['binary'],
            # 'boosting_type': ['rf'],
            # 'num_leaves': [5],
            # 'force_row_wise': [True],
            # 'learning_rate': [0.5],
            # 'metric': ['binary_logloss'],
            # 'bagging_fraction': [0.8],
            # 'feature_fraction': [0.8],
            # 'num_round' = [500]  
        }
        
    
        
        return param_grid
    
    def get_cross_validation(self):
        return TimeSeriesSplit(n_splits=3, test_size=20)
    
    def set_metrics(self):
        self.metrics = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']

    def define_model(self, model, param_grid, cv):
        best_model = GridSearchCV(estimator = model, cv=cv, param_grid=param_grid, 
                                scoring = self.metrics,
                                refit = 'neg_mean_absolute_error',  verbose = False)
        
        return best_model
    
    def train(self):
        print('Training...')
        model = self.create_model()
        param_grid = self.get_param_grid()
        cross_val_split = self.get_cross_validation()
        self.set_metrics()
        best_model_lgbm = self.define_model(model, param_grid, cross_val_split)
       
        best_model_lgbm.fit(self.X, self.y)
        print('Training completed...')

        return best_model_lgbm
    
    def get_results(self, model):
        return model.cv_results_
    
    def save_best_results(self, results):
        print('Saving results...')
        best_results = {}
        for metric in self.metrics:
            best_index = results[f'rank_test_{metric}'].argmin()
            best_score = round(float(results[f'mean_test_{metric}'][best_index]*-1),2)
            other_metric_score = round(float(results[f'mean_test_{self.opposite_metric[metric]}'][best_index]*-1),2)
            best_params = results['params'][best_index]
            best_model = lgbm.LGBMRegressor(**best_params, verbosity = -1)
            best_model.fit(self.X, self.y)
            if metric == 'neg_mean_absolute_error':
                best_results[metric] = (best_model, best_params, best_score, other_metric_score)
            else:
                best_results[metric] = (best_model, best_params, other_metric_score, best_score)

        self.best_results_to_df(best_results)

    def best_results_to_df(self, best_results):
        self.best_results = pd.DataFrame(best_results)
        self.best_results.rename(columns = {'neg_mean_absolute_error':'best_MAE', 'neg_root_mean_squared_error':'best_RMSE'}, 
                                inplace = True) 
        self.best_results.index = ['model', 'parameters', 'mae', 'rmse']   

    def make_predictions(self, metric):
        print(f'Making {metric} predictions...')
        best_metric_model = self.best_results.loc['model', metric]
        predictions = mf.get_pred_df(self.ts, best_metric_model)
        return predictions

    def save_prediction_to_csv(self, predictions, metric):
        print(f'Saving {metric} predictions...')
        if metric == 'best_MAE':
            predictions.to_csv(f'csv_predictions/{self.model_name}_{self.target}_best_mae.csv')
        else:
            predictions.to_csv(f'csv_predictions/{self.model_name}_{self.target}_best_rmse.csv')
        mf.save_pred_plot(self.model_name, self.ts, predictions, metric) # it does not show the pred because plt.show() is commented.
                                                   

    def get_current_time(self):
        return datetime.now().strftime('%H:%M:%S %d/%m/%Y')
    
    def ini_mlflow_reporitory(self):
        dagshub.init(repo_owner='JCOQUE', repo_name='TFG-ingenieria', mlflow=True) 
    
    def mlflow_connect(self):
        print('Connecting to mlflow...')
        mlflow.set_tracking_uri(uri='https://dagshub.com/JCOQUE/TFG-ingenieria.mlflow')
        mlflow.set_experiment(f'{self.target} LightGBM')

    def save_mlflow(self):
        print('Saving to mlflow...')
        current_time = self.get_current_time()
        for metric in self.best_results.columns:
            with mlflow.start_run(run_name =f'{metric}'):
                mlflow.set_tag('model_name', f'{self.model_name}_{metric}')
                mlflow.set_tag('Time', f'{current_time}')
                mlflow_dataset = mlflow.data.from_pandas(self.X.head(1)) # since I log the schema with a row is enough
                mlflow.log_input(mlflow_dataset, context = 'Training features')
                signature = infer_signature(self.X, self.y)
                mlflow.lightgbm.log_model(lgb_model = self.best_results.loc['model',metric], 
                                         artifact_path = f'{self.model_name}_{self.target}_{metric}',
                                         signature = signature)
                mlflow.log_params(self.best_results.loc['parameters', metric])
                mlflow.log_metric('MAE', self.best_results.loc['mae', metric])
                mlflow.log_metric('RMSE', self.best_results.loc['rmse', metric])
                mlflow.log_artifact(f'csv_predictions/{self.model_name}_{self.target}_{metric.lower()}.csv',
                                    artifact_path = 'predictions')
                mlflow.log_artifact(f'pred_plots/{self.model_name} {self.target} Prediction {metric.upper()}.png',
                    artifact_path="plots")
                
                
    def run(self):
        self.setting_attributes()
        best_model_lgbm = self.train()
        results = self.get_results(best_model_lgbm)
        self.save_best_results(results)

        predictions_mae = self.make_predictions('best_MAE')
        self.save_prediction_to_csv(predictions_mae, 'best_MAE')
        
        predictions_rmse = self.make_predictions('best_RMSE')
        self.save_prediction_to_csv(predictions_rmse, 'best_RMSE')

        self.init_mlflow_reporitory()
        self.mlflow_connect()
        self.save_mlflow()

        return None
        
@flow
def hello_world():
    return print('hello worldd')

if __name__ == '__main__':
    # my_lgbm = MyLightGBM(target = 'Ventas')
    # my_lgbm.run() 
    print('hello')
    hello_world()

        

        

        

