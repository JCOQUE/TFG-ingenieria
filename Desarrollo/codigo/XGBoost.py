import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import mlflow
from mlflow.models.signature import infer_signature
from prefect import flow, task
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

ABS_PATH_CSV = 'C:/Users/jcoqu/OneDrive/Documents/U-tad/Curso5/TFG/TFGinso/Desarrollo/codigo/csv_predictions'
ABS_PATH_PLOT = 'C:/Users/jcoqu/OneDrive/Documents/U-tad/Curso5/TFG/TFGinso/Desarrollo/codigo/pred_plots'


class MyXGBoost:

    def __init__(self, target):
        self.target = target
        self.model_name = 'XGBoost'
        self.ts = None
        self.X = None
        self.y = None
        self.metric = None
        self.opposite_metric = {'neg_mean_absolute_error':'neg_root_mean_squared_error',
                                'neg_root_mean_squared_error':'neg_mean_absolute_error'}
        self.best_results = None

    def setting_attributes(self):
        self.ts = mgts.get_ts(self.target)
        self.X, self.y = mpd.create_features(self.ts.copy(), target = self.target, informer = False)
    
    def create_model(self):
        return xgb.XGBRegressor()

    def get_param_grid(self):
        param_grid = {'max_depth': [3,5,10],
                        'n_estimators': [50, 500, 1000]#,
                        # 'tree_method':['exact', 'approx']
                        # 'learning_rate': [0.1, 0.01],
                        # 'n_estimators': [50, 500, 1000],
                        # 'colsample_bytree': [0.3,  0.7],
                        # 'max_depth': [3, 5, 10],
                        # 'max_leaves': [1,2,3,4,5]
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
        model = self.create_model()
        param_grid = self.get_param_grid()
        cross_val_split = self.get_cross_validation()
        self.set_metrics()
        best_model_xgb = self.define_model(model, param_grid, cross_val_split)

        best_model_xgb.fit(self.X, self.y)
        print('Training completed')

        
        return best_model_xgb
    
    def get_results(self, model):
        return model.cv_results_
    
    def save_best_results(self, results):
        best_results = {}
        for metric in self.metrics:
            best_index = results[f'rank_test_{metric}'].argmin()
            best_score = round(float(results[f'mean_test_{metric}'][best_index]*-1),2)
            other_metric_score = round(float(results[f'mean_test_{self.opposite_metric[metric]}'][best_index]*-1),2)
            best_params = results['params'][best_index]
            best_model = xgb.XGBRegressor(**best_params)
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
        best_metric_model = self.best_results.loc['model', metric]
        predictions = mf.get_pred_df(self.ts, best_metric_model)
        return predictions

    def save_predictions_to_csv(self, predictions, metric):
        print(f'Saving {metric} predictions...')
        if metric == 'best_MAE':
            predictions.to_csv(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_best_mae.csv')
        else:
            predictions.to_csv(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_best_rmse.csv')
        mf.save_pred_plot(self.model_name, self.ts, predictions, metric) # it does not show the pred because plt.show() is commented.

    def get_current_time(self):
        return datetime.now().strftime('%H:%M:%S %d/%m/%Y')
    
    def init_mlflow_repository(self):
        dagshub.init(repo_owner='JCOQUE', repo_name='TFG-ingenieria', mlflow=True) 
    
    def mlflow_connect(self):
        mlflow.set_tracking_uri(uri='https://dagshub.com/JCOQUE/TFG-ingenieria.mlflow')
        mlflow.set_experiment(f'{self.target} XGBoost v1')
        
    def save_mlflow(self):
        current_time = self.get_current_time()
        for metric in self.best_results.columns:
            with mlflow.start_run(run_name =f'{metric}'):
                mlflow.set_tag('model_name', f'{self.model_name}_{metric}')
                mlflow.set_tag('Time', f'{current_time}')
                mlflow_dataset = mlflow.data.from_pandas(self.X.head(1)) # since I log the schema with a row is enough
                mlflow.log_input(mlflow_dataset, context = 'Training features')
                signature = infer_signature(self.X, self.y)
                #https://github.com/mlflow/mlflow/issues/9659
                mlflow.xgboost.log_model(xgb_model = self.best_results.loc['model',metric], 
                                         artifact_path = f'{self.model_name}_{self.target}_{metric}',
                                         model_format = 'json', signature = signature)
                mlflow.log_params(self.best_results.loc['parameters', metric])
                mlflow.log_metric('MAE', self.best_results.loc['mae', metric])
                mlflow.log_metric('RMSE', self.best_results.loc['rmse', metric])
                mlflow.log_artifact(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_{metric.lower()}.csv', artifact_path = 'predictions')
                mlflow.log_artifact(f'{ABS_PATH_PLOT}/{self.model_name} {self.target} Prediction {metric.upper()}.png',
                                    artifact_path="plots")

    # def run(self):
    #     self.setting_attributes()
    #     best_model_lgbm = self.train()
    #     results = self.get_results(best_model_lgbm)
    #     self.save_best_results(results)

    #     predictions_mae = self.make_predictions('best_MAE')
    #     self.save_prediction_to_csv(predictions_mae, 'best_MAE')
        
    #     predictions_rmse = self.make_predictions('best_RMSE')
    #     self.save_prediction_to_csv(predictions_rmse, 'best_RMSE')

    #     self.mlflow_connect()
    #     self.save_mlflow()

    #     return None

# dagshub.init(repo_owner='JCOQUE', repo_name='TFG-ingenieria', mlflow=True)
# my_xgboost = MyXGBoost(target = 'Compras')
# my_xgboost.run()


# Prefect, at the moment, does not allow to use tasks in a class method. 
@task(task_run_name = 'Setting attributes', log_prints = True, retries = 2)
def set_attributes(xgboost):
    print('Setting attributes...')
    xgboost.setting_attributes()

@task(task_run_name = 'Train', log_prints = True, retries = 2)
def train(xgboost):
    print('Training...')
    return xgboost.train() 

@task(task_run_name = 'Get results', log_prints = True)
def get_results(xgboost, best_model_xgboost):
    return xgboost.get_results(best_model_xgboost)  

@task(task_run_name = 'Save best results', log_prints = True)
def save_best_results(xgboost, results):
    print('Saving best results...')
    xgboost.save_best_results(results)

@task(task_run_name = 'Make predictions {model}', log_prints = True)
def make_predictions(xgboost, model):
    print(f'Making {model} predictions...')
    return xgboost.make_predictions(model)

@task(task_run_name = 'Save predictions {model}', log_prints = True)
def save_predictions_to_csv(xgboost, predictions, model):
    print(f'Saving {model} predictions...')
    xgboost.save_predictions_to_csv(predictions, model)

@task(task_run_name = 'Init mlflow repository', log_prints = True)
def init_mlflow_repository(xgboost):
    xgboost.init_mlflow_repository()

@task(task_run_name = 'Connect to mlflow', log_prints = True)
def mlflow_connect(xgboost):
    print('Connecting to mlflow...')
    xgboost.mlflow_connect()

@task(task_run_name = 'Save results to mlflow', log_prints = True)
def save_mlflow(xgboost):
    print('Saving to mlflow...')
    xgboost.save_mlflow()
 
@flow(flow_run_name='XGBoost {target}')
def run(target):
        my_xgboost = MyXGBoost(target = target)
        set_attributes(my_xgboost)

        best_model_lgbm = train(my_xgboost)
        results = get_results(my_xgboost, best_model_lgbm)
        save_best_results(my_xgboost, results)

        predictions_mae = make_predictions(my_xgboost, 'best_MAE')
        save_predictions_to_csv(my_xgboost, predictions_mae, 'best_MAE')
        
        predictions_rmse = make_predictions(my_xgboost, 'best_RMSE')
        save_predictions_to_csv(my_xgboost, predictions_rmse, 'best_RMSE')

        init_mlflow_repository(my_xgboost)
        mlflow_connect(my_xgboost)
        save_mlflow(my_xgboost)

        return None
    

if __name__ == '__main__':
    run('Ventas')



























