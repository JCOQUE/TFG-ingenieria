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

ABS_PATH_CSV = 'C:/Users/jcoqu/OneDrive/Documents/U-tad/Curso5/TFG/TFG_ingenieria/Desarrollo/codigo/csv_predictions'
ABS_PATH_PLOT = 'C:/Users/jcoqu/OneDrive/Documents/U-tad/Curso5/TFG/TFG_ingenieria/Desarrollo/codigo/pred_plots'

'''
NOTE: In this same folder you have a .ipynb notebook where you can follow in an easier way the core of this code.
'''

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
        '''
        Initializes the correct attributes for the model. These are: getting 
        the time series to work with, as well as setting the features as 
        the training input and the target as the training output.
        '''
        self.target = mgts.target_cleaned(self.target)
        self.ts = mgts.get_ts(self.target)
        self.X, self.y = mpd.create_features(self.ts.copy(), target = self.target, informer = False)
    
    def create_model(self):
        return xgb.XGBRegressor()

    def get_param_grid(self):
        '''
        Returns all possible parameters values that the GridSearchCV method. 
        It will try all possible combinations.
        '''
        param_grid = {'max_depth': [3,5,10],
                        'n_estimators': [50, 500, 1000]#,
                        # 'tree_method':['exact', 'approx']
                        # 'learning_rate': [0.1, 0.01],
                        # 'n_estimators': [50, 500, 1000],
                        # 'colsample_bytree': [0.3,  0.7],
                        # 'max_depth': [3, 5, 10],
                        # 'max_leaves': [1,2,3,4,5],
                        # 'random_state':[None]
                    }
                            
        return param_grid
    
    
    def get_cross_validation(self):
        '''
        Creates the necessary splits for the cross validation
        method applied in the GridSearchCV method from sklearn.
        '''
        return TimeSeriesSplit(n_splits=3, test_size=20)
    
    def set_metrics(self):
        '''
        Sets the metrics that will be tracked during the training. In this case, 
        negative MAE, and negative RMSE. Later on, in save_best_results function, 
        these metrics are converted to positive (i.e. how they should be).
        '''
        self.metrics = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']

    def define_model(self, model, param_grid, cv):
        '''
        Creates the GridSearchCV object and the model, parameters and
        metrics to track are passed. Also the number of cross validations (cv) that we 
        want to split our data into for training.
        '''
        best_model = GridSearchCV(estimator = model, cv=cv, param_grid=param_grid, 
                                scoring = self.metrics,
                                refit = 'neg_mean_absolute_error',  verbose = False)
        
        return best_model

    def train(self):
        '''
        Once the model, its metrics to track and its possible parameters
        to try, this function executes the training.
        '''
        model = self.create_model()
        param_grid = self.get_param_grid()
        cross_val_split = self.get_cross_validation()
        self.set_metrics()
        best_model_xgb = self.define_model(model, param_grid, cross_val_split)

        best_model_xgb.fit(self.X, self.y)
        print('Training completed')

        
        return best_model_xgb
    
    def get_results(self, model):
        '''
        Returns the GridSearchCV results in a dictionary.
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
            best_model = xgb.XGBRegressor(**best_params)
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
        self.best_results.rename(columns = {'neg_mean_absolute_error':'best_MAE', 
                                            'neg_root_mean_squared_error':'best_RMSE'}, 
                                inplace = True)   
        self.best_results.index = ['model', 'parameters', 'mae', 'rmse'] 

    def make_predictions(self, metric):
        '''
        With the best models saved in the function best_results_to_df, this function
        is in charge of of calling get_pred_df that makes the predictions and passes them to 
        a pandas DataFrame.
        '''
        best_metric_model = self.best_results.loc['model', metric]
        predictions = mf.get_pred_df(self.ts, best_metric_model)
        return predictions

    def save_predictions_to_csv(self, predictions, metric):
        '''
        Saves the predictions into a .csv that will be useful in 
        the save_mlflow function to save them as an artifact in mflow.
        '''
        print(f'Saving {metric} predictions...')
        if metric == 'best_MAE':
            predictions.to_csv(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_best_mae.csv')
        else:
            predictions.to_csv(f'{ABS_PATH_CSV}/{self.model_name}_{self.target}_best_rmse.csv')
        mf.save_pred_plot(self.model_name, self.ts, predictions, metric) # it does not show the pred because plt.show() is commented.

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
        mlflow.set_experiment(f' {self.target} XGBoost')
        
    def save_mlflow(self):
        '''
        This functions logs all the important information about the best models obtained
        (for both MAE metric and RMSE metric) in mlflow.
        '''
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
                

# PREFECT CODE
# Prefect, at the moment, does not allow to use tasks in a class method. 
# That's why this redundant code needs to be made.
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
    run('Compras')



























