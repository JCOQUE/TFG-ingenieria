import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tfg_module.my_process_data as mpd

# This line solves the following warning:
# UserWarning: Starting a Matplotlib GUI outside of the main thread will likely fail.
# https://stackoverflow.com/questions/69924881/userwarning-starting-a-matplotlib-gui-outside-of-the-main-thread-will-likely-fa
matplotlib.use('Agg')


def get_future_features(dataset, informer = False):
    '''
    Returns the input (X) features needed for prediction
    '''
    dates = get_future_dates(dataset)
    df_future = pd.DataFrame({'date': dates})
    df_future = mpd.create_features(df_future, informer = informer)
    return df_future

def get_future_dates(ts):
    '''
    Returns the months for which the prediction will be made.
    '''
    start_pred_date = ts.tail(1)['date'].values[0]
    future_dates = pd.date_range(start=start_pred_date, periods=12, freq='ME')
    return future_dates

def get_pred_df(ts, model, informer = False):
    '''
    Creates a dataframe with two columns. A date column
    with the dates for the prediction (12 months ahead) and a pred 
    columns with the values predicted for each month.
    '''
    X_pred = get_future_features(ts, informer)
    df = pd.DataFrame()
    df['date'] = get_future_dates(ts)
    df['pred'] = np.around(model.predict(X_pred),2)
    df = concatante_past_future_values(ts, df)
    return df

def concatante_past_future_values(ts, df):
    '''
    Concatanates the last time series value with 
    the first predicted value.
    '''
    df.loc[df.index[0], 'pred'] = ts[ts.columns[1]].iloc[-1]
    return df

def save_pred_plot(model_name, ts, df_pred, metric):
    '''
    Plots the prediction results along with the time series (ts)
    itself using the matplotlib library for visual results. The absolute path 
    is needed if using Prefect.
    '''
    ABS_PATH_PLOT = 'C:/Users/jcoqu/OneDrive/Documents/U-tad/Curso5/TFG/TFG_ingenieria/Desarrollo/codigo/pred_plots'
    try:
        plt.plot(ts['date'], ts[ts.columns[1]], label = 'Actual', color = 'blue')
        plt.plot(df_pred['date'], df_pred['pred'], label = 'Predicted', color = 'orange')
        plt.title(f'{model_name} {metric.upper()}')
        plt.xlabel('Years')
        plt.ylabel(ts.columns[1])
        plt.box(False)
        plt.grid(False)
        plt.legend()
        plt.savefig(f'{ABS_PATH_PLOT}/{model_name} {ts.columns[1]} Prediction {metric.upper()}.png')
        plt.close()
        #plt.show()
    except Exception as e:
        print(f'Unexpected error: {e}')





