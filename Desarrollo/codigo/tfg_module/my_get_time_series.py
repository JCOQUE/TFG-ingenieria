from datetime import datetime
import pandas as pd
from azure.storage.blob import BlobServiceClient
import tfg_module.my_dataset_transformations as mdf
import tfg_module.my_azure as maz


def get_ts(target, type = 'train'):
    '''
    Returns the time series that the model needs to train
    and predict.
    '''
    dataset = maz.get_dataset(type)
    dataset = mdf.transform_dataset(dataset)
    ts = transform_data_to_ts(dataset, target, type)

    return ts

   
def transform_data_to_ts(dataset, target, type):
    '''
    This function transforms the dataset into a time series grouped by year and
    then month.
    '''
    dataset['Fecha'] = pd.to_datetime(dataset['Fecha'], format='%d/%m/%Y')
    month_prct_estimation = month_prct_target_estimation(dataset.copy(), target)
    dataset.sort_values('Fecha', ascending = True, inplace = True)
    dataset['Year'] = dataset['Fecha'].dt.year
    dataset['Month'] = dataset['Fecha'].dt.month
    ts = dataset.groupby(['Year', 'Month']).agg({target: 'sum'})
    dates = get_dates(ts.index[0][0], ts.index[-1][0],
                      ts.index[0][1], ts.index[-1][1]) # [0] = year, [1] month
    ts.set_index(dates, inplace = True)
    ts['date'] = ts.index
    ts.reset_index(drop = True,inplace = True) 
    if target == "Compras" and type == 'train':
        #because there are no purchases in the first month of september since the first data is on day 25
        ts.iloc[0,0] = ts[(ts['date'].dt.month == 9) & (ts['date'].dt.year > 2017)]['Compras'].mean()

    ts = change_order_columns(ts)
    ts = set_estimation_logic(ts, target, month_prct_estimation)   
    ts[target] = ts[target].round(2).astype('float32')
    return ts 

def month_prct_target_estimation(dataset, target):
    '''
    The logic behind this and the related functions is as follows:
    Since I am grouping by Year and Month, if the most recent day that 
    the dataset has record of is very recent, e.g. 5th, when the grouping
    is done, it will set a very low and unrealistic target value 
    for the most recent month. What I want to do is:
        - If the day is lower that a limit day (see function get_target_prct_estimation())
          then the most recent month will be predicted instead of grouped.
        - If the day is above the limit, let's say 20th, if the group by is performed, the 
          value is still unrealistic because the month hasn't ended yet. Therefore, I calculate,
          based on the past years, the percentage of the target value in that month up until that 
          that (20th in this case) compared to the total target value at the end of the month. This logic
          assumes that if the past years, up until the day 20th of a month, it represented the 80% of the
          sales value at the end of the month, then this year, the same pattern will occur for this month. 
    This prct is what this function returns in case the day is above the limit. Otherwise it returns -1, 
    and the most recent month will be considered too recent and will be predicted. See the function 
    set_estimation_logic() where this logic is applied.
    '''
    actual_day, actual_month, actual_year = get_actual_date_target(dataset, target)
    prct_estimation = get_target_prct_estimation(dataset, target, actual_day, actual_month, actual_year)
    return prct_estimation

def get_filtered_target(target):
    '''
    Filters the target to only get the desired one.
    '''
    filter_condition = {'Compras': '600',
                        'Ventas':'700'}
    return filter_condition[target]

def get_actual_date_target(dataset, target):
    '''
    Note: read first explanation from the month_prct_target_estimation function above.

    Returns the last day, its month and year for the last record of the target in the 
    dataset.
    '''
    actual_day = dataset[dataset['NoGrupo'] == (get_filtered_target(target))]\
        .tail(1)['Fecha'].iloc[0].day
    actual_month = dataset[dataset['NoGrupo'] == (get_filtered_target(target))]\
        .tail(1)['Fecha'].iloc[0].month
    actual_year = dataset[dataset['NoGrupo'] == (get_filtered_target(target))]\
        .tail(1)['Fecha'].iloc[0].year
    
    return actual_day, actual_month, actual_year

def  get_target_prct_estimation(dataset, target, actual_day, actual_month, actual_year):
    '''
    Read first explanation from the month_prct_target_estimation function above.

    This function sets a limit day to take into account in the time series the
    most actual month that the dataset has. For example, this function can set that
    if the most actual day in the most actual month that the dataset has is 4 or below, 
    then this month will not be taken into account in the present data, and will be predicted
    instead.
    If the day exceeds the limit_day, it returns the corresponding prct. Otherwise -1 indicating
    that this month will be predicted. This logic is done in the set_estimation_logic() below.
    '''
    LIMIT_DAY = 15
    dataset = dataset[dataset['NoGrupo'] == (get_filtered_target(target))]
    if actual_day < LIMIT_DAY:
        return -1
    else:
        total_sum_month = get_total_sum_month(dataset, target, actual_month, actual_year)
        total_sum_day = get_total_sum_day(dataset, target, actual_day,  actual_month, actual_year)
        prct_rounded = round(total_sum_day/total_sum_month,2)
        return prct_rounded

def get_total_sum_month(dataset, target, actual_month, actual_year):
    '''
    Note: read first explanation from the month_prct_target_estimation function above.

    It returns the total sum of the target (Compras or Ventas) for the most recent
    month that the dataset has. It does not count the actual year.
    '''
    return dataset[(dataset['Fecha'].dt.year < actual_year) & (dataset['Fecha'].dt.month == actual_month)][target].sum()

def get_total_sum_day(dataset, target, actual_day, actual_month, actual_year):
    '''
    Note: read first explanation from the month_prct_target_estimation function above.

    It returns the total sum of the target (Compras or Ventas) for the most actual month
    that the dataset has, but only until the most actual day that the dataset has.
    Example: If the last record of Ventas is May 7th, it returns the sum of Ventas for 
    every May in every year (except the actual year) up until the day 7th.
    '''
    return dataset[(dataset['Fecha'].dt.year < actual_year) & (dataset['Fecha'].dt.month == actual_month) & (dataset['Fecha']\
                    .dt.day <= actual_day)][target].sum()
    
def target_cleaned(target):
    '''
    Making the input case-insensitive.
    '''
    target = target.lower()
    if target == 'ventas':
        return 'Ventas'
    elif target =='compras':
        return 'Compras'
    else:
        raise ValueError('Incorrect target provided. Target can be either compras or ventas.')
    
def get_dates(start_year, end_year, start_month, end_month):
    '''
    Since I am grouping by year and month, this function sets the day of the month to the end.
    '''
    month_days = {1:'31',
        2:'28',
        3:'31', 
        4:'30',
        5:'31',
        6:'30', 
        7:'31',
        8:'31',
        9:'30',
        10:'31',
        11:'30',
        12:'31'}
    day_start = month_days[start_month]
    day_end = month_days[end_month]
    start_date = datetime.strptime(f'{day_start}/{start_month}/{start_year}', '%d/%m/%Y')
    end_date = datetime.strptime(f'{day_end}/{end_month}/{end_year}', '%d/%m/%Y')
    return pd.date_range(start = start_date,
                     end = end_date, freq = 'ME') #Warning: M is deprecated and will be removed in future versions. Use ME instead

def change_order_columns(dataset):
    '''
    The dataset received has two columns: date and target (the latter can be either Ventas 
    or Compras). It swaps the columns so the date is first (i.e. in the left side)
    '''
    return dataset[[dataset.columns[1], dataset.columns[0]]]

def set_estimation_logic(ts, target, prct_estimation):
    '''
    Note: read first explanation from the month_prct_target_estimation function above.
    '''
    if prct_estimation == -1:
        ts = ts.iloc[:-1]
    else:
        ts.loc[ts.index[-1], target]= ts[target].iloc[-1] / prct_estimation

    return ts