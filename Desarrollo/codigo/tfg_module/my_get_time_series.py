from datetime import datetime
import pandas as pd
from azure.storage.blob import BlobServiceClient
import tfg_module.my_dataset_transformations as mdf


def get_ts(target):
    dataset = get_dataset()
    dataset = mdf.transform_dataset(dataset)
    ts = transform_data_to_ts(dataset, target)

    return ts
    

def get_dataset():
    account_name = "blobstoragetfginso"
    account_key = "gd5nuYRJgr/SLkHdH7PIhh72OLQX/kwKuDlF5yO3grgfrrfyFigneBBd5VJPEuYZC6qlgzTBlvBS+AStpXySag=="
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    container_name = "containertfginso1"
    blob_name = "diario.csv"

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    return pd.read_csv(blob_client.download_blob())

   
def transform_data_to_ts(dataset, target):
    dataset['Fecha'] = pd.to_datetime(dataset['Fecha'], format='%d/%m/%Y')
    dataset['Year'] = dataset['Fecha'].dt.year
    dataset['Month'] = dataset['Fecha'].dt.month
    target = target_cleaned(target)
    ts = dataset.groupby(['Year', 'Month']).agg({target: 'sum'})
    dates = get_dates(ts.index[0][0], ts.index[-1][0],
                      ts.index[0][1], ts.index[-1][1]) # [0] = year, [1] month
    ts.set_index(dates, inplace = True)
    ts['date'] = ts.index
    ts.reset_index(drop = True,inplace = True) 

    if target == "Compras":
        #because there are no purchases in the first month of september since the first data is on day 25
        ts.iloc[0,0] = ts[(ts['date'].dt.month == 9) & (ts['date'].dt.year > 2017)]['Compras'].mean()
        # Last month Ventas/compras logic here

    ts = change_order_columns(ts)
         
    return ts 


def target_cleaned(target):
    target = target.lower()
    if target == 'ventas':
        return 'Ventas'
    elif target =='compras':
        return 'Compras'
    else:
        raise ValueError('Incorrect target provided. Target can be either compras or ventas')
    
def get_dates(start_year, end_year, start_month, end_month):
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
    return dataset[[dataset.columns[1], dataset.columns[0]]]