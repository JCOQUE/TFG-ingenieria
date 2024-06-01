import pandas as pd
from azure.storage.blob import BlobServiceClient
import os
from tfg_module import my_dataset_transformations as mdf
'''
This .py is in charge of pulling and pushing files from Azure Blob Storge
'''

'''
Guardar credenciales en Prefect??????
'''
ACCOUNT_NAME = "blobstoragetfginso"
ACCOUNT_KEY = "gd5nuYRJgr/SLkHdH7PIhh72OLQX/kwKuDlF5yO3grgfrrfyFigneBBd5VJPEuYZC6qlgzTBlvBS+AStpXySag=="
CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
CONTAINER_NAME = "containertfginso1"
def get_dataset(type = 'train'):
    '''
    Fetches via API the raw .csv from Azure Blob Store
    '''
    if type == 'train':
        blob_name = "diario.csv"
    elif type == 'test':
        blob_name = 'test.csv'
    else:
        raise ValueError('Incorrect type provided. Type can be either train or test.')
        
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    blob_client = container_client.get_blob_client(blob_name)
    return pd.read_csv(blob_client.download_blob())


def rename_columns(compras, ventas):
    compras.rename(columns = {'date':'Fecha', 'pred':'Compras'}, inplace = True)
    ventas.rename(columns = {'date':'Fecha','pred': 'Ventas'}, inplace = True)
    return compras, ventas

def get_final_datasets():
    train = get_dataset('train')
    train = mdf.transform_dataset(train)
    train['Fecha'] = pd.to_datetime(train['Fecha'], format='%d/%m/%Y')
    train = train[~((train['Fecha'].dt.year == 2024) & (train['Fecha'].dt.month == 1))]
    train['Fecha'] = train['Fecha'].dt.strftime('%d/%m/%Y')
    test = get_dataset('test')
    test = mdf.transform_dataset(test)
    test['Fecha'] = pd.to_datetime(test['Fecha'], format='%d/%m/%Y')
    test = test[~((test['Fecha'].dt.year == 2024) & (test['Fecha'].dt.month == 5))]
    test['Fecha'] = test['Fecha'].dt.strftime('%d/%m/%Y')
    return train, test

def sort_df(df):
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
    df.sort_values(by = 'Fecha', ascending = True, inplace = True)
    df.reset_index(drop=True, inplace=True)
    df['Fecha'] = df['Fecha'].dt.strftime('%d/%m/%Y')
    return df


def concatanate_train_test(train,test):
    df_concatanated = pd.concat([train, test]).reset_index(drop=True)
    df_concatanated = sort_df(df_concatanated)
    df_concatanated['ID'] = df_concatanated.index
    return df_concatanated


def set_data_labels(actual_df, type = 'Actual'):
    actual_df['type'] = type
    return actual_df



def prepare_push_dataset(compras_pred_df, ventas_pred_df):
    # Actual data
    train, test = get_final_datasets()
    actual_df = concatanate_train_test(train,test)
    actual_df = set_data_labels(actual_df, type = 'Actual')

    #Predicted data
    compras_pred_df, ventas_pred_df = rename_columns(compras_pred_df, ventas_pred_df)
    predictions_df = pd.concat([compras_pred_df, ventas_pred_df])
    predictions_df.loc[predictions_df['Compras'] > 0, 'Ventas'] = 0
    predictions_df.loc[predictions_df['Ventas'] > 0, 'Compras']  = 0
    predictions_df = sort_df(predictions_df)
    predictions_df = set_data_labels(predictions_df, type = 'Predicted')

    final_dataset = pd.concat([actual_df,  predictions_df]).reset_index(drop=True)
    for col in final_dataset.columns:
        if pd.api.types.is_numeric_dtype(final_dataset[col]):
            final_dataset[col] = final_dataset[col].fillna(0)
        else:
            final_dataset[col] = final_dataset[col].fillna('missing')
    final_dataset['ID'] = final_dataset.index
    final_dataset.to_parquet('../datasets/final_data.parquet')
    return final_dataset
    
    #return final_dataset


def push_dataset():
    file_to_upload = 'final_data.parquet'
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(container=CONTAINER_NAME)
    with open(file=os.path.join('C:\\Users\\jcoqu\\OneDrive\\Documents\\U-tad\\Curso5\\TFG\\TFG_ingenieria\\Desarrollo\\datasets', file_to_upload), mode="rb") as data:
        blob_client = container_client.upload_blob(name = file_to_upload, data=data, overwrite=True)

