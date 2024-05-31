import pandas as pd
from azure.storage.blob import BlobServiceClient
import os
'''
This .py is in charge of pulling and pushing files from Azure Blob Storge
'''

'''
Guardar credenciales en Prefect??????
'''

def get_dataset(type = 'train'):
    '''
    Fetches via API the raw .csv from Azure Blob Store
    '''
    account_name = "blobstoragetfginso"
    account_key = "gd5nuYRJgr/SLkHdH7PIhh72OLQX/kwKuDlF5yO3grgfrrfyFigneBBd5VJPEuYZC6qlgzTBlvBS+AStpXySag=="
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    container_name = "containertfginso1"
    if type == 'train':
        blob_name = "diario.csv"
    elif type == 'test':
        blob_name = 'test.csv'
    else:
        raise ValueError('Incorrect type provided. Type can be either train or test.')
        
    

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    return pd.read_csv(blob_client.download_blob())


file_to_upload = 'check_csv.csv'
def push_dataset(blob_service_client, container_name):
    container_client = blob_service_client.get_container_client(container=container_name)
    with open(file=os.path.join('C:/Users/jcoqu/OneDrive/Documents/U-tad/Curso5/TFG/TFG_INSO/Desarrollo/datasets', file_to_upload), mode="rb") as data:
        blob_client = container_client.upload_blob(name = file_to_upload, data=data, overwrite=True)