import pandas as pd


'''
This .py converts the raw .csv file from Azure Blob Store into the 
dataset used for prediction and in PowerBI. That means that applies
the necessary transformations using pandas to obtain a Compras, Ventas,
and NoGrupo columns. It also transforms some existing columns into 
the desired type and format.
'''

def format_with_commas(x):
    return '{:,.0f}'.format(x)

pd.options.display.float_format = format_with_commas

def add_grupo(asiento):
    grupos_plan_general_contable = {'1': '100',
                                '2':'200',
                                '3': '300',
                                '4':'400',
                                '5': '500',
                                '6':'600',
                                '7': '700',
                                '8':'800',
                                '9': '900'}
    first_account_number = asiento['NoCuenta'][:1]
    
    return grupos_plan_general_contable[first_account_number]


def create_purchases(asiento):
    if asiento['NoCuenta'].startswith('6'):
        return asiento['Debe'] - asiento['Haber']
    else:
        return 0
    
def create_sales(asiento):
    if asiento['NoCuenta'].startswith('7'):
        return asiento['Haber'] - asiento['Debe']
    else:
        return 0
    
def transform_dataset(dataset):
    dataset.drop(columns = ['Movimiento'], inplace = True)
    dataset.reset_index(inplace = True)
    dataset.rename(columns = {'index': 'ID'}, inplace = True)
    dataset['NoCuenta'] = dataset['NoCuenta'].astype(str).str[:3]
    dataset['Compras'] = dataset.apply(create_purchases, axis = 1)
    dataset['Ventas'] = dataset.apply(create_sales, axis = 1)
    dataset['NoGrupo'] = dataset.apply(add_grupo, axis =1)
    dataset['Fecha'] = pd.to_datetime(dataset['Fecha']).dt.strftime('%d/%m/%Y')

    return dataset




