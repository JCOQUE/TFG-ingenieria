




###############################################################################
###############################################################################
######### PAQUETES QUE SE NECESITAN: pandas, pyarrow y openpyxl ###############
###############################################################################
###############################################################################


import pandas as pd
import os

def format_with_commas(x):
    return '{:,.0f}'.format(x)


def add_grupo(asiento):
    print(asiento)
    grupo = asiento['NoCuenta'][:1]
    
    return grupos_plan_general_contable[grupo]


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

pd.options.display.float_format = format_with_commas


dataset = pd.read_csv('diario.csv')
dataset.reset_index(inplace = True)
dataset.rename(columns = {'index': 'ID'}, inplace = True)
dataset.drop(columns = ['Movimiento'], inplace = True)


grupos_plan_general_contable = {'1': '100',
                               '2':'200',
                               '3': '300',
                               '4':'400',
                               '5': '500',
                               '6':'600',
                               '7': '700',
                               '8':'800',
                               '9': '900'}


dataset['NoCuenta'] = dataset['NoCuenta'].astype(str).str[:3]
# dataset

dataset['Compras'] = dataset.apply(create_purchases, axis = 1)
# dataset

dataset['Ventas'] = dataset.apply(create_sales, axis = 1)
# dataset



dataset['Ventas'].sum()

dataset['Compras'].sum()

print(dataset.loc[110:170,:])

dataset['NoGrupo'] = dataset.apply(add_grupo, axis =1)

dataset[['Debe', 'Haber', 'Compras', 'Ventas']] = dataset[['Debe', 'Haber', 'Compras', 'Ventas']].round(2)

print('Make sure the excel is closed before executing the following code line')
dataset.to_parquet('diario.parquet')
dataset.to_csv('check_csv.csv', index = False)

