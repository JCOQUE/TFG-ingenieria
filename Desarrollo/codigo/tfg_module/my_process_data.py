import  pandas as pd
from sklearn.model_selection import train_test_split
import torch


def create_features(df, target = None, informer = False):
    '''
    This function is in charge of splitting the time series DataFrame
    (which contains two columns: date and target) into input features 
    for training (i.e. converting date, the input into hour, dayofweek,
    etc.) and output values (the target itself). Since the Informer
    model utilizes pytorch, and, therefore Tensors instead of pandas
    DataFrame, an additional argument is needed to treat each model
    in the proper way.
    '''
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    
    if target:
        if informer:
            X = torch.tensor(df.drop(columns=['date', target]).values, dtype = torch.float32)
            y = torch.tensor(df[target].values, dtype = torch.float32).unsqueeze(1)
            return X,y
            
        else:
            X = df.drop(columns=['date', target])
            y = df[target]
            return X, y
    else:
        if informer:
            X = torch.tensor(df.drop(columns=['date']).values, dtype = torch.float32)
            return X

        else:
            X = df.drop(columns=['date'])
            return X



def get_train_test(ts, test_size = 0.2):
    '''
    this functions utilizes the train_test_split function provided
    by sklearn to split the data into train and test. TCN is the only
    model that makes use of this function since the rest of the models
    use GridSearchCV from the sklearn library that uses Cross Validation
    for training.
    '''
    train, test = train_test_split(ts,  
                                    test_size = test_size,
                                    shuffle = False)
    return train, test