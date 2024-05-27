import  pandas as pd
from sklearn.model_selection import train_test_split
import torch


def create_features(df, target = None, informer = False):
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
    train, test = train_test_split(ts,  
                                    test_size = test_size,
                                    shuffle = False)
    return train, test