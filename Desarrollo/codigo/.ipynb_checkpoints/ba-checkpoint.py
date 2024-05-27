#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# In[4]:


dataset = pd.read_parquet('../datasets/diario.parquet')
dataset.head(6)


# In[5]:


dataset['Fecha'] = pd.to_datetime(dataset['Fecha'], format='%d/%m/%Y')
dataset


# In[6]:


dataset['Year'] = dataset['Fecha'].dt.year
dataset['Month'] = dataset['Fecha'].dt.month
dataset


# In[8]:


ts_ventas = dataset.groupby(['Year', 'Month']).agg({'Ventas': 'sum'})
ts_ventas


# In[9]:


start_date = ts_ventas.index[0]
start_date


# In[10]:


end_date = ts_ventas.index[-1]
end_date


# In[11]:


dias = {1:'31',
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

def get_dates(start, end):
    day_start = dias[start[1]]
    day_end = dias[end[1]]
    start_date = datetime.strptime(f'{day_start}/{start[1]}/{start[0]}', '%d/%m/%Y')
    end_date = datetime.strptime(f'{day_end}/{end[1]}/{end[0]}', '%d/%m/%Y')
    return pd.date_range(start = start_date,
                     end = end_date, freq = 'ME') #Warning: M is deprecated and will be removed in future versions. Use ME instead
    
    


# In[12]:


dates = get_dates(ts_ventas.index[0], ts_ventas.index[-1])
dates


# In[13]:


ts_ventas.reset_index(drop = True)
ts_ventas.set_index(dates, inplace = True)
ts_ventas


# In[14]:


ts_ventas['date'] = ts_ventas.index
ts_ventas.reset_index(drop = True, inplace = True)
ts_ventas


# In[15]:


plt.plot(ts_ventas['date'], ts_ventas['Ventas'])


# In[16]:


ts_compras = dataset.groupby(['Year', 'Month']).agg({'Compras': 'sum'})
ts_compras


# In[17]:


ts_compras.reset_index(drop = True)
ts_compras.set_index(dates, inplace = True)
ts_compras


# In[18]:


plt.plot(ts_compras.index, ts_compras['Compras'])


# In[19]:


import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = ts_ventas.copy()


# In[20]:


df


# In[25]:


class Informer(nn.Module):
    def __init__(self):
        super(Informer, self).__init__()
        self.fc1 = nn.Linear(3, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Informer()
epochs = 10000
learning_rate = 0.01
batch_size = 64


# In[26]:


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[27]:


df_dates = df.copy()
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df.drop(columns=['date'], inplace=True)
df


# In[28]:


train_size = int(0.8 * len(df))
train_df, test_df = df[:train_size], df[train_size:]


# In[29]:


X_train = torch.tensor(train_df.drop(columns=['Ventas']).values)
y_train = torch.tensor(train_df['Ventas'].values)
X_test = torch.tensor(test_df.drop(columns=['Ventas']).values)
y_test = torch.tensor(test_df['Ventas'].values)


# In[30]:


for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train.float())
    
    outputs = outputs.squeeze()  # Remove any singleton dimensions
    
    loss = criterion(outputs, y_train.float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every few epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')




# In[31]:


model.eval()

with torch.no_grad():
    predictions = model(X_test.float())

predictions = predictions.squeeze()


# In[32]:


mse = criterion(predictions, y_test.float()).item()
print(f"Mean Squared Error (MSE): {mse:.4f}")
test_dates = df_dates['date'].tail(len(y_test))
# Visualization
plt.figure(figsize=(10, 6))
plt.plot(test_dates, y_test, label='Actual')
plt.plot(test_dates, predictions, label='Predicted')
plt.title('Actual vs. Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()


# In[33]:


predictions


# In[34]:


y_test


# In[35]:


df.tail(15)


# In[36]:


sd = df_dates.tail(1)['date'].values[0]


# In[37]:


# Create a range of dates for one year starting from the given start date
daily = pd.date_range(start=sd, periods=365, freq='D')



# In[38]:


df_daily = pd.DataFrame({'date': daily, 'year':daily.year, 'month':daily.month, 'day':daily.day})


# In[40]:


df_daily.head(4)


# In[41]:


next_year_month = torch.tensor(df_daily.drop(columns = ['date']).values)

model.eval()

with torch.no_grad():
    next_year_predictions = model(next_year_month.float())

next_year_predictions = next_year_predictions.squeeze()

print("Predictions for the next month:")
print(next_year_predictions)


# In[42]:


df_daily['pred'] = next_year_predictions.numpy()


# In[43]:


df_daily


# In[44]:


next_year_pred_month = df_daily.groupby(['year', 'month'])['pred'].sum().reset_index()


# In[45]:


next_year_pred_month

