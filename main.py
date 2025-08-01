# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:38:46 2025

@author: thoma
"""

import requests
import pandas as pd 
from datetime import datetime,timedelta
import pickle
from shape import Shape,finder
import os 
# =============================================================================
# Data Update
# =============================================================================

def are_consecutive_months(start_period, end_period):
    """Check if end_period is exactly one month after start_period"""
    if pd.isna(start_period) or pd.isna(end_period):
        return False
    year_diff = end_period.year - start_period.year
    month_diff = end_period.month - start_period.month
    total_month_diff = year_diff * 12 + month_diff
    return total_month_diff == 1

df_tot_m = pd.read_csv('Hist.csv',parse_dates=True,index_col=0)
df_tot_m.index = df_tot_m.index.to_period('M')
df_tot_m=df_tot_m.fillna(0)

TOKEN = os.getenv('API_TOKEN')
if not TOKEN:
    raise ValueError("API_TOKEN environment variable is not set")


response = requests.get(f'https://helix-tools-api.idmcdb.org/external-api/idus/last-180-days/?client_id={TOKEN}&format=json')
json_data = response.json()
df_new=pd.DataFrame(json_data)
df_new.to_csv('new.csv')

df_new['displacement_start_date'] = pd.to_datetime(df_new['displacement_start_date'])
df_new['displacement_end_date'] = pd.to_datetime(df_new['displacement_end_date'])
df_new['start_month_year'] = df_new['displacement_start_date'].dt.to_period('M')
df_new['end_month_year'] = df_new['displacement_end_date'].dt.to_period('M')
consecutive_mask = df_new.apply(lambda row: are_consecutive_months(row['start_month_year'], row['end_month_year']), axis=1)
df_new['month_year'] = df_new['start_month_year']  
df_new.loc[consecutive_mask, 'month_year'] = df_new.loc[consecutive_mask, 'end_month_year'] 
same_month_mask = df_new['start_month_year'] == df_new['end_month_year']
valid_events_mask = same_month_mask | consecutive_mask
df_new = df_new[valid_events_mask].copy()
df_new = df_new.groupby(['month_year', 'iso3', 'event_id'])['figure'].mean().reset_index()
df_new = df_new.groupby(['month_year', 'iso3'])['figure'].sum().reset_index()
df_new = df_new.pivot(index='month_year', columns='iso3', values='figure')
df_new = df_new.fillna(0)
df_new = df_new.astype(int)

month = datetime.now().strftime("%m")
last_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
df_new= df_new.loc[:last_month,:].iloc[-5:,:]

df_tot_m = df_tot_m.loc[:df_new.index[0]].iloc[:-1,:]
df_tot_m = pd.concat([df_tot_m,df_new],axis=0)
df_tot_m.to_csv('Hist.csv')

# =============================================================================
# Forecast
# =============================================================================

df_tot_m.fillna(0,inplace=True)
df_tot_all=df_tot_m.copy()
tot_err=[]
h_train=12
dict_sce = {i :[[],[]] for i in df_tot_all.columns}
h=12
pred_tot=[]
for coun in range(len(df_tot_all.columns)):
    if not (df_tot_all.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        shape.set_shape(df_tot_all.iloc[-h_train:,coun]) 
        find = finder(df_tot_all.iloc[:-h,:],shape)
        find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2,min_mat=5)
        try:
            find.create_sce(h)
        except:
            find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2,min_mat=10)
            find.create_sce(h)
        sce_ts = find.val_sce
        sce_ts.columns = pd.date_range(start=df_tot_all.iloc[-h_train:,coun].index[-1].to_timestamp() + pd.DateOffset(months=1), periods=12, freq='M')
        sce_ts = sce_ts *(df_tot_all.iloc[-h_train:,coun].max()-df_tot_all.iloc[-h_train:,coun].min()) + df_tot_all.iloc[-h_train:,coun].min()
        sce_ts[sce_ts<0]=0
        dict_sce[df_tot_all.columns[coun]][0]=find.sequences
        dict_sce[df_tot_all.columns[coun]][1]=sce_ts
        
        pred = find.predict(h)
        pred = pred *(df_tot_all.iloc[-h_train:,coun].max()-df_tot_all.iloc[-h_train:,coun].min()) + df_tot_all.iloc[-h_train:,coun].min()
        pred_tot.append(pred)
    else:
        pred_tot.append(pd.Series([0]*h))
        
with open('Scenarios.pkl', 'wb') as f:
    pickle.dump(dict_sce, f)        

df_pred = pd.DataFrame(pred_tot).T
df_pred.index= pd.date_range(start=df_tot_all.index[-1].to_timestamp() + pd.DateOffset(months=1), periods=12, freq='M')
df_pred.columns = df_tot_all.columns
df_pred[df_pred<0]=0
df_pred.to_csv('Predictions.csv')
