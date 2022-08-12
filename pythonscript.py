import pandas as pd
import numpy as np
import seaborn as sns
import csv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import sys


from importlib import reload


# Utilisez load_env pour tracer le chemin de .env:
destination = os.environ.get("destinationimportcollab")


with open(destination + '\\' + sys.argv[1], "r",encoding="ISO-8859-1") as file_obj:
 data = pd.read_csv(file_obj, skipinitialspace = True,on_bad_lines='skip',sep=';')
#file_data["test"]=1

data.columns.values[0] = "document_number"

# Data Cleaning

data=data.replace(to_replace =["Ktn", "Stk"], value =["box","pcs"])

data=data.replace(to_replace=["EU ICO","EUROPA","NAT.","NAT. ICO","WELT"],value=["EU","EU","NAT","NAT","INT"])

data=data.rename(columns={"number_per_pick": "unit_per_pack"})

data=data.drop(['document_number','line_number','type','gross_weight','order_line','gesamt nett','gesamt brutt','number of position'],axis=1)


#drop any rows that have 0 in the amount and unit_per_pack columns
data.drop(data.index[data['unit_per_pack'] == 0], inplace=True)
data.drop(data.index[data['amount'] == 0], inplace=True)
data = data[data['amount'] > 0]
data = data[data['unit_per_pack'] > 0]

# #Add New Column
data['pics_number'] = data['amount']/data['unit_per_pack']

data['shipping_date']=data['shipping_date'].fillna(0)
data.drop(data.index[data['shipping_date'] == 0], inplace=True)

#corriger le type DATE
data['shipping_date'] = pd.to_datetime(data['shipping_date'],infer_datetime_format=True)
data['booking_date'] = pd.to_datetime(data['booking_date'],infer_datetime_format=True)

#corriger le type int
data['pics_number']=data['pics_number'].astype(int)

data['description_2']=data['description_2'].fillna('99999')

data['article_class']=data['article_class'].fillna(1000)

#Data Procecing

data.drop(data.loc[data['shipping_date']<='2016-05-20'].index, inplace=True)
data=data.loc[data['warehouse_code']=='M16']
data.drop(data.loc[(data['article_number'].astype(str).str.startswith('BH'))].index,inplace=True)
df5=data.iloc[:,15:]
df4=data.iloc[:,6:7]
df22=data.iloc[:,14:15]
df1=data.iloc[:,3:4]

DataFrame=pd.merge(df1,df22, left_index=True, right_index=True)
DataFrame=pd.merge(DataFrame,df4, left_index=True, right_index=True)
DataFrame=pd.merge(DataFrame,df5, left_index=True, right_index=True)

DataFrame.drop(DataFrame.loc[DataFrame['unit']=='m'].index, inplace=True)
DataFrame.drop(DataFrame.loc[DataFrame['unit']=='kg'].index, inplace=True)

#Regroupement par semaine

#La somme des pics vendus durant la semaine qui commence le Lundi indiqué (shipping_date)

DataFrame['shipping_date'] = pd.to_datetime(DataFrame['shipping_date']) - pd.to_timedelta(6, unit='d')
DataFrame=DataFrame.resample('W-MON', on='shipping_date').pics_number.sum()
DataFrame= pd.DataFrame(data=DataFrame)

DataFrame.drop(DataFrame[DataFrame['pics_number'] == 0].index, inplace = True)


ax = sns.boxplot(x=DataFrame["pics_number"])
