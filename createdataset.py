import pandas
import xarray
import requests
import datetime
import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch
from torch.utils import data
import os


def extract_features(row):
    point = ncep_data.sel(
        lon=row['longitude'],
        lat=row['latitude'],
        level=1000,
        method='nearest',
    )

    p1w = point.rolling(time=7).mean()
    p2w = point.rolling(time=14).mean()
    p3w = point.rolling(time=21).mean()

    date = row['date']
    v = point.sel(time=date)
    v1w = p1w.sel(time=date)
    v2w = p2w.sel(time=date)
    v3w = p3w.sel(time=date)

    year, month, day = row['date'].split('-') 

    return {
        'fire_type': row['fire_type'],
        'day': int(day),
        'month': int(month),
        'temperature': v.air.values.item(0),
        'humidity': v.rhum.values.item(0),
        'uwind': v.uwnd.values.item(0),
        't1w': v1w.air.values.item(0),
        't2w': v2w.air.values.item(0),
        't3w': v3w.air.values.item(0),
        'h1w': v1w.rhum.values.item(0),
        'h2w': v2w.rhum.values.item(0),
        'h3w': v3w.rhum.values.item(0)
    }



if os.path.isfile('wildfires_dataset.csv') == False:
    ncep_data = []
    years = list(range(2018, 2019))
    print(years)
    for year in years:
        for var in ('air', 'uwnd', 'rhum'):
            dataset_filename = 'data/ncep/{}.{}.nc'.format(var, year)
            ncep_data.append(xarray.open_dataset(dataset_filename))
    ncep_data = xarray.merge(ncep_data)

    ####################TRAIN DATASET################################################

    df_train = pandas.read_csv('data/wildfires_train.csv')
    df_subsample = df_train.query('(date > "2018") & (date < "2019")')#.sample(n=2000)

    df_features = []
    for i, row in tqdm(df_subsample.iterrows(), df_subsample.shape[0]):
        features = extract_features(row)
        df_features.append(features)
    df_features = pandas.DataFrame(df_features)
    df_features = df_features.dropna()
    df_features.to_csv("wildfires_dataset.csv")
else:
    df_features = pandas.read_csv("wildfires_dataset.csv")
    

if os.path.isfile('wildfires_test_dataset.csv') == False:
    ncep_data = []
    years = list(range(2019,2020))
    print(years)
    for year in years:
        for var in ('air', 'uwnd', 'rhum'):
            dataset_filename = 'data/ncep/{}.{}.nc'.format(var, year)
            ncep_data.append(xarray.open_dataset(dataset_filename))
    ncep_data = xarray.merge(ncep_data)

    ###################TEST DATASET#################################################

    df_test = pandas.read_csv('data/wildfires_check.csv')
    #df_subsample = df_test.query('(date == "2019")')#.sample(n=2000)
    df_subsample = df_test
    print(df_subsample.head())

    df_test_features = []
    for i, row in tqdm(df_subsample.iterrows(), total=df_subsample.shape[0]):
        features = extract_features(row)
        df_test_features.append(features)
    df_test_features = pandas.DataFrame(df_test_features)
    df_test_features = df_features.dropna()
    df_test_features.to_csv("wildfires_test_dataset.csv")
else:
    df_test_features = pandas.read_csv("wildfires_test_dataset.csv")

df_features = df_features.dropna()
fireDatas = torch.Tensor(df_features.drop(columns='fire_type').values)
fireTypes = torch.Tensor(df_features['fire_type'].tolist())
fireDataset = data.TensorDataset(fireDatas, fireTypes)
torch.save(fireDataset, "fireDataset")

df_test_features = df_test_features.dropna()
fireTestDatas = torch.Tensor(df_test_features.drop(columns='fire_type').values)
fireTestTypes = torch.Tensor(df_test_features['fire_type'].tolist())
fireTestDataset = data.TensorDataset(fireTestDatas, fireTestTypes)
torch.save(fireTestDataset, "fireTestDataset")