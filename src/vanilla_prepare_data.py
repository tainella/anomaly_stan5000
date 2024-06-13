import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import torch
import struct
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

from filter_data import filter_df
import get_mks_5000_ML_dataset

class Vanilla_Dataset(torch.utils.data.Dataset):   
    def __init__(self, dates, mode = 'train', create_feat_flg = True): 
        '''
        Инициализация датасета, вызывается для подгрузки данных для обучения и тестирования.
        
        Вход:
        dates - кортеж из двух дат начала и конца выгрузки данных в формате pd.Timastamp
        mode - строка 'train' или 'test' для режима загрузки данных
        create_feat_flg - флаг генерации фичей
        
        Выход:
        Объект класса Vanilla_Dataset, в поле Х хранится итоговый torch.Tensor данных
        '''
        temp_columns_ = ['[101:1]', '[101:4]', '[101:5]', '[104_28]', '[101:20]', '[101:25]']
        df = get_mks_5000_ML_dataset.get_mks_5000_ML_dataset(data_from_=dates[0], data_to_=dates[1], data_columns_=temp_columns_, s3_data_prefix = 'data/PRE_GSND_')
        df.rename(columns={"[101:1]": "Давление в системе", "[101:4]": "Уровень масла в баке", 
       "[101:5]": "Температура масла в баке"}, inplace=True)
    
        if mode == 'train':
            df = df[df['train_data_flg'] == True]
        df = df[df['stand_not_working_flg'] == False]
            
        df = df[["Давление в системе", "Уровень масла в баке", "Температура масла в баке"]]
        self.X, self.time_column, self.scaler = self.fit_form(df.copy(), create_feat_flg)
        self.columns_list = self.X.columns.to_list()
        self.X = torch.tensor(self.X.values, dtype=torch.float32)

    def __len__(self):
        '''
        Длина итогового датасета.
        '''
        return self.X.__len__()

    def __getitem__(self, index):
        '''
        Получение элемента датасета по индексу.
        '''
        return (self.X[index])
   
    def fit_form(self, df, create_feat_flg):
        '''
        Изменение гранулярности данных до минут, создание производных признаков, скалирование данных от -1 до 1.
        '''
        df.reset_index(inplace=True)
        df.loc[:,'Time'] = pd.to_datetime(df['Time'])
        df = self.resample(df, "1Min")
        
        if create_feat_flg:
            df = self.create_new_features(df)

        scaler = StandardScaler()
        time_df = df['Time']
        column_list = df.columns.to_list()
        column_list.remove('Time')
        df = pd.DataFrame(scaler.fit_transform(df.drop(columns = ['Time'])), columns=column_list)
        return df, time_df, scaler
    
    def create_new_features(self, df):
        '''
        Генерация новых признаков.
        '''
        #Давление
        col = 'Давление в системе'
        df[f"{col}_var"] = df[col].rolling(2*60).var() #2 часов

        #Температура
        col = 'Температура масла в баке'
        df[f'{col}_trend'] = df[col].rolling(12*60).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=False)
        # Уровень
        col = 'Уровень масла в баке'
        if col in df.columns:
            #негативный тренд за час, если положительный, то 0
            df[f'{col}_trend_hour'] = df[col].rolling(60).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if np.polyfit(np.arange(len(x)), x, 1)[0] < 0 else 0, raw=False)
            #негативный тренд за 12 часов, если положительный, то 0
            df[f'{col}_trend_12hour'] = df[col].rolling(12*60).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if np.polyfit(np.arange(len(x)), x, 1)[0] < 0 else 0, raw=False)
            #падение за минуту, если положительный, то 0
            df[f'{col}_diff_M'] = df[col].rolling(2).apply(lambda x: (x.iloc[1] - x.iloc[0]) if (x.iloc[1] - x.iloc[0]) < 0 else 0)
            df.drop(columns = ['Уровень масла в баке'], inplace=True)

        df.reset_index(inplace=False)
        df.dropna(inplace=True)
        return df
    
    def resample(self, df, aggreg):
        '''
        Изменение гранулярности данных до минут.
        '''
        df = df.resample(aggreg, on='Time').mean()
        df.reset_index(inplace=True)
        df = df.dropna()
        return df
    