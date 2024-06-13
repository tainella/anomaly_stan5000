#!c1.8
import boto3
from boto3.s3.transfer import TransferConfig

import os
from os import path

import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import pyarrow.parquet as pq
import numpy as np

from datetime import datetime, timedelta
import calendar

from tqdm import tqdm
import datefinder

# Настройки
s3_config = {
    'access_key_id': 'YCAJEY4AMKLqu-EeFVa0DO5JM',
    'access_key': os.environ['storage-editor'],
    'endpoint': "https://storage.yandexcloud.net",
    'cloud_region_name': "ru-central1"
    }
s3_bucket_name = 'preprocessed-selection'
s3_operation_name = 'list_objects_v2'

s3_data_prefix = 'data/PRE_MPM_'
s3_default_first_file = 'data/PRE_MPM_2022-01-01_2022-01-10.parquet'
s3_data_prefix_2 = 'data/PRE_GSND_'
s3_default_first_file_2 = 'data/PRE_GSND_2022-01-01_2022-01-31.parquet'

s3_mnt_data_path = '/home/jupyter/mnt/s3/preprocessed-selection'

#Конфигурирование отбора нештаток по умолчанию
cfg_outage_xls_file = 'config/2023-12-01 Простои МКС-5000.xlsx'
cfg_outage_pre_loc =  '12h'
cfg_outage_post_loc = '4h'

#Условия для выбора НШС, которые исключаются из данных для обучения
cfg_outage_loc_filter_ = slice(None, None, None) #Взять все НШС - по умолчанию
# cfg_outage_loc_filter_ = outage_df_filtered['Узел/деталь'].isin(["Гидронажимное устройство СО", "Гидронажимное устройство СП"]) #Пример: только отдельные узлы

#Готовим соединение
def get_s3_client_lf(s3_config):
    session = boto3.session.Session()

    session = boto3.Session(
        aws_access_key_id=(s3_config['access_key_id']),
        aws_secret_access_key=(s3_config['access_key']),
        region_name=s3_config['cloud_region_name'],
    )

    return session.client("s3", endpoint_url=s3_config['endpoint'])

#Готовим пагинатор, чтобы вытаскивать список файлов
def get_s3_pages_lf(s3, start_after_=s3_default_first_file, operation_name_=s3_operation_name, bucket_=s3_bucket_name, s3_data_prefix_=s3_data_prefix):
    paginator = s3.get_paginator(operation_name_)
    return paginator.paginate(Bucket=bucket_, Prefix=s3_data_prefix_, StartAfter=start_after_)
    
def get_mks_5000_preprocecced_file_list( \
    s3, data_from_, data_to_, \
    #Функция специализированная, поэтому бакет S3 и маска файлов должны быть известны из настроек
    bucket_=s3_bucket_name, operation_name_=s3_operation_name, s3_data_prefix_=s3_data_prefix):
    
    return_file_list = []
    if data_from_>data_to_:
        raise RuntimeError(f'Начальная дата {data_from_} больше конечной {data_to_}')
    #Формат названия файла data/PRE_MPM_2022-01-01_2022-01-10.parquet
    
    first_file = f'{s3_data_prefix_}{data_from_[0:7]}-01_{data_from_[0:7]}-01.parquet' #Приведем к началу месяца
    pages = get_s3_pages_lf(s3, start_after_=first_file, s3_data_prefix_= s3_data_prefix_)
    for page in pages:
        if page['KeyCount']==0:
            return []
        for obj in page['Contents']:
            k = obj['Key'].split('_')
            start_date = k[2]
            end_date = k[3].split('.')[0]
            if end_date <= data_to_ :
                if start_date <= data_from_ <= end_date or start_date >= data_from_:
                    return_file_list.append(obj['Key'])
            else:
                if start_date <= data_to_:
                    return_file_list.append(obj['Key'])
                return return_file_list
    return return_file_list

def get_preprocecced_parquet_data(file_name_, data_columns_, mnt_data_path_=s3_mnt_data_path):
    error_text = ""
    current_df = pd.DataFrame()

    try:
        current_df = pd.read_parquet(f'{mnt_data_path_}/{file_name_}',columns=data_columns_)
#         current_df = current_df.set_index('Time') #Это делать не нужно, указание на индекс сохранено в parquet
    except Exception as e:
        #Вообще, все файлы должны читаться, а новых колонок, которых нет в PRE_MPM-файлах никто не должен просить
        error_text = e.args[0][0:400]
        raise RuntimeError(f'Ошибка чтения файла {file_name_}.\nТекст ошибки:\n{error_text}')
        
    return current_df

def process_mks_5000_preprocessed_files(data_file_list_, data_from_, data_to_, data_columns_, cfg_groupby_ = False):
    current_df = pd.DataFrame()
    return_df = pd.DataFrame()
    return_df_list  = []
    data_from_dttm = datetime.strptime(f'{data_from_} 00:00:00.000000', '%Y-%m-%d %H:%M:%S.%f')
    data_to_dttm   = datetime.strptime(  f'{data_to_} 23:59:59.999999', '%Y-%m-%d %H:%M:%S.%f')
    last_index = datetime.strptime(f'{data_from_} 00:00:00.000000', '%Y-%m-%d %H:%M:%S.%f')
    for i in tqdm(range(len(data_file_list_))):
        i_file = data_file_list_[i]
        #В файле могут быть начальные или конечные хвосты по 'Time', которые не попадают в интервал data_from_-data_to_
        current_df = get_preprocecced_parquet_data(file_name_=i_file, data_columns_=data_columns_)
        #Сортировка по индексу важна. Если индекс не монотонный, то фильтрация по датам с помощью loc падает с ошибкой на смене тренда
        current_df = current_df.groupby('Time', sort=True).mean()
        #Это должно убрать дубли между файлами
#         return_df_list.append(current_df.loc[last_index:])
        return_df = pd.concat([return_df,current_df.loc[last_index:]])
        last_index = current_df.index.max() + pd.Timedelta('100000microseconds')
    return return_df

#Загружаем список НШС, от него будем формировать разметку для обучения
def get_outage_df(from_, to_, use_default_source=True, file_path="", sheet_num = 0):
    local_outage_df = []
    
    if use_default_source == True:
        file_path = s3_mnt_data_path+"/"+cfg_outage_xls_file
    else:
        if len(file_path)==0:
            raise RuntimeError("Не указано имя файла с НШС и не включен режим 'По умолчанию'")
        
    try:
        local_outage_df = pd.read_excel(file_path, sheet_name=sheet_num, index_col=0)
    except Exception as e:
        print(f'Exception in {file_path}: {e}')
        local_outage_df = None
        raise RuntimeError("Проблема чтения файла")
    
    #Принудительно выставим формат трем колонкам, которые точно должны быть и должны сконвертиться хорошо

    local_outage_df["Дата регистрации"] = pd.to_datetime(local_outage_df["Дата регистрации"],format='%Y-%m-%d %H:%M:%S')
    local_outage_df["Дата начала"] = pd.to_datetime(local_outage_df["Дата начала"],format='%Y-%m-%d %H:%M:%S')
    local_outage_df["Дата окончания"] = pd.to_datetime(local_outage_df["Дата окончания"],format='%Y-%m-%d %H:%M:%S')
    
    #Фильтрация по обрабатываемым датам

    from_ts = pd.Timestamp(from_)
    to_ts = (pd.Timestamp(to_) + pd.Timedelta('1d')) #Так надо, тк данные мы запрашиваем по датам. Есть риск потерять НШС из последнего дня
    #Оставим только НШС, которые с учетом окрестностей пересекаются с обрабатываемым интервалом
    local_outage_df = local_outage_df.loc[(from_ts <= (local_outage_df["Дата окончания"] + pd.Timedelta(cfg_outage_post_loc))) & (to_ts >= (local_outage_df["Дата начала"] -pd.Timedelta(cfg_outage_pre_loc)))]
    
    return local_outage_df

#Вызывать функцию нужно, передавая минимально data_from_, data_to_, data_columns_, cfg_mill_part_
#Остальное можно оставлять по умолчанию
def get_mks_5000_ML_dataset(
    #Начальная дата (включительно). data - это "данные", а не транслит от "дата"
    data_from_, \
    #Конечная дата (включительно)
    data_to_, \
    #Список колонок, которые понадобятся в датасете
    data_columns_=['Time'], \
    outage_df=pd.DataFrame(), \
    s3_config = s3_config, \
    s3_bucket_name=s3_bucket_name, \
    s3_data_prefix=s3_data_prefix \
    ):
#Функция возвращает pandas.DataFrame с запрошенными data_columns_, включая ['Time'] в качестве индекса
#Добавляются колонки:
# stand_not_working_flg (T/F) - Флаг "Клеть НЕ в работе"
# outage_flg (T/F) - Флаг "Период НШС и его окрестности"
# train_data_flg (T/F) - Флаг "Данные для обучения"
# Колонки из дополнительного списка cfg_add_columns: ['Time','[16:0]','[16_11]','[17_36']

#Значения колонок (сигналов) не трансформируются относительно исходных данных, не заменяются на NaN
    
    #Начало алгоритма
    cfg_add_columns = ['Time','[16:0]','[16_11]','[17_36]']
    local_file_list=[]
    local_data_df = pd.DataFrame()
    #Получим соединение с S3
    s3 = get_s3_client_lf(s3_config)
    #Получим список файлов для загрузки с помощью пагинатора S3, закроем соединение
    #Даты - текстовые в формате '%Y-%m-%d'
    local_file_list = get_mks_5000_preprocecced_file_list(s3=s3, data_from_=data_from_, data_to_=data_to_, 
                                                          s3_data_prefix_ = s3_data_prefix_2)
    #Данные для фильтра клеть не в работе
    local_file_list_stand = get_mks_5000_preprocecced_file_list(s3=s3, data_from_=data_from_, data_to_=data_to_,
                                                               s3_data_prefix_ = s3_data_prefix)
    s3.close()
    
    #Принудительно добавим колонки основных режимов и Time
    data_columns_ = list(set(data_columns_) | set(['Time']))
    
    #Забираем данные
    local_data_df = process_mks_5000_preprocessed_files(data_file_list_=local_file_list, data_from_=data_from_, data_to_=data_to_, data_columns_=data_columns_)
    #Данные для фильтра клеть не в работе
    local_data_df_stand = process_mks_5000_preprocessed_files(data_file_list_=local_file_list_stand,
                                                        data_from_=data_from_, data_to_=data_to_, 
                                                        data_columns_=cfg_add_columns)
    
    #Постобработка данных - флаги стандартной фильтрации
    #Флаг "Клеть не в работе"
    local_data_df = local_data_df.join(local_data_df_stand)
    local_data_df["stand_not_working_flg"] = True
    #По условиям из ТЗ
    # filter_indexes = local_data_df_stand[(local_data_df_stand['[17_36]']==1) & (local_data_df_stand['[16:0]']>201)].index
    local_data_df.loc[(local_data_df['[17_36]']==1) & (local_data_df['[16:0]']>201), 'stand_not_working_flg'] = False #Будет конфигурироваться через Airflow

    #Флаг "Период НШС и его окрестности"
    #Выставляется на основе списка НШС, который либо извлекается из мастер-версии по умолчанию, либо берется из outage_df, если кто-то хочет поигараться с выкидыванием только некоторых НШС из данных для обучения, а не всех
    if outage_df.empty: #Не ожидаем, что в качестве outage_df кто-то будет передавать df неприемлемого формата, а не полученный из get_outage_df
        outage_df = get_outage_df(from_=data_from_, to_=data_to_, use_default_source=True)
        
    #Условия для выбора НШС, которые исключаются из данных для обучения. Просто запараметризовать не получается другим способом
    cfg_outage_loc_filter_ = slice(None, None, None) #Взять все НШС - по умолчанию
    # cfg_outage_loc_filter_ = outage_df_filtered['Узел/деталь'].isin(["Гидронажимное устройство СО", "Гидронажимное устройство СП"]) #Пример: только отдельные узлы
    
    outage_df = outage_df.loc[cfg_outage_loc_filter_]
    
    local_data_df["outage_flg"] = False
    for _, (current_outage_start, current_outage_end) in outage_df[['Дата начала', 'Дата окончания']].iterrows():
        local_data_df.loc[current_outage_start-pd.Timedelta(cfg_outage_pre_loc) : current_outage_end+pd.Timedelta(cfg_outage_post_loc), "outage_flg"] = True
    
    #Флаг "Данные для обучения"
    local_data_df["train_data_flg"] = False
    local_data_df.loc[((local_data_df['outage_flg']==False)&(local_data_df['stand_not_working_flg']==False)), 'train_data_flg'] = True #Будет конфигурироваться через Airflow
    
    return local_data_df