import os
import torch
from torch import nn
from torch.nn import MSELoss, L1Loss
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from datetime import date
from glob import glob
import sys
import pandas as pd
import pickle

sys.path.append('../src')
from ae import Autoencoder
import vanilla_prepare_data
import analyse
from datetime import datetime

def get_quarter_start_end(year, quarter):
    """ Возвращает начальный и 'конечный' месяцы для заданного квартала. 
    'Конечный' месяц - это первый месяц следующего квартала или года. """
    quarter_start_map = {
        '1q': ('01', year),
        '2q': ('04', year),
        '3q': ('07', year),
        '4q': ('10', year)
    }
    quarter_end_map = {
        '1q': ('04', year),
        '2q': ('07', year),
        '3q': ('10', year),
        '4q': ('01', year + 1)
    }

    start_month, _ = quarter_start_map[quarter]
    end_month, end_year = quarter_end_map[str(int(quarter[0])) + 'q'] #'1q' if quarter == '4q' else 
    return f'{year}-{start_month}', f'{end_year}-{end_month}'

def get_training_period(target_y, target_q):
    """ Возвращает период для тренировочных данных. """
    year = target_y
    quarter_number = int(target_q[0])
    
    # Вычисляем квартал для начала тренировочного периода
    if quarter_number > 2:
        training_start_q = str(quarter_number - 2) + 'q'
    else:
        training_start_q = str(quarter_number + 2) + 'q'
        year -= 1  # Переход на предыдущий год, если нужно
    
    training_start, _ = get_quarter_start_end(year, training_start_q)
    test_start, _ = get_quarter_start_end(target_y, target_q)
    return training_start, test_start

def train_model(model, df, model_path, epoches=10):
    '''
    Обучение модели на одном участке данных.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Обучение
    num_epochs = epoches
    for epoch in range(num_epochs):
        losses_tr = []
        for x in torch.utils.data.DataLoader(df, batch_size = 32, shuffle = False):
            x = x.to(device)
            outputs = model(x)
            loss = criterion(outputs, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_tr.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {torch.mean(torch.tensor(losses_tr))}')
    torch.save(model, model_path)
    print(f'Model {model_path} has been saved')
    return model

def execute_full_train(target_y=2022, target_q='3q', 
                       architecture = 'simple', model_tag='z11feat', create_feat_flg = True, epoches = 20,
                       train_flg = True, thresh=0.3, test_data_from_=None,
                       test_data_to_=None):
    '''
    Подгрузка данных для обучения, обучение, тестирование, создание реестра аномалий для тестовых данных.
    '''
    # # Вычисление периодов для обучения
    if test_data_to_ is None:
        train_data_from_, test_data_from_ = get_training_period(target_y, target_q)
        train_data_to_ = (pd.Timestamp(test_data_from_) - pd.Timedelta('1d')).strftime('%Y-%m-%d')
        # Вычисление периодов для тестирования
        test_start, test_end = get_quarter_start_end(target_y, target_q)
        test_data_to_ = (pd.Timestamp(test_end) - pd.Timedelta('1d')).strftime('%Y-%m-%d')

        train_data_from_ = pd.Timestamp(train_data_from_).strftime('%Y-%m-%d')
        test_data_from_ = pd.Timestamp(test_data_from_).strftime('%Y-%m-%d')
        print(f"Период обучения: с {train_data_from_} по {train_data_to_}")
        print(f"Период тестирования: с {test_data_from_} по {test_data_to_}")

    test_df = vanilla_prepare_data.Vanilla_Dataset((test_data_from_, test_data_to_), 'test', 
                                                   create_feat_flg = create_feat_flg)
    test_time_column, test_scaler = test_df.time_column.reset_index(drop=True), test_df.scaler
    
    #отскалировать обратно аномальные значения
    test_true_df = test_scaler.inverse_transform(test_df.X)
    test_true_df = pd.DataFrame(test_true_df, columns = test_df.columns_list)
    test_true_df.set_index(test_time_column, inplace = True)
    test_true_df = test_true_df['Температура масла в баке']
    print('ПРИЗНАКИ ДЛЯ ОБУЧЕНИЯ:', test_df.columns_list)

    model_path = f'../models/{architecture}-{model_tag}_stand_{target_y}_{target_q}.pth'
    if train_flg:
        train_df = vanilla_prepare_data.Vanilla_Dataset((train_data_from_, train_data_to_), 'train', 
                                                        create_feat_flg = create_feat_flg)
        train_time_column, train_scaler = train_df.time_column.reset_index(drop=True), train_df.scaler

        input_size = train_df.X.shape[1]
        model = Autoencoder(input_size, architecture=architecture)
        train_model(model, train_df, model_path, epoches=epoches)

    true_vals, pred_vals = eval_model(model_path, test_df)
    mae = mean_absolute_error(true_vals, pred_vals)
    mse = mean_squared_error(true_vals, pred_vals)
    mape = mean_absolute_percentage_error(true_vals, pred_vals)
    r2 = r2_score(true_vals, pred_vals)
    metrics = {'MAE': mae, 'MSE': mse, 'MAPE': mape, 'r2_score': r2}
    print(f'Метрики тестирования: {metrics}')
        
    mae_df = get_mae(true_vals, pred_vals, test_time_column, test_df.columns_list)
    mae_df.to_pickle(f'../reports/thr={thresh}_{model_tag}_{target_y}_{target_q}_MAE.pickle')
    anomaly_vals = analyse.convert_anomaly_scores_to_p_value_with_signal_contribution(mae_df.copy(), 
                                                                                  mae_df.columns.to_list(), 
                                                                                  long_window=20000, 
                                                                                  short_window=120,
                                                                                  min_period=6,
                                                                                  alpha=thresh)
    
    combined_df = get_pred_fact(true_vals, pred_vals, test_time_column, test_df.columns_list, test_scaler)
    alert_df =  alert_df = analyse.get_alerts(anomaly_vals, test_data_from_, test_data_to_, thresh = thresh)
    return alert_df, combined_df, anomaly_vals
       
def rescale(test_df, test_scaler):
    timestamps = test_df.index
    original_values = test_scaler.inverse_transform(test_df)
    original_df = pd.DataFrame(original_values, index=timestamps, columns=test_df.columns)
    return original_df
                                                                                     
def eval_model(model_path, test_dataset):
    '''
    Тестирование модели на переданных данных.
    '''
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    
    true_vals = test_dataset
    pred_vals = []
    for x in torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = False):
        x_pred = model(x)
        pred_vals.extend(x_pred.detach().numpy())
    true_vals = [x.detach().numpy() for x in true_vals]
    return true_vals, pred_vals

def get_mae(true_vals, pred_vals, test_time_column, column_list):
    '''
    Возвращает значения абсолютной ошибки для тестируемых данных.
    '''
    mae_test = analyse.row_mae(true_vals, pred_vals)
    mae_df = pd.DataFrame(mae_test, columns = column_list)
    mae_df['Time'] = test_time_column
    mae_df = mae_df.set_index('Time')
    return mae_df

def get_pred_fact(true_vals, pred_vals, test_time_column, column_list, test_scaler):
    '''
    Создание объединенного набора данных с реальными и предсказанными значениями.
    '''
    predict_df = pd.DataFrame(pred_vals, columns = column_list)
    predict_df['Time'] = test_time_column
    predict_df = predict_df.set_index('Time')

    test_df = pd.DataFrame(true_vals, columns = column_list)
    test_df['Time'] = test_time_column
    test_df = test_df.set_index('Time')
    
    test_df = rescale(test_df, test_scaler)
    predict_df = rescale(predict_df, test_scaler)
    
    combined_df = test_df.join(predict_df, how='outer', lsuffix='_test', rsuffix='_pred')
    return combined_df
      
def iterate_quarters(start_year, start_quarter, end_year, end_quarter):
    """ Перебирает все кварталы между двумя заданными временными точками. """
    quarters = ['1q', '2q', '3q', '4q']
    for year in range(start_year, end_year + 1):
        start_q_index = quarters.index(start_quarter) if year == start_year else 0
        end_q_index = quarters.index(end_quarter) if year == end_year else 3

        for q_index in range(start_q_index, end_q_index + 1):
            yield year, quarters[q_index]

def execute(test_y_start=2022, test_q_start='3q', test_y_end=2023, test_q_end='3q', 
            architecture = 'simple', model_tag='z11feat', create_feat_flg = True,epoches=20, train_flg = True,
            thresh = 0.3, test_data_from_=None, test_data_to_=None):
    '''
    Целевая общая функция запуска обучения или тестирования.
    '''
    alert_total = pd.DataFrame()
    fact_predict_total = pd.DataFrame()
    anomaly_vals_total = pd.DataFrame()
    if test_data_from_ is not None:
        alert_total, fact_predict_total, anomaly_vals_total = execute_full_train(
                                                                         architecture = architecture,
                                                                         model_tag=model_tag, 
                                                                         create_feat_flg=create_feat_flg,
                                                                         epoches=epoches, train_flg=train_flg,
                                                                         thresh = thresh, 
                                                                         test_data_from_=test_data_from_,
                                                                         test_data_to_=test_data_to_)
    else:
        for year, quarter in iterate_quarters(test_y_start, test_q_start, test_y_end, test_q_end):
            print(year, quarter)
            local_alerts, combined_df, anomaly_vals = execute_full_train(target_y=year, target_q=quarter,
                                                                         architecture = architecture,
                                                                         model_tag=model_tag, 
                                                                         create_feat_flg=create_feat_flg,
                                                                         epoches=epoches, train_flg=train_flg,
                                                                         thresh = thresh)
            # Объединение полученных датафреймов с общими
            alert_total = pd.concat([alert_total, local_alerts])
            alert_total.reset_index(drop= True , inplace= True )
            fact_predict_total = pd.concat([fact_predict_total, combined_df])
            combined_df.to_pickle(f'../reports/thr={thresh}_{model_tag}_{year}_{quarter}_combined_df.pickle')
            anomaly_vals_total = pd.concat([anomaly_vals_total, anomaly_vals])

    alert_total.to_excel(f'../reports/thr={thresh}_{model_tag}_{test_y_start}_{test_q_start}_{test_y_end}_{test_q_end}_alerts.xlsx')
    fact_predict_total.to_pickle(f'../reports/thr={thresh}_{model_tag}_{test_y_start}_{test_q_start}_{test_y_end}_{test_q_end}_fact_predict_total.pickle')
    anomaly_vals_total.to_pickle(f'../reports/thr={thresh}_{model_tag}_{test_y_start}_{test_q_start}_{test_y_end}_{test_q_end}_anomaly_vals_total.pickle')
    # Возвращение итоговых объединенных датафреймов
    return alert_total, fact_predict_total, anomaly_vals_total