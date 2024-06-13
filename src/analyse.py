import numpy as np
import pandas as pd
from scipy.stats import chi2
import operator

from get_mks_5000_ML_dataset import *

def flatten_timeseries(data):
    '''
    Из исходного списка списков для обучения сделать один плоский список
    '''
    new_data = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            new_data.append(data[i][j])
    return new_data

def row_mae(list1, list2):
    return [abs(list1[i] - list2[i]) for i in range(len(list1))]

def convert_anomaly_scores_to_p_value(inp_df, score_cols, long_window=20000, short_window=120, min_period=10, alpha=0.05):
    """
    Преобразование оценок риска в p-value и метки аномалий для многомерных данных

    Входные данные:
        inp_df: DataFrame, содержащий предсказания модели (индекс должен быть датой)
        score_cols: список строк, имена столбцов, содержащих оценки риска
    Вывод:
        inp_df: тот же DataFrame с новыми столбцами 'p_value' и 'is_anomaly'
    """
    # Вычисление скользящего среднего и ковариационной матрицы для длинного окна
    mean_distribution = inp_df[score_cols].rolling(window=long_window, min_periods=min_period).mean()
    cov_distribution = inp_df[score_cols].rolling(window=long_window, min_periods=min_period).cov(pairwise=True)

    p_values = np.full(inp_df.shape[0], np.nan)
    is_anomaly = np.full(inp_df.shape[0], np.nan, dtype=bool)

    for i in range(inp_df.shape[0] - short_window + 1):
        # Вычисление среднего значения для короткого окна
        short_window_data = inp_df.iloc[i:i+short_window][score_cols]
        short_window_mean = short_window_data.mean().values

        # Получение среднего значения и ковариационной матрицы для длинного окна
        long_window_mean = mean_distribution.iloc[i + short_window - 1].values
        cov_matrix = cov_distribution.iloc[len(score_cols) * (i + short_window - 1):(len(score_cols) * (i + short_window))].values

        if not np.any(np.isnan(long_window_mean)) and not np.any(np.isnan(short_window_mean)) and not np.any(np.isnan(cov_matrix)):
            # Вычисление расстояния Махаланобиса
            delta = short_window_mean - long_window_mean
            mahalanobis_distance = np.dot(np.dot(delta.T, np.linalg.inv(cov_matrix)), delta)

            # Вычисление p-value, используя распределение хи-квадрат
            p_value = 1 - chi2.cdf(mahalanobis_distance, df=len(score_cols))
            p_values[i + short_window - 1] = p_value

            # Определение, является ли наблюдение аномалией
            is_anomaly[i + short_window - 1] = p_value < alpha

    inp_df["p_value"] = p_values
    inp_df["is_anomaly"] = is_anomaly

    inp_df["p_value"].fillna(method='bfill', inplace=True)
    inp_df["is_anomaly"].fillna(False, inplace=True)

    return inp_df

def convert_anomaly_scores_to_p_value_with_signal_contribution(inp_df, score_cols, long_window=20000, short_window=120, min_period=10, alpha=0.05, fill_control_flg = False):
    """
    Преобразование оценок риска в p-value и метки аномалий для многомерных данных с расчетом вклада каждого сигнала

    Входные данные:
        inp_df: DataFrame, содержащий предсказания модели (индекс должен быть датой)
        score_cols: список строк, имена столбцов, содержащих оценки риска
    Вывод:
        inp_df: тот же DataFrame с новыми столбцами 'p_value', 'is_anomaly' и вкладами каждого сигнала
    """
    
    # Вычисление скользящего среднего и ковариационной матрицы для длинного окна
    mean_distribution = inp_df[score_cols].rolling(window=long_window, min_periods=min_period).mean()
    cov_distribution = inp_df[score_cols].rolling(window=long_window, min_periods=min_period).cov(pairwise=True)

    p_values = np.full(inp_df.shape[0], np.nan)
    is_anomaly = np.full(inp_df.shape[0], np.nan, dtype=bool)

    # Для хранения вкладов сигналов
    signal_contributions = {col: [] for col in score_cols}

    for i in range(inp_df.shape[0] - short_window + 1):
        # if i % 10000 == 0:
        #     print(i)
        # Вычисление среднего значения для короткого окна
        short_window_data = inp_df.iloc[i:i+short_window][score_cols]
        short_window_mean = short_window_data.mean().values

        # Получение среднего значения и ковариационной матрицы для длинного окна
        long_window_mean = mean_distribution.iloc[i + short_window - 1].values
        cov_matrix = cov_distribution.iloc[len(score_cols) * (i + short_window - 1):(len(score_cols) * (i + short_window))].values

        if not np.any(np.isnan(long_window_mean)) and not np.any(np.isnan(short_window_mean)) and not np.any(np.isnan(cov_matrix)):
            # Вычисление расстояния Махаланобиса
            delta = short_window_mean - long_window_mean
            mahalanobis_distance = np.dot(np.dot(delta.T, np.linalg.inv(cov_matrix)), delta)

            # Вычисление p-value, используя распределение хи-квадрат
            p_value = 1 - chi2.cdf(mahalanobis_distance, df=len(score_cols))
            p_values[i + short_window - 1] = p_value

            # Определение, является ли наблюдение аномалией
            is_anomaly[i + short_window - 1] = p_value < alpha

            # Вычисление вклада каждого сигнала
            for idx, col in enumerate(score_cols):
                modified_delta = np.array(delta)
                modified_delta[idx] = 0  # Исключение вклада текущего сигнала
                modified_distance = np.dot(np.dot(modified_delta.T, np.linalg.inv(cov_matrix)), modified_delta)
                contribution = mahalanobis_distance - modified_distance
                signal_contributions[col].append(contribution)

    inp_df["p_value"] = p_values
    inp_df["is_anomaly"] = is_anomaly
    for col, contributions in signal_contributions.items():
        inp_df[f"{col}_contribution"] = contributions + [np.nan] * (inp_df.shape[0] - len(contributions))

    inp_df["p_value"].fillna(method='bfill', inplace=True)
    inp_df["is_anomaly"].fillna(False, inplace=True)
    for col in score_cols:
        inp_df[f"{col}_contribution"].fillna(method='bfill', inplace=True)

    return inp_df

def get_alerts(anomaly_vals: pd.DataFrame, date_from: pd.Timestamp, date_to: pd.Timestamp, thresh = 0.5, temp_column = None):
    '''
    Создание реестра модели
    
    filename - путь до файла для сохранения
    anomaly_vals - датафрейм с pvalue и вкладом фичей
    date_from - начало периода обработки
    date_to - конец периода обработки
    thresh - граница alert для pvalue
    '''
    result_dict = {'time_start' : [],
                   'time_end' : [],
                   'real_stop_desc' : [],
                   'real_stop_time' : [],
                   'good_alert' : [],
                   'top_signals' : [],
                   'diff_time' : []}
    #реальные остановы
    outage_df = get_outage_df(from_=date_from, to_=date_to, use_default_source=True)
    cfg_part_list = ["Гидростанция системы высокого давления"]
    outage_list = []
    for _, (current_outage_start, current_outage_end,o_part_,o_type_,o_comment_,o_vnu) in outage_df[['Дата начала', 'Дата окончания', 'Узел/деталь','Служба','Комментарий оператора/мастера', 'Не ВНУ']].iterrows():
        outage_list.append({"from": current_outage_start, 'to': current_outage_end, "text": o_part_+"; "+o_type_+"; "+o_comment_[0:100], "vnu" : o_vnu})
    
    #обработка алертов
    anomaly_vals = anomaly_vals.loc[date_from: date_to]
    anomaly_vals['p_value'] = anomaly_vals['p_value'].rolling('2H').mean().fillna(value=np.nan)
    is_alert = False
    ignore = False
    alert_start_time = ''
    alert_stop_time = ''
    for index, row in anomaly_vals.iterrows():
        if alert_start_time != '':
            if index - alert_start_time > pd.Timedelta('2H') and not ignore:
                is_alert = False
                ignore = True
                #поиск информации по остановам в период alert
                descr, times, diffs  = '', '', ''
                i = 0
                continue_flg = False
                for el in outage_list:
                    if el['from'] < alert_start_time < el['to'] + pd.Timedelta('1H'):
                        alert_stop_time = ''
                        alert_start_time = ''
                        ignore = False
                        continue_flg = True
                        break
                    if el['from'] >= alert_start_time and el['to'] <= (alert_stop_time + pd.Timedelta('10d')) and 'Гидростанция системы высокого давления' in el['text'] and el['vnu'] != 1:
                        i += 1
                        descr += str(i) + ')' + el['text'] + '\n'
                        times += str(i) + ')' + el['from'].strftime('%Y-%m-%d %H:%M') + '\n'
                        diffs += str(i) + ')' + str(el['from'] - alert_start_time) + '\n'
                if continue_flg:
                    continue
                result_dict['time_end'].append(alert_stop_time.strftime('%Y-%m-%d %H:%M'))
                result_dict['time_start'].append(alert_start_time.strftime('%Y-%m-%d %H:%M'))
                signals = []
                for column in anomaly_vals.columns.to_list():
                    if 'contrib' in column:
                        signals.append({'column' : column,
                                    'value' : anomaly_vals.loc[alert_start_time][column]})
                signals.sort(key=operator.itemgetter('value'), reverse=True)
                signals = [el['column'][:-13] for el in signals]
                result_dict['top_signals'].append('\n'.join(signals[:5]))
                if len(descr) > 0:
                    result_dict['good_alert'].append(True)
                else:
                    result_dict['good_alert'].append(False)
                result_dict['real_stop_desc'].append(descr)
                result_dict['real_stop_time'].append(times)
                result_dict['diff_time'].append(diffs)
                alert_stop_time = ''
                continue
        
        if alert_start_time != '':
            if index - alert_start_time > pd.Timedelta('12H'):
                ignore = False
                alert_start_time = ''
                alert_stop_time = ''
        
        if not is_alert:   
            if row['p_value'] < thresh and not ignore:
                is_alert = True
                alert_start_time = index
                alert_stop_time = index
        else: #идет останова
            if row['p_value'] < thresh:
                alert_stop_time = index

    df = pd.DataFrame.from_dict(result_dict)
    df = df.rename(columns={'time_start' : 'Дата и время начала срабатывания предупреждения модели',
                   'time_end' : 'Дата и время окончания срабатывания предупреждения модели',
                   'real_stop_desc' : 'НШС узла в течение 10 дней',
                   'real_stop_time' : 'Время НШС',
                   'good_alert' : 'Признак: спрогнозировано ли НШС',
                   'top_signals' : 'Показатели, сработавшие в модели как определяющие предупреждение (список)',
                   'diff_time' : 'Срок от предупреждения до начала НШС'})
    return df