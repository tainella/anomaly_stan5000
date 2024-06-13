import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO

from get_mks_5000_ML_dataset import *

def plot_custom_p_value(true_vals: pd.DataFrame, pred_vals: pd.DataFrame, anomaly_vals: pd.DataFrame, column: str, \
                        date_from: pd.Timestamp, date_to: pd.Timestamp, alert_start, alert_end,
                        label='mae', thresh = 0.5, draw_stand = False, ax=None):
    '''
    Для одной колонки отрисовка реальных данных, предсказанных данных, подсчитанного по ним pvalue, столбцов останов по узлам
    
    Входные данные:
    true_vals - реальные тестовые данные
    pred_vals - предсказания модели
    anomaly_vals - датафрейм с pvalue и вкладом фичей
    column - колонка для отрисовки
    date_from - начало периода отрисовки
    date_to - конец периода отрисовки
    thresh - граница alert для pvalue
    '''
    true_vals = true_vals.loc[date_from: date_to, column].rolling('2h').mean()
    true_vals = true_vals.resample('1Min').first().fillna(value=np.nan)
    pred_vals = pred_vals.loc[date_from: date_to, column].rolling('2h').mean()
    pred_vals = pred_vals.resample('1Min').first().fillna(value=np.nan)
    anomaly_vals = anomaly_vals.resample('1Min').first().fillna(value=np.nan)
    anomaly_vals['p_value'] = anomaly_vals['p_value'].rolling('2H').mean().fillna(value=np.nan) # сглаживание
    
    ax.set_title(column)
    lines = []
    line1, = ax.plot(true_vals, alpha=0.3, color = 'g', label = 'факт')
    line2, = ax.plot(pred_vals, alpha=0.3, color = 'b', label = 'прогноз')
    ax2 = ax.twinx()
    ax.set_ylabel("Значение показателя")
    ax2.set_ylabel("Уровень аномальности")
    line3, = ax2.plot((1 - anomaly_vals.loc[date_from: date_to, 'p_value']) * 100, color = 'r', label = 'функция аномальности')
    line4 = ax2.axhline(y=(1 - thresh) * 100, color='k', linestyle='-', label = 'граница')
    lines = [line1, line2, line3, line4] # line_alert
        
    #закрасить клеть не в работе
    if draw_stand:
        stand_not_loaded = get_mks_5000_ML_dataset(data_from_=date_from, data_to_=date_to, data_columns_=[])
        stand_not_loaded = stand_not_loaded['stand_not_working_flg']
        st_start = ''
        not_loaded = False
        for index, row in stand_not_loaded.items():
            if (row == False) and (not_loaded == True):
                line7 = ax.axvspan(st_start, index, alpha=0.2, color='c', label='Клеть не в работе')
                not_loaded = False
                st_start = ''
            if (row == True) and (not_loaded == False):
                st_start = index
                not_loaded = True
        lines.append(line7)  
    ax.legend(bbox_to_anchor=(-0.04, 1), handles=lines, loc='upper right')
    ax.grid()
    
def text_to_rgba(text, filename):
    fig = plt.Figure(facecolor="none")
    text = text.split('\n')
    for i in range(len(text)):
        text[i] = text[i].replace('_trend_12hour',' (тренд за 12 часов)')
        text[i] = text[i].replace('_trend_12hour',' (тренд за 12 часов)')
        text[i] = text[i].replace('_trend_hour',' (тренд за 1 час)')
        text[i] = text[i].replace('_trend',' (тренд за 1 час)')
        text[i] = text[i].replace('_var',' (дисперсия за 12 часов)')
        text[i] = '• ' + text[i]
    text = '\n'.join(text)
    
    fig.text(0, 0, text, color = 'black')
    with BytesIO() as buf:
        fig.savefig(buf, dpi=200, format="png", bbox_inches="tight",
                    pad_inches=0)
        buf.seek(0)
        rgba = plt.imread(buf)
        plt.imsave(filename, rgba)
        
def plot_multiple_cols(COLS, test_df, predict_df, anomaly_vals_total, alert_start, 
                       alert_end, label='mae', thresh=70, draw_stand=False,
                      save_folder = '', main_features = None):
    # Расчет временного диапазона
    date_from = alert_start
    date_to = alert_end

    # Создание сетки подграфиков
    fig, axes = plt.subplots(nrows=len(COLS), ncols=1, figsize=(20, len(COLS) * 5))

    # Отрисовка графиков для каждого столбца
    for i, col in enumerate(COLS):
        ax = axes[i]
        # Пример вызова функции:
        plot_custom_p_value(test_df, predict_df, anomaly_vals_total, col, date_from, date_to, 
                                      label=label, thresh=thresh, draw_stand=draw_stand, ax=ax,
                                      alert_start=alert_start, alert_end=alert_end)

    plt.tight_layout()
    if not os.path.exists(f"../reports/images/{save_folder}"):
        os.mkdir(f"../reports/images/{save_folder}")
    plt.savefig(f"../reports/images/{save_folder}/hhp_{alert_start.strftime('%Y-%m-%d %H_%M_%S')}.png")
    plt.close()
    
    #Файл с названием топ главных фичей
    if main_features != None:
        visualize_1.text_to_rgba(main_features, f"../reports/images/{save_folder}/features_{alert_start.strftime('%Y-%m-%d %H_%M_%S')}.png")