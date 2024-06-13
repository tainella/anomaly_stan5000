import pandas as pd

def get_idels(mode: str, from_: pd.Timestamp, to_: pd.Timestamp, units: list,
              idels_file_path=str) -> pd.DataFrame:
    idle_data = pd.read_csv(idels_file_path, parse_dates=[0,1,2], dayfirst=True, index_col=0)
    
    if len(units) > 0:
        idle_data = idle_data[idle_data['unit'].isin(units)]

    if mode == 'train':
        mask_type = idle_data['type'] == 'Пневм. и гидрооборудование'
        mask_type &= ~idle_data['not_internal_reliability']
        mask_type &= ~idle_data['fast_growth']
        
        idle_data.loc[mask_type, 'date_start'] = idle_data.loc[mask_type, 'date_start'] - pd.Timedelta('12h')
        idle_data.loc[mask_type, 'date_end'] = idle_data.loc[mask_type, 'date_end'] + pd.Timedelta('4h')

        idle_data.loc[~mask_type, 'date_start'] = idle_data.loc[~mask_type, 'date_start'] - pd.Timedelta('1h')
        idle_data.loc[~mask_type, 'date_end'] = idle_data.loc[~mask_type, 'date_end'] + pd.Timedelta('1h')
        
    elif mode == 'test':
        idle_data['date_end'] = idle_data['date_end'] + pd.Timedelta('1h')
        
    else:
        raise Exception(f'incorrect mode: "{mode}"')

    idle_data = idle_data[(idle_data['date_end'] > from_) & (idle_data['date_start'] < to_)]
    return idle_data

def filter_by_idels(df: pd.DataFrame, idle_data) -> pd.DataFrame:
    last_idle_end = df.index[0]
    data = []
    for _, (current_idle_start, current_idle_end) in idle_data.iloc[1:-1][['date_start', 'date_end']].iterrows():
        data.append(df.loc[last_idle_end: current_idle_start - pd.Timedelta('100ms')])
        last_idle_end = current_idle_end + pd.Timedelta('100ms')
    data.append(df.loc[last_idle_end:])
    return pd.concat(data)

def filter_by_values(df: pd.DataFrame, filter_values: dict, remove_filter_values: bool) -> pd.DataFrame:
    for tag in filter_values:
        sign, value = filter_values[tag].split(' ')
        value = eval(value)

        if sign == '>':
            mask = df[tag] > value
        elif sign == '<':
            mask = df[tag] < value
        elif sign == '==':
            mask = df[tag] == value
        else:
            raise Exception(f'incorrect sign: "{sign}"')

        df = df[mask]
        
    if remove_filter_values:
        df = df.drop(columns=filter_values.keys())
    return df

def filter_df(df: pd.DataFrame, mode: str, filter_units=[], filter_values={}, remove_filter_values=False, idels_file_path=\
              '/home/jupyter/mnt/s3/preprocessed-selection/data_markup/idles_with_markup_4_unit_2023_11_21.csv') ->pd.DataFrame:
    idle_data = get_idels(mode=mode, from_=df.index[0], to_=df.index[-1], units=filter_units, idels_file_path=idels_file_path)
    df = filter_by_idels(df, idle_data)
    df = filter_by_values(df, filter_values, remove_filter_values=remove_filter_values)
    return df