import sys
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import textwrap
import numpy as np
import os
sys.path.append('../src')
from anomaly_report_plotter_config2 import *
import time

def run_plot(
        data_df, zones, signals,
        description={"value": ''}, 
        legend = {"visible": False},
        config_map= {
            "max_advance": np.timedelta64(10, 'D'),
            "min_advance": np.timedelta64(20, 'm')
        }
    ):
    return Plot(
        data_df,
        zones,
        legend,
        signals,
        description,
        config_map
    )

def customwrap(s,width=30):
    return "<br>".join(textwrap.wrap(s,width=width))

def annotation_create(i, width):
    return f"<b>{customwrap(i['text'], width)}</b>" if 'text' in i else ''

font_size = 12

def calc_width(object_count, all_count, container_width = 1238):
    return int(round((container_width/all_count)/font_size*object_count, 0))

class Plot:
    def __init__(self, data_df, zones, legend, signals, description, config_map):
        
        if isinstance(data_df, pd.Series):
            data_df = data_df.to_frame()
        
        self.fig = go.Figure()
        self.data_df = data_df
        self.indexes = data_df.index.values
        column_names = list(self.data_df.columns)
        self.min_value = data_df[column_names[0]].min() 

        self.zones = zones
        self.legend = legend
        self.signals = signals
        self.config_map = config_map
        self.description = description
        
        self.__add_main_trace()
        self.__add_zones()
        self.__add_signals()
        self.__add_layout()
        self.__add_description()

    def __add_main_trace(self):
        column_names = list(self.data_df.columns)
        
        ut = time.time()

        if len(column_names) >= 2:
            self.data_df[str(ut) + '_wrk'] = self.data_df[
                ~self.data_df[column_names[1]]
#                 | self.data_df[column_names[1]].shift(1).fillna(True)
#                 | self.data_df[column_names[1]].shift(-1)
            ][column_names[0]]

            self.data_df[str(ut) + '_empty'] = self.data_df[
                self.data_df[column_names[1]]
                | self.data_df[column_names[1]].shift(1).fillna(True)
                | self.data_df[column_names[1]].shift(-1)
            ][column_names[0]]
            
        else: 
            self.data_df[str(ut) + '_wrk'] = self.data_df
            self.data_df[str(ut) + '_empty'] = np.nan

        self.fig.add_trace(go.Scatter(
            x=self.data_df.index.values,
            y=self.data_df[str(ut) + '_wrk'],

            **main_layout
        ))

        self.fig.add_trace(go.Scatter(
            x=self.data_df.index.values,
            y=self.data_df[str(ut) + '_empty'],
            **no_data_layout,
        ))
        
        self.data_df.drop(columns=[str(ut) + '_wrk', str(ut) + '_empty'], inplace=True)

    def __add_zones(self):
        if('outage' in self.zones):
            for i in self.zones['outage']:
                if 'from' in i and 'to' in i:
                    indexes_need = len(list(filter(lambda x: i['from'] <= x <=i['to'], self.indexes)))

                    self.fig.add_vrect(
                        x0=i['from'], x1=i['to'],
                        annotation_text=annotation_create(i, 90),
                        **outage_layout
                    )

                    self.fig.add_vrect(
                        x0=i['from'] - self.config_map['min_advance'], x1=i['from'] - self.config_map['max_advance'],
                        **orange_layout
                    )

        if('outage_add' in self.zones):
            for i in self.zones['outage_add']:
                if 'from' in i and 'to' in i:
                    indexes_need = len(list(filter(lambda x: i['from'] <= x <=i['to'], self.indexes)))

                    self.fig.add_vrect(
                        x0=i['from'], x1=i['to'],
                        annotation_text=annotation_create(i, 90),
                        **green_layout
                    )
        
    def __add_description(self):
        if 'text' in self.description:
            self.fig.add_annotation(
#                  x=self.indexes[0],
#                  y=self.min_value,
                text=annotation_create(self.description, 200), 
#                 xshift=14 * font_size,
#                 yshift= 2 * font_size,
                **description_layout
            )
            

    def __add_layout(self):
        layout_tmp = dict(
            width=1280,#1280,
            height=820,#720,
            margin=dict(l=10, r=15, t=15, b=450),
        )
        self.fig.update_layout(
            legend={**self.legend, **legend_layout},
            **layout_tmp,
            xaxis={
                "range": [self.indexes[0], self.indexes[-1]]
            }
        )
    
    def __add_signals(self):
        width = 90
        if 'treshold' in self.signals:
            for i in self.signals['treshold']:
                self.fig.add_hline(
                    y=i['value'],
                    annotation_text=annotation_create(i, width),
                    **treshold_layout
                )

        if 'signal' in self.signals:
            for i in self.signals['signal']:
                self.fig.add_vrect(
#                     x=(i['value'] - np.timedelta64(3, 'h')).astype(np.int64) / 1000,
                    x0=i['value'],x1=i['value'],
                    annotation_text=annotation_create(i, width),
                    **signal_layout
                )
        
    def show(self):
        self.fig.show()

    def save(self, name, ext='pdf', only_name=False):      
        if only_name:
            filename = f"{name}.{ext}"
        else:
            if not os.path.exists("images"):
                os.mkdir("images")
            filename = f"images/{name}.{ext}"
        try:
            os.remove(filename)
        except OSError:
            pass
        self.fig.write_image(filename)
    