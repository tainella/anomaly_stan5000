import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class Data():
    def __init__(
        self, 
        dataset_path: str,
        columns: list,
        seq_len: int
    ):
        self.dataset_path = dataset_path
        self.columns = columns
        self.seq_len = seq_len
        
    def read_dataset(self):
        if self.dataset_path[-2:] =='gz':
            df = pd.read_csv(self.dataset_path, compression='gzip', usecols=self.columns)
        elif self.dataset_path[-3:] =='csv':
            df = pd.read_csv(self.dataset_path, usecols=self.columns)
        else:
            raise Exception("Формат отличается от .csv и .csv.gz!")
        return df
    
    def scale_data(self):
        scaler = StandardScaler()
        scaling_df = scaler.fit_transform(self.read_dataset())
        return scaler, scaling_df

    def prepare_data(
        self, 
        data: np.ndarray, 
    ):
        tensor = torch.tensor(data, dtype=torch.float32)
        tensor = tensor.reshape((-1, self.seq_len, data.shape[1]))
        return [tensor[i] for i in range(tensor.size(0))]

    def train_test_split(
        self,
        test_size: float,
    ):
        scaler, data = self.scale_data()
        train_count = int(int(test_size * len(data)) // self.seq_len * self.seq_len)
        test_count = int((len(data) - train_count) // self.seq_len * self.seq_len)
        train_data = self.prepare_data(data[:train_count])
        test_data = self.prepare_data(data[train_count:train_count+test_count])
        return train_data, test_data, scaler