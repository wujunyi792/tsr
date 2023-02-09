import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class TrainingDataset(Dataset):
    def __init__(self):
        root = 'Dataset_2'
        path = os.path.join(root, 'Train.csv')
        data = pd.read_csv(path)

        self.data_path = data['Path']
        self.label = data['ClassId']
        for i in range(data.shape[0]):
            self.data_path[i] = os.path.join(root, self.data_path[i])
        self.x = np.load('x.npy').tolist()

    def __getitem__(self, item):
        # print(x)
        x = self.x[item]
        x = torch.tensor(x, dtype=torch.float)
        x = x.ravel()

        label = [0 for i in range(43)]
        label[self.label[item]] = 1
        label = torch.Tensor(label)

        return x, label

    def __len__(self):
        return len(self.data_path)


class TestingDataset(Dataset):
    def __init__(self):
        root = 'Dataset_2'
        path = os.path.join(root, 'Test.csv')
        data = pd.read_csv(path)

        self.data_path = data['Path']
        self.label = data['ClassId']
        for i in range(data.shape[0]):
            self.data_path[i] = os.path.join(root, self.data_path[i])
        self.x = np.load('test_x.npy').tolist()

    def __getitem__(self, item):
        # print(x)
        x = self.x[item]
        x = torch.tensor(x, dtype=torch.float)
        x = x.ravel()

        label = [0 for i in range(43)]
        label[self.label[item]] = 1
        label = torch.Tensor(label)

        return x, label

    def __len__(self):
        return len(self.data_path)


if __name__ == '__main__':
    test = TrainingDataset()
    test = DataLoader(test)

    Max = [0 for i in range(785)]
    Min = [9999 for i in range(785)]

    for x, label in test:
        for i in range(x.shape[1]):
            if Min[i] > x[0, i].item():
                Min[i] = x[0, i].item()

            if Max[i] < x[0, i].item():
                Max[i] = x[0, i].item()
    print(Min)
    print(Max)
