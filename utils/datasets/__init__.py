import os
from scipy.io import arff
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

class MTSDataset(Dataset):
    def __init__(self, inp, out, time_length, feature_length):
        self.input = inp
        self.output = out
        self.dims = time_length
        self.channels = feature_length

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        image = (self.input[index]).reshape(1,self.dims,self.channels)
        label = self.output[index]
        return image, label

class DatasetLoader:
    def __init__(self, folder_path):
        self.path = os.path.normpath(folder_path)
        self.train_path = os.path.join(self.path, os.path.basename(self.path) + '_TRAIN.arff')
        self.test_path = os.path.join(self.path, os.path.basename(self.path) + '_TEST.arff')

    @staticmethod
    def _load_arff(arff_path):
        return arff.loadarff(arff_path)

    def load_to_df(self):
        train_data = pd.DataFrame(self._load_arff(self.train_path)[0])
        test_data = pd.DataFrame(self._load_arff(self.test_path)[0])
        return train_data, test_data

    def load_to_nparray(self):
        train_data, test_data = self.load_to_df()
        train_X = np.array([pd.DataFrame(train_data.loc[i][0]).values for i in range(len(train_data))],dtype=np.float)
        train_y = np.array([int(float(train_data.loc[i][1])) -1 for i in range(len(train_data))],dtype=np.long)
        test_X = np.array([pd.DataFrame(test_data.loc[i][0]).values for i in range(len(test_data))],dtype=np.float)
        test_y = np.array([int(float(test_data.loc[i][1])) - 1 for i in range(len(test_data))],dtype=np.long)
        return train_X, train_y, test_X, test_y

    def get_torch_dataset_loader_auto(self, train_batch_size, test_batch_size):
        X_train, y_train, X_test, y_test = self.load_to_nparray()
        train_dataset = MTSDataset(X_train, y_train, X_train.shape[-1], X_train.shape[-2])
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        test_dataset = MTSDataset(X_test, y_test, X_test.shape[-1], X_test.shape[-2])
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        dataloaders = {}
        dataloaders['train'] = train_loader
        dataloaders['val'] = test_loader

        dataset_sizes = {}
        dataset_sizes['train'] = len(train_dataset)
        dataset_sizes['val'] = len(test_dataset)

        return dataloaders, dataset_sizes

    def get_torch_dataset_loader_custom(self, time_length, feature_length, train_batch_size, test_batch_size):
        X_train, y_train, X_test, y_test = self.load_to_nparray()
        train_dataset = MTSDataset(X_train, y_train, time_length, feature_length)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        test_dataset = MTSDataset(X_test, y_test, time_length, feature_length)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        dataloaders = {}
        dataloaders['train'] = train_loader
        dataloaders['val'] = test_loader

        return dataloaders
