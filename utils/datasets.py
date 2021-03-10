from scipy.io import arff
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
        train_y = np.array([int(float(train_data.loc[i][1])) for i in range(len(train_data))],dtype=np.float)
        test_X = np.array([pd.DataFrame(test_data.loc[i][0]).values for i in range(len(test_data))],dtype=np.float)
        test_y = np.array([int(float(test_data.loc[i][1])) for i in range(len(test_data))],dtype=np.float)
        return train_X, train_y, test_X, test_y
