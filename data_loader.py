from torch.utils.data import Dataset
import sys
import torch
import os
import re
import numpy as np

class StaticSinrDataset(Dataset):
    def __init__(self, folder) -> None:
        super().__init__()
        file_path = []
        for (dirpath, dirnames, filenames) in os.walk(folder):
            x_path = None
            y_path = None
            for f in filenames:
                if f.endswith('x.npy'):
                    x_path = dirpath + f
                if f.endswith('y.npy'):
                    y_path = dirpath + f
            if x_path != None and y_path != None:
                file_path.append((x_path, y_path)) 
        self.xs = np.load(file_path[0][0])
        self.ys = np.load(file_path[0][1])
        self.len = len(self.xs)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return {'x': self.xs[idx], 'y': self.ys[idx]}

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    folder = '/home/rui/work/static-sinr-model/data/train/'
    dataset = StaticSinrDataset(folder)
    print(dataset[100])
