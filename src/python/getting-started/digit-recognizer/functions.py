import pandas as pd
import numpy as np

def read_data_from_csv():
    data1 = pd.read_csv('train.csv')
    train_data = data1.values[0:, 1:]
    train_label = data1.values[0:, 0]
    
    data2 = pd.read_csv('test.csv')
    test_data = data2.values[0:, 0:]
    
    return train_data, train_label, test_data