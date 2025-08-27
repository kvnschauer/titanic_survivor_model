from analyzer import Analyzer
from model import Model
import pandas as pd

trn_data = pd.read_csv('./Data/train.csv')

analyzer = Analyzer(trn_data)
model = Model(trn_data)

user_input = None
valid_user_input = {'1': analyzer.analyze_data, '2': model.train_model}

while user_input not in valid_user_input:
    user_input = input('Enter: \n'
                    '1. To analyze the data\n'
                    '2. To train the model\n')

valid_user_input[user_input]()
