from analyzer import Analyzer
from model import Model
import pandas as pd

trn_data = pd.read_csv('./Data/train.csv')

analyzer = Analyzer(trn_data)
model = Model(trn_data)

continue_input = True
user_input = None
valid_user_input = {'1': analyzer.analyze_data, '2': lambda: model.train_model(save_model=True), '3': model.predict}

while continue_input:
    while user_input not in valid_user_input:
        user_input = input('Enter: \n'
                        '1. To analyze the data\n'
                        '2. To train the model\n'
                        '3. To predict with the model\n')

    valid_user_input[user_input]()

    while user_input != 'y' and user_input != 'n':
        user_input = input('Do you want to continue? (y/n)\n').lower()

    if user_input == 'n':
        continue_input = False
