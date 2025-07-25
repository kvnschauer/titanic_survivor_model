import os

import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame


class Analyzer:

    data: DataFrame
    folder_path = './Analysis'
    def __init__(self, data):
        self.data = data

    def __build_age_bar(self, save_graph):
        grouped_ages = self.data.Age.value_counts().sort_index()

        fig, ax = plt.subplots()
        counter = 1
        for index, count in grouped_ages.items():
            ax.bar(counter, count, label=index)
            counter += 1

        plt.ylabel('Count')
        plt.xlabel('Age')
        if save_graph:
            plt.savefig(self.folder_path + '\\age_bar_graph.png', format='png')
        plt.show()

    def __build_male_female_pie(self, save_graph):
        grouped_sexes = self.data.Sex.value_counts()
        plt.pie([grouped_sexes.iloc[0], grouped_sexes.iloc[1]], labels=[grouped_sexes.index[0], grouped_sexes.index[1]], autopct='%1.1f%%')
        if save_graph:
            plt.savefig(self.folder_path + '\\male_female_pie.png', format='png')
        plt.show()

    def analyze_data(self):
        valid_input = ['y', 'n']
        user_input = None
        save_data = False

        while user_input not in valid_input:
            user_input = input('Save analyzed data (y/n)?\n').lower()

        if user_input == 'y':
            save_data = True

        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', None)
        trn_data = self.data.copy()

        print(f'Columns: {trn_data.columns}')
        print(f'Data shape {trn_data.shape}')
        print(f'Head: {trn_data.head()}')

        # Analyze Age feature
        self.__build_age_bar(save_data)
        self.__build_male_female_pie(save_data)