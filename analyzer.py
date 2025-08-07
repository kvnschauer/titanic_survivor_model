import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

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

    def __build_fare_bar(self, save_graph):
        fig, ax = plt.subplots()

        bucket_size = 10
        counter = 10

        while counter < 200:
            fare_count = self.data.loc[(counter > self.data.Fare) & (self.data.Fare >= (counter - bucket_size))]
            ax.bar(counter, len(fare_count), label=counter, width=10)
            counter += bucket_size

        plt.ylabel('Count')
        plt.xlabel('Ticket Price')
        plt.xticks(range(0, 200, bucket_size))
        if save_graph:
            plt.savefig(self.folder_path + '\\fares.png', format='png')
        plt.show()

    def __build_transformed_scaled_bar(self, save_graph):
        fig, ax = plt.subplots()

        bucket_size = 0.1
        counter = 0.1
        max_val = 1

        # first transform data via log
        data_copy = self.data.copy()
        zero_cost_fares: DataFrame = data_copy.loc[data_copy.Fare == 0]
        data_copy = data_copy.loc[data_copy.Fare != 0]
        data_copy.Fare = data_copy.Fare.map(lambda x: np.log(x))

        # now min-max scale
        scaler = MinMaxScaler()
        data_copy.Fare = scaler.fit_transform(data_copy[['Fare']].values)

        while counter < max_val:
            fare_count = data_copy.loc[(counter > data_copy.Fare) & (data_copy.Fare >= (counter - bucket_size))]
            ax.bar(counter, len(fare_count), label=counter, width=bucket_size)
            counter += bucket_size
        ax.bar(0, len(zero_cost_fares), label=0, width=bucket_size)
        plt.ylabel('Count')
        plt.xlabel('Ticket Price')

        if save_graph:
            plt.savefig(self.folder_path + '\\fares_transformed_scaled.png', format='png')
        plt.show()

    def analyze_data(self):
        valid_input = ['y', 'n']
        user_input = None
        save_data = False

        while user_input not in valid_input:
            user_input = input('Save analyzed data (y/n)?\n').lower()

        if user_input == 'y':
            save_data = True

        # allow columns to display fully
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', None)
        trn_data = self.data.copy()

        high_level_analysis_txt = ('Data high level analysis:\n'
                                   f'Columns: {trn_data.columns}\n'
                                   f'Data shape {trn_data.shape}\n'
                                   f'Head: {trn_data.head()}\n\n')

        print(high_level_analysis_txt)


        null_cabins = self.data.loc[pd.isnull(self.data.Cabin)]
        non_null_cabins = self.data.loc[~pd.isnull(self.data.Cabin)]

        # analyze cabins feature
        cabin_feature_txt_analysis = ('Cabins feature analysis:\n'
                                      f'null cabins count {len(null_cabins)}\n'
                                      f'non null cabins count {len(non_null_cabins)}\n'
                                      f'null cabins survived ratio: {np.sum(null_cabins.Survived) / len(null_cabins)}\n'
                                      f'non null cabins survived ratio: {np.sum(non_null_cabins.Survived) / len(non_null_cabins)}\n\n')

        print(cabin_feature_txt_analysis)

        # analyze fare
        zero_fare = self.data.loc[self.data.Fare == 0]
        print(f'zero cost fare survival: {len(zero_fare.loc[zero_fare.Survived == 1]) / len(zero_fare)}')

        if save_data:
            os.remove('./Analysis/stats.txt')
            with open('./Analysis/stats.txt', 'a') as f:
                f.write(high_level_analysis_txt + cabin_feature_txt_analysis)
                f.close()

        # build graphs
        self.__build_age_bar(save_data)
        self.__build_male_female_pie(save_data)
        self.__build_fare_bar(save_data)
        self.__build_transformed_scaled_bar(save_data)