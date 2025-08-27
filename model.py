
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler, minmax_scale
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

class Model():
    data: DataFrame

    def __init__(self, data):
        self.data = data.copy()

    @staticmethod
    def __ticket_prefix_feature_extraction(data: DataFrame):
        def extract_ticket_prefix(ticket: str):
            ticket_prefix = ''
            split_ticket = ticket.split()

            if len(split_ticket) == 1:
                if split_ticket[0] == 'LINE':
                    return split_ticket[0]
                else:
                    return ''

            # everything up to the last split would be ticket prefix
            for index, x in enumerate(split_ticket):
                if index < len(split_ticket) - 1:
                    ticket_prefix = ticket_prefix + x

            return ticket_prefix

        data['TicketPrefix'] = data['Ticket'].map(lambda x: extract_ticket_prefix(x))
        return data

    @staticmethod
    def __ticket_number_feature_extraction(data: DataFrame):
        def extract_ticket_number(ticket: str):
            split_ticket = ticket.split()

            # last value is always the ticket num unless its not numeric
            last_split = split_ticket[len(split_ticket) - 1]
            if last_split.isdecimal():
                return last_split
            else:
                return 0 # fill in ticket num with some value so it can be min maxed

        data['TicketNumber'] = data['Ticket'].map(lambda x: extract_ticket_number(x))
        return data

    @staticmethod
    def __graph_ticket_num():
        plt.subplots()

    def __get_preprocessor(self):
        def fare_log_transform(data: DataFrame):
            data['Fare'] = np.where(data['Fare'] > 0, data['Fare'], 1e-10)
            data['Fare'] = np.log(data['Fare'])
            return  data

        return ColumnTransformer(transformers=[
            ('Hot_encoded', OneHotEncoder(sparse_output=True), ['Sex', 'Parch', 'SibSp', 'Pclass', 'Embarked', 'TicketPrefix']),
            ('Fare', Pipeline([
                ('log_transform', FunctionTransformer(func=fare_log_transform)),
                ('min_max', MinMaxScaler())
            ]), ['Fare']),
            ('MinMax', MinMaxScaler(), ['TicketNumber', 'Age']),
            ('drop_cols', 'drop', ['Name', 'Ticket', 'Cabin', 'Survived'])
        ])

    def train_model(self):
        """
            Handles training the model and analyzing performance
            :return: None
        """

        self.__ticket_prefix_feature_extraction(self.data)
        self.__ticket_number_feature_extraction(self.data)

        # fill na values for Age feature with average for that sex
        female_average_age = np.average(
            self.data.loc[(self.data.Sex == 'female') & (~pd.isnull(self.data.Age)), ['Age']])
        male_average_age = np.average(self.data.loc[(self.data.Sex == 'male') & (~pd.isnull(self.data.Age)), ['Age']])

        female_average_age = np.round(female_average_age, decimals=2)
        male_average_age = np.round(male_average_age, decimals=2)

        self.data.Age = np.where((self.data.Sex == 'female') & (pd.isnull(self.data.Age)), female_average_age, self.data['Age'])
        self.data.Age = np.where((self.data.Sex == 'male') & (pd.isnull(self.data.Age)), male_average_age, self.data['Age'])

        pre_processor = self.__get_preprocessor()
        X = pre_processor.fit_transform(self.data)
        y = self.data.loc[:, 'Survived']

        #SGDClassifier
        #LinearRegression
        #LinearSVC
        #DecisionTreeClassifier

        sgd_classifier = SGDClassifier()
        linear_regressor = LinearRegression()
        svc = LinearSVC()
        decision_tree = DecisionTreeClassifier()

        scores = cross_val_score(decision_tree, X, y, cv=5)

        print(scores)