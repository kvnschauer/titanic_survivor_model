from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler

class Model():
    data: DataFrame
    model_file_name = 'rand_forest_titanic_classifier.pkl'
    preprocessor: ColumnTransformer

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

        # feature importance is very low for any parch value other than 0 or 1, combine all others into a value '2'
        def __reduce_parch_noise(data: DataFrame):
            data['Parch'] = np.where(data['Parch'] > 2, 2, data['Parch'])
            return data

        def __reduce_sibsp_noise(data: DataFrame):
            data['SibSp'] = np.where(data['SibSp'] > 2, 2, data['SibSp'])
            return data

        return ColumnTransformer(transformers=[
            ('Hot_encoded', OneHotEncoder(sparse_output=True, handle_unknown='ignore'), ['Sex', 'Pclass', 'Embarked']),
            ('Parch', Pipeline([
                 ('reduce_noise', FunctionTransformer(func=__reduce_parch_noise, feature_names_out='one-to-one')),
                 ('hot_encode', OneHotEncoder(sparse_output=True, handle_unknown='ignore'))
            ]), ['Parch']),
            ('SibSp', Pipeline([
                ('reduce_noise', FunctionTransformer(func=__reduce_sibsp_noise, feature_names_out='one-to-one')),
                ('hot_encode', OneHotEncoder(sparse_output=True, handle_unknown='ignore'))
            ]), ['SibSp']),
            ('MinMax', MinMaxScaler(), ['TicketNumber', 'Age']),
            ('passthrough', 'passthrough', ['Fare']),# since we are using random forest, no need to scale
            ('drop_cols', 'drop', ['Name', 'Ticket', 'Cabin'])
        ], remainder='drop')

    def train_model(self, save_model = False):
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
        self.preprocessor = pre_processor.fit(self.data)
        X = pre_processor.transform(self.data)
        y = self.data.loc[:, 'Survived']

        print(f'feature names: {pre_processor.get_feature_names_out()}')

        rand_forest = RandomForestClassifier(random_state=22, n_jobs=-1, oob_score=True, max_depth=10, min_samples_split=6, n_estimators=100, max_features="log2")

        rand_forest.fit(X, y)

        print(f'Train score: {accuracy_score(rand_forest.predict(X), y)}')
        print(f'OOB score: {rand_forest.oob_score_}')

        if save_model:
            joblib.dump(rand_forest, self.model_file_name)

    def predict(self):
        # for now we'll assume it's a random forest classifier since that is what we've been working with
        predictor: RandomForestClassifier = joblib.load(self.model_file_name)

        test_data = pd.read_csv('./Data/test.csv')

        print(test_data.shape)

        self.__ticket_number_feature_extraction(test_data)

        X = self.preprocessor.transform(test_data)

        predictions = predictor.predict(X)

        prediction_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})

        date_time_now = datetime.now()
        day_time = '{}_{}_{}_{}_{}_{}'.format(date_time_now.month,
                                                 date_time_now.day,
                                                 date_time_now.year,
                                                 date_time_now.hour,
                                                 date_time_now.minute,
                                                 date_time_now.second)
        prediction_df.to_csv(f'./Data/predictions_{day_time}.csv', index=False)