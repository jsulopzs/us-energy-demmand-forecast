from metaflow import FlowSpec, step


class DemandForecastFlow(FlowSpec):

    @step
    def start(self):
        import pandas as pd

        self.df_base = pd.read_csv('../data/general/pjm_pivot.csv', index_col=0, parse_dates=True)
        self.regions = ['AE', 'AEP', 'AP', 'ATSI', 'BC']
        # [ 'AE', 'AEP', 'AP', 'ATSI', 'BC', 'CE', 'DAY', 'DEOK', 'DOM', 'DPL', 
        # 'DUQ', 'EKPC', 'JC', 'ME', 'PE', 'PEP', 'PL', 'PN', 'PS', 'RECO']
        self.next(self.prepare_data, foreach='regions')

    @step
    def prepare_data(self):
        from sklearn.model_selection import train_test_split
        import os

        self.region = self.input
        self.df = self.df_base.loc[:, [self.region]]
        self.train, self.test = train_test_split(self.df, test_size=0.3, shuffle=False)
        self.test, self.prod = train_test_split(self.test, test_size=0.5, shuffle=False)
        self.path_region = f'../data/regions/{self.region}'

        if not os.path.exists(self.path_region):
            os.makedirs(self.path_region)

        for df, name in zip([self.train, self.test, self.prod], ['train', 'test', 'prod']):
            df.to_csv(f'{self.path_region}/{name}.csv')
        self.next(self.create_sequences)

    @step
    def create_sequences(self):
        from sklearn.preprocessing import MinMaxScaler
        from us_energy_demmand_forecast.utils import create_sequences

        scaler = MinMaxScaler(feature_range=(0, 1))
        self.train_norm = scaler.fit_transform(self.train)
        self.test_norm = scaler.transform(self.test)
        self.sequence_length = 24 # Use 24 hours prior to predict the following hour
        self.X_train, self.y_train = create_sequences(self.train_norm, self.sequence_length)
        self.X_test, self.y_test = create_sequences(self.test_norm, self.sequence_length)
        self.next(self.train_model)

    @step
    def train_model(self):
        import lightgbm as lgb
        self.train_data = lgb.Dataset(self.X_train, label=self.y_train)
        self.val_data = lgb.Dataset(self.X_test, label=self.y_test)
        self.lgb_params = {'metric': {'mae'}, 'num_leaves': 10, 'learning_rate': 0.01, 'max_depth': 5}
        self.num_round=10
        # self.model = lgb.train(self.lgb_params, self.train_data, self.num_round, valid_sets=[self.val_data])
        self.x = 1
        self.next(self.combine)

    @step
    def combine(self, inputs):
        print('total is %d' % sum(input.x for input in inputs))
        self.next(self.end)

    @step
    def end(self):
        print('Demand forecast trained succesfully')
 

if __name__ == '__main__':
    DemandForecastFlow()