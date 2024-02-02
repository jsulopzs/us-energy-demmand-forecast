from metaflow import FlowSpec, Parameter, step


class DemandForecastFlow(FlowSpec):

    # Optional parameters to monitor the models with NannyML Cloud
    # Get free access using the Azure Manage Application: 
    # https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nannyml1682590100745.nannyml-managed?tab=Overview
    # Follow the docs to set up the app: https://nannyml.gitbook.io/cloud/deployment/azure/azure-managed-application
    NANNYML_CLOUD_INSTANCE_URL = Parameter('NANNYML_CLOUD_INSTANCE_URL')
    NANNYML_CLOUD_API_TOKEN = Parameter('NANNYML_CLOUD_API_TOKEN')

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
        self.X_prod, _ = create_sequences(self.test_norm, self.sequence_length)
        self.next(self.train_model)

    @step
    def train_model(self):
        from lightgbm import LGBMRegressor
        self.model = LGBMRegressor(num_leaves=10, learning_rate=0.01, max_depth=5)
        self.model.fit(self.X_train[:, :, 0], self.y_train, eval_metric='mae', eval_set=[(self.X_test[:, :, 0], self.y_test)])
        self.model.booster_.save_model(f'../models/{self.region}_model.h5')
        self.next(self.make_predictions)
    
    @step
    def make_predictions(self):
        self.y_prod_pred = self.model.predict(self.X_prod[:, :, 0])
        self.next(self.prepare_monitoring_data)
    
    @step
    def prepare_monitoring_data(self):
        import pandas as pd
        self.y_test_pred = self.model.predict(self.X_test[:, :, 0])
        self.feature_names = [f"demand_{str(i + 1)}_days_ago" for i in range(self.sequence_length)]
        
        self.reference_df = pd.merge(pd.DataFrame(self.X_test[:, :, 0], columns=self.feature_names),
                                     pd.DataFrame({'demand':self.y_test[:, 0], 'predicted_demand':self.y_test_pred}), 
                                     left_index=True, right_index=True)
        self.reference_df['timestamp'] = pd.to_datetime(self.test.index[self.sequence_length - 1:-1], format="%Y%m%d%H%M%S")
        self.reference_df['id'] = self.reference_df['timestamp']


        self.prod_df = pd.merge(pd.DataFrame(self.X_prod[:, :, 0], columns=self.feature_names),
                                pd.DataFrame({'predicted_demand':self.y_prod_pred}), 
                                left_index=True, right_index=True)
        self.prod_df['timestamp'] = pd.to_datetime(self.prod.index[self.sequence_length - 1:-1], format="%Y%m%d%H%M%S")
        self.prod_df['id'] = self.prod_df['timestamp']

        self.next(self.monitor_model)

    @step
    def monitor_model(self):
        """
        For this step you need access to NannyML Cloud. You can try it for free
        using the Azure Manage Application: 
        https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nannyml1682590100745.nannyml-managed?tab=Overview
        Follow the docs to set up the app: https://nannyml.gitbook.io/cloud/deployment/azure/azure-managed-application
        """
        import nannyml_cloud_sdk as nml_sdk

        if self.NANNYML_CLOUD_INSTANCE_URL and self.NANNYML_CLOUD_API_TOKEN:

            nml_sdk.url = self.NANNYML_CLOUD_INSTANCE_URL
            nml_sdk.api_token = self.NANNYML_CLOUD_API_TOKEN
            schema = nml_sdk.Schema.from_df(
                problem_type='REGRESSION',
                df=self.reference_df,
                target_column_name='demand',
                prediction_column_name='predicted_demand',
                timestamp_column_name='timestamp',
                identifier_column_name='id'
            )

            nml_sdk.Model.create(
                name=f'MarvelousMLOps Demand Forecasting - Region {self.region}',
                schema=schema,
                chunk_period='WEEKLY',
                reference_data=self.reference_df,
                analysis_data=self.prod_df,
                main_performance_metric='MAE',
            )

        self.next(self.combine)

    @step
    def combine(self, inputs):
        # print('total is %d' % sum(input.x for input in inputs))
        self.next(self.end)

    @step
    def end(self):
        print('Demand forecast trained succesfully')
 

if __name__ == '__main__':
    DemandForecastFlow()