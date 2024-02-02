# us-energy-demmand-forecast

## Installion
This project uses NannyML Cloud for monitoring ML models. It automates data ingestion by leveraging the NannyML Cloud SDK. Unfortunately, the SDK hasn't been published on PyPI yet, which means you cannot install it via the regular Python channels. Instead, you'll have to clone the repository and install it from your local copy.

To install all the appropriate dependencies for this project, follow these steps:

1. Clone the NannyML Cloud SDK and install it in your environment.
2. Clone this project and install it in your environment.

```bash
git clone https://github.com/NannyML/nannyml-cloud-sdk.git
cd nannyml-cloud-sdk
pip install .

cd ..
git clone https://github.com/jsulopzs/us-energy-demmand-forecast.git
cd us-energy-demmand-forecast
pip install .
```

## Running the Demand Forecasting Flow
### Without ML Monitoring

```bash
python demand_forecast_flow.py run
```

### With ML Monitoring

```bash
python demand_forecast_flow.py run --NANNYML_CLOUD_INSTANCE_URL 'NANNYML_CLOUD_INSTANCE_URL' --NANNYML_CLOUD_API_TOKEN 'NANNYML_CLOUD_INSTANCE_URL'
```
Get free access to [NannyML Cloud](https://www.nannyml.com/nannyml-cloud) using the [Azure Manage Application](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/nannyml1682590100745.nannyml-managed?tab=Overview)

Check out the [docs](https://nannyml.gitbook.io/cloud/deployment/azure/azure-managed-application) to set up the app and get the instance url and api token.