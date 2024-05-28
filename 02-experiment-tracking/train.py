import os
import pickle
import click
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
from scipy.sparse import issparse

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Enable autologging -- but sklearn autolog seems to have issue with sparse data
# mlflow.sklearn.autolog()

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# set experiment
mlflow.set_experiment("nyc_taxi_experiment_green")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def ensure_dense(data):
    if issparse(data):
        return data.toarray()
    return data

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    try:
        print("Loading training data...")
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        print("Data loaded successfully.")

        print("Data shapes after loading:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)

        # Ensure data is dense
        X_train = ensure_dense(X_train)
        X_val = ensure_dense(X_val)

        with mlflow.start_run():
            print("Starting MLflow run...")
            rf = RandomForestRegressor(max_depth=3, random_state=0)
            print("fitting model")
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            print("Model fitted successfully.")
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            print(f"RMSE: {rmse}")
            # Log additional metrics manually if needed
            mlflow.log_metric("rmse", rmse)

    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == '__main__':
    run_train()
