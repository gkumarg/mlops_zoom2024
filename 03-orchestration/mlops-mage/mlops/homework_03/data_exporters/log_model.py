import mlflow
import pickle

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")
    with mlflow.start_run():

        _, dv, lr = data
        mlflow.set_tag("developer", "GG")

        # local_dir = "/tmp/artifact_downloads"
        # if not os.path.exists(local_dir):
        #     os.mkdir(local_dir)
        with open("preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor.b", artifact_path="preprocessor")

        model_info = mlflow.sklearn.log_model(sk_model=lr, artifact_path="models")

        with open("lin_reg.bin", "wb") as f_out:
            pickle.dump(lr, f_out)
        mlflow.log_artifact(local_path="lin_reg.bin", artifact_path="models")

