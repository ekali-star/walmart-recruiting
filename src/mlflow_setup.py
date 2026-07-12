import os
from dotenv import load_dotenv
import mlflow

def init_tracking():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=env_path)

    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if username is None or token is None:
        raise ValueError(f"Missing credentials — check .env exists at {os.path.abspath(env_path)}")

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(f"https://dagshub.com/{username}/walmart-recruiting.mlflow")