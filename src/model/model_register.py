# register model

import json
import mlflow
import logging
import os

# Set up MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-18-141-141-49.ap-southeast-1.compute.amazonaws.com:5000")


# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug(f'Model info logged from {file_path}')
        return model_info
    except FileNotFoundError:
        logger.error(f'File not found at {file_path}')
        raise
    except Exception as e:
        logger.error(f'Error occurred while loading model info from {file_path}: {e}')
        raise

def register_model(model_name: str, model_info: dict) -> None:
    """Register the model to the Mlflow Model Registry"""
    try:
        model_uri = model_info['model_uri']
        
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging'
        )
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error(f'Error during model registration: {e}')
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        model_name = 'yt_chrome_plugin_model'
        register_model(model_info=model_info, model_name=model_name)
    
    except Exception as e:
        logger.error(f'Failed to complete the model registration process: {e}')
        print(f'Error: {e}')

if __name__ == "__main__":
    main()