import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature
import lightgbm as lgb

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path: str) -> pd.DataFrame:
    """Load preprocessed test data for evaluation"""
    try:
        df = pd.read_csv(data_path)
        df.fillna('', inplace = True)
        logger.debug(f'Test data loaded successfully fromm {data_path}')
        return df
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading test data from {data_path}: {e}')
        raise

def load_model(model_path: str) -> lgb.LGBMClassifier:
    """Load trained model"""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f'Model loaded successfully from {model_path}')
        return model
    except Exception as e:
        logger.error(f'Unexpected error while loading model from {model_path}: {e}')
        raise

def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load TF-IDF vectorizer"""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug(f'Vectorizer loaded successfully from {vectorizer_path}')
        return vectorizer
    
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading vectorizer at {vectorizer_path}: {e}')
        raise

def load_params(params_path: str) -> dict:
    """Load params from a path"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Parameters loaded from {params_path}')
        return params
    except Exception as e:
        logger.error(f'Error loading parameters from {params_path}: {e}')
        raise

def evaluate_model(model: lgb.LGBMClassifier, X_test: np.ndarray, y_test: np.ndarray) -> tuple[dict, np.ndarray]:
    """Evaluate the model and log classification metrics and confusion matrix"""
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logger.debug(f'Model evaluation completed')
        return report, cm

    except Exception as e:
        logger.error(f'Error occurred while evaluating model: {e}')
        raise

def log_confusion_matrix(cm: np.ndarray, dataset_name) -> None:
    """Log confusion matrix as an artifact"""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion matrix for {dataset_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        cm_file_path = f'confusion_matrix_{dataset_name}.png'
        plt.savefig(cm_file_path)
        mlflow.log_artifact(cm_file_path)
        plt.close()
        logger.debug('Confusion matrix logged successfully')
    except Exception as e:
        logger.error(f'Unexpected Error occurred while logging confusion matrix: {e}')

def save_model_info(run_id: str, model_uri: str, file_path: str) -> None:
    """Save the model run_id, local path, and S3 path to a JSON file."""
    try:
        model_info = {
            'run_id': run_id,
            'model_uri': model_uri,
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug(f'Model info saved to {file_path}')
    
    except Exception as e:
        logger.error(f'Error occurred while saving the model info to {file_path}: {e}')
        raise

def main():
    mlflow.set_tracking_uri('http://ec2-18-141-141-49.ap-southeast-1.compute.amazonaws.com:5000')
    mlflow.set_experiment('dvc-pipeline-runs')

    with mlflow.start_run() as run:
        try:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_preprocessed.csv'))
            model = load_model(os.path.join(root_dir, 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            for param, value in params.items():
                mlflow.log_param(param, value)

            X_test = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            input_example = pd.DataFrame(X_test.toarray()[:5], columns=vectorizer.get_feature_names_out())
            signature = infer_signature(input_example, model.predict(X_test[:5]))

            mlflow.sklearn.log_model(
                model,
                'lgbm_model',
                signature=signature,
                input_example=input_example
            )

            # Get the S3 path of the logged model
            model_uri = mlflow.get_artifact_uri('lgbm_model')
            save_model_info(run.info.run_id, model_uri, 'experiment_info.json')

            mlflow.log_artifact(os.path.join(root_dir, 'tfidf_vectorizer.pkl'))

            report, cm = evaluate_model(model, X_test, y_test)

            for label, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    mlflow.log_metrics({
                        f'test_{label}_precision': metrics['precision'],
                        f'test_{label}_recall': metrics['recall'],
                        f'test_{label}_f1-score': metrics['f1-score']
                    })
                elif isinstance(metrics, (int, float)):
                    mlflow.log_metric(f'test_{label}', metrics)
            
            log_confusion_matrix(cm, "Test Data")

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")
        
        except Exception as e:
            logger.error(f'Failed to complete the model evaluation: {e}')
            print(f'Error: {e}')

if __name__ == '__main__':
    main()


            