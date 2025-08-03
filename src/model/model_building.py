import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger('model building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_root_directory() -> str:
    """Get the root directory of the project"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def load_params(params_path: str) -> dict:
    """Load paramters from a YAML file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Params loaded at {params_path}')
        return params
    
    except FileNotFoundError:
        logger.error(f'Yaml file not found at {params_path}')
        raise

    except yaml.YAMLError as e:
        logger.error(f'YAML error: {e}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a path"""
    try:
        df = pd.read_csv(data_path)
        df.fillna('', inplace=True)
        logger.debug(f'Data loaded successfully from {data_path}')
        return df
    
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the csv file: {e}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error while loading data: {e}')
        raise

def apply_tfidf(data: pd.DataFrame, ngram_range: tuple, max_features: int) -> tuple:
    """Apply TF-IDF vectorizer to the comments"""
    try:

        vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        X_train = data['clean_comment'].values
        y_train = data['category'].values

        X_train_vec = vectorizer.fit_transform(X_train)

        logger.debug(f'TF-IDF transformation completed. Train shape: {X_train_vec.shape}')

        with open(os.path.join(get_root_directory(), 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF applied with trigrams and data transformed')
        return X_train_vec, y_train
    
    except Exception as e:
        logger.error('Error during TF-IDF tranformation: %s', e)
        raise

def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM model"""
    try:
        best_model = lgb.LGBMClassifier(
            objective = 'multiclass',
            num_class = 3,
            metric = 'multi_logloss',
            is_unbalance = True,
            class_weight = 'balanced',
            reg_alpha = 0.1,
            reg_lambda = 0.1,
            learning_rate = learning_rate,
            max_depth = max_depth,
            n_estimators = n_estimators
        )
        best_model.fit(X_train, y_train)
        logger.debug(f'LightGBM model training completed')
        return best_model
    
    except Exception as e:
        logger.error(f'Error while training the LightGBM model: {e}')
        raise

def save_model(model: lgb.LGBMClassifier, model_path: str) -> None:
    """Save a trained LightGBM model"""
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f'Model saved to {model_path}')
    
    except Exception as e:
        logger.error(f'Error occurred while saving the model: {e}')

def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))

        data = load_data(os.path.join(root_dir, 'data/interim/train_preprocessed.csv'))

        ngram_range = tuple(params['model_building']['ngram_range'])
        # ngram_range = eval(ngram_range_str)  # Convert string to tuple
        max_features = params['model_building']['max_features']
        X_train, y_train = apply_tfidf(data, ngram_range=ngram_range, max_features=max_features)

        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']
        model = train_lgbm(X_train=X_train, y_train=y_train, learning_rate = learning_rate, max_depth=max_depth, n_estimators=n_estimators)

        save_model(model, os.path.join(root_dir, 'lgbm_model.pkl'))

    except Exception as e:
        logger.error(f'Failed to complete the feature engineering and the model building process: {e}')
        print(f'Error: {e}')

if __name__ == '__main__':
    main()


