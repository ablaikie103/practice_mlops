import argparse
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from mlem.api import save
from src.utils.logs import MyLogger

def train_diabetes_model(config, params):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    with open(params, 'r') as f:
        params = yaml.safe_load(f)
    logger = MyLogger(config['base']['log_file'])

    logger.info('Loading diabetes dataset')
    X_train = pd.read_csv(config['data_clean']['X_train'])
    y_train = pd.read_csv(config['data_clean']['y_train'])

    # Convert y_train to a 1D array
    y_train = y_train.values.ravel()

    logger.info('Training random forest model')
    model = RandomForestRegressor(n_estimators=params['train']['n_estimators'], max_depth=params['train']['max_depth'], random_state=config['base']['random_state'])
    
    model.fit(X_train, y_train)

    logger.info('Saving trained model to file')
    save(model, config['train']['model_file'], sample_data=X_train)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train random forest model on diabetes dataset')
    parser.add_argument('--config', type=str, help='path to YAML configuration file')
    parser.add_argument('--params', type=str, help='path to YAML parameter file')
    args = parser.parse_args()
    train_diabetes_model(args.config, args.params)