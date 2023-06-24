import argparse
import yaml
from sklearn.datasets import load_diabetes
import pandas as pd
from src.utils.logs import MyLogger

def save_diabetes_dataset(config):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    logger = MyLogger(config['base']['log_file'])

    logger.info('Loading diabetes dataset')
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    logger.info('Saving diabetes dataset to file')
    df.to_csv(config['data_load']['dataset_csv'], index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save diabetes dataset to file')
    parser.add_argument('--config', type=str, help='path to YAML configuration file')
    args = parser.parse_args()
    save_diabetes_dataset(args.config)