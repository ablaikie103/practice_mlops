import argparse
import yaml
from mlem.api import load
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from src.utils.logs import MyLogger

def evaluate_diabetes_model(config):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    logger = MyLogger(config['base']['log_file'])
    logger.info('Loading diabetes dataset')
    X_test = pd.read_csv(config['data_clean']['X_test'])
    y_test = pd.read_csv(config['data_clean']['y_test'])

    logger.info('Loading trained model')
    model = load(config['train']['model_file'])

    logger.info('Making predictions on test data')
    y_pred = model.predict(X_test)

    logger.info('Calculating evaluation metrics')
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f'MSE: {mse:.2f}, R2: {r2:.2f}')
    logger.info('Saving evaluation metrics to file')
    with open(config['evaluate']['metrics_file'], 'w') as f:
        f.write(f"MSE: {mse:.2f}\nR2: {r2:.2f}")

    logger.info('Generating evaluation plots')
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.savefig(config['evaluate']['scatter_plot_file'])
    plt.clf()
    plt.plot(y_test, label='True Values')
    plt.plot(y_pred, label='Predictions')
    plt.legend()
    plt.savefig(config['evaluate']['line_plot_file'])

    logger.info('Evaluation complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate random forest model on diabetes dataset')
    parser.add_argument('--config', type=str, help='path to YAML configuration file')
    args = parser.parse_args()
    evaluate_diabetes_model(args.config)