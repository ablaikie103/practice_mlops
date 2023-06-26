import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logs import MyLogger


def clean_diabetes_dataset(config):
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    logger = MyLogger(config["base"]["log_file"])

    logger.info("Loading diabetes dataset from data/")

    df = pd.read_csv(config["data_load"]["dataset_csv"])

    logger.info("Cleaning diabetes dataset")
    df.dropna(inplace=True)

    logger.info("Splitting diabetes dataset into test and control groups")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data_clean"]["test_size"],
        random_state=config["base"]["random_state"],
    )

    # Save data to CSV files
    logger.info("Saving diabetes dataset to files")
    X_train.to_csv(config["data_clean"]["X_train"], index=False)
    X_test.to_csv(config["data_clean"]["X_test"], index=False)
    y_train.to_csv(config["data_clean"]["y_train"], index=False)
    y_test.to_csv(config["data_clean"]["y_test"], index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean diabetes dataset and split into test and control groups"
    )
    parser.add_argument("--config", type=str, help="path to YAML configuration file")
    args = parser.parse_args()
    clean_diabetes_dataset(args.config)
