from src.config import load_config
from src.data.features import basic_feature_engineering
from src.data.loader import load_raw_data
from src.data.preprocess import prepare_data
from src.env import get_env_variable, load_env
from src.logger import setup_logger
from src.models.evaluate import evaluate_model
from src.models.train import train_model


def main():
    load_env()

    config = load_config()
    logger = setup_logger(config)

    env = get_env_variable("ENV")
    logger.info(f"Starting {config['project']['name']} in {env} mode..")

    df = load_raw_data(config)
    logger.info(f"Loaded data with shape : {df.shape}")

    df = basic_feature_engineering(df)  
    logger.info(f"Data after feature engineering {df.shape}")
    #logger.info(f"Columns after features : {df.columns.tolist()}")

    X_train,X_test,y_train,y_test=prepare_data(df)
    logger.info(f"Train shape:{X_train.shape},Test shape{X_test.shape}")
    
    model= train_model(X_train,y_train)
    accuracy,report=evaluate_model(model,X_test,y_test)
    logger.info(f"Model Accuracy :{accuracy}")
    logger.info(f"\n{report}")

if __name__=='__main__':
    main()