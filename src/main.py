from src.config import load_config
from src.logger import setup_logger
from src.env import load_env,get_env_variable

def main():
    load_env()

    config = load_config()
    logger = setup_logger(config)

    env = get_env_variable("ENV")
    
    logger.info(f"Starting {config['project']['name']} in {env} mode..")

if __name__=='__main__':
    main()