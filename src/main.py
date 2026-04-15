from src.config import load_config
from src.data.features import basic_feature_engineering
from src.data.loader import load_raw_data
from src.data.preprocess import prepare_data
from src.env import get_env_variable, load_env
from src.logger import setup_logger
from src.models.evaluate import evaluate_model, plot_precision_recall
from src.models.explain import (
    compute_shap_values,
    explain_single_prediction,
    get_shap_dict,
    plot_feature_importance,
)
from src.models.train import train_model
from src.recovery.generator import generate_message, generate_message_llm
from src.rag.chain import generate_rag_answer
from src.rag.ingest import load_documents,create_vector_store
from src.rag.retriever import retrieve_context


def main():
    load_env()
    thresholds = [0.5, 0.4, 0.3, 0.2]

    config = load_config()
    logger = setup_logger(config)

    docs=load_documents("data/external/reviews.txt")
    vectorstore = create_vector_store(docs)

    query = "Why do users abandon due to price?"

    context = retrieve_context(vectorstore,query)
    
    answer = generate_rag_answer(context,query)

    print("\nRAG ANSWER:\n",answer)

    env = get_env_variable("ENV")
    logger.info(f"Starting {config['project']['name']} in {env} mode..\n\n")

    df = load_raw_data(config)
    logger.info(f"Loaded data with shape : {df.shape}\n\n")

    df = basic_feature_engineering(df)  
    logger.info(f"Data after feature engineering {df.shape}\n\n")
    logger.info(f"Columns after features : {df.columns.tolist()}\n\n")

    X_train,X_test,y_train,y_test=prepare_data(df)
    logger.info(f"Train shape:{X_train.shape},Test shape{X_test.shape}\n\n")
    
    model= train_model(X_train,y_train)
    for t in thresholds:
        accuracy,report=evaluate_model(model,X_test,y_test,threshold=t)
        logger.info(f"AT THRESHOLD: {t}")
        logger.info(f"Accuracy :{accuracy}")
        logger.info(f"\n{report}\n\n")
    
    plot_precision_recall(model,X_test,y_test)
    
    X_sample=X_test.sample(200,random_state=42)
    shap_values=compute_shap_values(model,X_sample)

    plot_feature_importance(shap_values)

    explain_single_prediction(model,X_sample,index=0)

    shap_dict=get_shap_dict(shap_values,X_sample,index=0)
    logger.info(f"Top SHAP features: {shap_dict}\n\n")
    message=generate_message(shap_dict)
    logger.info(f"Generated message: {message}\n\n")

    message=generate_message_llm(shap_dict,config)
    logger.info(f"LLM Message: {message}\n\n")
    
if __name__=='__main__':
    main()