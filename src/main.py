from src.config import load_config
from src.data.features import basic_feature_engineering
from src.data.loader import load_raw_data
from src.data.preprocess import prepare_data
from src.env import get_env_variable, load_env
from src.logger import setup_logger
from src.models.evaluate import (
    calibrate_model,
    compute_auc_pr,
    compute_auc_roc,
    evaluate_model,
    find_best_threshold,
    get_threshold_metrics,
    plot_calibration_curve,
    plot_precision_recall,
)
from src.models.explain import (
    compute_shap_values,
    explain_single_prediction,
    get_shap_dict,
    plot_feature_importance,
)
from src.models.train import train_model
from src.rag.chain import generate_rag_answer
from src.rag.ingest import create_vector_store, load_documents
from src.rag.retriever import retrieve_context
from src.recovery.generator import generate_message, generate_message_llm
from src.services import clear_cache
from src.simulation.ab_test import (
    print_ab_test_summary,
    run_ab_test_simulation,
    save_ab_test_results,
)


def main():
    load_env()
    clear_cache()  # Ensure fresh state for CLI run
    config = load_config()
    logger = setup_logger(config)

    thresholds = config.get("model.thresholds", [0.5, 0.4, 0.3, 0.2])

    docs = load_documents(config.get("paths.reviews_file", "data/external/reviews.txt"))
    vectorstore = create_vector_store(docs, config)

    env = get_env_variable("ENV")
    logger.info(f"Starting {config.get('project.name')} in {env} mode..\n\n")

    # Use default query for non-interactive runs
    query = "Why do users abandon carts due to price and engagement?"
    context = retrieve_context(vectorstore, query, config.get("rag.retriever.k", 3), config)
    answer = generate_rag_answer(context, query, config)
    # Handle Unicode encoding for Windows console
    safe_answer = answer.encode('ascii', 'replace').decode('ascii')
    logger.info(f"RAG ANSWER: {safe_answer}\n")

    df = load_raw_data(config)
    logger.info(f"Loaded data with shape: {df.shape}\n\n")

    df = basic_feature_engineering(df, config)
    logger.info(f"Data after feature engineering: {df.shape}\n\n")
    logger.info(f"Columns after features: {df.columns.tolist()}\n\n")

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df, config)
    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}\n\n")

    model = train_model(X_train, y_train, config)

    # AUC metrics (un-calibrated)
    auc_roc = compute_auc_roc(model, X_test, y_test)
    auc_pr = compute_auc_pr(model, X_test, y_test)
    logger.info(f"AUC-ROC (uncalibrated): {auc_roc:.4f}")
    logger.info(f"AUC-PR (uncalibrated): {auc_pr:.4f}\n")

    # Calibrate model on validation set
    logger.info("=== Calibrating Model (Isotonic Regression) ===")
    calibrated_model = calibrate_model(model, X_val, y_val, method="isotonic")
    
    # AUC metrics (calibrated)
    auc_roc_cal = compute_auc_roc(calibrated_model, X_test, y_test)
    auc_pr_cal = compute_auc_pr(calibrated_model, X_test, y_test)
    logger.info(f"AUC-ROC (calibrated): {auc_roc_cal:.4f}")
    logger.info(f"AUC-PR (calibrated): {auc_pr_cal:.4f}\n")

    # Calibration curve
    logger.info("Plotting calibration curve...")
    plot_calibration_curve(y_test, calibrated_model.predict_proba(X_test)[:, 1])

    # Tune threshold on VALIDATION set (not test) - use calibrated model
    logger.info("=== Threshold Tuning on Validation Set (Calibrated) ===")
    best_threshold, best_f1 = find_best_threshold(calibrated_model, X_val, y_val, thresholds)
    logger.info(f"Best threshold: {best_threshold} (F1: {best_f1:.4f})\n")

    # Final evaluation on TEST set with best threshold (calibrated)
    logger.info("=== Final Evaluation on Test Set (Calibrated) ===")
    accuracy, report = evaluate_model(calibrated_model, X_test, y_test, best_threshold)
    logger.info(f"AT THRESHOLD: {best_threshold}")
    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"\n{report}\n\n")

    # All thresholds on test for reference (calibrated)
    logger.info("=== All Thresholds on Test Set (Calibrated) ===")
    threshold_metrics = get_threshold_metrics(calibrated_model, X_test, y_test, thresholds)
    for m in threshold_metrics:
        logger.info(
            f"Threshold {m['threshold']}: "
            f"Acc={m['accuracy']*100:.2f}% | "
            f"Class1 P/R/F1={m['precision_1']:.3f}/{m['recall_1']:.3f}/{m['f1_1']:.3f} | "
            f"Class0 P/R/F1={m['precision_0']:.3f}/{m['recall_0']:.3f}/{m['f1_0']:.3f}"
        )

    plot_precision_recall(calibrated_model, X_test, y_test)

    sample_size = config.get("model.evaluation.sample_size_for_shap", 200)
    X_sample = X_test.sample(sample_size, random_state=config.get("data.random_state", 42))
    # Use original model for SHAP (calibrated model wrapper not supported)
    shap_values = compute_shap_values(model, X_sample)

    plot_feature_importance(shap_values)

    explain_single_prediction(model, X_sample, index=0)

    top_n = config.get("model.evaluation.shap_top_n", 5)
    shap_dict = get_shap_dict(shap_values, X_sample, index=0, top_n=top_n)
    logger.info(f"Top SHAP features: {shap_dict}\n\n")
    message = generate_message(shap_dict)
    logger.info(f"Generated message: {message}\n\n")

    message = generate_message_llm(shap_dict, config)
    safe_message = message.encode('ascii', 'replace').decode('ascii')
    logger.info(f"LLM Message: {safe_message}\n\n")

    # A/B Test Simulation (use calibrated model)
    if config.get("simulation.ab_test.enabled", True):
        logger.info("=== Running A/B Test Simulation ===")
        n_sim = config.get("simulation.ab_test.n_simulations", 1000)
        X_sim = X_test.sample(min(n_sim, len(X_test)), random_state=config.get("data.random_state", 42))
        ab_results = run_ab_test_simulation(X_sim, calibrated_model, config)
        print_ab_test_summary(ab_results)
        
        output_path = config.get("paths.ab_test_results_file", "data/ab_test_results.json")
        save_ab_test_results(ab_results, output_path)
        logger.info(f"A/B test results saved to: {output_path}\n")


if __name__ == "__main__":
    main()