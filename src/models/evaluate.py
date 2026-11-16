import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test, y_test, threshold):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, report


def find_best_threshold(model, X_val, y_val, thresholds):
    """Find best threshold on validation set using F1 score."""
    y_probs = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def calibrate_model(model, X_val, y_val, method: str = "isotonic"):
    """Calibrate model probabilities using validation set.
    
    Args:
        model: Trained classifier with predict_proba
        X_val: Validation features
        y_val: Validation labels
        method: 'isotonic' (more flexible) or 'sigmoid' (Platt scaling)
    
    Returns:
        Calibrated model with predict_proba method
    """
    try:
        # New sklearn API (>=1.3): use ensemble=False for prefit-like behavior
        calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=5)
        calibrated.fit(X_val, y_val)
        return calibrated
    except TypeError:
        # Fallback: simple isotonic calibration via calibration_curve interpolation
        from scipy.interpolate import interp1d
        
        y_prob = model.predict_proba(X_val)[:, 1]
        prob_true, prob_pred = calibration_curve(y_val, y_prob, n_bins=10, strategy="uniform")
        
        # Create interpolation function
        if method == "isotonic":
            # Ensure monotonic
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(prob_pred, prob_true)
            
            class CalibratedModel:
                def __init__(self, base_model, calibrator):
                    self.base_model = base_model
                    self.calibrator = calibrator
                
                def predict_proba(self, X):
                    probs = self.base_model.predict_proba(X)[:, 1]
                    calibrated_probs = self.calibrator.predict(probs)
                    return np.vstack([1 - calibrated_probs, calibrated_probs]).T
                
                def predict(self, X):
                    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
            
            return CalibratedModel(model, ir)
        else:
            # Sigmoid/Platt scaling - simple logistic calibration
            from sklearn.linear_model import LogisticRegression
            
            # Fit logistic regression on validation probabilities
            lr = LogisticRegression()
            lr.fit(y_prob.reshape(-1, 1), y_val)
            
            class CalibratedModel:
                def __init__(self, base_model, calibrator):
                    self.base_model = base_model
                    self.calibrator = calibrator
                
                def predict_proba(self, X):
                    probs = self.base_model.predict_proba(X)[:, 1]
                    calibrated_probs = self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
                    return np.vstack([1 - calibrated_probs, calibrated_probs]).T
                
                def predict(self, X):
                    return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
            
            return CalibratedModel(model, lr)


def compute_calibration_curve(y_true, y_prob, n_bins: int = 10):
    """Compute calibration curve data."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return prob_true, prob_pred


def plot_calibration_curve(y_true, y_prob, title: str = "Calibration Curve"):
    """Plot calibration curve (reliability diagram)."""
    prob_true, prob_pred = compute_calibration_curve(y_true, y_prob)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "s-", label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
    
    return prob_true, prob_pred


def compute_auc_roc(model, X_test, y_test):
    """Compute AUC-ROC score."""
    y_probs = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_probs)


def compute_auc_pr(model, X_test, y_test):
    """Compute AUC-PR score."""
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    return auc(recall, precision)


def plot_roc_curve(model, X_test, y_test):
    """Plot ROC curve."""
    y_probs = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()
    
    return fpr, tpr, thresholds, roc_auc


def plot_precision_recall(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid()

    plt.show()

    return precision, recall, thresholds


def get_threshold_metrics(model, X_test, y_test, thresholds):
    """Compute metrics for multiple thresholds."""
    y_probs = model.predict_proba(X_test)[:, 1]
    
    results = []
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results.append({
            "threshold": t,
            "accuracy": accuracy,
            "precision_0": report["0"]["precision"],
            "recall_0": report["0"]["recall"],
            "f1_0": report["0"]["f1-score"],
            "precision_1": report["1"]["precision"],
            "recall_1": report["1"]["recall"],
            "f1_1": report["1"]["f1-score"],
        })
    
    return results