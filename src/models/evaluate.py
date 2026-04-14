from sklearn.metrics import accuracy_score,classification_report,precision_recall_curve
import matplotlib.pyplot as plt

def evaluate_model(model,X_test,y_test,threshold=0.5):
    
    #Getting probabilities for class 1
    y_probs = model.predict_proba(X_test)[:,1]

    #Applying custom threshold
    y_pred=(y_probs>=threshold).astype(int)
    
    accuracy= accuracy_score(y_test,y_pred)
    report=classification_report(y_test,y_pred)

    return accuracy,report

def plot_precision_recall(model,X_test,y_test):
    y_probs=model.predict_proba(X_test)[:,1]

    precision,recall,thresholds=precision_recall_curve(y_test,y_probs)

    plt.figure()
    plt.plot(recall,precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precisio-Recall Curve")
    plt.grid()
    
    plt.show()

    return precision,recall,thresholds
