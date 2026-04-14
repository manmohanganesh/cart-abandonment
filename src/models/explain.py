import shap
import matplotlib.pyplot as plt

def plot_shap_bar(shap_values):
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    return fig


def compute_shap_values(model,X_Sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_Sample)

    return shap_values

def plot_feature_importance(shap_values):
    shap.plots.bar(shap_values)

def explain_single_prediction(model,X_Sample,index=0):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_Sample)

    #Plot explanation for one instance
    shap.plots.waterfall(shap_values[index])

def get_shap_dict(shap_values,X_sample,index=0,top_n=5):

    """
    Return top N feature contributions for a given instance
    """

    values = shap_values.values[index]
    feature_name = X_sample.columns

    #Creating feature -> value mapping
    shap_dict = dict(zip(feature_name,values))

    sorted_items = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_features = dict(sorted_items[:top_n])

    return top_features