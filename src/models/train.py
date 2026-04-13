from sklearn.linear_model import LogisticRegression

def train_model(X_train,y_train):

    model = LogisticRegression(
        max_iter=2000,
        solver='lbfgs')
    model.fit(X_train,y_train)

    return model