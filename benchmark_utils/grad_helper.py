from scipy.special import expit


def gradient_linreg(X, y, beta):
    return X.T @ (X @ beta - y)


def gradient_logreg(X, y, beta):
    return X.T @ (expit(X @ beta) - y)
