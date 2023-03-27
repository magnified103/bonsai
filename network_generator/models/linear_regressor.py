import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin



def loss_fn(X, Y, beta):
    return cp.pnorm(X @ beta - Y, p=2) ** 2


def regularizer(beta):
    return cp.pnorm(beta, p=2) ** 2


def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, Y, beta).value


def objective_fn(X, Y, beta, lambd):
    return loss_fn(X, Y, beta) + lambd * regularizer(beta)


class LinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lambd=0, transformer=None):
        super(LinearRegressor, self).__init__()
        self.beta = None
        self.lambd = lambd
        self.transformer = transformer

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        if self.transformer is not None:
            y = self.transformer.fit_transform(y.reshape(-1, 1))
            y = y.reshape(-1)
        n, d = X.shape
        beta = cp.Variable(d)
        lambd = cp.Parameter(nonneg=True, value=self.lambd)
        zero = cp.Parameter(value=self.zero())
        prob = cp.Problem(cp.Minimize(objective_fn(X, y, beta, lambd)), [X @ beta >= zero])
        try:
            prob.solve(solver=cp.ECOS)
            self.beta = beta.value
        except cp.SolverError:
            self.beta = np.zeros(d)
        return self

    def predict(self, X):
        return self.transformer.inverse_transform((X @ self.beta).reshape(-1, 1)).reshape(-1)

    def zero(self):
        return - self.transformer.mean_[0] / np.sqrt(self.transformer.var_[0])

    def score(self, X, y):
        #This does not work for some reason, takes too long
        return mse(X, y, self.beta)
