import os

import joblib as joblib
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.svm import NuSVR, SVR
from category_encoders import TargetEncoder

from network_generator.models.FFNN.ffnn import FFNNRegressor
from .linear_regressor import LinearRegressor


class BaseModel(object):
    """Base model class for all models."""

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        """Fit a model to data."""
        raise NotImplementedError("Model.fit() must be implemented in a subclass.")

    def predict(self, *args, **kwargs):
        """Predict using a model."""
        raise NotImplementedError("Model.predict() must be implemented in a subclass.")

    def score(self, *args, **kwargs):
        """Score a model."""
        raise NotImplementedError("Model.score() must be implemented in a subclass.")

    def save(self, model_name):
        """Save a model to disk.

        :param model_name: Name of the model to save.
        :type model_name: str
        """
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        joblib.dump(self, model_name)


class ModelLoadingError(Exception):
    """Raised when a model cannot be loaded from disk."""

    def __init__(self, message):
        super(ModelLoadingError, self).__init__(message)


# Test this
def load_model(model_name):
    """Load a model from disk.

    :param model_name: Name of the model to load.
    :type model_name: str

    :return: Model.
    :rtype: BaseModel
    """
    try:
        model = joblib.load(model_name)
    except IOError as e:
        raise ModelLoadingError("Unable to load model %s from disk: %s" % model_name % e)
    return model


class SimpleFFNNModel(BaseModel):
    """Simple feed-forward neural network model.

    """

    def __init__(self, model_kwargs):
        super(SimpleFFNNModel, self).__init__()
        transformer = None

        _model_kwargs = model_kwargs.copy()
        if 'transform' in _model_kwargs:
            to_scale_features = _model_kwargs.get('to_scale_features', None)
            to_transform_features = _model_kwargs.get('to_transform_features', None)
            to_categorical_features = _model_kwargs.get('to_categorical_features', None)
            if _model_kwargs['transform'] == 'standard':  # default
                in_transformer = StandardScaler()
                out_transformer = StandardScaler()
                transformer = 'standard'
            elif _model_kwargs['transform'] == 'power':
                in_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                out_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                transformer = 'power'
            elif _model_kwargs['transform'] == 'quantile_uniform':
                in_transformer = QuantileTransformer(output_distribution='uniform')
                out_transformer = QuantileTransformer(output_distribution='uniform')
                transformer = 'quantile_uniform'
            elif _model_kwargs['transform'] == 'quantile_normal':
                in_transformer = QuantileTransformer(output_distribution='normal')
                out_transformer = QuantileTransformer(output_distribution='normal')
                transformer = 'quantile_normal'
            elif _model_kwargs['transform'] == 'standard_and_power':
                in_transformer = ColumnTransformer([('scaler', StandardScaler(), to_scale_features),
                                                    ('transformer', PowerTransformer(method='yeo-johnson', standardize=True), to_transform_features),
                                                    ('categorical', TargetEncoder(), to_categorical_features)])
                out_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                transformer = 'standard_and_power'
            del _model_kwargs['transform']

        if 'to_scale_features' in _model_kwargs:
            del _model_kwargs['to_scale_features']
        if 'to_transform_features' in _model_kwargs:
            del _model_kwargs['to_transform_features']

        if transformer is None:
            regressor = FFNNRegressor(**_model_kwargs)
            self.model = Pipeline([('regression', regressor)])
        else:
            regressor = FFNNRegressor(**_model_kwargs, scaler=out_transformer)
            self.model = Pipeline([('scaler', in_transformer), ('regression', regressor)])

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    def score(self, X, y):
        # loss, training_loss = self.model.score(X, y)
        loss, training_loss = self.model.score(X, y)
        pred = self.predict(X)
        test_r2 = r2_score(y, pred)
        test_rmse = mean_squared_error(y, pred, squared=False)
        test_mae = mean_absolute_error(y, pred)
        test_mape = mean_absolute_percentage_error(y, pred)
        return training_loss, loss, test_r2, test_rmse, test_mae, test_mape


class SimpleLinearModel(BaseModel):
    """Simple linear model.

    """

    def __init__(self, model_kwargs, transformer=None):
        super(SimpleLinearModel, self).__init__()
        in_scaler = StandardScaler()
        out_scaler = StandardScaler()
        regressor = LinearRegressor(**model_kwargs, transformer=out_scaler)
        if transformer is None:
            self.model = Pipeline([('scaler', in_scaler), ('regression', regressor)])
        else:
            self.model = Pipeline([('scaler', in_scaler), ('transformer', transformer), ('regression', regressor)])

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        pred = self.model.predict(X)
        test_r2 = r2_score(y, pred)
        test_rmse = mean_squared_error(y, pred, squared=False)
        return test_r2, test_rmse


class SimpleSVRModel(BaseModel):
    """Simple support vector regression model.

    """

    def __init__(self, model_kwargs={}, nusvr=False):
        super(SimpleSVRModel, self).__init__()
        in_scaler = StandardScaler()
        out_scaler = StandardScaler()
        if nusvr:
            regressor = NuSVR(**model_kwargs)
        else:
            regressor = SVR(**model_kwargs)
        self.model = TransformedTargetRegressor(
            regressor=Pipeline([('scaler', in_scaler), ('regression', regressor)]), transformer=out_scaler)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        # loss = self.model.score(X, y)
        pred = self.model.predict(X)
        test_r2 = r2_score(y, pred)
        test_rmse = mean_squared_error(y, pred, squared=False)
        return test_r2, test_rmse
