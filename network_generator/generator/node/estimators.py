import numpy as np
from .base import CapacityEstimator


class RandomEstimator(CapacityEstimator):
    """ Random estimator.

Parameters
----------
min_value : int
    Minimum value between source and destination node.
max_value : int
    Maximum value between source and destination node.
seed : int, optional
    Seed for the random number generator.

    """
    def __init__(self, min_value, max_value, seed=None):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.random = np.random.RandomState(seed=seed)

    def estimate(self, node):
        return self.random.randint(self.min_value, self.max_value)


class NormalEstimator(CapacityEstimator):
    """ Normal estimator.

Parameters
----------
dataset : dict, optional
    Dataset of mean and standard deviation of the value between source and destination node.
mean : int, optional
    Mean value between source and destination node.
std : int, optional
    Standard deviation of the value between source and destination node.
seed : int, optional
    Seed for the random number generator.

    """
    def __init__(self, dataset=None, mean=None, std=None, seed=None):
        super().__init__()
        self.data = dataset
        self.mean = mean
        self.std = std
        if self.data is None:
            assert self.mean is not None
            assert self.std is not None
        self.random = np.random.RandomState(seed=seed)

    def estimate(self, node):
        if self.data is not None:
            properties = self.data.get(node)
            self.mean, self.std = properties.mean, properties.std
        while True:
            value = self.random.normal(self.mean, self.std)
            if value > 0:
                return value


class FixedEstimator(CapacityEstimator):
    """ Fixed estimator.

Parameters
----------
dataset : dict, optional
    Dataset of mean and standard deviation of the value between source and destination node.
value : int, optional
    value between source and destination node.

    """

    def __init__(self, dataset=None, value=None):
        super().__init__()
        self.data = dataset
        self.value = value
        if self.data is None:
            assert self.value is not None

    def estimate(self, node):
        if self.data is not None:
            return self.data.get(node).mean
        return self.value


class PredictiveEstimator(CapacityEstimator):
    """ Predictive estimator.

    Parameters
    ----------
    model : BaseModel
        Model to predict some value between source and destination node.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def estimate(self, node):
        X = self.model.to_features(node)
        return self.model.predict(X)
