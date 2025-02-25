import numpy as np
import pandas as pd
import torch

from bonsai.types import NetworkNode

def from_config(config):
    estimator = None
    if 'estimate' in config:
        if config['estimator'] == 'random':
            estimator = RandomLatencyEstimator(config['params']['min_value'], config['params']['max_value'], config['params']['seed'])
        elif config['estimator'] == 'normal':
            estimator = NormalLatencyEstimator(mean=config['params']['mean'], std=config['params']['std'], seed=config['params']['seed'])
        elif config['estimator'] == 'fixed':
            estimator == FixedLatencyEstimator(value=config['params']['value'])
        else:
            raise ValueError('LatencyEstimator type not recognized')
    
    return LatencyGenerator(estimator=estimator)

class LatencyGenerator:
    """
    Base class for latency generators.

    Latency generators are used to generate the latency matrix for a network.
    """

    def __init__(self, estimator=None):
        """
        :param estimator: Latency estimator.
        :type estimator: LatencyEstimator
        """
        self.estimator = estimator if estimator is not None else RandomLatencyEstimator(10, 100)

    def generate(self, nodes: NetworkNode):
        """
        Generate the latency matrix for the given nodes and network specification.
        :param nodes: Nodes.
        :type nodes: list[NetworkNode]
        :return: nodes, nodes_x_nodes, latencies
        :rtype: tuple[list[NetworkNode], list[tuple[NetworkNode, NetworkNode]], list[float]]
        """

        #print("------- Generating latency matrix for %d nodes ---------" % (len(nodes)))
        #start = time.time()
        nodes_x_nodes = [(node1, node2) for i, node1 in enumerate(nodes) for node2 in nodes[:i] if node1 != node2]
        #print("nodes_x_nodes took %f seconds" % (time.time() - start))
        #start = time.time()
        latencies = self.estimator.estimateBatch(nodes_x_nodes)
        #print("estimateBatch %d took %f seconds" % (len(nodes_x_nodes), time.time() - start))
        if len(nodes_x_nodes) > len(latencies):
            raise ValueError("LatencyGenerator.estimateBatch() returned less latencies than expected."
                             "\n Expected: %d, got: %d" 
                             "\n Check your dataset configuration." % (len(nodes_x_nodes), len(latencies)))


        return nodes, nodes_x_nodes, latencies

        # this goes to the exported
        #latency_matrix = LatencyMatrix(nodes, nodes_x_nodes, latencies, parallel=self.parallel)
        #return latency_matrix


        # latMat = LatencyMatrix(len(nodes))
        # for i, node1 in enumerate(nodes):
        #     for j, node2 in enumerate(nodes[:i]):
        #         if i == j:
        #             continue
        #         latMat.set_latency(i, j, self.estimator.estimate(node1, node2))
        # return latMat

    def config(self):
        """
        Return the configuration of the latency generator.

        :return: Configuration.
        :rtype: dict
        """
        return {
            "estimator": self.estimator.config()
        }

class LatencyEstimator:

    def __init__(self):
        pass

    def estimate(self, node1, node2):
        raise NotImplementedError("LatencyEstimator.estimate() must be implemented in a subclass.")

    def estimateBatch(self, nodes, parallel=1):
        raise NotImplementedError("LatencyEstimator.estimateBatch() must be implemented in a subclass.")

    def config(self):
        """
        Return the configuration of the latency estimator.

        :return: Configuration.
        :rtype: dict
        """
        raise NotImplementedError("LatencyEstimator.config() must be implemented in a subclass.")


class RandomLatencyEstimator(LatencyEstimator):
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

    def estimate(self, src, dst):
        return self.random.randint(self.min_value, self.max_value)

    def estimateBatch(self, nodes):
        return self.random.randint(self.min_value, self.max_value, size=len(nodes))

    def config(self):
        """
        Return the configuration of the latency estimator.

        :return: Configuration.
        :rtype: dict
        """
        return {
            "name": "RandomLatencyEstimator",
            "description": "Random latency estimator, estimates latency between min and max values at random.",
            "params": {
                "min_value": self.min_value,
                "max_value": self.max_value,
                "seed": self.random.seed
            }
        }



class NormalLatencyEstimator(LatencyEstimator):
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

    def __init__(self, dataset=None, mean=None, std=None, unknown=RandomLatencyEstimator(10, 400), seed=None):
        super().__init__()
        self.data = dataset
        self.mean = mean
        self.std = std
        self.unknown = unknown
        if self.data is None and (self.mean is None or self.std is None):
            raise ValueError("Either dataset or mean and std must be specified.")
        self.random = np.random.RandomState(seed=seed)

    def estimate(self, src, dst):
        if self.data is not None:
            try:
                properties = self.data.get(src, dst)
                self.mean, self.std = properties.mean, properties.std
            except KeyError:
                return self.unknown.estimate(src, dst)
        while True:
            value = np.median(self.random.normal(self.mean, self.std, 3))
            if value > 0:
                return np.round(value, 3) / 2

    def estimateBatch(self, nodes): # TODO: find a way to make this faster
        return [self.estimate(src, dst) for src, dst in nodes]

    def config(self):
        """
        Return the configuration of the latency estimator.

        :return: Configuration.
        :rtype: dict
        """
        return {
            "name": "NormalLatencyEstimator",
            "description": "Normal latency estimator, estimates latency with a normal distribution.",
            "params": {
                "mean": self.mean,
                "std": self.std,
                "seed": self.random.seed
            }
        }

class NormalBatchLatencyEstimator(LatencyEstimator):
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

    def __init__(self, dataset=None, mean=None, std=None, unknown=RandomLatencyEstimator(10, 400), seed=None, batch_size=-1):
        super().__init__()
        self.data = dataset

        self.mean = mean
        self.std = std
        self.unknown = unknown
        self.batch_size = batch_size
        if self.data is None and (self.mean is None or self.std is None):
            raise ValueError("Either dataset or mean and std must be specified.")
        if seed is not None:
            torch.manual_seed(seed)


    def estimate(self, src, dst):
        if self.data is not None:
            try:
                properties = self.data.get(src, dst)
                self.mean, self.std = properties.mean, properties.std
            except KeyError:
                return self.unknown.estimate(src, dst)
        while True:
            value = np.median(self.random.normal(self.mean, self.std, 3))
            if value > 0:
                return np.round(value, 3) / 2

    def estimateNormalBatch(self, means_stds):
        import torch.distributions as D
        means_stds = np.array(means_stds)
        means = torch.tensor(means_stds[:, 0].reshape(-1, 1))
        stds = torch.tensor(means_stds[:, 1].reshape(-1, 1))
        ones = torch.ones(means.shape)
        normalBatch = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(ones),
            component_distribution=D.Normal(means, stds))
        samples = []
        for i in range(3):
            samples.append(normalBatch.sample().cpu().numpy())
        return np.median(samples, axis=0) / 2

    def deal_with_negative(self, latencies, X):
        ## get index of latency < 0
        idx = np.where(latencies < 0)[0]
        if len(idx) == 0:
            return latencies
        ## get the features of the negative latencies
        X_neg = np.array(X)[idx]

        ## regenerate the negative latencies
        latencies_neg = self.processBatch(X_neg)
        ## replace the negative latencies with the regenerated ones
        # print("Negative latencies: ", len(idx))
        # print(latencies_neg.shape)
        # print(latencies.shape)
        # print(latencies[idx].shape)
        latencies[idx] = latencies_neg
        return latencies

    def processBatch(self, nodes):
        normal = self.estimateNormalBatch([(mean, std) for _, mean, std in nodes])
        return self.deal_with_negative(normal, nodes)

    def estimateBatch(self, nodes):
        def get_props(src, dst):
            if self.data is not None:
                try:
                    properties = self.data.get(src, dst)
                    return properties.mean, properties.std
                except KeyError:
                    return None, None
            return self.mean, self.std

        latencies = np.zeros(len(nodes))
        idx_means_std = [(i, get_props(src, dst)) for i, (src, dst) in enumerate(nodes)]
        normal_idx_means_std = [(i, mean, std) for i, (mean, std) in idx_means_std if mean is not None]
        unknown_idx = [i for i, (mean, std) in idx_means_std if mean is None]
        unknown = self.unknown.estimateBatch([(nodes[i][0], nodes[i][1]) for i in unknown_idx])
        latencies[unknown_idx] = unknown
        if self.batch_size > 0:
            for i in range(0, len(normal_idx_means_std), self.batch_size):
                #start = time.time()
                #print("Predicting batch {} of {}".format(i, len(X)))
                latencies[[i for i, _, _ in normal_idx_means_std[i:i + self.batch_size]]] = self.processBatch(normal_idx_means_std[i:i + self.batch_size])
                #print("Done predicting batch {} of {}, elapsed time: {}".format(i, len(X), time.time() - start))
                #sample = self.model.sample(X[i:i + self.batch_size], self.n_samples)
        else:
            latencies[[i for i, _, _ in normal_idx_means_std]] = self.processBatch(normal_idx_means_std)

        return latencies

    def config(self):
        """
        Return the configuration of the latency estimator.

        :return: Configuration.
        :rtype: dict
        """
        return {
            "name": "NormalLatencyEstimator",
            "description": "Normal latency estimator, estimates latency with a normal distribution.",
            "params": {
                "mean": self.mean,
                "std": self.std,
                "seed": self.random.seed
            }
        }


class FixedLatencyEstimator(LatencyEstimator):
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

    def estimate(self, src, dst):
        if self.data is not None:
            return self.data.get(src, dst).mean
        return self.value

    def estimateBatch(self, nodes):
        return [self.estimate(src, dst) for src, dst in nodes]

    def config(self):
        """
        Return the configuration of the latency estimator.

        :return: Configuration.
        :rtype: dict
        """
        return {
            "name": "FixedLatencyEstimator",
            "description": "Fixed latency estimator, estimates latency with a fixed value.",
            "params": {
                "value": self.value
            }
        }


class PredictiveLatencyEstimator(LatencyEstimator):
    """ Predictive estimator.

    Parameters
    ----------
    model : BaseModel
        Model to predict some value between source and destination node.
    """

    def __init__(self, model, batch_size=-1):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.model.set_batch_size(batch_size)

    def estimate(self, src, dst):
        X, features = self.model.compute_features(src, dst)
        X = pd.DataFrame(X, columns=features)
        return self.model.predict(X)

    def estimateBatch(self, nodes):
        return np.round(self.model.predict(nodes).reshape(-1), 3)

    def config(self):
        """
        Return the configuration of the latency estimator.

        :return: Configuration.
        :rtype: dict
        """
        return {
            "name": "PredictiveLatencyEstimator",
            "description": "Predictive latency estimator, estimates latency with a predictive model.",
            "params": {
                "model": self.model.config()
            }
        }

