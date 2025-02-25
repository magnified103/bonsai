import numpy as np
import pandas as pd
from bonsainet.predictor.bonsai_predictor import load_predictor, BonsaiSelector
from tqdm import tqdm

from bonsai.generator.latency.base import LatencyGenerator
from bonsai.types import NetworkNode


# import sys
# sys.path.append('../bonsai-network/bonsainet')
# sys.path.append('../bonsai-network/bonsainet/predictor')


def from_config(config):
    return BonsaiLatencyGenerator(mode=config['mode'], batch_size=config['batch_size'], model_path=config['model_path'])


class BonsaiLatencyGenerator(LatencyGenerator):
    def __init__(self, mode='most_likely', batch_size=None, model_path=None):
        super()
        self.batch_size = batch_size
        if model_path is not None:
            self.predictor  = load_predictor(model_path)

        self.to_sample = 32
        self.mode = mode

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def generate(self, nodes: list, verbose=False):
        ## prepare list of NetworkNode to feed into the predictor
        global chosen
        coords, countries = [], []
        for node in nodes:
            node: NetworkNode = node
            lat, lon = node.get_coordinates()
            country = node.get_country()
            coords.append({'lat': lat, 'lon': lon})
            countries.append({'country': country})
        coords = pd.DataFrame(coords)
        countries = pd.DataFrame(countries)
        _nodes = self.predictor.prepare_graph_data(coords, countries)
        mu, sigma, pi, target_edges, lower_bound = self.predictor.predict(_nodes, batch_size=self.batch_size, verbose=verbose)


        if self.batch_size is None:
            samples = self.predictor.sample_from(mu, sigma, pi, n_samples=self.to_sample)
            selector = BonsaiSelector(mu, sigma, pi, samples, lower_bound, self.predictor.target_transformer)
            if self.mode == 'most_likely':
                chosen = selector.most_likely() / 2
            elif self.mode == 'random':
                chosen = selector.random() / 2
            elif self.mode == 'median':
                chosen = selector.median() / 2
            elif self.mode == 'avg':
                chosen = selector.avg() / 2
            else:
                raise ValueError('Mode not recognized')
        else:
            for i in range(0, len(mu), self.batch_size) if not verbose else tqdm(range(0, len(mu), self.batch_size)):
                _samples = self.predictor.sample_from(mu[i:i+self.batch_size], sigma[i:i+self.batch_size], pi[i:i+self.batch_size], n_samples=self.to_sample)
                selector = BonsaiSelector(mu[i:i+self.batch_size], sigma[i:i+self.batch_size], pi[i:i+self.batch_size], _samples, lower_bound[i:i+self.batch_size], self.predictor.target_transformer)
                if self.mode == 'most_likely':
                    _chosen = selector.most_likely() / 2
                elif self.mode == 'random':
                    _chosen = selector.random() / 2
                elif self.mode == 'median':
                    _chosen = selector.median() / 2
                elif self.mode == 'avg':
                    _chosen = selector.avg() / 2
                else:
                    raise ValueError('Mode not recognized')

                if i == 0:
                    chosen = _chosen
                else:
                    chosen = np.concatenate((chosen, _chosen), axis=0)

        return nodes, target_edges.numpy(), chosen

## Test the BonsaiLatencyGenerator
## then generate IPFS network
## then adapt bonsai-to-kollaps config
## then run our exp managemnt system