import time

from bonsai import NetworkGenerator
from bonsai.exporter import DefaultExporter
from bonsai.generator import Generator
from bonsai.generator.latency.base import LatencyGenerator, PredictiveLatencyEstimator
from bonsai.generator.node.generators import MultinomialNodeGenerator
from bonsai.interpreter import DefaultInterpreter
from bonsai.models import load_model
from bonsai.parser import YAMLParser

config = '''
        nodes: 5000
        continents:
            EU: 30
            NA: 30
            AS: 20
            SA: 10
            OC: 10            
        countries:
            DE: 10
            FR: 10
            CN: 5
            JP: 5
            US: 20                
        '''

start = time.time()
bonsai_mdn = load_model('bonsai/models/bonsai-network/bonsai_mdn_layers-100-100-100_output_12/model.dill')
print('Loading model took {} seconds'.format(time.time() - start))
t = time.time()
bonsai_network_generator = NetworkGenerator(parser=YAMLParser(),
                                            interpreter=DefaultInterpreter(),
                                            generator=Generator(
                                                MultinomialNodeGenerator(),
                                                latency_estimator=PredictiveLatencyEstimator(bonsai_mdn),
                                            ),
                                            exporter=DefaultExporter())
print('Loading network generator took {} seconds, total time: {}'.format(time.time() - t, time.time() - start))
t = time.time()
nodes, latencies = bonsai_network_generator.generate(config)
print('Generating network took {} seconds, total time: {}'.format(time.time() - t, time.time() - start))
t = time.time()
generator_config = bonsai_network_generator.config()
print('Getting config took {} seconds, total time: {}'.format(time.time() - t, time.time() - start))

