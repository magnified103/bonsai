import os

import pandas as pd
from joblib import delayed, Parallel
from tqdm import tqdm

from bonsai.datasets.ripe.ripe import RipeDataSet
from bonsai.exporter.link import LinkExporter
from bonsai.exporter.matrix import SimpleMatrixExporter
from bonsai.generator.latency.base import LatencyGenerator, NormalBatchLatencyEstimator, PredictiveLatencyEstimator, \
    RandomLatencyEstimator
from bonsai.generator.node.generators import MultinomialNodeGenerator, UniformNodeGenerator
from bonsai.models.bonsai.bonsai_estimator import load_model, load_graph_model
from bonsai.types import MetaNode, NetworkNode, NetworkCapacity, ComputationCapacity, StorageCapacity


def create_node_generator(node_generator):
    if node_generator == 'uniform':
        return UniformNodeGenerator()
    elif node_generator == 'multinomial':
        return MultinomialNodeGenerator()
    else:
        raise ValueError('Unknown node generator: {}'.format(node_generator))


def create_latency_estimator(latency_estimator, mode='median'):
    if latency_estimator == 'predictive':
        bonsai_mdn = load_model('bonsai/models/bonsai-network/connected/6e11d_00243_layers_80_output_9', mode=mode)
        #bonsai_gnn_mdn = load_graph_model('bonsai/models/bonsai-network/gnn/bonsai-gnn-mdn-128-0.3-0', mode=mode)
        return PredictiveLatencyEstimator(bonsai_mdn, batch_size=10000)
    elif latency_estimator == 'uniform':
        min_value = 1
        max_value = 400
        return RandomLatencyEstimator(min_value, max_value)
    elif latency_estimator == 'normal':
        rtt_dataset = RipeDataSet(granularity='country')
        return NormalBatchLatencyEstimator(dataset=rtt_dataset, batch_size=1000000)


def run_config(nodes_file, latency_estimators):
    nodes = []
    df = pd.read_csv(nodes_file, delimiter=';')
    for idx, row in df.iterrows():
        meta_node = MetaNode(idx, row['src_country'], row['src_lat'], row['src_lon'], row['src_asn'])
        nodes.append(NetworkNode(meta_node, NetworkCapacity(1000, 1000), ComputationCapacity(0,0), StorageCapacity(0)))


    #nodes=nodes[:5]
    #random permutation
    #nodes = np.random.permutation(nodes)
    #nodes = list(nodes)[:3000]

    data = None
    nodes_x_nodes = None
    def _run_estimator(latency_estimator, mode):
        latency_estimator = create_latency_estimator(latency_estimator, mode)
        _nodes, _nodes_x_nodes, latency = LatencyGenerator(latency_estimator).generate(nodes)
        return _nodes, _nodes_x_nodes, latency

    #output = Parallel(n_jobs=len(latency_estimators), verbose=13)(delayed(_run_estimator)(latency_estimator, 'median') for latency_estimator in latency_estimators)
    output = [_run_estimator(latency_estimator, 'median') for latency_estimator in latency_estimators]
    for latency_estimator, output in zip(latency_estimators, output):
        data = output[2]
        if nodes_x_nodes is None:
            nodes_x_nodes = output[1]

    # latencies = Parallel(n_jobs=len(latency_estimators), verbose=13)(
    #     delayed(_run_estimator)('predictive', mode) for mode in ['mlh'])
    # for latency_estimator, latency in zip(['random', 'mlh'], latencies):
    #     data["predictive-"+latency_estimator] = latency

    return nodes, nodes_x_nodes, data


if __name__ == '__main__':
    n_rounds = 1
    n_jobs = 1

    latency_estimators = ['predictive']
    #output = 'performance_ds2_latency_results_connected'
    output = 'punchr_data/generated'
    #output = 'gnn/performance_vs_ds2_latency_results_connected'
    #output = 'gnn/punchr_data'
    #output = 'gnn-test'

    #configs = ['nodes_ds2.csv']
    #configs = ['all_nodes.csv']
    configs = ['punchr_data/punchr_nodes.csv']

    exporter = SimpleMatrixExporter()
    link_exporter = LinkExporter()

    rounds = [i for i in range(n_rounds)]
    os.makedirs(output, exist_ok=True)
    _output = output
    for i in tqdm(rounds, desc='Rounds'):
        output = os.path.join(_output, str(i))
        os.makedirs(output, exist_ok=True)
        #results = Parallel(n_jobs=n_jobs, verbose=13)(delayed(run_config)(config, latency_estimators) for config in configs)
        results = [run_config(config, latency_estimators) for config in configs]
        for config, result in zip(configs, results):
            _nodes, _nodes_x_nodes, latencies = result
            data = pd.DataFrame(latencies)
            data.to_csv(os.path.join(output, 'latencies.csv'))
            nodes = pd.DataFrame([node.to_dict() for node in _nodes])
            nodes.to_csv(os.path.join(output, 'nodes.csv'))
            exporter.export(_nodes, _nodes_x_nodes, latencies, os.path.join(output, 'matrix.csv'))
            link_exporter.export(_nodes, _nodes_x_nodes, latencies, os.path.join(output, 'links.csv'))









