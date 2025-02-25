import argparse
import bonsai

parser = argparse.ArgumentParser(description='Bonsai Network Model Generator')
parser.add_argument('--net_config', type=str, default='')
parser.add_argument('--generator_config', type=str, default='')
parser.add_argument('--output_dir', type=str, default='')


default_config = {
    'parser': {
        'type': 'yaml',
        'config': {}
    },
    'interpreter': {
        'type': 'default',
        'config': {}
    },
    'generator': {
        'type': 'default',
        'config': {
            'node_generator': {
                'type': 'multinomial',
                'config': {}
            },
            'latency_generator': {
                'type': 'bonsai',
                'config': {
                    'mode': 'most_likely',
                    'batch_size': 100000,
                    'model_path': 'bonsainet/models/bonsai-gnn-mdn-64-3-0.4-10-knn-20'
                }
            }
        }
    },
    'exporter': {
        'type': 'link',
        'config': {}
    }
}


if __name__ == '__main__':
    args = parser.parse_args()
    generator_config = args.generator_config
    net_config = args.net_config
    output_dir = args.output_dir
    
    bonsaiNMG = bonsai.from_config(generator_config, default_config)
    nodes, edges, latency = bonsaiNMG.generate(net_config)
    
    bonsaiNMG.export(nodes, edges, latency, output_dir)
