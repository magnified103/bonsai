import argparse

parser = argparse.ArgumentParser(description='Bonsai Network Model Generator')
parser.add_argument('--config', type=str, default='')
parser.add_argument('--output', type=str, default='')

if __name__ == '__main__':
    args = parser.parse_args()
    config_dir = args.config
    output_dir = args.output
