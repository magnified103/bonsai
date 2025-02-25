from .network_generator import NetworkGenerator
import yaml


def from_config(config_file, default_config=None):
    """
    Instantiate a NetworkGenerator object from a configuration file

    :param config_file: path to the configuration file
    :type config_file: str
    :param default_config: default configuration
    :type default_config: dict
    :return: NetworkGenerator object
    """
    if default_config == None and config_file == '':
        raise ValueError('Either config_file or default_config must be provided')
    elif config_file == '':
        return NetworkGenerator.from_config(default_config)
    else:
        #parse the config file
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)    
        return NetworkGenerator.from_config(config)
