from .parsers import YAMLParser

def from_config(config):
    """
    Instantiate a Parser object from a configuration dictionary
    
    :param config: configuration dictionary
    """
    if config['type'] == 'yaml':
        return YAMLParser(config['config'])
    else:
        raise ValueError('Parser type not recognized')