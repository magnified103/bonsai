
def from_config(config):
    """Instanciate a LatencyGenerator object from a configuration dictionary
    
    :param config: configuration dictionary
    """
    
    if config['type'] == 'simple':
        from .base import from_config as base_from_config
        return base_from_config(config['config'])
    elif config['type'] == 'bonsai':
        from .bonsai_estimator import from_config as bonsai_from_config
        return bonsai_from_config(config['config'])
    else:
        raise ValueError('LatencyGenerator type not recognized')
    