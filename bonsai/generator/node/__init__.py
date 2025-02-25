
def from_config(config):
    """Instanciate a NodeGenerator object from a configuration dictionary
    
    :param config: configuration dictionary
    """
    
    network_capacity_estimator = None
    if 'network_capacity_estimator' in config:
        from .estimators import from_config as capacity_estimator_from_config
        network_capacity_estimator = capacity_estimator_from_config(config['network_capacity_estimator'])
    
    storage_capacity_estimator = None
    if 'storage_capacity_estimator' in config:
        from .estimators import from_config as capacity_estimator_from_config
        storage_capacity_estimator = capacity_estimator_from_config(config['storage_capacity_estimator'])
    
    computational_capacity_estimator = None        
    if 'computational_capacity_estimator' in config:
        from .estimators import from_config as capacity_estimator_from_config
        computational_capacity_estimator = capacity_estimator_from_config(config['computational_capacity_estimator'])
    
    if config['type'] == 'uniform':
        from .generators import UniformNodeGenerator
        return UniformNodeGenerator(network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator)
    elif config['type'] == 'multinomial':
        from .generators import MultinomialNodeGenerator
        return MultinomialNodeGenerator(network_capacity_estimator, computational_capacity_estimator, storage_capacity_estimator)
    else:
        raise ValueError('NodeGenerator type not recognized')
    