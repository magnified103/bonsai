from .generator import Generator

def from_config(config):
    """Instanciate a Generator object from a configuration dictionary
    
    :param config: configuration dictionary
    """
    from .node import from_config as node_from_config
    node_generator = node_from_config(config['node_generator'])
    from .latency import from_config as latency_from_config
    latency_generator = latency_from_config(config['latency_generator'])
    
    return Generator(node_generator, latency_generator)