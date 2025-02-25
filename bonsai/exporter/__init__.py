from .default import DefaultExporter

def from_config(config):
    """Instantiate an Exporter object from a configuration dictionary
    
    :param config: configuration dictionary
    """
    if config['type'] == 'default':
        return DefaultExporter()
    elif config['type'] == 'matrix':
        from .matrix import MatrixExporter
        return MatrixExporter(config['matrix'])
    elif config['type'] == 'simpleTC':
        from .simpleTC import SimpleTCExporter
        return SimpleTCExporter(config['baseBandwidth'], config['subnet'])
    elif config['type'] == 'link':
        from .link import LinkExporter
        return LinkExporter()
    else:
        raise ValueError('Exporter type not recognized')