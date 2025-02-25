from .interperters import *

def from_config(config):
    """
    Instantiate a Interpreter object from a configuration dictionary
    
    :param config: configuration dictionary
    """
    
    if config['type'] == 'default':
        return DefaultInterpreter()
    else:
        raise ValueError('Interpreter type not recognized')