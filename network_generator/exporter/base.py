
class BaseExporter(object):
    """Base class for all exporters. """

    def __init__(self, *args, **kwargs):
        pass

    def export(self, *args, **kwargs):
        """Export a network of nodes and edges. """
        raise NotImplementedError("Exporter.export() must be implemented in a subclass.")