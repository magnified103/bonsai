from .base import BaseExporter

class DefaultExporter(BaseExporter):
    """Default exporter. """

    def export(self, nodes, edges, *args):
        """Export a network of nodes and edges. """
        return nodes, edges