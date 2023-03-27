
class Dataset:
    """ Dataset base class.

    Parameters
    ----------

    Methods
    -------
    get(src, dst)
        Get properties between two nodes.

    """

    def __init__(self):
        pass

    def get(self, *args):
        raise NotImplementedError("Dataset.get() must be implemented in a subclass.")