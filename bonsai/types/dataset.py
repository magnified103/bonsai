class Dataset:
    """ Dataset base class.

    Parameters
    ----------

    Methods
    -------
    probabilities(by=None)
        Get probabilities.
    values(by=None)
        Get values.

    """

    def __init__(self, name, data):
        """ Initialize the dataset.

        Parameters
        ----------
        name : str
            Name.
        data : dict | pandas.DataFrame
            Data.
        """
        self.name = name
        self.data = data
        self.computed_probabilities = {}

    def properties(self):
        """ Get properties.

        Returns
        -------
        properties : list
            Properties.

        """
        raise NotImplementedError()

    def get_property(self, property_name):
        """ Get property.

        Parameters
        ----------
        property_name : str
            Property name.

        Returns
        -------
        property : any
            Property.

        """
        raise NotImplementedError()

    def probabilities(self, by=None):
        """ Get probabilities.

        Parameters
        ----------
        by : str
            Group by.

        Returns
        -------
        probabilities : dict
            Probabilities.

        """
        raise NotImplementedError()

    def values(self, by=None):
        """ Get values.

        Parameters
        ----------
        by : str
            Group by.

        Returns
        -------
        values : dict
            Values.

        """
        raise NotImplementedError()
