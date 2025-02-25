from bonsai.types import NetworkCapacity, ComputationCapacity, StorageCapacity
import haversine as hs


class BaseNode:
    """ BaseNode class
    Defines a node in the network.

    Attributes:
    id (int): The id of the node.


    """

    def __init__(self, node_id):
        self.id = node_id

    def __str__(self):
        return "Node id: {}".format(self.id)

    def __repr__(self):
        return "Node id: {}".format(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def __le__(self, other):
        return self.id <= other.id

    def __gt__(self, other):
        return self.id > other.id

    def __ge__(self, other):
        return self.id >= other.id

    def __ne__(self, other):
        return self.id != other.id


class MetaNode(BaseNode):
    """ MetaNode class
    Defines a node in the network.

    Attributes:
    id (int): The id of the node.
    country_code (str): The country code of the node.
    latitude (float): The latitude of the node.
    longitude (float): The longitude of the node.
    asn (int): The ASN of the node.

    """

    def __init__(self, node_id, country_code, latitude, longitude, asn):
        super().__init__(node_id)
        self.country_code = country_code
        self.latitude = latitude
        self.longitude = longitude
        self.asn = asn

    def distance(self, other):
        """
        Compute the distance between two nodes.
        :param other: MetaNode
        :return: distance in Km (Haversine distance)
        """
        return self.haversine(self.latitude, self.longitude, other.latitude, other.longitude)

    def haversine(self, latitude, longitude, latitude1, longitude1):
        """
        Compute the distance between two coordinates.
        :param latitude: float
        :param longitude: float
        :param latitude1: float
        :param longitude1: float
        :return: distance in Km (Haversine distance)
        """
        return hs.haversine((latitude, longitude), (latitude1, longitude1))


class NetworkNode:
    """ NetworkNode class
    Defines a node in for emulated network.

    Attributes:
    meta_node (MetaNode): The meta node of the network node, contains the node's meta information.
    network_capacity (NetworkCapacity): The network capacity of the network node.
    computation_capacity (ComputationCapacity): The computation capacity of the network node.
    storage_capacity (StorageCapacity): The storage capacity of the network node.


    """

    def __init__(self, meta_node, network_capacity, computation_capacity, storage_capacity):
        self.meta_node: MetaNode = meta_node
        self.network_capacity: NetworkCapacity = network_capacity
        self.computation_capacity: ComputationCapacity = computation_capacity
        self.storage_capacity: StorageCapacity = storage_capacity

    def get_id(self):
        return self.meta_node.id

    def __eq__(self, other):
        return self.meta_node.__eq__(other.meta_node)

    def __hash__(self):
        return self.meta_node.__hash__()

    def __lt__(self, other):
        return self.meta_node.__lt__(other.meta_node)

    def __le__(self, other):
        return self.meta_node.__le__(other.meta_node)

    def __gt__(self, other):
        return self.meta_node.__gt__(other.meta_node)

    def __ge__(self, other):
        return self.meta_node.__ge__(other.meta_node)

    def __ne__(self, other):
        return self.meta_node.__ne__(other.meta_node)

    def get_country(self):
        return self.meta_node.country_code

    def get_asn(self):
        return self.meta_node.asn

    def get_coordinates(self):
        return self.meta_node.latitude, self.meta_node.longitude

    def to_dict(self):
        return {
            'id': self.get_id(),
            'country': self.get_country(),
            'asn': self.get_asn(),
            'coordinates': self.get_coordinates(),
            'network_capacity': self.network_capacity,
            'computation_capacity': self.computation_capacity,
            'storage_capacity': self.storage_capacity
            }
