import numpy as np

from ..types import NetworkNode
from ..utils.location import compute_distance

def compute_features(src: NetworkNode, dst: NetworkNode) -> np.ndarray:
    """
    Compute features between two nodes.
    :param src: source node
    :param dst: dst node
    :return: array of features
    """
    distance = compute_distance((src.meta_node.latitude, src.meta_node.longitude),
                                (dst.meta_node.latitude, dst.meta_node.longitude))

    ## hops

    return np.array([src.network_capacity.upload, src.network_capacity.download, src.meta_node.latitude, src.meta_node.longitude, src.meta_node.asn,
                     dst.network_capacity.upload, dst.network_capacity.download, dst.meta_node.latitude, dst.meta_node.longitude, dst.meta_node.asn,
                     distance])

