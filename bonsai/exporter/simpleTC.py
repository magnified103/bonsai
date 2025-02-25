import ipaddress

import numpy as np

from .base import BaseExporter
from ..types import LatencyMatrix


class SimpleTCExporter(BaseExporter):
    """ Simple exporter for TC. """

    def __init__(self, baseBandwidth, subnet):
        self.baseBandwidth = baseBandwidth
        self.subnet = ipaddress.ip_network(subnet)

    def __defineIP__(self, nodes):
        idxs = [i + 1 for i in range(len(nodes))]
        ips = [str(self.subnet[idx]) for idx in idxs]
        return ips

    def __tcrule__(self, node_idx, ips, latencies):
        rules = []
        rules.append("modprobe ifb numifbs=1")
        rules.append("ip link add ifb0 type ifb")
        rules.append("ip link set dev ifb0 up")
        rules.append("tc qdisc add dev eth0 handle ffff: ingress")
        rules.append(
            "tc filter add dev eth0 parent ffff: protocol ip u32 match u32 0 0 action mirred egress redirect dev ifb0")
        rules.append("tc qdisc add dev ifb0 root handle 1: htb default 1")
        rules.append("tc class add dev ifb0 parent 1: classid 1:1 htb rate {}mbit".format(self.baseBandwidth))
        rules.append("tc qdisc add dev eth0 root handle 1: htb default 1")
        rules.append("tc class add dev eth0 parent 1: classid 1:1 htb rate {}mbit".format(self.baseBandwidth))
        for i, lat in enumerate(latencies):
            if i == node_idx:
                continue
            rules.append("tc class add dev eth0 parent 1:1 classid 1:{} htb rate {}mbit".format(i + 2, self.baseBandwidth))
            rules.append(
                "tc qdisc add dev eth0 parent 1:{} handle {} netem delay {}ms".format(i + 2, i + 2, latencies[i]))
            rules.append(
                "tc filter add dev eth0 protocol ip parent 1:0 prio 1 u32 match ip dst {} flowid 1:{}".format(ips[i],
                                                                                                              i + 2))
        return rules

    def export(self, nodes, nodes_x_nodes, latencies, *args):
        """Export a network of nodes and edges. """
        latencyMatrix = LatencyMatrix(nodes, nodes_x_nodes, latencies)
        ips = self.__defineIP__(nodes)
        rules = {}
        for i, node in enumerate(nodes):
            latencies = latencyMatrix.get_matrix()[i]
            rules[node] = self.__tcrule__(i, ips, latencies)
        return rules, ips

    def config(self):
        """Return the configuration of the exporter.

        :return: Configuration.
        :rtype: dict
        """
        return {
            "name": "SimpleTCExporter",
            "description": "Simple exporter for TC",
            "parameters": {
                "baseBandwidth": self.baseBandwidth,
                "subnet": self.subnet,
            }
        }
