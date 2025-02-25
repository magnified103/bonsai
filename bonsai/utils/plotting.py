import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import bonsai.utils.location as utils_location
from bonsai.types import NetworkNode


def plot_nodes_on_map(nodes, figsize=(20, 10), show=True, save=False, filename='nodes.png'):
    fig, ax = plt.subplots(figsize=figsize)
    utils_location.country_bounds.plot(ax=ax)
    for node in nodes:
        ax.plot(node.meta_node.longitude, node.meta_node.latitude, 'ro', markersize=1)
    if save:
        plt.savefig(filename)
    if show:
        plt.show()

    return fig, ax


def order_latencies_by_node_property(nodes, latency, rows, property):
    node_df = pd.DataFrame(rows)
    node_df = node_df.sort_values(by=[property])
    nodes = [nodes[i] for i in node_df.index]
    latency = latency[node_df.index, node_df.index]
    return nodes, latency


def order_latencies(nodes: list[NetworkNode], latency, order_by):
    if order_by == 'country':
        rows = [
            {'country': n.meta_node.country_code} for n in nodes
        ]
        nodes, latency = order_latencies_by_node_property(nodes, latency, rows, 'country')
    elif order_by == 'continent':
        rows = [
            {'continent': utils_location.get_continent(n.meta_node.country_code)} for n in nodes
        ]
        nodes, latency = order_latencies_by_node_property(nodes, latency, rows, 'continent')
    elif order_by == 'latency':
        pass
    elif order_by == 'distance':
        pass
    elif order_by == 'asn':
        rows = [
            {'asn': n.meta_node.country_code} for n in nodes
        ]
        nodes, latency = order_latencies_by_node_property(nodes, latency, rows, 'asn')

    return nodes, latency


def plot_latency_heatmap(nodes, latency, figsize=(20, 10), order_by=None, show=True, save=False,
                         filename='latency.png'):
    nodes, latency = order_latencies(nodes, latency, order_by)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(latency, ax=ax)
    if save:
        plt.savefig(filename)
    if show:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close(fig)
