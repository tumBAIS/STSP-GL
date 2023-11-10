import networkx as nx
from itertools import product, permutations
from random import sample
from scipy.spatial import distance


def random_nodes(graph_size, n):
    """
    Return n random tuples from the grid graph of size graph_size
    """
    return sample(list(product(range(graph_size), repeat=2)), k=n)


def build_graph(graph_size, n):
    nodes = random_nodes(graph_size, n)
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node, pos=node)
    # G.add_nodes_from(nodes)
    edges = permutations(nodes, 2)

    for edge in edges:
        G.add_edge(edge[0], edge[1], cost=round(distance.euclidean(edge[0], edge[1]), 2))

    return G
