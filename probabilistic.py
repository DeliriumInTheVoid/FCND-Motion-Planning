import numpy.linalg as LA
import networkx as nx
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point, LineString


def prune_path_3d(path, polygons):
    pruned_path = [p for p in path]
    i = 0
    while i < len(pruned_path) - 2:
        p1 = pruned_path[i]
        p2 = pruned_path[i + 1]
        p3 = pruned_path[i + 2]

        if can_connect(p1, p3, polygons):
            pruned_path.remove(p2)
            i = 0
            continue
        i += 1
    return pruned_path


def can_connect(n1, n2, polygons):
    l = LineString([n1, n2])
    for p, h in polygons:
        if p.crosses(l) and h >= min(n1[2], n2[2]):
            return False
    return True


def create_graph(nodes, k, polygons):
    graph = nx.Graph()
    tree = KDTree(nodes)
    for n1 in nodes:
        # for each node connect try to connect to k nearest nodes
        idxs = tree.query([n1], k, return_distance=False)[0]

        for idx in idxs:
            n2 = nodes[idx]
            if n2 == n1:
                continue

            if can_connect(n1, n2, polygons):
                graph.add_edge(n1, n2, weight=1)
    return graph
