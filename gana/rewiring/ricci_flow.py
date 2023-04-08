import numpy as np
import numpy.random as npr
import networkx as nx

import torch
from matplotlib import pyplot as plt
from torch.distributions import Categorical

from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci


def find_min_edge(edges, key='formanCurvature'):
    val = float('inf')
    target = None
    for e in edges:
        ricci = edges[e][key]
        if ricci < val:
            val = ricci
            target = e

    return target


def find_max_edge(edges, c_plus, key='formanCurvature'):
    val = -float('inf')
    target = None
    for e in edges:
        ricci = edges[e][key]
        if ricci > val and ricci > c_plus:
            val = ricci
            target = e

    return target


def cartesian(x, y):
    return [{i, j} for i in x for j in y]


def curvature_factory(key, graph):
    if key == 'formanCurvature':
        return FormanRicci(graph, verbose='ERROR')
    elif key == 'ricciCurvature':
        return OllivierRicci(graph, verbose='ERROR')


def stable_softmax(x, tau=1.0):
    x = tau * (x - max(x))
    x = np.exp(x)
    x = x / sum(x)

    return x


class SDRF:
    """
    stochastic discrete ricci flow

    C stands for curvature
    """

    def __init__(
            self,
            initial_G: nx.Graph,
            C_type: str = 'formanCurvature',
    ):
        self.C_type = C_type
        self.C = curvature_factory(C_type, initial_G)

    def fit(self, iters=100):
        for t in range(iters):
            self.step()

        return self.C.G

    def step(
            self,
            tau: float = 100.,
            c_plus: float = 5.0
    ):
        # add edge
        self.C.compute_ricci_curvature()

        ij_edge = find_min_edge(self.C.G.edges, self.C_type)

        b0 = self.C.G.neighbors(ij_edge[0])
        b1 = self.C.G.neighbors(ij_edge[1])
        # subgraph = nx.complete_graph(list(b0)+list(b1))
        # subgraph.remove_edge(*ij_edge)
        # kl_edges = subgraph.edges
        kl_edges = cartesian(b0, b1)

        xs = np.zeros(len(kl_edges))
        new_riccis = np.zeros(len(kl_edges))

        old_ricci = self.C.G.edges[ij_edge][self.C_type]

        for i, kl_edge in enumerate(kl_edges):
            if kl_edge == set(ij_edge):
                continue

            self.C.G.add_edge(*kl_edge, weight=1.0)
            self.C.compute_ricci_curvature()
            new_riccis[i] = self.C.G.edges[kl_edge][self.C_type]
            xs[i] = new_riccis[i] - old_ricci
            self.C.G.remove_edge(*kl_edge)

        xs = stable_softmax(xs, tau)
        # idx = npr.multinomial(1, xs)
        self.dist = Categorical(probs=torch.FloatTensor(xs))
        idx = self.dist.sample()

        # rewiring
        print(f'adding {kl_edges[idx]}')
        self.C.G.add_edge(
            *kl_edges[idx],
            weight=1.0,
            formanCurvature=new_riccis[idx]
        )

        # remove edge
        e = find_max_edge(self.C.G.edges, c_plus, self.C_type)
        if e:
            self.C.G.remove_edge(*e)
            print(f'removing {e}')

        improvement = sum(xs) / len(xs)

        return improvement


if __name__ == '__main__':
    _dk = dict(
        with_labels=True,
        # font_size=2,
        # node_size=5,
    )

    # graph = nx.newman_watts_strogatz_graph(32, 4, 0.1)
    # graph = nx.barbell_graph(100, 200)
    com1 = nx.complete_graph(50)
    com2 = nx.complete_graph(50)
    graph = nx.disjoint_union(com1, com2)
    graph.add_edge(0, len(com1))
    graph.add_edge(1, 1+len(com1))
    graph.add_edge(2, 2+len(com1))

    nx.draw_kamada_kawai(graph, **_dk)
    plt.show()

    algo = SDRF(graph)
    algo.fit(30)

    nx.draw_kamada_kawai(algo.C.G, **_dk)
    plt.show()
