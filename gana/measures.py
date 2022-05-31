import networkx as nx


def homophily_index(graph, key='class'):
    def f(node):
        ret = 0.
        ns = nx.all_neighbors(graph, node)
        for n in ns:
            if graph.nodes[node][key] == graph.nodes[n][key]:
                ret += 1

        return ret

    h = 0.
    for i in graph.nodes:
        d = graph.degree[i]
        n = f(i)
        h += (n / d)

    h = h / len(graph.nodes)

    return h
