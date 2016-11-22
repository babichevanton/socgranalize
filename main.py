import json
import os.path
from copy import deepcopy


def sample_extract(file):
    with open(file, 'rt') as input:
        strings = input.readlines()

    # list of elems. Elem is dict with 'id' as node index and 'value' as list of neighbours
    list_sample = map(lambda x: json.loads(x), strings)

    print list_sample[0]

    # convert to dict of nodes with node index as key and list of neighbours as value
    sample = {}
    for node in list_sample:
        sample[node['id']] = node['value']

    return sample


def make_undirected(graph):
    res = {}
    for node in graph:
        res[node] = set(graph[node])

    for node in graph:
        for neighbour in graph[node]:
            res[neighbour].add(node)

    for node in res:
        res[node] = sorted(list(res[node]))

    return res


def neighbour_nodes(graph, node, dist):
    nodes = set([node])
    seen = set([])
    neighbours = set([node])
    for i in xrange(dist):
        for node in nodes:
            seen.add(node)
            neighbours.update(set(graph[node]))
        nodes = neighbours.difference(seen)

    return list(neighbours)


def neighbourhood(graph, nodes):
    subgraph = {}

    for node in nodes:
        subgraph[node] = filter(lambda x: x in nodes, graph[node])

    return subgraph


def main():
    # samplefile = 'data/graphSample.json'
    samplefile = 'data/tmpsample.json'

    sample = sample_extract(samplefile)

    undir_sample = make_undirected(sample)

    nodes = sorted(neighbour_nodes(undir_sample, 1, 2))

    print nodes

    subgraph = neighbourhood(sample, nodes)

    for node in subgraph:
        print node, subgraph[node]

    return 0


if __name__ == "__main__":
    main()