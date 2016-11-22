import json
import os.path


class Graph:
    def __init__(self, file):
        self.graph = self._extract(file)
        self.undir = self._make_undirected(self.graph)

    def _extract(self, file):
        with open(file, 'rt') as input:
            strings = input.readlines()

        # list of elems. Elem is dict with 'id' as node index and 'value' as list of neighbours
        list_graph = map(lambda x: json.loads(x), strings)

        # print list_graph[0]

        # convert to dict of nodes with node index as key and list of neighbours as value
        graph = {}
        for node in list_graph:
            graph[node['id']] = node['value']

        return graph

    def _make_undirected(self, graph):
        res = {}
        for node in graph:
            res[node] = set(graph[node])

        for node in graph:
            for neighbour in graph[node]:
                res[neighbour].add(node)

        for node in res:
            res[node] = sorted(list(res[node]))

        return res

    def _neighbour_nodes(self, node, dist):
        nodes = set([node])
        seen = set([])
        neighbours = set([node])
        for i in xrange(dist):
            for node in nodes:
                seen.add(node)
                neighbours.update(set(self.undir[node]))
            nodes = neighbours.difference(seen)

        return list(neighbours)

    def neighbourhood(self, node, dist):
        subgraph = {}

        nodes = self._neighbour_nodes(node, dist)

        for node in nodes:
            subgraph[node] = filter(lambda x: x in nodes, self.graph[node])

        return subgraph


def main():
    # samplefile = 'data/graphSample.json'
    samplefile = 'data/tmpsample.json'

    sample = Graph(samplefile)

    subgraph = sample.neighbourhood(1, 2)

    for node in subgraph:
        print node, subgraph[node]

    return 0


if __name__ == "__main__":
    main()