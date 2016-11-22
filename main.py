import json
import sys
import os.path
from itertools import combinations


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
        nodes = {node}
        seen = set([])
        neighbours = {node}
        for i in xrange(dist):
            for node in nodes:
                seen.add(node)
                neighbours |= set(self.undir[node])
            nodes = neighbours - seen

        return list(neighbours)

    def neighbourhood(self, node, dist):
        dir_subg = {}
        undir_subg = {}

        nodes = self._neighbour_nodes(node, dist)

        for node in nodes:
            dir_subg[node] = filter(lambda x: x in nodes, self.graph[node])
            undir_subg[node] = filter(lambda x: x in nodes, self.undir[node])

        return (dir_subg, undir_subg)

    def _create_stat_str(self):
        stat = {}

        stat[1] = 0  # for (a->b)(a->c)
        stat[2] = 0  # for (b->a)(c->a)
        stat[3] = 0  # for (a->b)(b->c)
        stat[4] = 0  # for (a->b)(a->c)(c->b)
        stat[5] = 0  # for (a->b)(b->c)(c->a)
        stat[6] = 0  # for (a<->b)(a->c)
        stat[7] = 0  # for (a<->b)(c->a)(c->b)
        stat[8] = 0  # for (a<->b)(a->c)(b->c)
        stat[9] = 0  # for (a<->b)(b->c)(c->a)
        stat[10] = 0  # for (a<->b)(a<->c)
        stat[11] = 0 # for (a<->b)(a<->c)(b->c)
        stat[12] = 0 # for (a<->b)(a<->c)(b<->c)

        return stat

    def _recognize_graphlet(self, nodes, neighb, stat):
        # there are some edges between node1 & node2 and between node2 & node3 as well.
        node1, node2, node3 = nodes

        twodir_edges = 0
        simple_edges = 0
        twodir_edge = ()
        if node2 in neighb[node1] and node1 in neighb[node2]:
            twodir_edges = 1
            twodir_edge = (node1, node2)
        else:
            simple_edges = 1
        if node3 in neighb[node2] and node2 in neighb[node3]:
            twodir_edges += 1
            if twodir_edges == 1:
                twodir_edge = (node2, node3)
            else:
                twodir_edge = ()
        else:
            simple_edges += 1
        if node1 in neighb[node3]:
            if node3 in neighb[node1]:
                twodir_edges += 1
                if twodir_edges == 1:
                    twodir_edge = (node1, node3)
                else:
                    twodir_edge = ()
            else:
                simple_edges += 1
        elif node3 in neighb[node1]:
            simple_edges += 1

        if twodir_edges == 0:
            if simple_edges == 2:
                # 1-3 types
                if node2 in neighb[node1] and node3 in neighb[node1] or \
                    node1 in neighb[node2] and node3 in neighb[node2] or \
                    node1 in neighb[node3] and node2 in neighb[node3]:
                    stat[1] += 1
                elif node1 in neighb[node2] and node1 in neighb[node3] or \
                    node2 in neighb[node1] and node2 in neighb[node3] or \
                    node3 in neighb[node1] and node3 in neighb[node2]:
                    stat[2] += 1
                else:
                    stat[3] += 1
            else:
                if node2 in neighb[node1] and node3 in neighb[node1] or \
                    node1 in neighb[node2] and node3 in neighb[node2] or \
                    node1 in neighb[node3] and node2 in neighb[node3]:
                    stat[4] += 1
                else:
                    stat[5] += 1
        elif twodir_edges == 1:
            if simple_edges == 1:
                stat[6] += 1
            else:
                # 7-9 types
                if twodir_edge == (node1, node2) and node1 in neighb[node3] and node2 in neighb[node3] or \
                    twodir_edge == (node2, node3) and node2 in neighb[node1] and node3 in neighb[node1] or \
                    twodir_edge == (node1, node3) and node1 in neighb[node2] and node3 in neighb[node2]:
                    stat[7] += 1
                elif twodir_edge == (node1, node2) and node3 in neighb[node1] and node3 in neighb[node2] or \
                    twodir_edge == (node2, node3) and node1 in neighb[node2] and node1 in neighb[node3] or \
                    twodir_edge == (node1, node3) and node2 in neighb[node1] and node2 in neighb[node3]:
                    stat[8] += 1
                else:
                    stat[9] += 1
        elif twodir_edges == 2:
            if simple_edges == 0:
                stat[10] += 1
            else:
                stat[11] += 1
        else:
            stat[12] += 1

    def eval_stat(self, node, dist):
        dir, undir = self.neighbourhood(node, dist)

        stat = self._create_stat_str()

        graphlets = combinations(dir.keys(), 3)

        try:
            while 1:
                node1, node2, node3 = graphlets.next()
                # minimum 2 edges required
                if (node1 in dir[node2] or node2 in dir[node1]) and (node1 in dir[node3] or node3 in dir[node1]):
                    self._recognize_graphlet((node2, node1, node3), dir, stat)
                elif (node1 in dir[node2] or node2 in dir[node1]) and (node2 in dir[node3] or node3 in dir[node2]):
                    self._recognize_graphlet((node1, node2, node3), dir, stat)
                elif (node1 in dir[node3] or node3 in dir[node1]) and (node2 in dir[node3] or node3 in dir[node2]):
                    self._recognize_graphlet((node1, node3, node2), dir, stat)
        except StopIteration:
            pass

        return stat


def main():    
    datadir = 'data/'
    if not os.path.isfile(datadir + sys.argv[1]):
        print 'No such file'
        return 1.
    samplefile = datadir + sys.argv[1]

    sample = Graph(samplefile)

    # subgraph, tmp = sample.neighbourhood(, 2)
    #
    # for node in subgraph:
    #     print node, subgraph[node]

    stat = sample.eval_stat(sample.graph.keys()[0], 2)

    print 'stat:'
    for ind in stat:
        print ind, stat[ind]

    return 0


if __name__ == "__main__":
    main()