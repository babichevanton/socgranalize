import json
import sys
import os.path
from itertools import combinations


class Graph:
    def __init__(self, file):
        with open(file, 'rt') as input:
            strings = input.readlines()

        # list of elems. Elem is dict with 'id' as node index and 'value' as list of neighbours
        list_graph = map(lambda x: json.loads(x), strings)

        # convert to dict of nodes with node index as key and dict of neighbours as value
        self.graph = {}
        for node in list_graph:
            if node['id'] not in self.graph:
                self.graph[node['id']] = {'to': [], 'from': []}
            for target in node['value']:
                if target not in self.graph:
                    self.graph[target] = {'to': [], 'from': []}
                self.graph[node['id']]['to'].append(target)
                self.graph[target]['from'].append(node['id'])

    def _undir(self, graph, node):
        neighbours = list(set(graph[node]['to']) | set(graph[node]['from']))
        return neighbours

    def _neighbour_nodes(self, node, dist):
        nodes = {node}
        seen = set([])
        neighbours = {node}
        for i in xrange(dist):
            for node in nodes:
                seen.add(node)
                neighbours |= set(self._undir(self.graph, node))
            nodes = neighbours - seen

        return list(neighbours)

    def neighbourhood(self, node, dist):
        neighbourhood = {}

        nodes = self._neighbour_nodes(node, dist)

        for node in nodes:
            neighbourhood[node] = {}
            neighbourhood[node]['to'] = filter(lambda x: x in nodes, self.graph[node]['to'])
            neighbourhood[node]['from'] = filter(lambda x: x in nodes, self.graph[node]['from'])

        return neighbourhood

    def _create_stat_str(self):
        stat = {}

        stat[1]  = {'val': 0, 'desc': '(a->b)(a->c)'}
        stat[2]  = {'val': 0, 'desc': '(b->a)(c->a)'}
        stat[3]  = {'val': 0, 'desc': '(a->b)(b->c)'}
        stat[4]  = {'val': 0, 'desc': '(a->b)(a->c)(c->b)'}
        stat[5]  = {'val': 0, 'desc': '(a->b)(b->c)(c->a)'}
        stat[6]  = {'val': 0, 'desc': '(a<->b)(a->c)'}
        stat[7]  = {'val': 0, 'desc': '(a<->b)(c->a)(c->b)'}
        stat[8]  = {'val': 0, 'desc': '(a<->b)(a->c)(b->c)'}
        stat[9]  = {'val': 0, 'desc': '(a<->b)(b->c)(c->a)'}
        stat[10] = {'val': 0, 'desc': '(a<->b)(a<->c)'}
        stat[11] = {'val': 0, 'desc': '(a<->b)(a<->c)(b->c)'}
        stat[12] = {'val': 0, 'desc': '(a<->b)(a<->c)(b<->c)'}

        return stat

    def _recognize_graphlet(self, nodes, neighb, stat):
        # there are some edges between node1 & node2 and between node2 & node3 as well.
        node1, node2, node3 = nodes

        twodir_edges = 0
        simple_edges = 0
        twodir_edge = ()
        if node2 in neighb[node1]['to']:
            if node1 in neighb[node2]['to']:
                twodir_edges = 1
                twodir_edge = (node1, node2)
            else:
                simple_edges += 1
        elif node1 in neighb[node2]['to']:
            simple_edges += 1
        if node3 in neighb[node2]['to']:
            if node2 in neighb[node3]['to']:
                twodir_edges += 1
                if twodir_edges == 1:
                    twodir_edge = (node2, node3)
                else:
                    twodir_edge = ()
            else:
                simple_edges += 1
        elif node2 in neighb[node3]['to']:
            simple_edges += 1
        if node1 in neighb[node3]['to']:
            if node3 in neighb[node1]['to']:
                twodir_edges += 1
                if twodir_edges == 1:
                    twodir_edge = (node1, node3)
                else:
                    twodir_edge = ()
            else:
                simple_edges += 1
        elif node3 in neighb[node1]['to']:
            simple_edges += 1

        if twodir_edges == 0:
            if simple_edges == 2:
                # 1-3 types
                if node2 in neighb[node1]['to'] and node3 in neighb[node1]['to'] or \
                        node1 in neighb[node2]['to'] and node3 in neighb[node2]['to'] or \
                        node1 in neighb[node3]['to'] and node2 in neighb[node3]['to']:
                    stat[1]['val'] += 1
                elif node1 in neighb[node2]['to'] and node1 in neighb[node3]['to'] or \
                        node2 in neighb[node1]['to'] and node2 in neighb[node3]['to'] or \
                        node3 in neighb[node1]['to'] and node3 in neighb[node2]['to']:
                    stat[2]['val'] += 1
                else:
                    stat[3]['val'] += 1
            else:
                if node2 in neighb[node1]['to'] and node3 in neighb[node1]['to'] or \
                        node1 in neighb[node2]['to'] and node3 in neighb[node2]['to'] or \
                        node1 in neighb[node3]['to'] and node2 in neighb[node3]['to']:
                    stat[4]['val'] += 1
                else:
                    stat[5]['val'] += 1
        elif twodir_edges == 1:
            if simple_edges == 1:
                stat[6]['val'] += 1
            else:
                # 7-9 types
                if twodir_edge == (node1, node2) and node1 in neighb[node3]['to'] and node2 in neighb[node3]['to'] or \
                        twodir_edge == (node2, node3) and node2 in neighb[node1]['to'] and node3 in neighb[node1]['to'] or \
                        twodir_edge == (node1, node3) and node1 in neighb[node2]['to'] and node3 in neighb[node2]['to']:
                    stat[7]['val'] += 1
                elif twodir_edge == (node1, node2) and node3 in neighb[node1]['to'] and node3 in neighb[node2]['to'] or \
                        twodir_edge == (node2, node3) and node1 in neighb[node2]['to'] and node1 in neighb[node3]['to'] or \
                        twodir_edge == (node1, node3) and node2 in neighb[node1]['to'] and node2 in neighb[node3]['to']:
                    stat[8]['val'] += 1
                else:
                    stat[9]['val'] += 1
        elif twodir_edges == 2:
            if simple_edges == 0:
                stat[10]['val'] += 1
            else:
                stat[11]['val'] += 1
        else:
            stat[12]['val'] += 1

    def eval_stat(self, node, dist):
        subgraph = self.neighbourhood(node, dist)

        stat = self._create_stat_str()

        for node1 in subgraph:
            seen = set([])
            for node2 in self._undir(subgraph, node1):
                # hold edge (node1,node2)
                nodes3 = list((set(self._undir(subgraph, node1)) | set(self._undir(subgraph, node2))) - \
                              (seen | {node1, node2}))
                for node3 in nodes3:
                    self._recognize_graphlet((node1, node2, node3), subgraph, stat)
                # remove edge (node1,node2) from subgraph
                if node2 in subgraph[node1]['to']:
                    subgraph[node1]['to'].remove(node2)
                if node2 in subgraph[node1]['from']:
                    subgraph[node1]['from'].remove(node2)
                if node1 in subgraph[node2]['to']:
                    subgraph[node2]['to'].remove(node1)
                if node1 in subgraph[node2]['from']:
                    subgraph[node2]['from'].remove(node1)
                # mark node2 as 'seen' to prevent repeated analize of graphlets with node1 and node2
                seen.add(node2)

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
        print stat[ind]['desc'], stat[ind]['val']

    return 0


if __name__ == "__main__":
    main()