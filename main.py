import json
import sys
import threading
from time import time


class Graph:
    def __init__(self, file):
        print 'Construct graph from sample file'
        begin_time = time()

        with open(file, 'rt') as input:
            strings = input.readlines()

        # list of elems. Elem is dict with 'id' as node index and 'value' as list of neighbours
        list_graph = map(lambda x: json.loads(x), strings)

        # convert to dict of nodes with node index as key and dict of neighbours as value
        self.graph = {}
        i = 0
        for node in list_graph:
            i += 1
            print '\r    line {0}/{1}  [{2}%]'.format(i, len(list_graph),  i * 100 / len(list_graph)),
            if node['id'] not in self.graph:
                self._create_node(node['id'])
            for target in node['value']:
                if target not in self.graph:
                    self._create_node(target)
                self.graph[node['id']]['to'].append(target)
                self.graph[target]['from'].append(node['id'])
        print

        self.node1_iter = iter(self.graph)

        cur_time = time()
        print 'process took {0} sec'.format(cur_time - begin_time)
        print

    def _create_node(self, id):
        self.graph[id] = {'stat': self._create_stat_structure(), 'to': [], 'from': []}

    def _undir(self, graph, node):
        """
        Compute neighbours of node as if the graph is undirected.

        :param graph: analized graph treated as directed;
        :param node: target node which neighbours are computing
        :return: list of neighbour of target node as if the graph is undurected.
        """
        neighbours = list(set(graph[node]['to']) | set(graph[node]['from']))
        return neighbours

    def _neighbour_nodes(self, graph, node, dist):
        """
        Compute the nodes in target node's neighbourhood in graph.

        :param graph: analizing graph where neighbourhoods are;
        :param node: target node which neighbourhood is computed;
        :param dist: radius of neighbourhood;
        :return: list of nodes in the traget node's neighbourhood.
        """
        # print '  Find nodes in neighbourhood:'
        # begin_time = time()

        nodes = {node}
        seen = set([])
        neighbours = {node}
        for i in xrange(dist):
            # j = 0
            for node in nodes:
                # j += 1
                # print '\r    step {0}/{1}: node {2}/{3}  [{4}%]'.format(i + 1, dist, j, len(nodes), j * 100 / len(nodes)),
                seen.add(node)
                neighbours |= set(self._undir(graph, node))
            nodes = neighbours - seen
        # print
        # cur_time = time()
        # print '    process took {0} sec'.format(cur_time - begin_time)

        return list(neighbours)

    def _create_stat_structure(self):
        stat = {}

        stat[1]  = {'val': 0, 'desc': '(a->b)(a->c)'}
        stat[2]  = {'val': 0, 'desc': '(b->a)(c->a)'}
        stat[3]  = {'val': 0, 'desc': '(a->b)(b->c)'}
        stat[4]  = {'val': 0, 'desc': '(a->b)(a->c)(c->b)'}
        stat[5]  = {'val': 0, 'desc': '(a->b)(b->c)(c->a)'}
        stat[6]  = {'val': 0, 'desc': '(a<->b)(a->c)'}
        stat[7]  = {'val': 0, 'desc': '(a<->b)(c->a)'}
        stat[8]  = {'val': 0, 'desc': '(a<->b)(c->a)(c->b)'}
        stat[9]  = {'val': 0, 'desc': '(a<->b)(a->c)(b->c)'}
        stat[10]  = {'val': 0, 'desc': '(a<->b)(b->c)(c->a)'}
        stat[11] = {'val': 0, 'desc': '(a<->b)(a<->c)'}
        stat[12] = {'val': 0, 'desc': '(a<->b)(a<->c)(b->c)'}
        stat[13] = {'val': 0, 'desc': '(a<->b)(a<->c)(b<->c)'}

        for type in stat:
            stat[type]['lock'] = threading.Lock()

        return stat

    def _add_graphlet_to_stat(self, stat_lst, type):
        """
        Add 1 graphlet to stat structure for every node in list.

        :param stat_lst: list of nodes which stat structures need to be modified;
        :param type: type of graphlet to add.
        """
        for node in stat_lst:
            # multiple threads can write - lock access to the stat structure
            self.graph[node]['stat'][type]['lock'].acquire()
            self.graph[node]['stat'][type]['val'] += 1
            self.graph[node]['stat'][type]['lock'].release()

    def _recognize_graphlet(self, nodes, stats):
        """
        Recognize type of graphlet and fill stats structures.

        :param nodes: nodes describing the analizing graphlet;
        :param stats: list of nodes which neighbourhoods contain the analizing graphlet.
        """
        # there are some edges between node1 & node2 and between node2 & node3 as well.
        node1, node2, node3 = nodes

        twodir_edges = 0
        simple_edges = 0
        twodir_edge = ()
        if node2 in self.graph[node1]['to']:
            if node1 in self.graph[node2]['to']:
                twodir_edges = 1
                twodir_edge = (node1, node2)
            else:
                simple_edges += 1
        elif node1 in self.graph[node2]['to']:
            simple_edges += 1
        if node3 in self.graph[node2]['to']:
            if node2 in self.graph[node3]['to']:
                twodir_edges += 1
                if twodir_edges == 1:
                    twodir_edge = (node2, node3)
                else:
                    twodir_edge = ()
            else:
                simple_edges += 1
        elif node2 in self.graph[node3]['to']:
            simple_edges += 1
        if node1 in self.graph[node3]['to']:
            if node3 in self.graph[node1]['to']:
                twodir_edges += 1
                if twodir_edges == 1:
                    twodir_edge = (node1, node3)
                else:
                    twodir_edge = ()
            else:
                simple_edges += 1
        elif node3 in self.graph[node1]['to']:
            simple_edges += 1

        if twodir_edges == 0:
            if simple_edges == 2:
                # 1-3 types
                if node2 in self.graph[node1]['to'] and node3 in self.graph[node1]['to'] or \
                        node1 in self.graph[node2]['to'] and node3 in self.graph[node2]['to'] or \
                        node1 in self.graph[node3]['to'] and node2 in self.graph[node3]['to']:
                    self._add_graphlet_to_stat(stats, 1)
                elif node1 in self.graph[node2]['to'] and node1 in self.graph[node3]['to'] or \
                        node2 in self.graph[node1]['to'] and node2 in self.graph[node3]['to'] or \
                        node3 in self.graph[node1]['to'] and node3 in self.graph[node2]['to']:
                    self._add_graphlet_to_stat(stats, 2)
                else:
                    self._add_graphlet_to_stat(stats, 3)
            else:
                if node2 in self.graph[node1]['to'] and node3 in self.graph[node1]['to'] or \
                        node1 in self.graph[node2]['to'] and node3 in self.graph[node2]['to'] or \
                        node1 in self.graph[node3]['to'] and node2 in self.graph[node3]['to']:
                    self._add_graphlet_to_stat(stats, 4)
                else:
                    self._add_graphlet_to_stat(stats, 5)
        elif twodir_edges == 1:
            if simple_edges == 1:
                if twodir_edge == (node1, node2) and (node3 in self.graph[node1]['to'] or node3 in self.graph[node2]['to']) or \
                        twodir_edge == (node2, node3) and (node1 in self.graph[node2]['to'] or node1 in self.graph[node3]['to']) or \
                        twodir_edge == (node1, node3) and (node2 in self.graph[node1]['to'] or node2 in self.graph[node3]['to']):
                    self._add_graphlet_to_stat(stats, 6)
                else:
                    self._add_graphlet_to_stat(stats, 7)
            else:
                # 7-9 types
                if twodir_edge == (node1, node2) and node1 in self.graph[node3]['to'] and node2 in self.graph[node3]['to'] or \
                        twodir_edge == (node2, node3) and node2 in self.graph[node1]['to'] and node3 in self.graph[node1]['to'] or \
                        twodir_edge == (node1, node3) and node1 in self.graph[node2]['to'] and node3 in self.graph[node2]['to']:
                    self._add_graphlet_to_stat(stats, 8)
                elif twodir_edge == (node1, node2) and node3 in self.graph[node1]['to'] and node3 in self.graph[node2]['to'] or \
                        twodir_edge == (node2, node3) and node1 in self.graph[node2]['to'] and node1 in self.graph[node3]['to'] or \
                        twodir_edge == (node1, node3) and node2 in self.graph[node1]['to'] and node2 in self.graph[node3]['to']:
                    self._add_graphlet_to_stat(stats, 9)
                else:
                    self._add_graphlet_to_stat(stats, 10)
        elif twodir_edges == 2:
            if simple_edges == 0:
                self._add_graphlet_to_stat(stats, 11)
            else:
                self._add_graphlet_to_stat(stats, 12)
        else:
            self._add_graphlet_to_stat(stats, 13)

    def _neighbours_containes(self, graphlet, dist):
        """
        Evaluate all nodes which neighbourhoods contain analizing graphlet.

        :param graphlet: tuple of nodes desribing graphlet;
        :param dist: radius of neighbourhood;
        :return: list of nodes which neighbourhoods contain graphlet.
        """
        node1, node2, node3 = graphlet
        result = list(set(self._neighbour_nodes(self.graph, node1, dist)) &
                      set(self._neighbour_nodes(self.graph, node2, dist)) &
                      set(self._neighbour_nodes(self.graph, node3, dist)))
        return result

    def _next_node1(self):
        try:
            return self.node1_iter.next()
        except StopIteration:
            return None

    def eval_stat(self, dist):
        """
        Evaluate 3-graphlet statistics for each graph node's neighbouhood.

        :param dist: radius of neighbourhood
        """
        print 'Compute 3-graphlets occurance.' \
              ''
        begin_time = time()

        i = 0
        node1 = self._next_node1()
        while node1 is not None:
            i += 1
            seen = set([])
            print '\r    node {0}/{1}  [{2}%]'.format(i, len(self.graph), i * 100 / len(self.graph)),
            nodes2 = self._undir(self.graph, node1)
            for node2 in nodes2:
                # hold edge (node1,node2)
                nodes3 = list((set(self._undir(self.graph, node1)) | set(self._undir(self.graph, node2))) - \
                              (seen | {node1, node2}))
                for node3 in nodes3:
                    stats = self._neighbours_containes((node1, node2, node3), dist)
                    self._recognize_graphlet((node1, node2, node3), stats)
                seen.add(node2)
            node1 = self._next_node1()
        print
        cur_time = time()
        print '    process took {0} sec'.format(cur_time - begin_time)
        print

    def save_result(self, outfile):
        print 'Save result to the file.'
        begin_time = time()

        graph_elems = []
        i = 0
        for node in self.graph:
            i += 1
            print '\r    line {0}/{1}  [{2}%]'.format(i, len(self.graph), i * 100 / len(self.graph)),

            for type in self.graph[node]['stat']:
                self.graph[node]['stat'][type]['val'] /= 3
                del self.graph[node]['stat'][type]['lock']

            graph_elems.append({'id': node, 'stat': self.graph[node]['stat']})
        print

        print 'convert list to strings'
        graph_elems = map(lambda x: json.dumps(x) + '\n', graph_elems)

        # print strings into output file
        print 'Print sample into file'
        with open(outfile, 'wt') as output:
            output.writelines(graph_elems)


class MyThread(threading.Thread):
    def __init__(self, id, dist, graph):
        threading.Thread.__init__(self)
        self.id = id
        self.dist = dist
        self.graph = graph

    def run(self):
        print 'thread {0} started'.format(self.id)
        self.graph.eval_stat(self.dist)
        print 'thread {0} finished'.format(self.id)


def main(infile, outfile, dist, thread_num):
    sample = Graph(infile)

    threads = []
    for i in xrange(thread_num):
        threads.append(MyThread(i + 1, dist, sample))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    sample.save_result(outfile)
    print 'main thread finished'

    return 0


if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    dist = int(sys.argv[3])
    thread_num = int(sys.argv[4])
    main(infile, outfile, dist, thread_num)
