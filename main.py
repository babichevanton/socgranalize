import json


def sample_extract(file):
    with open(file, 'rt') as input:
        strings = input.readlines()

    # list of elems. Elem is dict with 'id' as node index and 'value' as list of neighbours
    list_sample = map(lambda x: json.loads(x), strings)

    # convert to dict of nodes with node index as key and list of neighbours as value
    sample = {}
    for node in list_sample:
        sample[node['id']] = node['value']

    return sample


def neighbour_nodes(graph, node, dist):
    nodes = set([node])
    neighbours = set([])
    for i in xrange(dist):
        for node in nodes:
            neighbours = neighbours.union(set(graph[node]))
        nodes = neighbours
        neighbours = set([])

    return list(nodes)



def main():
    samplefile = 'data/graphSample.json'

    sample = sample_extract(samplefile)

    return 0


if __name__ == "__main__":
    main()