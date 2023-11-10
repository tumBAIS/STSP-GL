import numpy as np
import tsplib95
import networkx as nx
import os


def load_demand(instance):

    paths = {
        14: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/burma14_tsp_14_4_4_6_44_50_6_112_0_0_0_100_3687662_0_0'),
        16: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/ulysses16_tsp_16_4_4_6_44_50_8_132_0_0_0_100_7700916_0_0'),
        17: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/gr17_tsp_17_4_4_6_44_50_8_142_0_0_0_100_830042_0_0'),
        21: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/gr21_tsp_21_4_4_6_44_50_11_182_0_0_0_100_1099468_0_0'),
        22: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/ulysses22_tsp_22_4_4_6_44_50_11_192_0_0_0_100_7710422_0_0'),
        24: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/gr24_tsp_24_4_4_6_44_50_13_212_0_0_0_100_3906865_0_0'),
        26: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/fri26_tsp_26_4_4_6_44_50_14_232_0_0_0_100_5757882_0_0'),
        29: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/bayg29_tsp_29_4_4_6_44_50_16_262_0_0_0_100_8704652_0_0'),
        42: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/dantzig42_tsp_42_4_4_6_44_50_24_392_0_0_0_100_7450950_0_0'),
        48: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/att48_tsp_48_4_4_6_44_50_28_452_0_0_0_100_7499019_0_0'),
        51: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/eil51_tsp_51_4_4_6_44_50_30_482_0_0_0_100_2602220_0_0'),
        52: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/berlin52_tsp_52_4_4_6_44_50_30_492_0_0_0_100_7857992_0_0'),
        70: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/st70_tsp_70_4_4_6_44_50_42_672_0_0_0_100_2639062_0_0'),
        76.1: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/eil76_tsp_76_4_4_6_44_50_45_732_0_0_0_100_7767899_0_0'),
        76.2: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/pr76_tsp_76_4_4_6_44_50_45_732_0_0_0_100_1542109_0_0'),
        96: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/gr96_tsp_96_4_4_6_44_50_58_932_0_0_0_100_1631895_0_0'),
        99: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/rat99_tsp_99_4_4_6_44_50_60_962_0_0_0_100_1547366_0_0'),
        100.1: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/kroA100_tsp_100_4_4_6_44_50_61_972_0_0_0_100_2603610_0_0'),
        100.2: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/kroB100_tsp_100_4_4_6_44_50_61_972_0_0_0_100_4359226_0_0'),
        100.3: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/kroC100_tsp_100_4_4_6_44_50_61_972_0_0_0_100_1276170_0_0'),
        100.4: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/kroD100_tsp_100_4_4_6_44_50_61_972_0_0_0_100_9406420_0_0'),
        100.5: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/rd100_tsp_100_4_4_6_44_50_61_972_0_0_0_100_8768484_0_0'),
        101: os.path.join(os.path.dirname(__file__), '../data/TSPGL2/eil101_tsp_101_4_4_6_44_50_61_982_0_0_0_100_3537282_0_0'),
    }

    if instance not in paths:
        print(f'Instance has to be in {paths.keys()}')
        raise ValueError
    else:
        path = paths[instance]

    with open(path, 'r') as f:

        line_counter = -1

        req = 0
        demand = []
        probability = []

        dist_matrix = list()
        demand_dict = dict()
        for line in f:
            if line[0] == 'c':
                continue

            if line_counter == -1:
                num_nodes = int(line)
                line_counter += 1

            elif line_counter < num_nodes:
                dist = [int(num) for num in line.split()]
                dist.insert(line_counter, 0)
                dist_matrix.append(dist)
                line_counter += 1

            elif line_counter == num_nodes:
                line_counter += 1

            elif len(line.split()) == 2:
                req = (int(line.split()[0]), int(line.split()[1]))

            elif len(line.split()) > 2:
                demand = [int(x) for x in line.split()[::2]]
                probability = [float(x) for x in line.split()[1::2]]

            if len(demand) > 0:
                demand_dict[req] = dict()
                demand_dict[req]['demand'] = demand
                demand_dict[req]['probability'] = probability

                req = 0
                demand = []
                probability = []

    return dist_matrix, demand_dict


def generate_graph(instance):

    paths = {
        14: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/burma14.tsp/burma14.tsp'),
        16: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/ulysses16.tsp/ulysses16.tsp'),
        17: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/gr17.tsp/gr17.tsp'),
        21: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/gr21.tsp/gr21.tsp'),
        22: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/ulysses22.tsp/ulysses22.tsp'),
        24: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/gr24.tsp/gr24.tsp'),
        26: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/fri26.tsp/fri26.tsp'),
        29: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/bayg29.tsp/bayg29.tsp'),
        42: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/dantzig42.tsp/dantzig42.tsp'),
        48: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/att48.tsp/att48.tsp'),
        51: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/eil51.tsp/eil51.tsp'),
        52: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/berlin52.tsp/berlin52.tsp'),
        70: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/st70.tsp/st70.tsp'),
        76.1: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/eil76.tsp/eil76.tsp'),
        76.2: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/pr76.tsp/pr76.tsp'),
        96: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/gr96.tsp/gr96.tsp'),
        99: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/rat99.tsp/rat99.tsp'),
        100.1: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/kroA100.tsp/kroA100.tsp'),
        100.2: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/kroB100.tsp/kroB100.tsp'),
        100.3: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/kroC100.tsp/kroC100.tsp'),
        100.4: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/kroD100.tsp/kroD100.tsp'),
        100.5: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/rd100.tsp/rd100.tsp'),
        101: os.path.join(os.path.dirname(__file__), '../data/TSPLIB/eil101.tsp/eil101.tsp')
    }
    if instance not in paths:
        print(f'Instance has to be in {paths.keys()}')
        raise ValueError
    else:
        problem = tsplib95.load(paths[instance])

    G = nx.DiGraph()

    # add basic graph metadata
    G.graph['name'] = problem.name
    G.graph['dimension'] = problem.dimension

    # set up a map from original node name to new node name
    nodes = list(problem.get_nodes())
    try:
        # TSPGL2 instances go from 0 to n-1 nodes while TSPLIB instances go from 1 to n nodes. This fixes that
        names = {n - 1: tuple(problem.node_coords.get(n)) for n in nodes}

    # If no coordinates are given we just take arbitrary coords to keep the datastructure of nodes == tuples
    except TypeError:

        # If there are no node coords given in the TSPLIB instance, the nodes either start from 0 or 1
        # We have to consider both cases
        if min(nodes) == 0:
            names = {n: (n, n) for n in nodes}
        else:
            names = {n - 1: (n, n) for n in nodes}

    # add every node with some associated metadata
    for n in names:
        G.add_node(names[n], pos=problem.node_coords.get(n))

    # add every edge with some associated metadata
    for a, b in problem.get_edges():
        if a != b:
            weight = problem.get_weight(a, b)

            if len(problem.node_coords) > 0:
                G.add_edge(names[a - 1], names[b - 1], cost=weight)

            # If there are no node coords given in the TSPLIB instance, the nodes either start from 0 or 1
            # We have to consider both cases
            else:
                if min(nodes) == 0:
                    G.add_edge(names[a], names[b], cost=weight)
                else:
                    G.add_edge(names[a - 1], names[b - 1], cost=weight)

    # return the graph object
    return G, names


def generate_demand(names, instance, num_scenarios, value_stochastic_solution=False):
    dist_data, demand_data = load_demand(instance)

    demand = dict()
    for n in range(num_scenarios):
        demand[n] = dict()
        for req, data in demand_data.items():

            # Make sure origin != destination
            if req[0] == req[1]:
                continue

            # Ensure the probabilities sum to 1
            data['probability'] = np.array(data['probability'])
            data['probability'] /= data['probability'].sum()

            # Get a random demand for a reqeuest based on the given probability
            demand_val = int(np.random.choice(data['demand'], 1, p=data['probability'])[0])

            # Only add positiv demand to the demand dict
            if demand_val > 0:
                try:
                    demand[n][(names[req[0]], names[req[1]])] = demand_val
                except KeyError:
                    print('here')
    return demand


def get_average_demand(demand):
    average_demand = dict()
    for scenario in demand:
        for req in demand[scenario]:
            try:
                average_demand[req] += demand[scenario][req]
            except KeyError:
                average_demand[req] = demand[scenario][req]

    for req in average_demand:
        average_demand[req] = average_demand[req] / len(demand)

    demand = {0: average_demand}

    return demand


def get_compulsory_stops(graph, demand, percentage):
    all_requests = set([req for scenario in demand for req in demand[scenario].keys()])
    all_demands = {req: sum(demand[scenario].get(req, 0) for scenario in demand) for req in all_requests}
    node_utilization = {node: sum([all_demands[req] for req in all_demands if node in req]) for node in graph.nodes()}

    sorted_nodes = sorted(node_utilization.items(), key=lambda x: x[1], reverse=True)

    compulsory_stops_sorted = sorted_nodes[:int(len(sorted_nodes) * percentage)]
    compulsory_stops = [item[0] for item in compulsory_stops_sorted]
    return compulsory_stops
