import networkx as nx

import solver.symmetric_TSP as stsp
import random

import time


def is_minimum_cover(demand, cover: list, theta: float, rho: float, minimum_demand_served: dict, served_demand: dict, compulsory_stops: list):
    '''
    Check if a cover is a minimal feasible cover. Return boolean and set which is either the minimum cover (True, cover)
    or the subset which is still feasibile (False, subset).
    '''
    if served_demand is None:
        served_demand = do_sum(demand, cover)

    # check if the cover is feasible
    if not is_feasible_cover(demand, cover, theta, rho, minimum_demand_served, served_demand, compulsory_stops):
        return False, None, served_demand

    served_demand_subset = dict()
    for i in range(len(cover)):
        subset = cover[:i] + cover[i + 1:]

        # calculate new served demand
        for scenario in served_demand:
            try:
                served_demand_subset[scenario] = served_demand[scenario] - demand[scenario][cover[i]]
            except KeyError:
                served_demand_subset[scenario] = served_demand[scenario]

        # if that is the case our original cover is not a minimal cover
        if is_feasible_cover(demand, subset, theta, rho, minimum_demand_served, served_demand_subset, compulsory_stops):
            return False, subset, served_demand_subset

    # if all subsets of the original cover are no feasible covers, the original cover is a minimal cover
    return True, cover, served_demand


def is_minimum_node_cover(demand: dict, cover: frozenset, theta: float, rho: float, nodes, minimum_demand_served: dict, compulsory_stops: list):
    # check if the cover is feasible
    if not is_feasible_cover(demand, cover, theta, rho, minimum_demand_served, compulsory_stops):
        return False, None, None

    # check if any subset of the original cover is still a feasible cover
    for i in range(len(nodes)):
        subset = remove_nodes_from_cover(cover, [nodes[i]])

        # if that is the case our original cover is not a minimal cover
        if is_feasible_cover(demand, subset, theta, rho, minimum_demand_served, compulsory_stops):
            return False, subset, nodes[i]

    # if all subsets of the original cover are no feasible covers, the original cover is a minimal cover
    return True, cover, None


def do_sum(demand, cover):
    dict_served = dict()
    for scenario in demand:
        dict_served[scenario] = 0

        for req in cover:
            try:
                dict_served[scenario] += demand[scenario][req]
            except KeyError:
                continue

    return dict_served


def is_feasible_cover(demand: dict, cover, theta: float, rho: float, minimum_demand_served: dict, served_demand: dict, compulsory_stops: list, node_cover=list()) -> bool:
    '''
    Check if a subset of reqeusts (cover) is a feasible cover
    '''

    nodes = list(get_nodes_from_cover(cover, node_cover))

    # Check if all compulsory stops are in the cover
    for stop in compulsory_stops:
        if stop not in nodes:
            print('Not all compulsory stops in node cover!')
            return False

    if len(minimum_demand_served) == 0:
        minimum_demand_served = {scenario: sum(demand[scenario].values()) * theta for scenario in demand}

    if served_demand is None:
        served_demand = do_sum(demand, cover)

    num_scenarios_service_level_met = 0

    for scenario in demand:
        # epsilon needed because otherwise floats may cause weird things, e.g., 9.95000000000001 > 9.950000000000003
        if served_demand[scenario] >= minimum_demand_served[scenario] - 0.000001:
            num_scenarios_service_level_met += 1

    if num_scenarios_service_level_met >= (1 - rho) * len(demand):
        return True
    else:
        return False


def get_feasible_subsets(demand, cover, theta, rho, compulsory_stops):
    # check if the cover is feasible
    if not is_feasible_cover(demand, cover, theta, rho, compulsory_stops):
        return False

    feasible_subsets = [cover]
    for req in cover:
        subset = cover - {req}

        # if that is the case our original cover is not a minimal cover
        if is_feasible_cover(demand, subset, theta, rho, compulsory_stops):
            feasible_subsets.append(subset)

    return feasible_subsets


def get_minimum_cover_from_cover(demand: dict, cover: list, theta: float, rho: float, minimum_demand_served: dict, compulsory_stops: list, served_demand=None):
    if served_demand:
        assert is_feasible_cover(demand, cover, theta, rho, minimum_demand_served, served_demand, compulsory_stops), 'The given cover is not feasible!'
        is_minimum, cover, served_demand = is_minimum_cover(demand, cover, theta, rho, minimum_demand_served, served_demand, compulsory_stops)
    else:
        assert is_feasible_cover(demand, cover, theta, rho, minimum_demand_served,
                                 served_demand, compulsory_stops), 'The given cover is not feasible!'
        is_minimum, cover, served_demand = is_minimum_cover(demand, cover, theta, rho, minimum_demand_served, None, compulsory_stops)

    while not is_minimum:
        is_minimum, cover, served_demand = is_minimum_cover(demand, cover, theta, rho, minimum_demand_served, served_demand, compulsory_stops)

    return frozenset(cover)


def get_minimum_node_cover_from_cover(demand: dict, cover, D, theta: float, rho: float):
    cover = set(cover)
    nodes = list(get_nodes_from_cover(cover))

    cover = [req for req in D if req[0] in nodes and req[1] in nodes]
    random.shuffle(cover)
    random.shuffle(nodes)

    is_minimum, cover, removed_node = is_minimum_node_cover(demand, cover, theta, rho, nodes)
    a = list()
    while not is_minimum:
        is_minimum, cover, removed_node = is_minimum_node_cover(demand, cover, theta, rho,
                                                                [node for node in nodes if node != removed_node])

    return get_minimum_cover_from_cover(demand, list(cover), theta, rho)


def is_node_cover(demand, nodes, D, theta, rho, minimum_demand_served, compulsory_stops: list):
    cover = [req for req in D if req[0] in nodes and req[1] in nodes]

    return is_feasible_cover(demand, cover, theta, rho, minimum_demand_served, None, compulsory_stops), frozenset(cover)


def get_nodes_from_cover(cover, node_cover=set()):
    used_nodes = set(node_cover)

    for req in cover:
        used_nodes.add(req[0])
        used_nodes.add(req[1])

    return frozenset(used_nodes)


def remove_nodes_from_cover(cover, nodes):
    cover = set(cover)
    for req in list(cover):
        if req[0] in nodes or req[1] in nodes:
            cover.remove(req)

    return cover


def add_nodes_to_cover(cover, used_nodes, new_nodes, D, demand, theta, rho, dual_requests, minimum_demand_served):
    cover = set(cover)
    all_nodes = set(new_nodes).union(set(used_nodes))

    allowed_new_req = [req for req in D if req not in cover and req[0] in all_nodes and req[1] in all_nodes]

    while not is_feasible_cover(demand, cover, theta, rho, minimum_demand_served):
        new_req = min(allowed_new_req, key=dual_requests.get)
        cover.add(new_req)
        allowed_new_req.remove(new_req)

    return cover


def get_new_minimum_cover(demand, D, cover, theta, rho, option=None,
                          seed=0, node_cover=frozenset(), minimum_demand_served={},
                          compulsory_stops=[]):
    '''
    Input feasibility cover
    Output new feasibility cover
    '''

    # Basic form
    if option is None:

        new_cover = list(cover)
        set_removed_req = set()

        served_demand = do_sum(demand, new_cover)
        served_demand_subset = dict(served_demand)
        # If req cover is still feasible remove nodes until cover is not feasible again
        is_feasible = is_feasible_cover(demand, new_cover, theta, rho, minimum_demand_served, served_demand, compulsory_stops)
        while is_feasible:
            removed_req = new_cover.pop(seed % len(new_cover))
            set_removed_req.add(removed_req)

            # calculate new served demand
            for scenario in served_demand:
                try:
                    served_demand_subset[scenario] = served_demand_subset[scenario] - demand[scenario][removed_req]
                except KeyError:
                    served_demand_subset[scenario] = served_demand_subset[scenario]

            is_feasible = is_feasible_cover(demand, new_cover, theta, rho, minimum_demand_served, served_demand_subset, compulsory_stops)

        # add 'nodes_change' many random request to cover added requests should not be removed req
        all_req_of_nodecover = [req for req in D if req[0] in node_cover and req[1] in node_cover]

        all_req = set(all_req_of_nodecover) - cover
        if len(all_req) == 0:
            all_req = set(D) - set(new_cover)

        # If new req cover is not feasible, add random requests to req cover until it is feasible again
        served_demand = do_sum(demand, new_cover)
        while not is_feasible:

            if len(all_req) == 0:
                all_req = set(D) - set(new_cover)

            # Added if D is the only feasible cover
            if len(all_req) == 0:
                return D

            new_req = all_req.pop()
            new_cover.append(new_req)

            # calculate new served demand
            for scenario in served_demand:
                try:
                    served_demand_subset[scenario] = served_demand_subset[scenario] + demand[scenario][new_req]
                except KeyError:
                    served_demand_subset[scenario] = served_demand_subset[scenario]

            is_feasible = is_feasible_cover(demand, new_cover, theta, rho, minimum_demand_served, served_demand_subset, compulsory_stops)

        random.shuffle(new_cover)

        return get_minimum_cover_from_cover(demand, new_cover, theta, rho, minimum_demand_served, compulsory_stops, served_demand_subset)


def get_upper_bound_routing(demand, cover, network, theta, alpha, edges, lower_bound_design):
    # get upper bound by finding shortest path on TSP solution
    arcs_sym_TSP = [tuple(edge) for edge in edges]

    edge_subgraph = network.edge_subgraph(arcs_sym_TSP)
    paths = dict(nx.shortest_path_length(edge_subgraph.to_undirected(), weight='cost'))
    path_arcs = dict(nx.shortest_path(edge_subgraph.to_undirected(), weight='cost'))

    upper_bound_routing = 1 / len(demand) * sum(demand[n].get(req, 0) / (theta * sum(demand[n].values())) *
                                                paths[req[0]][req[1]] for n in demand for req in cover)

    upper_bound = (1 - alpha) * lower_bound_design + alpha * upper_bound_routing
    return upper_bound, paths, path_arcs


def get_lower_bound_routing(demand, cover, network, theta, alpha, lower_bound_design):
    lower_bound_routing = 1 / len(demand) * sum(demand[n].get(req, 0) / (theta * sum(demand[n].values())) *
                                                network.edges[req]['cost'] for n in demand for req in cover)

    lower_bound = (1 - alpha) * lower_bound_design + alpha * lower_bound_routing

    return lower_bound


def bounds_cover(demand, cover, network, theta, alpha, compulsory_stops=set()):
    t = time.time()
    relevant_nodes = get_nodes_from_cover(cover, compulsory_stops)

    relevant_edges_h = [edge for edge in network.edges() if edge[0] in relevant_nodes and edge[1] in relevant_nodes]
    relevant_edges = set(frozenset(arc) for arc in relevant_edges_h)

    # solve S-TSP to get a lower bound for the design part
    sym_TSP = stsp.S_TSP(network, relevant_edges, relevant_nodes)
    sym_TSP.calculate_dist_dict()
    sym_TSP.build_model()
    sym_TSP.solve_model()

    lower_bound_design = sym_TSP.m.objVal
    edges_TSP = {edge: value.X for edge, value in sym_TSP.x.items() if value.X > 0.5}

    lower_bound = get_lower_bound_routing(demand, cover, network, theta, alpha, lower_bound_design)

    upper_bound, paths, path_arcs = get_upper_bound_routing(demand, cover, network, theta, alpha, edges_TSP, lower_bound_design)

    print(f'Time to calculate bound: {time.time() - t}')
    return lower_bound, upper_bound, edges_TSP, lower_bound_design, sym_TSP.m._subtours, paths, path_arcs
