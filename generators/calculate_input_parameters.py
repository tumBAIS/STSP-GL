def calculate_D(demand):
    D = set()
    for scenario, demand_dict in demand.items():
        for req, demand_value in demand_dict.items():
            D.add(req)

    return frozenset(D)


def calculate_q(demand, graph, theta, D, reachable_arcs):
    print('Calculate q dict...')
    # Determine q in subproblem
    temp_demand_sum = {n: sum(demand[n].values()) for n in demand}  # needed for runtime (~80% reduction)
    q = dict()
    for req in D:
        q[req] = {arc: 0 for arc in reachable_arcs[req]}
        for arc in reachable_arcs[req]:
            for n in demand:
                try:
                    q[req][arc] += graph.edges[arc]['cost'] / len(demand) * \
                                        demand[n][req] / (theta * temp_demand_sum[n])
                except KeyError:
                    continue
    return q

def calculate_reachable_arcs(graph, demand, paths, distance_scale):
    '''
    Derive all arcs which can be reached within distance_scale * shortest path from origin to destination for each
    request. We utilize the fact that we have a fully connected graph.
    '''
    all_requests = set([req for scenario in demand for req in demand[scenario].keys()])

    # reachable_nodes = {req: [arc for arc in graph.edges() if (req[0] == arc[0] and graph[arc[0]][arc[1]]['cost'] < distance_scale * graph[req[0]][req[1]]['cost'])
    #                                                           or graph[req[0]][arc[0]]['cost'] < distance_scale * graph[req[0]][req[1]]['cost']] for req in all_requests}

    reachable_arcs = {req: list() for req in all_requests}
    for req in all_requests:
        for arc in graph.edges():

            # Do not add arcs going into origin
            if req[0] == arc[1]:
                continue

            # outgoing arcs origin
            if req[0] == arc[0]:

                # direct arc
                if req[1] == arc[1]:
                    reachable_arcs[req].append(arc)

                elif graph[req[0]][arc[1]]['cost'] + graph[arc[1]][req[1]]['cost'] < \
                        distance_scale * paths[req[0]][req[1]]:
                    reachable_arcs[req].append(arc)

            # ingoing arcs destination
            elif req[1] == arc[1]:

                if graph[req[0]][arc[0]]['cost'] + graph[arc[0]][req[1]]['cost'] < \
                        distance_scale * paths[req[0]][req[1]]:
                    reachable_arcs[req].append(arc)

            elif graph[req[0]][arc[0]]['cost'] + graph[arc[0]][arc[1]]['cost'] + graph[arc[1]][req[1]]['cost'] < \
                    distance_scale * paths[req[0]][req[1]]:
                reachable_arcs[req].append(arc)

            # elif graph[arc[1]][req[1]]['cost'] < distance_scale * graph[req[0]][req[1]]['cost']:
            #     reachable_arcs[req].append(arc)

    return reachable_arcs

def calculate_all_arcs(graph, demand):
    all_requests = set([req for scenario in demand for req in demand[scenario].keys()])
    reachable_arcs = {req: list() for req in all_requests}
    for req in all_requests:
        for arc in graph.edges():
            reachable_arcs[req].append(arc)

    return reachable_arcs