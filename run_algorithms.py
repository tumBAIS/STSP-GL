import numpy.random
import random
import sys

import solver.MIP_solver
import generators.TSPLIB_demand
import generators.calculate_input_parameters as calc
import time
from pathlib import Path
import solver.advanced_branch as ab
import solver.check_minimum_covers as mc
import pickle


def calculate_path(route_edges):
    edges = route_edges
    path = [tuple(route_edges[0])[0], tuple(route_edges[0])[1]]
    edges.remove(route_edges[0])

    while len(edges) > 0:
        new_edge = False
        for edge in edges:
            if tuple(edge)[0] == path[-1]:
                path.append(tuple(edge)[1])
                edges.remove(edge)
                new_edge = True

            elif tuple(edge)[1] == path[-1]:
                path.append(tuple(edge)[0])
                edges.remove(edge)
                new_edge = True

        if not new_edge:
            raise ValueError('Optimal solution has subtours! We assume demand patterns where this does not happen!')


    return path


if __name__ == "__main__":
    print("Number of arguments:", len(sys.argv), "arguments.")
    print("Argument List:", str(sys.argv))

    dimension = float(sys.argv[1])  # [14, 16, 17, 21, 22, 24, 26, 29, 42, 48, 51, 52, 70, 76.1, 76.2, 96, 99, 100.1, 100.2, 100.3, 100.4, 100.5, 101]
    if dimension not in [14, 16, 17, 21, 22, 24, 26, 29, 42, 48, 51, 52, 70, 76.1, 76.2, 96, 99, 100.1, 100.2, 100.3, 100.4, 100.5, 101]:
        raise ValueError('Dimension has to be in [14, 16, 17, 21, 22, 24, 26, 29, 42, 48, 51, 52, 70, 76.1, 76.2, 96, 99, 100.1, 100.2, 100.3, 100.4, 100.5, 101]')

    scenarios = int(sys.argv[2])  # 1, 5, 10, 20, ..., 50
    if not scenarios > 1:
        raise ValueError('Scenarios have to be a positive integer')

    seed = int(sys.argv[3])  # between 0 - 5

    theta = float(sys.argv[4])  # 0 - 1
    if not 0 <= theta <= 1:
        raise ValueError('Theta has to be between 0 and 1')

    rho = float(sys.argv[5])  # 0 - 1
    if not 0 <= rho <= 1:
        raise ValueError('Rho has to be between 0 and 1')

    alpha = float(sys.argv[6]) # 0 - 1
    if not 0 <= alpha <= 1:
        raise ValueError('Alpha has to be between 0 and 1')

    run_option = sys.argv[7] # MIP, CG, heuristic, hybrid
    if run_option not in ['MIP', 'CG', 'heuristic', 'hybrid']:
        raise ValueError('Run option has to be in ["MIP", "CG", "heuristic", "hybrid"]')

    arcs = sys.argv[8]  # all, reachable
    if arcs not in ["all", "reachable"]:
        raise ValueError('Arcs has to be in ["all", "reachable"]')

    value_of_stochastic_solution = (sys.argv[9] == 'True') # True, False
    if value_of_stochastic_solution not in [True, False]:
        raise ValueError('value of stochastic solution has to be in [True, False]')

    computation_time = 1 * 60 # Max computation time of the algorithm

    # dimension = 21  # 15 - 35
    # scenarios = 20  # 1, 5, 10, 20, ..., 50
    # seed = 0  # between 0 - 10
    # theta = 0.95  # 0.6 - 1
    # rho = 0.05  # 0.0 - 0.3
    # alpha = 0.25
    # max_iter=5
    # new_covers_per_iteration=5
    # run_option='hybrid'
    # arcs = 'all'
    # value_of_stochastic_solution = False

    print('\n########################################### Input parameters ############################################')
    print(f'dimension: {dimension}')
    print(f'scenarios: {scenarios}')
    print(f'seed: {seed}')
    print(f'theta: {theta}')
    print(f'rho: {rho}')
    print(f'alpha: {alpha}')
    print(f'run option: {run_option}')
    print(f'arcs: {arcs}')
    print(f'value of stochastic solution: {value_of_stochastic_solution}')
    print('#########################################################################################################\n')

    print('\n################################################# SETUP #################################################')



    # fix seed for the instance
    numpy.random.seed(seed)
    random.seed(seed)

    # build graph and generate demand
    graph, node_names = generators.TSPLIB_demand.generate_graph(dimension)

    # First generate demand to get the same compulsory nodes, i.e., identical instances for value_of_stochastic_solution == False and value_of_stochastic_solution == True
    demand = generators.TSPLIB_demand.generate_demand(node_names, dimension, scenarios, False)

    instance_name = f'{seed}-{len(graph.nodes)}-{len(demand)}-{theta}-{rho}'
    filename = Path.cwd() / 'results' / f'Demand_{instance_name}'# linux
    with open(f'{filename}.pickle', 'wb') as f:
        pickle.dump(demand, f, pickle.HIGHEST_PROTOCOL)

    # Find compulsory stops
    compulsory_stops = generators.TSPLIB_demand.get_compulsory_stops(graph, demand, percentage=0.2)

    # If we want to solve the deterministic problem find average demand over all scenarios
    if value_of_stochastic_solution:
        demand = generators.TSPLIB_demand.get_average_demand(demand)
        print(demand)

    # Determine set of all request D
    D = calc.calculate_D(demand)

    # Precalculate costs of requests to not recalculate them in the different subproblems
    lower_bound, upper_bound, edges_TSP, lower_bound_design, subtours, paths, path_arcs = mc.bounds_cover(demand, D, graph, theta, alpha, compulsory_stops)
    q = calc.calculate_q(demand, graph, theta, D, calc.calculate_all_arcs(graph, demand))

    # If wanted, reduce the set of arcs for each requests ('reachable') or not ('all')
    if arcs == 'all':
        reachable_arcs = calc.calculate_all_arcs(graph, demand)
    elif arcs == 'reachable':
        reachable_arcs = calc.calculate_reachable_arcs(graph, demand, paths, distance_scale=2)

    print('#########################################################################################################\n')

    print('######################################### RUN ALGORITHM #################################################\n')
    # Solve the STSP-GL as a MIP, i.e., our benchmark
    if run_option == 'MIP':
        problem = solver.MIP_solver.Model(seed, graph, demand, theta, rho, alpha, q, compulsory_stops, reachable_arcs, 0.02, computation_time)
        t = time.time()
        problem.build_problem()
        print(f'Time to build MIP: {time.time() - t}')
        # time runtime of algorithm
        t = time.time()
        problem.solve()
        solution_time = time.time() - t
        print(solution_time)
        problem.save_solution()

        print(
            '#########################################################################################################\n')

        # Print information about run
        try:
            print('\n########################################### SOLUTION ###########################################')
            print(f'Computation time: {round(time.time() - t, 2)} seconds')
            print(f'Upper bound: {round(problem.m.objVal, 2)}')
            print(f'Lower bound: {round(problem.m.ObjBound, 2)}')
            print(f'Optimality gap: {round(problem.m.MIPGap * 100, 2)} %')
            print(f'Design cost: {round(problem.calculate_design_cost(), 2)}')
            print(f'Feasibility cover size: {problem.calculate_feasibility_cover_size()}')
            route = calculate_path(problem.get_edges())
            print(f'Node cover size: {len(route) - 1}')
            print(f'Route: {route}')

        except AttributeError:
            print('Run unsucessfull')

        print('################################################################################################\n')


    # Solve the STSP-GL with one of our approaches column generation based branch and price ('CG'),
    # local search based heuristic ('heuristic'), or hybrid approach ('hybrid')
    else:
        t = time.time()
        a = ab.AdvancedBranch(seed, graph, demand, theta, rho, alpha, D, q, compulsory_stops, reachable_arcs, 0.02, computation_time)

        if run_option == 'CG':
            a.solve_CG()

        elif run_option == 'heuristic':
            a.solve(explore_steps=5, explore_option='moderate')

        elif run_option == 'hybrid':
            a.solve_CG_with_local_search()

        print(
            '#########################################################################################################\n')
        # Print information about run
        try:
            print('\n########################################### SOLUTION ###########################################')
            print(f'Computation time: {round(time.time() - t, 2)} seconds')
            print(f'Upper bound: {round(a.upper_bound, 2)}')
            if run_option != 'heuristic':
                print(f'Lower bound: {round(max(a.lower_bounds), 2)}')
                print(f'Optimality gap: {round(a.optimality_gap * 100, 2)} %')
            print(f'Design cost: {round(a.solved_nodes[0].design_cost, 2)}')
            print(f'Feasibility cover size: {len(a.solved_nodes[0].feasibility_cover)}')
            print(f'Node cover size: {len(a.solved_nodes[0].node_cover)}')
            print(f'Route: {calculate_path(a.solved_nodes[0].route_edges)}')

            print('################################################################################################\n')

        except AttributeError:
            print(
                f'Terminated after {time.time() - t} with an upper bound of {round(a.upper_bound, 2)}')



    print("End of execution.")



# run this code via "python .\run_algorithms.py 21 5 0 0.95 0.05 0.25 CG all False"
