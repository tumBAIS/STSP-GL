import gurobipy as gp
from gurobipy import GRB
import solver.BBC_problems as bbc
import solver.symmetric_TSP as tsp

import math
import time


class AlgorithmBBC1:
    def __init__(self, demand, cover, network, alpha, theta, q, reachable_arcs):
        self.demand = demand
        self.cover = cover
        self.network = network
        self.alpha = alpha
        self.theta = theta
        self.reachable_arcs = reachable_arcs

        # Relevant elements
        self.relevant_nodes = None
        self.relevant_arcs = None
        self.relevant_edges = None
        self.q = q

        # Master
        self.master = None

        # Subproblems
        self.dual_problems = dict()

        # Bounds
        self.lower_bounds = list()
        self.upper_bounds = list()
        self.optimality_gap = list()
        self.same_solution_x = False

        # Memory
        self.x = dict()

    def get_relevant_elements(self):
        # Determine relevant nodes in subnetwork
        self.relevant_nodes = set()
        for (h, k) in self.cover:
            self.relevant_nodes.add(h)
            self.relevant_nodes.add(k)

        # Determine relevant arcs in subnetwork
        self.relevant_arcs = set()
        for (i, j) in self.network.edges():
            if i in self.relevant_nodes and j in self.relevant_nodes:
                self.relevant_arcs.add((i, j))

        # Determine undirected edges in subnetwork
        self.relevant_edges = set(frozenset(arc) for arc in self.relevant_arcs)

        # Determine q in subproblem
        # self.q = dict()
        # for req in self.cover:
        #     self.q[req] = dict()
        #     for arc in self.relevant_arcs:
        #         self.q[req][arc] = self.network.edges[arc]['cost'] / len(self.demand) * \
        #                            sum(self.demand[n].get(req, 0) /
        #                                (self.theta * sum(self.demand[n].values())) for n in self.demand)

        # # Determine q in subproblem (done only once for all in algorithm)
        # self.q = dict()
        # temp_demand_sum = {n: sum(self.demand[n].values()) for n in self.demand} #needed for runtime (~80% reduction)
        #
        # for req in self.cover:
        #     self.q[req] = {arc: 0 for arc in self.relevant_arcs}
        #     for arc in self.relevant_arcs:
        #         for n in self.demand:
        #             try:
        #                 self.q[req][arc] += self.network.edges[arc]['cost'] / len(self.demand) * \
        #                                    self.demand[n][req] / (self.theta * temp_demand_sum[n])
        #             except KeyError:
        #                 continue

        return

    def setup_master_BBC1(self):
        # Build the masterproblem of BBC1
        self.master = bbc.MasterProblemBBC(self.alpha)
        self.master.build_problem_BBC1(self.relevant_nodes, self.relevant_edges, self.network)

    def setup_master_BBC2(self):
        # Build the masterproblem of BBC1
        self.master = bbc.MasterProblemBBC(self.alpha)
        self.master.build_problem_BBC2(self.relevant_nodes, self.relevant_edges, self.network)

    def solve_master(self):
        self.master.m.Params.LogToConsole = 0
        self.master.solve()
        new_x = {edge: self.master.x[edge].X for edge in self.relevant_edges if self.master.x[edge].X > 0}
        if new_x == self.x:
            self.same_solution_x = True

        self.x = new_x

    def initialize_h2(self):
        '''
        This function initializes BBC1 by solving the S-TSP
        '''
        s_tsp = tsp.S_TSP(self.network, self.relevant_edges, self.relevant_nodes)
        s_tsp.calculate_dist_dict()
        s_tsp.build_model()
        s_tsp.solve_model()

        for edge in s_tsp.edges:
            if s_tsp.x[edge].X > 0.5:
                self.x[edge] = s_tsp.x[edge].X

        self.lower_bounds.append((1 - self.alpha) * s_tsp.m.objVal)

    def initialize_with_edges(self, edges: dict, lower_bound: float):
        for edge, value in edges.items():
            self.x[edge] = value

        self.master.m.update()

        self.lower_bounds.append(lower_bound)
        self.upper_bounds.append(math.inf)

    def solve_dual_subproblems(self):
        rhs = 0
        lhs = {edge: 0 for edge in self.relevant_edges}
        bound = 0

        for h, k in self.cover:

            # Store the subproblems in a dict (Idea: only update objective function)
            if (h, k) not in self.dual_problems:
                self.dual_problems[(h, k)] = bbc.DualSubproblemBBC1(self.x, h, k)
                self.dual_problems[(h, k)].build_subproblem(self.relevant_arcs, self.relevant_edges,
                                                            self.relevant_nodes,
                                                            self.q[(h, k)])

            # If the subproblem already exist update the objective function
            else:
                self.dual_problems[(h, k)].update_objective(self.relevant_edges, self.x)

            # solve the subproblem
            self.dual_problems[(h, k)].solve_subproblem()

            # return an extreme ray if decomposed subproblem is unbounded
            if self.dual_problems[(h, k)].m.status == GRB.Status.UNBOUNDED:
                return 'feasibility_cut', {'rho': {n: self.dual_problems[(h, k)].rho[n].UnbdRay
                                                   for n in self.relevant_nodes},
                                           'chi': {arc: self.dual_problems[(h, k)].chi[arc].UnbdRay
                                                   for arc in self.relevant_arcs}}, \
                       (h, k), None

            # save soltuion values if decomposed subproblem is bounded (may be needed for optimality cut)
            elif self.dual_problems[(h, k)].m.status == GRB.OPTIMAL:
                bound += self.dual_problems[(h, k)].m.objVal
                rhs += self.dual_problems[(h, k)].rho[h].X - self.dual_problems[(h, k)].rho[k].X

                for i, j in self.relevant_edges:
                    lhs[frozenset([i, j])] += self.dual_problems[(h, k)].chi[(i, j)].X + \
                                              self.dual_problems[(h, k)].chi[(j, i)].X

        return 'optimality_cut', rhs, lhs, bound

    def solve_dual_subproblems_o3(self):
        '''
        optimality cuts as described in o3 of Errico et al Paper
        '''
        lhs = {n: {edge: 0 for edge in self.relevant_edges} for n in self.relevant_nodes}
        rhs = {n: 0 for n in self.relevant_nodes}
        bound = 0

        for h, k in self.cover:
            # print(f'Solve dual subproblem for {h, k}')

            # Store the subproblems in a dict (Idea: only update objective function)
            if (h, k) not in self.dual_problems:
                self.dual_problems[(h, k)] = bbc.DualSubproblemBBC1(self.x, h, k, self.reachable_arcs)
                self.dual_problems[(h, k)].build_subproblem(self.relevant_arcs, self.relevant_edges,
                                                            self.relevant_nodes,
                                                            self.q[(h, k)])

            # If the subproblem already exist update the objective function
            else:
                self.dual_problems[(h, k)].update_objective(self.relevant_edges, self.x)

            # solve the subproblem
            self.dual_problems[(h, k)].solve_subproblem()
            # return an extreme ray if decomposed subproblem is unbounded (should not happen with subtourelim before)
            if self.dual_problems[(h, k)].m.status == GRB.Status.UNBOUNDED:
                return 'feasibility_cut', {'rho': {n: self.dual_problems[(h, k)].rho[n].UnbdRay
                                                   for n in self.relevant_nodes},
                                           'chi': {arc: self.dual_problems[(h, k)].chi[arc].UnbdRay
                                                   for arc in self.relevant_arcs}}, \
                       (h, k), None

            elif self.dual_problems[(h, k)].m.status == GRB.OPTIMAL:
                bound += self.dual_problems[(h, k)].m.objVal
                rhs[h] += self.dual_problems[(h, k)].rho[h].X - self.dual_problems[(h, k)].rho[k].X

                for i, j in self.relevant_edges:
                    lhs[h][frozenset([i, j])] += self.dual_problems[(h, k)].chi[(i, j)].X + \
                                                 self.dual_problems[(h, k)].chi[(j, i)].X

        return 'optimality_cut', rhs, lhs, bound

    def add_feasibility_cut(self, ray, h, k):
        # Add cut
        self.master.feasibility_cuts.append(
            self.master.m.addConstr(ray['rho'][h] - ray['rho'][k] - gp.quicksum((ray['chi'][(i, j)] +
                                                                                 ray['chi'][(j, i)]) *
                                                                                self.master.x[frozenset([i, j])]
                                                                                for i, j in self.relevant_edges) <= 0))

    def add_feasibility_cut_subtour(self, subtours):
        # Add cut
        for tour in subtours:
            # add subtour elimination constr. for every pair of cities in subtour
            self.master.feasibility_cuts.append(
                self.master.m.addConstr(gp.quicksum(self.master.x[edge] for edge in tour)
                                        <= len(tour) - 1))

        # Update the model
        self.master.m.update()

    def add_optimality_cut(self, rhs, lhs):
        self.master.optimality_cuts.append(self.master.m.addConstr(self.master.eta +
                                                                   gp.quicksum(lhs[edge] * self.master.x[edge]
                                                                               for edge in self.relevant_edges) >= rhs))

    def add_optimality_cut_o3(self, rhs, lhs):
        for h in self.relevant_nodes:
            self.master.optimality_cuts.append(self.master.m.addConstr(self.master.eta[h] +
                                                                       gp.quicksum(lhs[h][edge] * self.master.x[edge]
                                                                                   for edge in self.relevant_edges) >=
                                                                       rhs[h]))


def run_BBC1(demand: dict, cover: frozenset, graph, alpha: float, theta: float, edges: dict, lower_bound: float, q: dict):
    error = 1e-6
    alg = AlgorithmBBC1(demand, cover, graph, alpha, theta, q)
    alg.get_relevant_elements()
    alg.setup_master_BBC1()
    # alg.initialize_h2()
    alg.initialize_with_edges(edges, lower_bound)

    while True:
        t = time.time()
        cut_type, a, b, bound = alg.solve_dual_subproblems()

        if cut_type == 'feasibility_cut':
            # print(f'feasibility cut after {time.time() - t}')
            alg.add_feasibility_cut(a, b[0], b[1])

        elif cut_type == 'optimality_cut':
            # print(f'optimality cut after {time.time() - t}')
            alg.add_optimality_cut(a, b)

            alg.upper_bounds.append(bound + alg.lower_bounds[-1])

        # import helper.plot_graph as plt
        #
        # plt.plot_route_graph(alg.network, alg.x)

        if min(alg.upper_bounds) - max(alg.lower_bounds[1:], default=0) < error:
            break

        alg.solve_master()

        if alg.same_solution_x:
            break

        alg.lower_bounds.append(alg.master.m.objVal)

    return alg


def run_BBC2(demand: dict, cover: frozenset, graph, alpha: float, theta: float, edges: dict, lower_bound: float, upper_bound: float, q: dict, subtours: list, reachable_arcs: dict):
    error = 1e-6
    alg = AlgorithmBBC1(demand, cover, graph, alpha, theta, q, reachable_arcs)
    alg.get_relevant_elements()
    alg.setup_master_BBC2()
    # alg.initialize_h2()
    alg.initialize_with_edges(edges, lower_bound)
    alg.add_feasibility_cut_subtour(subtours)

    n = 0
    t_start = time.time()
    while time.time() - t_start < 20*60:
        if max(alg.lower_bounds, default=0) >= upper_bound:
            break
        n += 1
        t = time.time()
        #################### SUBTOUR ELIMINATION ############################
        if n > 1:
            used_edges = gp.tuplelist(edge for edge, val in alg.master.x.items() if val.X > 0.999)
            subtours = tsp.subtour(used_edges, alg.relevant_nodes)
            # print(f'Add calculated subtours in {time.time() - t}')
        if n > 1 and len(subtours) > 1:
            t = time.time()
            alg.add_feasibility_cut_subtour(subtours)
            # print(f'Added feaibility cut in {time.time() - t}')

        #################### OPTIMALITY CUTS ################################
        else:
            # print('Solve subproblems...')
            t = time.time()
            cut_type, rhs, lhs, bound = alg.solve_dual_subproblems_o3()
            # print(f'Solved subproblems in {time.time() - t}')

            if cut_type != 'optimality_cut':
                raise AssertionError
            t = time.time()
            alg.add_optimality_cut_o3(rhs, lhs)
            # print(f'Add optimality cut in {time.time() - t}')
            alg.upper_bounds.append(bound + alg.lower_bounds[-1])

            # import helper.plot_graph as plt
            #
            # plt.plot_route_graph(alg.network, alg.x)

            if min(alg.upper_bounds) - max(alg.lower_bounds[1:], default=0) < error:
                break

        t = time.time()
        alg.solve_master()
        # print(f'Solved master in {time.time() - t}')

        if alg.same_solution_x:
            break

        alg.lower_bounds.append(alg.master.m.objVal)

    return alg
