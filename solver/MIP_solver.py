import gurobipy as gp
from gurobipy import GRB
import csv
import time
from pathlib import Path
import solver.check_minimum_covers as mc

import matplotlib.pyplot as plt

def data_cb(model, where):
    if where == gp.GRB.Callback.MIP:
        cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
        cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)

        # Did objective value or best bound change?
        if model._obj != cur_obj or model._bd != cur_bd:
            model._obj = cur_obj
            model._bd = cur_bd
            model._data.append([time.time() - model._start, cur_obj, cur_bd])


class Model:

    def __init__(self, seed, network, demand, theta, rho, alpha, q, compulsory_stops, reachable_arcs, goal_optimality_gap: float,
                 computation_time: int):

        # Input parameters
        self.seed = seed
        self.network = network
        self.demand = demand
        self.theta = theta
        self.rho = rho
        self.q = q
        self.alpha = alpha
        self.compulsory_stops = compulsory_stops
        self.reachable_arcs = reachable_arcs
        self.goal_optimality_gap = goal_optimality_gap
        self.computation_time = computation_time
        self.D = None

        # Model
        self.m = None

        # Decision variables
        self.x = None
        self.z = None
        self.f = None
        self.Y = None
        self.w = None

        # Constraints
        self.compulsory_stops_constr = None
        self.service_flow = None
        self.service_out = None
        self.flow_passengers = None
        self.selective_bound = None
        self.passenger_flow_bound = None
        self.chance_quality = None
        self.node_constr_origin = None
        self.node_contr_dest = None

        # These constraints have to be generated
        self.subtour_elimination = dict()

        self.subtours = None

        self.counter = 0

        # analyze algorithm
        self.it_subtours = 0

    @staticmethod
    def get_all_requests(demand):
        D = set()
        for scenario, demand_dict in demand.items():
            for req, demand_value in demand_dict.items():
                D.add(req)

        return list(D)

    @staticmethod
    def flow_lookup(z, req, i):
        if i == req[0]:
            return z[req]

        elif i == req[1]:
            return -z[req]

        else:
            return 0

    def build_problem(self):
        self.D = self.get_all_requests(self.demand)
        m = gp.Model("Stochastic Program")

        #self.Determine undirected edges in network
        edges = set(frozenset(edge) for edge in self.network.edges())

        # Variables
        self.x = {edge: m.addVar(vtype=GRB.BINARY, lb=0, ub = 1,
                                 obj=(1 - self.alpha) * 2 * self.network.edges[edge]['cost'],
                                 name=f'x_{edge}') for edge in edges}

        self.z = {req: m.addVar(vtype=GRB.BINARY, lb=0, name=f'z_{req}') for req in self.D}
        self.f = {(edge, req): m.addVar(vtype=GRB.CONTINUOUS, lb=0,
                                        obj=self.alpha * self.q[req][edge],
                                        name=f'f_{edge}_{req}') for edge in self.network.edges() for req in self.D
                  if edge in self.reachable_arcs[req]}
        self.Y = {n: m.addVar(vtype=GRB.BINARY, name=f'Y_{n}') for n in self.demand}
        self.w = {i: m.addVar(vtype=GRB.BINARY, lb=0, name=f'w_{i}') for i in self.network.nodes()}

        # Constraints
        self.compulsory_stops_constr = {i: m.addConstr(self.w[i] == 1, name=f'compuslory_stop_{i}') for i in self.compulsory_stops}
        self.service_flow = {i: m.addConstr(gp.quicksum(self.x[edge] for edge in edges if i in edge) - 2*self.w[i] == 0,
                                            name=f'service_flow_{i}')
                             for i in self.network.nodes()}

        self.flow_passengers = {
            (req, i): m.addConstr(gp.quicksum(self.f[(edge, req)] for edge in self.network.out_edges(i)
                                              if edge in self.reachable_arcs[req]) -
                                  gp.quicksum(self.f[(edge, req)] for edge in self.network.in_edges(i)
                                              if edge in self.reachable_arcs[req]) ==
                                  self.flow_lookup(self.z, req, i), name=f'flow_passenger_{i}')
            for req in self.D for i in self.network.nodes()}

        self.selective_bound = {
            n: m.addConstr(gp.quicksum(self.z[req] * self.demand[n][req] for req in self.demand[n]) >=
                           self.theta * self.Y[n] * sum(self.demand[n].values()),
                           name=f'selective_bound_{n}')
            for n in self.demand}

        self.passenger_flow_bound = {(edge, req): m.addConstr(self.f[(edge, req)] <=
                                                              self.x[frozenset(edge)],
                                                              name=f'passenger_flow_bound_{(edge, req)}')
                                     for edge in self.network.edges()
                                     for req in self.D
                                     if edge in self.reachable_arcs[req]}

        self.chance_quality = m.addConstr(gp.quicksum(self.Y[n] for n in self.demand) >=
                                          (1 - self.rho) * len(self.demand),
                                          name='chance quality')

        self.node_constr_origin = {h: m.addConstr(self.w[h] - self.z[(h, k)] >= 0) for (h, k) in self.D}
        self.node_contr_dest = {k: m.addConstr(self.w[k] - self.z[(h, k)] >= 0) for (h, k) in self.D}

        self.m = m
        return

    def solve(self):
        self.m.update()

        # add time limit 1h
        self.m.setParam('TimeLimit', self.computation_time)
        self.m.setParam('Presolve', 0)
        self.m.setParam('MIPGap', self.goal_optimality_gap)
        self.m._obj = None
        self.m._bd = None
        self.m._data = []
        self.m._start = time.time()
        self.m.optimize(callback=data_cb)
        self.m._data.append([time.time() - self.m._start, self.m.ObjVal, self.m.ObjBound])

        filename = Path.cwd() / 'results' / f'{self.seed}-{len(self.network.nodes)}-{len(self.demand)}-{self.theta}-{self.rho}.csv' # linux
        # filename = f'Bounds_MIP_{len(self.network.nodes)}-{len(self.demand)}-{self.theta}-{self.rho}.csv' #windows
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.m._data)
        return

    def find_subtours(self):
        used_edges = [edge for edge in self.network.edges() if self.x[edge].X == 1]
        self.subtours = list()

        while len(used_edges) > 0:
            tour = [used_edges[0]]
            used_edges.remove(used_edges[0])

            while True:
                last_node = tour[-1][1]
                next_edge = [edge for edge in used_edges if edge[0] == last_node][0]
                tour.append(next_edge)
                used_edges.remove(next_edge)

                if next_edge[1] == tour[0][0]:
                    self.subtours.append(tour)
                    break

        return

    def add_subtour_elimination(self):
        for tour in self.subtours:
            self.m.addConstr(gp.quicksum(self.x[edge] for edge in tour) <=
                             len(tour) - 2, name=f'subtour_elimination_{self.counter}')
            # -2 because len(tour) = |#nodes in tour| +1

            self.counter += 1

        return

    def calculate_design_cost(self):
        return sum([self.x[edge].X * (1 - self.alpha) * 2 * self.network.edges[edge]['cost'] for edge in self.x])

    def calculate_feasibility_cover_size(self):
        return sum([1 for req in self.z if self.z[req].X > 0.9])

    def calculate_node_cover_size(self):

        return

    def get_edges(self):
        return [edge for edge in self.x if self.x[edge].X > 0.9]

    def save_solution(self):

        try:
            feasibility_cover = frozenset([req for req in self.z if self.z[req].X >= 0.99])
            node_cover = mc.get_nodes_from_cover(feasibility_cover, self.compulsory_stops)
            route_edges = [edge for edge in self.x if self.x[edge].X > 0.9]
        except:
            print('No solution could be found')
            return

        instance_name = f'{self.seed}-{len(self.network.nodes)}-{len(self.demand)}-{self.theta}-{self.rho}_MIP'


        design_cost = self.calculate_design_cost()
        filename = Path.cwd() / 'results' / f'Design_{instance_name}.csv'  # linux
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['compulsory stops', 'node cover', 'feasibility cover', 'route edges', 'design cost'])
            writer.writerows([[list(self.compulsory_stops), list(node_cover),
                               list(feasibility_cover), list(route_edges), design_cost]])
        return



