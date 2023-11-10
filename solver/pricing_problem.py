import gurobipy as gp
from gurobipy import GRB
import solver.check_minimum_covers as mc


def non_minimum_cover_elim(model, where):
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._vars)
        demand = model._demand
        theta = model._theta
        rho = model._rho
        compulsory_stops = model._compulsory_stops

        cover = [req for req in vals if vals[req] >= 0.99]

        is_minimum, _, _ = mc.is_minimum_cover(demand, cover, theta, rho, {}, None, compulsory_stops)

        if not is_minimum:
            model.cbLazy(gp.quicksum(model._vars[req] for req in cover) <= len(cover) - 1)


class ModelCoversPricing:
    def __init__(self, network, demand, iota, epsilon, alpha_p, theta, rho, D, alpha, compulsory_stops):
        self.network = network
        self.demand = demand
        self.iota = iota
        self.epsilon = epsilon
        self.alpha_p = alpha_p
        self.theta = theta
        self.rho = rho
        self.D = D
        self.alpha = alpha
        self.compulsory_stops = compulsory_stops

        # Model
        self.m = None

        # Decision variables
        self.l = None
        self.q = None
        self.gamma = None

        # Constraints
        self.compulsory_stops_constr = None
        self.constr_node_origin = None
        self.constr_node_dest = None
        self.constr_service_level = None
        self.constr_feasibility_level = None
        self.constr_solved_covers = {}
        self.constr_node_covers = {}

    def demand_lookup(self, scenario, request):
        try:
            return self.demand[scenario][request]

        except KeyError:
            return 0

    def build_problem(self):
        m = gp.Model("Pricing Problem Cover")

        # Variables
        self.l = {node: m.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f'l_{node}')
                  for node in self.network.nodes()}
        self.q = {req: m.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f'q_{req}')
                  for req in self.D}
        self.gamma = {scenario: m.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f'gamma_{scenario}')
                      for scenario in self.demand}

        # Constraints
        self.compulsory_stops_constr = {node: m.addConstr(self.l[node] == 1, name=f'compulsory_stops_{node}') for node in self.compulsory_stops}

        self.constr_node_origin = {(origin, destination): m.addConstr(self.l[origin] >= self.q[(origin, destination)],
                                                                      name=f'node_origin_{origin}')
                                   for (origin, destination) in self.D}

        self.constr_node_dest = {(origin, destination):
                                     m.addConstr(self.l[destination] >= self.q[(origin, destination)],
                                                 name=f'node_origin_{origin}')
                                 for (origin, destination) in self.D}

        self.constr_service_level = {scenario:
                                         m.addConstr(gp.quicksum(self.q[req] * self.demand_lookup(scenario, req)
                                                                 for req in self.D) >=
                                                     self.gamma[scenario] * self.theta * gp.quicksum(
                                             self.demand_lookup(scenario, req) for req in self.D), name=f'Service_level_{scenario}')
                                     for scenario in self.demand}

        self.constr_feasibility_level = m.addConstr(gp.quicksum(self.gamma[scenario] for scenario in self.demand) >=
                                                    (1 - self.rho) * len(self.demand), name=f'feasibility_level')

        # Add objective function
        # Need to add "if node not in self.compulsory_stops" if we want to throw out compulsory stops
        obj = 2*gp.quicksum(self.l[node] * self.iota[node] for node in self.network.nodes()) + \
              gp.quicksum(self.q[req] * (self.epsilon[(req, req[0])] - self.epsilon[(req, req[1])]) for req in self.D) \
              - self.alpha_p

        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('Cutoff', 0)
        m.setParam('MIPGap', 0.02)
        m.setParam('TimeLimit', 20 * 60)

        m.Params.LazyConstraints = 1
        m._vars = self.q
        m._demand = self.demand
        m._theta = self.theta
        m._rho = self.rho
        m._compulsory_stops = self.compulsory_stops
        self.m = m
        return

    def solve_pricing(self):
        print('Solve pricing problem')
        self.m.update()
        self.m.Params.LogToConsole = 0
        self.m.optimize(non_minimum_cover_elim)
        return

    def update_obj_function(self, iota, epsilon, alpha_p):
        self.iota = iota
        self.epsilon = epsilon
        self.alpha_p = alpha_p

        # Need to add "if node not in self.compulsory_stops" if we want to throw out compulsory stops
        obj = 2 * gp.quicksum(self.l[node] * self.iota[node] for node in self.network.nodes()) + \
              gp.quicksum(self.q[req] * (self.epsilon[(req, req[0])] - self.epsilon[(req, req[1])]) for req in self.D) \
              - self.alpha_p

        self.m.setObjective(obj, GRB.MINIMIZE)

    def add_solved_covers_constr(self, solved_covers):
        self.constr_solved_covers = {cover: self.m.addConstr(gp.quicksum(self.q[req] for req in cover) <= len(cover) -1)
                                     for cover in solved_covers}

    def remove_solved_covers_constr(self):
        for cover in self.constr_solved_covers:
            self.m.remove(self.constr_solved_covers[cover])

        self.constr_solved_covers = {}

    def add_node_cover_constraint(self, node_cover):
        self.constr_node_covers[node_cover] = self.m.addConstr(gp.quicksum(self.l[node] for node in node_cover) <=
                                                               len(node_cover) - 1)

    def remove_node_cover_constraint(self, node_cover):
        self.m.remove(self.constr_node_covers[node_cover])

    def get_cover(self):
        # Account for rounding errors
        return frozenset([req for req in self.D if self.q[req].X >= 0.99])

    def get_node_cover(self):
        return [node for node in self.l if self.l[node].X >= 0.99]



