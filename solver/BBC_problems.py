import gurobipy as gp
from gurobipy import GRB
import time


class MasterProblemBBC:
    def __init__(self, alpha):
        self.alpha = alpha

        # Model
        self.m = None

        # Decision variables
        self.x = None
        self.eta = None

        # Constraints
        self.degree_constr = None

        # Cuts
        self.optimality_cuts = list()
        self.feasibility_cuts = list()

    def build_problem_BBC1(self, nodes, edges, network):
        self.m = gp.Model("Master Benders")

        # Variables
        self.x = {edge: self.m.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f'x_{edge}')
                  for edge in edges}
        self.eta = self.m.addVar(vtype=GRB.CONTINUOUS, name='eta')

        # Constraints
        self.degree_constr = {n: self.m.addConstr(gp.quicksum(self.x[edge] for edge in edges if n in edge) == 2)
                              for n in nodes}

        # Add objective function
        obj = (1 - self.alpha) * 2 * gp.quicksum(network.edges[edge]['cost'] * self.x[edge] for edge in edges) + \
              self.alpha * self.eta
        self.m.setObjective(obj, GRB.MINIMIZE)

    def build_problem_BBC2(self, nodes, edges, network):
        self.m = gp.Model("Master Benders")

        # Variables
        self.x = {edge: self.m.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f'x_{edge}')
                  for edge in edges}
        self.eta = {n: self.m.addVar(vtype=GRB.CONTINUOUS, name=f'eta_{n}') for n in nodes}

        # Constraints
        self.degree_constr = {n: self.m.addConstr(gp.quicksum(self.x[edge] for edge in edges if n in edge) == 2)
                              for n in nodes}

        # Add objective function
        obj = (1 - self.alpha) * 2 * gp.quicksum(network.edges[edge]['cost'] * self.x[edge] for edge in edges) + \
              self.alpha * gp.quicksum(self.eta[n] for n in self.eta)
        self.m.setObjective(obj, GRB.MINIMIZE)

    def solve(self):
        self.m.Params.LogToConsole = 0
        self.m.optimize()


class DualSubproblemBBC1:

    def __init__(self, x, h, k, reachable_arcs):
        self.x = x
        self.h = h
        self.k = k
        self.reachable_arcs = reachable_arcs

        # Model
        self.m = None

        # Decision variables
        self.rho = None
        self.chi = None

        # Constraints
        self.dual_constraint = None

    def build_subproblem(self, arcs, edges, nodes, q_hk):
        self.m = gp.Model("Subproblem Benders")

        # Variables
        self.rho = {n: self.m.addVar(vtype=GRB.CONTINUOUS, name=f'rho_{n}') for n in nodes}
        self.chi = {arc: self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'chi_{arc}') for arc in arcs}

        # Constraints
        self.dual_constraint = {(i, j): self.m.addConstr(self.rho[i] - self.rho[j] - self.chi[(i, j)] <=
                                                         q_hk[(i, j)]) for (i, j) in arcs}

        # Add objective function
        obj = self.rho[self.h] - self.rho[self.k] - gp.quicksum((self.chi[(i, j)] + self.chi[(j, i)]) *
                                                                self.x.get(frozenset([i, j]), 0)
                                                                for i, j in edges)
        self.m.setObjective(obj, GRB.MAXIMIZE)

    def update_objective(self, edges, x):
        self.x = x
        # Add objective function
        obj = self.rho[self.h] - self.rho[self.k] - gp.quicksum((self.chi[(i, j)] + self.chi[(j, i)]) *
                                                                self.x.get(frozenset([i, j]), 0)
                                                                for i, j in edges)
        self.m.setObjective(obj, GRB.MAXIMIZE)

    def solve_subproblem(self):
        self.m.Params.LogToConsole = 0
        self.m.Params.NetworkAlg = 1
        self.m.Params.DualReductions = 0
        self.m.Params.InfUnbdInfo = 1
        t = time.time()
        self.m.optimize()
        # print(f'Solved dual subproblem {(self.h, self.k)} in {time.time() - t} seconds')



