import gurobipy as gp
from gurobipy import GRB


def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist(edge for edge in model._edges
                                if vals[edge] > 0.5)
        # find the shortest cycle in the selected edge list
        subtours = subtour(selected, model._nodes)
        if len(subtours) > 1:
            for tour in subtours:
                # add subtour elimination constr. for every pair of cities in subtour
                model.cbLazy(gp.quicksum(model._vars[edge] for edge in tour)
                             <= len(tour) - 1)
                model._subtours.append(tour)


# Given a tuplelist of edges, find the shortest subtour
def subtour(used_edges, nodes):
    subtours = list()

    while len(used_edges) > 0:
        tour_nodes = set([list(used_edges[0])[0], list(used_edges[0])[1]])
        tour_edges = [used_edges[0]]
        used_edges.remove(used_edges[0])

        while True:
            next_edges = [edge for edge in used_edges if list(edge)[0] in tour_nodes or list(edge)[1] in tour_nodes]
            for edge in next_edges:
                tour_nodes.update(edge)
                tour_edges.append(edge)
                used_edges.remove(edge)

            if len(next_edges) == 0:
                subtours.append(tour_edges)
                break

    return subtours


class S_TSP:
    def __init__(self, network, relevant_edges, relevant_nodes):
        self.network = network
        self.edges = relevant_edges
        self.nodes = relevant_nodes

        self.dist = None

        # Model
        self.m = None

        # Variables
        self.x = None

        # Constraints
        self.degree_constr = None
        self.subtour_constr = None
        self.flow_constr = None

    def calculate_dist_dict(self):
        self.dist = {edge: self.network.edges[edge]['cost'] for edge in self.edges}

    def build_model(self):
        self.m = gp.Model('S-TSP')

        # Variables
        self.x = {edge: self.m.addVar(vtype=GRB.BINARY, obj=2*self.dist[edge], lb=0, name=f'x_{edge}')
                  for edge in self.edges}

        # Constraints
        self.degree_constr = {
            n: self.m.addConstr(gp.quicksum(self.x[edge] for edge in self.edges if n in edge) == 2,
                           name=f'service_flow_{n}')
            for n in self.nodes}


    def solve_model(self):
        self.m._vars = self.x
        self.m._nodes = self.nodes
        self.m._edges = self.edges
        self.m._subtours = list()
        self.m.Params.LogToConsole = 0
        self.m.Params.lazyConstraints = 1
        self.m.optimize(subtourelim)

