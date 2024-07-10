import gurobipy as gp
from gurobipy import GRB


def termination_criterion(model, where):
    if where == GRB.Callback.SIMPLEX:
        model._bestprimal = model.cbGet(GRB.Callback.SPX_OBJVAL)
        model._bestdual = model.cbGet(GRB.Callback.SPX_DUALINF)
        model._time = model.cbGet(GRB.Callback.RUNTIME)

        if model._time > 1 and model._bestprimal < GRB.INFINITY and model._bestdual > 100:
            model.terminate()

def calculate_incidence_matrix(network):
    matrix = dict()
    for node in network.nodes:
        matrix[node] = {}

    for arc in network.arcs:
        matrix[arc.start][arc] = 1
        matrix[arc.end][arc] = -1

    return matrix


def convert_service_covers_to_dict(covers):
    s = range(len(covers))
    lst_covers = [cover for cover in covers]
    return {frozenset(lst_covers[k]): lst_covers[k] for k in s}


class ModelCoversRMP:
    def __init__(self, network, demand, service_covers, D, alpha, theta, q, compulsory_stops, reachable_arcs):
        self.network = network
        self.demand = demand
        self.service_covers = convert_service_covers_to_dict(service_covers)
        self.D = D
        self.alpha = alpha
        self.theta = theta
        self.q = q
        self.compulsory_stops = compulsory_stops
        self.reachable_arcs = reachable_arcs

        # Model
        self.m = None

        # Decision variables
        self.chi = None
        self.x = None
        self.f = None

        # Constraints
        self.cover_constr = None
        self.service_flow = None
        self.service_out = None
        self.passenger_flow_bound = None
        self.flow_passengers = None

        # Used for Branch Tree
        self.solved_cover_constr = dict()
        self.branch_constr = None

    def node_lookup(self, cover, node):
        """
        If node is compulsory stop or covered by any request in the cover return 1, otherwise return 0
        :param cover: dict of set of requests forming a minimum service cover
        :param node: node of the network
        :return: 0 or 1
        """
        if node in self.compulsory_stops:
            return 1

        for req in self.service_covers[cover]:
            if node in req:
                return 1

        return 0

    def req_lookup(self, cover, request, node):
        if request not in self.service_covers[cover]:
            return 0

        elif node == request[0]:
            return 1

        elif node == request[1]:
            return -1

        else:
            return 0

    def build_problem(self):
        m = gp.Model("Cover Formulation")
        m.setParam('Method', 1)

        # Determine undirected edges in network
        edges = set(frozenset(edge) for edge in self.network.edges())

        # Variables
        self.chi = {cover: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1)
                    for cover in self.service_covers}
        self.x = {edge: m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                 obj=(1 - self.alpha) * 2 * self.network.edges[edge]['cost'],
                                 name=f'x_{edge}')
                  for edge in edges}

        self.f = {(edge, req): m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1,
                                        obj=self.alpha * self.q[req][edge],
                                        name=f'f_{edge}_{req}')
                  for edge in self.network.edges() for req in self.D
                  if edge in self.reachable_arcs[req]} # only consider thereachable arcs

        # Constraints
        self.cover_constr = m.addConstr(gp.quicksum(self.chi[cover] for cover in self.service_covers) == 1)

        print('Add service flow constraints...')
        self.service_flow = {i: m.addConstr(gp.quicksum(self.x[edge] for edge in edges
                                                       if i in edge) -
                                           2*gp.quicksum(self.chi[cover] * self.node_lookup(cover, i)
                                                       for cover in self.service_covers) == 0, name=f'service_flow_{i}')
                            for i in self.network.nodes()}

        print('Add passenger flow constraints...')
        self.flow_passengers = {
            (req, i): m.addConstr(gp.quicksum(self.f[(edge, req)] for edge in self.network.out_edges(i)
                                              if edge in self.reachable_arcs[req]) -
                                  gp.quicksum(self.f[(edge, req)] for edge in self.network.in_edges(i)
                                              if edge in self.reachable_arcs[req]) -
                                  gp.quicksum(self.chi[cover] * self.req_lookup(cover, req, i)
                                              for cover in self.service_covers) == 0, name=f'flow_passenger_{i}_{req}')
            for req in self.D for i in self.network.nodes()}

        print('Add passenger flow bound constraints...')
        self.passenger_flow_bound = {(edge, req): m.addConstr(self.f[(edge, req)] <=
                                                              self.x[frozenset(edge)],
                                                              name=f'passenger_flow_bound_{(edge, req)}')
                                     for edge in self.network.edges()
                                     for req in self.D
                                     if edge in self.reachable_arcs[req]}

        m.setParam('MIPGap', 0.03)
        self.m = m
        return

    def add_new_cover(self, new_cover):
        # add new cover to the model
        cover = len(self.service_covers)
        self.service_covers[cover] = new_cover

        c = gp.Column()

        # Add new chi to cover constraint
        c.addTerms(1, self.cover_constr)

        # Add new chi to service out contraints
        for node in self.network.nodes():
            # if node not in self.compulsory_stops:
            c.addTerms(-2*self.node_lookup(cover, node), self.service_flow[node])

        # Add new chi to flow passenger constraints
        for req in self.D:
            for i in self.network.nodes():
                c.addTerms(-self.req_lookup(cover, req, i), self.flow_passengers[(req, i)])

        # Add new chi as variable to the model
        self.chi[new_cover] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'chi_{cover}', column=c)

        return

    def solve_lp(self):
        print('Solve master problem')
        self.m.update()
        self.m.Params.LogToConsole = 0
        self.m.setParam('TimeLimit', 40 * 60)
        self.m.optimize()
        return

    def get_iota(self):
        return {node: self.service_flow[node].Pi for node in self.network.nodes()}

    def get_epsilon(self):
        return {(req, node): self.flow_passengers[(req, node)].Pi for req in self.D for node in self.network.nodes()}

    def get_alpha(self):
        return self.cover_constr.Pi

    def get_dual_solved_cover_constrs(self):
        return {cover: self.solved_cover_constr[cover].Pi for cover in self.solved_cover_constr}
