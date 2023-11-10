import itertools
from queue import PriorityQueue
import solver.check_minimum_covers as mc
import solver.BBC_algorithm as bbc1

import solver.restricted_master_problem as rmp
import solver.pricing_problem as pricing

import time
import math
import csv
from pathlib import Path
import os

import random

import bisect

instance_name = None

def save_data(d_bounds, d_subproblems, d_cover=None):
    '''
    Save
    '''
    global instance_name

    if d_bounds is not None:
        try:
            # write saved data_bounds to csv file
            filename = Path.cwd() / 'results' / f'Bounds_BB_{instance_name}.csv'

            with open(filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(d_bounds)

        except FileNotFoundError:
            filename = Path.cwd() / 'results' / f'Bounds_BB_{instance_name}.csv'
            with open(filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(d_bounds)

    if d_subproblems:
        try:
            filename = Path.cwd() / 'results' / f'Subbproblems_BB_{instance_name}.csv'
            with open(filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(d_subproblems)

        except FileNotFoundError:
            filename = Path.cwd() / 'results' / f'Subbproblems_BB_{instance_name}.csv'
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(d_subproblems)

    if d_cover:
        try:
            filename = Path.cwd() / 'results' / f'Cover_{instance_name}.csv'
            with open(filename, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(d_cover)

        except FileNotFoundError:
            filename = Path.cwd() / 'results' / f'Cover_{instance_name}.csv'
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(d_cover)

    return


class Node:
    '''
    One node of the solution tree either for branch-and-price, heuristic, or hybrid
    '''
    graph = None
    demand = None
    D = None
    theta = None
    rho = None
    alpha = None
    sym_TSP = dict()
    minimum_demand_served = None
    q = None

    def __init__(self, parent, node_cover: frozenset, feasibility_cover: frozenset, reachable_arcs: dict):
        self.parent = parent
        self.node_cover = node_cover
        self.feasibility_cover = feasibility_cover
        self.reachable_arcs = reachable_arcs

        self.upper_bound = None
        self.lower_bound = None

        self.cost = math.inf

        self.TSP_edges = None

        self.route_edges = None

        self.solve_time = None

        self.design_cost = None

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.feasibility_cover == other.feasibility_cover

    def __str__(self):
        return f'size nodes: {len(self.node_cover)}, size req: {len(self.feasibility_cover)}, ub: {self.upper_bound}, lb: {self.lower_bound}'

    def set_class_attributes(self, graph, demand: dict, D: frozenset, theta: float, rho: float, alpha: float, q: dict):
        Node.graph = graph
        Node.demand = demand
        Node.D = D
        Node.theta = theta
        Node.rho = rho
        Node.alpha = alpha
        Node.minimum_demand_served = {scenario: int(sum(demand[scenario].values()) * theta) + 1 for scenario in demand}
        Node.q = q

    def get_bounds(self):
        # If the node cover already exists, use information about the TSP from it
        try:
            lower_bound_design = Node.sym_TSP[self.node_cover]['cost']
            self.TSP_edges = Node.sym_TSP[self.node_cover]['edges']

            self.upper_bound, paths, path_arcs = mc.get_upper_bound_routing(self.demand, self.feasibility_cover, self.graph, self.theta,
                                                          self.alpha, self.TSP_edges, lower_bound_design)

            self.lower_bound = mc.get_lower_bound_routing(self.demand, self.feasibility_cover, self.graph, self.theta,
                                                          self.alpha, lower_bound_design)

        # If it does not exist, compute min TSP and safe information for later use
        except KeyError:
            self.lower_bound, self.upper_bound, self.TSP_edges, lower_bound_design, subtours, paths, path_arcs = \
                mc.bounds_cover(self.demand, self.feasibility_cover,self.graph, self.theta, self.alpha, self.node_cover)

            Node.sym_TSP[self.node_cover] = dict()
            Node.sym_TSP[self.node_cover]['cost'] = lower_bound_design
            Node.sym_TSP[self.node_cover]['edges'] = self.TSP_edges
            Node.sym_TSP[self.node_cover]['subtours'] = subtours
            Node.sym_TSP[self.node_cover]['path arcs'] = path_arcs

    def solve(self, upper_bound):
        # This code can be used to solve the TSP-GL subproblems via BBC1 as introduced by Errico et al
        # print(f'Solve node with BBC1, {str(self)}...')
        # t = time.time()
        # alg = bbc1.run_BBC1(self.demand, self.feasibility_cover, self.graph, self.alpha, self.theta, self.TSP_edges, self.lower_bound)
        # t_BENDERS = time.time() - t
        # self.cost = alg.master.m.objVal
        # print(f'Solved in {round(t_BENDERS, 2)} seconds, objVal: {self.cost}')

        # Solve the TSP-GL subproblem via better BBC2 as introduced by Errico et al
        print(f'Solve node with BBC2, {str(self)}...')
        t = time.time()

        # get all already found subtours for the nodecover to add as feasibility covers to the TSP-GL
        subtours = Node.sym_TSP[self.node_cover]['subtours']
        alg = bbc1.run_BBC2(self.demand, self.feasibility_cover, self.graph, self.alpha, self.theta, self.TSP_edges, self.lower_bound, upper_bound, self.q, subtours, self.reachable_arcs)
        t_BENDERS = time.time() - t
        self.cost = alg.master.m.objVal
        self.upper_bound = alg.master.m.objVal

        # >0 needed for numerical errors in gurobi e.g. 0.9999998. All alg.x variables are or should be binary
        self.route_edges = [edge for edge in alg.x if alg.x[edge] > 0.9]
        self.design_cost = alg.master.m.objVal - self.alpha * sum(alg.master.eta[n].X for n in alg.master.eta)
        self.solve_time = t_BENDERS
        print(f'Solved in {round(t_BENDERS, 2)} seconds, objVal: {self.cost}')
        return


class AdvancedBranch:
    def __init__(self, seed, graph, demand: dict, theta: float, rho: float, alpha: float, D: frozenset, q: dict,
                 compulsory_stops: list, reachable_arcs: dict, goal_optimality_gap: float, computation_time: int):
        self.seed = seed
        self.graph = graph
        self.demand = demand
        self.theta = theta
        self.rho = rho
        self.alpha = alpha
        self.D = D
        self.compulsory_stops = compulsory_stops
        self.reachable_arcs = reachable_arcs

        self.found_node_covers = set()
        self.found_feasibility_covers = set()
        self.solved_node_covers = set()
        self.solved_feasibility_covers = set()
        self.node_cover_feas_cover_dict = dict()

        self.solved_nodes = list()

        self.upper_bound = math.inf
        self.lower_bounds = list()
        self.incumbents = list()

        self.queue = PriorityQueue()
        self.root = None

        self.min_nodeset_size = math.inf

        self.local_nodes = 0
        self.explore_nodes = 0

        self.incumbent_improvements = list()
        self.needed_improvement = None
        self.new_incumbents_with_too_small_improvement = 0
        self.goal_optimality_gap = goal_optimality_gap

        # For CG
        self.rmp = None
        self.pricing = None
        self.iota = None
        self.epsilon = None
        self.alpha_p = None
        self.q = q

        # Data
        self.data_bounds = list()
        self.data_subproblems = list()
        self.start_time = None

        # Algorithm control
        self.max_computation_time = computation_time

    def get_all_requests(self):
        D = set()
        for scenario, demand_dict in self.demand.items():
            for req, demand_value in demand_dict.items():
                D.add(req)

        return frozenset(D)

    def calculate_q(self):
        print('Calculate q dict...')
        # Determine q in subproblem
        temp_demand_sum = {n: sum(self.demand[n].values()) for n in self.demand}  # needed for runtime (~80% reduction)

        for req in self.D:
            self.q[req] = {arc: 0 for arc in self.graph.edges()}
            for arc in self.graph.edges():
                for n in self.demand:
                    try:
                        self.q[req][arc] += self.graph.edges[arc]['cost'] / len(self.demand) * \
                                            self.demand[n][req] / (self.theta * temp_demand_sum[n])
                    except KeyError:
                        continue
        return

    ##################################### Column generation start ######################################################
    def initilization_CG(self):
        t = time.time()
        self.setup_rmp()
        print(f'Setup RMP in {time.time() - t} seconds')
        self.rmp.solve_lp()
        self.get_duals_of_rmp()
        self.setup_pricing_problem()
        self.update_pricing_solved_covers()
        self.found_feasibility_covers.add(self.D)
        self.lower_bounds.append(0.000001)

    def setup_rmp(self):
        # Build restricted master problem of column generation with the set of all requests
        self.rmp = rmp.ModelCoversRMP(self.graph, self.demand, set([frozenset(self.D)]), self.D, self.alpha, self.theta,
                                      self.q, self.compulsory_stops, self.reachable_arcs)
        self.rmp.build_problem()

    def setup_pricing_problem(self):
        self.pricing = pricing.ModelCoversPricing(self.graph, self.demand, self.iota,
                                                  self.epsilon, self.alpha_p, self.theta, self.rho, self.D, self.alpha,
                                                  self.compulsory_stops)
        self.pricing.build_problem()

    def get_duals_of_rmp(self):
        self.iota = self.rmp.get_iota()
        self.epsilon = self.rmp.get_epsilon()
        self.alpha_p = self.rmp.get_alpha()

    def solve_rmp(self):
        self.rmp.solve_lp()

    def update_pricing_duals(self):
        self.get_duals_of_rmp()

        # Since we only build the pricing problem once, we have to update the objective function every time we resolve
        self.pricing.update_obj_function(self.iota, self.epsilon, self.alpha_p)

    def update_pricing_solved_covers(self):
        # Ensure that all found req covers are cut from the feasible region to avoid finding the same cover again
        # (and dual variables of branching constraints)
        new_solved_covers = [cover for cover in self.found_feasibility_covers if cover not in self.pricing.constr_solved_covers]

        self.pricing.add_solved_covers_constr(new_solved_covers)

    def solve_pricing(self):
        self.pricing.solve_pricing()

    ##################################### Column generation end ########################################################

    ##################################### Nodes of queue start #########################################################
    def solve_node(self):
        node = self.queue.get()[2]

        # Do not solve nodes whose lower bound is bigger than the current minimum upper bound
        if node.lower_bound < self.upper_bound:
            node.solve(self.upper_bound)

        # Check if node is new incumbent
        if len(self.solved_nodes) == 0:
            self.upper_bound = node.cost

        elif len(self.solved_nodes) == 0 or node.cost < self.upper_bound:
            self.upper_bound = node.cost
            self.incumbent_improvements.append((self.solved_nodes[0].cost / node.cost) - 1)

            # If the incumbent improvement is too small raise the count by one
            if self.incumbent_improvements[-1] < self.needed_improvement:
                self.new_incumbents_with_too_small_improvement += 1

            # If the incumbent improvement is big enough reset count to 0
            else:
                self.new_incumbents_with_too_small_improvement = 0

        bisect.insort(self.solved_nodes, node)

        self.solved_node_covers.add(node.node_cover)
        self.solved_feasibility_covers.add(node.feasibility_cover)
        self.add_feas_cover_to_node_cover_dict(node.node_cover, node.feasibility_cover)
        return node

    def add_feas_cover_to_node_cover_dict(self, node_cover, feasibility_cover):
        try:
            self.node_cover_feas_cover_dict[node_cover].append(feasibility_cover)

        except KeyError:
            self.node_cover_feas_cover_dict[node_cover] = list()
            self.node_cover_feas_cover_dict[node_cover].append(feasibility_cover)

    def try_to_add_node_to_queue(self, node: Node) -> bool:
        # Cut nodes with unsuitable bounds right away by not putting them into the queue
        if node.lower_bound > self.upper_bound:
            print(
                f'Lower bound of found node too high, size nodes: {len(node.node_cover)}, size req: {len(node.feasibility_cover)}, ub: {node.upper_bound}, lb: {node.lower_bound}')
            self.found_node_covers.add(frozenset(node.node_cover))
            self.found_feasibility_covers.add(frozenset(node.feasibility_cover))
            self.explore_nodes += 1
            return False

        else:
            # Add new node to the queue sorted from lowest upper bound to highest upper bound
            print(
                f'New node added to the queue, size nodes: {len(node.node_cover)}, size req: {len(node.feasibility_cover)}, ub: {node.upper_bound}, lb: {node.lower_bound}')
            self.queue.put((node.upper_bound, time.time(), node))
            self.found_node_covers.add(frozenset(node.node_cover))
            self.found_feasibility_covers.add(frozenset(node.feasibility_cover))
            self.explore_nodes += 1
            if node.upper_bound < self.upper_bound:
                self.upper_bound = node.upper_bound
                self.incumbents.append(node)
                save_data([[time.time() - self.start_time, node.upper_bound, max(self.lower_bounds, default=0)]], None)
            print(f'Time: {time.time() - self.start_time}')
            return True

    ##################################### Nodes of queue end ###########################################################

    ##################################### Explore and Local search start ###############################################
    def initialization(self):
        '''
        Add set of all requests as a feasibility cover to our queue
        '''

        # calculate node cover of set D
        node_cover = set()
        for (h, k) in self.D:
            node_cover.add(h)
            node_cover.add(k)

        root = Node(None, frozenset(node_cover), self.D, self.reachable_arcs)
        root.set_class_attributes(self.graph, self.demand, self.D, self.theta, self.rho, self.alpha, self.q)
        root.get_bounds()
        self.root = root

        self.found_node_covers.add(frozenset(node_cover))
        self.found_feasibility_covers.add(frozenset(root.feasibility_cover))

        # Add root to priority queue
        self.queue.put((root.upper_bound, time.time(), root))

        self.upper_bound = root.upper_bound
        self.incumbents.append(root)

    def branch_new_node_cover_without_cover(self, size_nodecover: int, max_tries: int):
        # Find a nodecover which so far has not been added to the queue

        # Create subset of nodes in random order
        random_nodes = list(self.graph.nodes())
        random.shuffle(random_nodes)
        random_generator = itertools.combinations(random_nodes, size_nodecover)

        n = 0
        while True:
            n += 1

            if n > max_tries:
                print(f'No new node cover could be found for size {size_nodecover} in {max_tries} tries')
                break

            # Next nodecover from the random generator
            nodecover = next(random_generator, None)

            if nodecover is None:
                break

            # Do not check a nodecover which has already been found
            if nodecover in self.found_node_covers:
                print(f'Found nodecover of size {len(nodecover)} already exists')
                continue

            is_feasible, req_cover = mc.is_node_cover(self.demand, nodecover, self.D, self.theta, self.rho, Node.minimum_demand_served, self.compulsory_stops)

            # Make sure set of nodes is a feasible node cover
            if not is_feasible:
                continue

            # Find a req cover based on this node_cover
            new_cover = mc.get_minimum_cover_from_cover(self.demand, list(req_cover), self.theta, self.rho, Node.minimum_demand_served, self.compulsory_stops)

            # Find nodes of this new cover (it could be that all request adjecent to a node have been eliminated)
            nodes_of_new_cover = mc.get_nodes_from_cover(new_cover)

            # If this node cover has not been found so far, add node to the queue
            if nodes_of_new_cover not in self.found_node_covers:
                new_node = Node(self.root, frozenset(nodes_of_new_cover), new_cover, self.reachable_arcs)

                # Find bounds of the new node
                new_node.get_bounds()

                if self.try_to_add_node_to_queue(new_node):
                    break

    def find_new_request_cover_with_node(self, node: Node, max_tries: int, option=None):
        n = 0
        while True:
            n += 1
            new_cover = mc.get_new_minimum_cover(self.demand, self.D, node.feasibility_cover, self.theta, self.rho, option,
                                                 seed=len(self.found_feasibility_covers) + len(self.solved_nodes) + n,
                                                 node_cover=node.node_cover, minimum_demand_served=node.minimum_demand_served,
                                                 compulsory_stops=self.compulsory_stops)

            if new_cover not in self.found_feasibility_covers:
                # Find nodes of this new cover
                nodes_of_new_cover = mc.get_nodes_from_cover(new_cover)

                new_node = Node(node, frozenset(nodes_of_new_cover), new_cover, self.reachable_arcs)

                # Find bounds of the new node
                new_node.get_bounds()

                if self.try_to_add_node_to_queue(new_node):
                    break

            if n > max_tries:
                print(f'No new req cover could be found for node cover size {len(node.node_cover)} in {max_tries} tries')
                break

    def explore(self, new_node_covers_min=5, new_smaller_node_covers=1):
        min_nodecover = min([len(nodecover) for nodecover in self.found_node_covers])

        # Try to find new node covers with the size of the smalles node cover found so far
        print(f'Explore new node cover, size: {min_nodecover}, new node covers to find: {new_node_covers_min}')
        for i in range(new_node_covers_min):
            self.branch_new_node_cover_without_cover(min_nodecover, new_node_covers_min)

        # Try to find even smaller node covers than the smallest node cover found so far
        print(f'Explore new node cover, size: {min_nodecover - 1}, new node covers to find: {new_smaller_node_covers}')
        for i in range(new_smaller_node_covers):
            self.branch_new_node_cover_without_cover(min_nodecover-1, new_smaller_node_covers)

    def local_search(self, new_feasibility_covers=2, best_nodes=2, option=None):
        '''
        Try to find n new covers for the best "best_nodes" many nodes which we already solved
        :param new_feasibility_covers: number of new cover tries per node
        :param best_nodes: number of best nodes on which the new covers are based on
        :param option: how to find the new covers
        '''

        print(f'Local search, new covers to find: {new_feasibility_covers}')
        for node in self.solved_nodes[:max(len(self.solved_nodes), best_nodes)]:
            for i in range(new_feasibility_covers):
                self.find_new_request_cover_with_node(node, new_feasibility_covers, option)

    @staticmethod
    def set_explore_parameters(explore_option='light'):
        if explore_option == 'light':
            new_node_covers_min = 5
            new_smaller_node_covers = 1

        elif explore_option == 'moderate':
            new_node_covers_min = 10
            new_smaller_node_covers = 5

        elif explore_option == 'high':
            new_node_covers_min = 20
            new_smaller_node_covers = 10

        return new_node_covers_min, new_smaller_node_covers

    @staticmethod
    def set_local_search_parameters(local_search_option):
        if local_search_option == 'light':
            new_feasibility_covers = 2
            best_nodes = 1

        elif local_search_option == 'moderate':
            new_feasibility_covers = 5
            best_nodes = 2

        elif local_search_option == 'high':
            new_feasibility_covers = 10
            best_nodes = 3

        return new_feasibility_covers, best_nodes

    ##################################### Explore and Local search end #################################################

    ##################################### Solve only explore and local search start ####################################
    def solve(self, explore_steps=10, explore_option='light',
              local_search_steps=5, local_search_option='light',
              max_small_improvements=10, min_improvements_incumbents=0.0015):

        global instance_name

        self.initialization()

        self.needed_improvement = min_improvements_incumbents

        self.start_time = time.time()
        terminate = False
        instance_name = f'{self.seed}-{len(self.graph.nodes)}-{len(self.demand)}-{self.theta}-{self.rho}_heuristic_{explore_option}_{local_search_option}'

        # Set the explore options
        new_node_covers_min, new_smaller_node_covers = self.set_explore_parameters(explore_option)

        # Set the local search options
        new_feasibility_covers, best_nodes = self.set_local_search_parameters(local_search_option)

        # Populate the queue with node covers of different sizes
        for i in range(2, len(self.graph.nodes())):
            self.branch_new_node_cover_without_cover(i, 10)
            print(f'Time: {time.time() - self.start_time}')

        # Explore more extensively around the smallest found node cover
        self.explore(new_node_covers_min=new_node_covers_min, new_smaller_node_covers=new_smaller_node_covers)

        # Perform a local search
        self.local_search(new_feasibility_covers=new_feasibility_covers, best_nodes=best_nodes)

        # Solve nodes in the queue and perform exploration and local search steps
        while not self.queue.empty() and time.time() - self.start_time < self.max_computation_time:

            t = time.time()
            node = self.solve_node()
            self.data_subproblems.append([time.time() - t])

            if node == self.solved_nodes[0]:
                self.data_bounds.append(
                    [time.time() - self.start_time, node.cost, -math.inf])

            # Repopulate the queue with new node covers in repopulate step
            if len(self.solved_feasibility_covers) % explore_steps == 0:
                self.explore(new_node_covers_min=new_node_covers_min, new_smaller_node_covers=new_smaller_node_covers)

            # Perform local search on the best found feasibility covers
            if len(self.solved_feasibility_covers) % local_search_steps == 0:
                self.local_search(new_feasibility_covers=new_feasibility_covers, best_nodes=best_nodes)

            # Explore and local search if the queue is empty
            if self.queue.empty():
                n = 0
                while self.queue.empty():
                    n += 1
                    self.explore(new_node_covers_min=new_node_covers_min, new_smaller_node_covers=new_smaller_node_covers)
                    self.local_search(new_feasibility_covers=new_feasibility_covers, best_nodes=best_nodes)

                    if n == 1 and self.queue.empty():
                        terminate = True
                        break

            # If no new covers can be found terminate the algorithm
            if terminate:
                break

            #
            if self.new_incumbents_with_too_small_improvement >= max_small_improvements:
                print(f'Termination because of insuficient improvements in solutions {self.incumbent_improvements[-max_small_improvements:]}')
                break

            d_cover = [[len(self.found_feasibility_covers), len(self.found_node_covers),
                       sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                       sum([1 for node in self.solved_nodes if node.solve_time]),
                       sum([1 for node in self.solved_nodes if not node.solve_time])]]

            save_data(self.data_bounds, self.data_subproblems, d_cover)
            self.data_bounds = list()
            self.data_subproblems = list()

        filename = Path.cwd() / 'results' / f'Design_{instance_name}.csv'  # linux
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['compulsory stops', 'node cover', 'feasibility cover', 'route edges'])
            writer.writerows([[list(self.compulsory_stops), list(self.solved_nodes[0].node_cover),
                               list(self.solved_nodes[0].feasibility_cover), list(self.solved_nodes[0].route_edges)]])
    ##################################### Solve only explore and loval search end ######################################

    ##################################### Solve only column generation start ###########################################
    def solve_nodes_from_queue(self, num_nodes=5):
        for i in range(num_nodes):
            self.optimality_gap = self.upper_bound / max(self.lower_bounds) - 1
            if self.optimality_gap <= self.goal_optimality_gap and len(self.solved_nodes) > 0:
                return

            if self.queue.empty():
                return

            t = time.time()
            node = self.solve_node()
            self.data_subproblems.append([time.time() - t])

            if node == self.solved_nodes[0]:
                self.data_bounds.append([time.time() - self.start_time, node.cost, max(self.lower_bounds)])

    def solve_CG(self, covers_per_iter=5):
        global instance_name
        self.start_time = time.time()
        # Initialize Node class
        self.initialization()

        self.data_bounds.append(
            [time.time() - self.start_time, self.upper_bound, -math.inf])

        # initialize RMP and Pricing problem
        self.initilization_CG()

        self.needed_improvement = 0.0015

        added_covers = [self.D]

        instance_name = f'{self.seed}-{len(self.graph.nodes)}-{len(self.demand)}-{self.theta}-{self.rho}_new_CG'

        num_CG = 0
        while time.time() - self.start_time < self.max_computation_time:
            num_CG += 1
            data_CG = []
            print('Next iteration')
            for i in range(covers_per_iter):

                # Solve rmp first such that after branching constraints we get new updated duals
                t = time.time()
                self.rmp.solve_lp()
                t_rmp = time.time() - t

                # Update the pricing problem cover constraints and dual variables
                self.update_pricing_solved_covers()
                self.update_pricing_duals()

                # Solve the pricing problem
                t = time.time()
                self.solve_pricing()
                t_pricing = time.time() - t

                # Terminate for loop if no new cover can be found in pricing problem through branch constraints
                if self.pricing.m.status == 3 or self.pricing.m.status == 6:
                    break
                # Lower bound can be higher than upper bound if we forbid too many feasibility covers
                try :
                    self.lower_bounds.append(min(self.rmp.m.ObjBound + self.pricing.m.objVal, self.upper_bound))
                except AttributeError:
                    print('Here')

                if self.rmp.m.ObjBound + self.pricing.m.objVal == max(self.lower_bounds):
                    self.data_bounds.append([time.time() - self.start_time, self.upper_bound, self.rmp.m.ObjBound + self.pricing.m.objVal])
                    d_cover = [[len(self.found_feasibility_covers), len(self.found_node_covers),
                                sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                                sum([1 for node in self.solved_nodes if node.solve_time]),
                                sum([1 for node in self.solved_nodes if not node.solve_time])]]

                    save_data(self.data_bounds, self.data_subproblems, d_cover)
                    self.data_bounds = list()
                    self.data_subproblems = list()

                new_cover = self.pricing.get_cover()
                new_node_cover = self.pricing.get_node_cover()
                if not mc.is_feasible_cover(self.demand, [a for a in new_cover], self.theta, self.rho, [], None, self.compulsory_stops, new_node_cover):
                    print([a for a in new_cover])
                    print('Compulsory Stops')
                    print(self.compulsory_stops)
                    raise AssertionError

                # Terminate the for loop if no new cover can be found
                if self.pricing.m.objVal > -1e-4:
                    print('Pricing found no new cover')
                    self.lower_bounds.append(min(self.rmp.m.ObjBound + self.pricing.m.objVal, self.upper_bound))

                    # special case in which all reqeusts have to be fulfilled
                    if len(self.solved_nodes) > 0 and self.solved_nodes[0].feasibility_cover == self.D:
                        self.lower_bounds.append(self.solved_nodes[0].cost)
                    break

                # Make sure no cover is found multiple times
                assert new_cover not in self.rmp.chi

                self.found_feasibility_covers.add(new_cover)
                self.rmp.add_new_cover(new_cover)

                data_CG.append([t_pricing, t_rmp, 0, 0, 0])

            # Solve rmp once more after last pricing iteration
            self.rmp.solve_lp()

            # Add all covers in the basis of the RMP to the queue (if they have not been added so far)
            relevant_covers = [cover for cover in self.rmp.chi if self.rmp.chi[cover].X > 0]

            for cover in relevant_covers:

                if cover not in added_covers:
                    new_node = Node(None, mc.get_nodes_from_cover(cover, self.compulsory_stops), cover, self.reachable_arcs)
                    new_node.get_bounds()
                    self.try_to_add_node_to_queue(new_node)
                    added_covers.append(cover)

                    # Make sure that every cover is at most once in the basis of the RMP
                    self.rmp.solved_cover_constr[cover] = self.rmp.m.addConstr(self.rmp.chi[cover] == 0)

            # If no new promising feasibility covers can be found and all candidates have been solved, the optimal solution has been found
            if relevant_covers == [self.D]:
                if len(self.solved_nodes) == 0:
                    self.lower_bounds.append(self.rmp.m.objVal)
                    self.solve_nodes_from_queue()


                self.lower_bounds.append(self.solved_nodes[0].cost)
                self.optimality_gap = self.upper_bound / max(self.lower_bounds) - 1
                self.data_bounds.append(
                    [time.time() - self.start_time, self.upper_bound, max(self.lower_bounds)])

                d_cover = [[len(self.found_feasibility_covers), len(self.found_node_covers),
                            sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                            sum([1 for node in self.solved_nodes if node.solve_time]),
                            sum([1 for node in self.solved_nodes if not node.solve_time])]]

                save_data(self.data_bounds, self.data_subproblems, d_cover)
                break

            if time.time() - self.start_time < self.max_computation_time:
                self.solve_nodes_from_queue()

            # Terminate the algorithm if the desired optimality gap is reached
            self.optimality_gap = self.upper_bound / max(self.lower_bounds) - 1
            if self.optimality_gap <= self.goal_optimality_gap:
                print(self.lower_bounds)

                # Save data on bounds, subproblems and cover, not needed for algorithm
                self.data_bounds.append(
                    [time.time() - self.start_time, self.upper_bound, max(self.lower_bounds)])

                d_cover = [[len(self.found_feasibility_covers), len(self.found_node_covers),
                            sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                            sum([1 for node in self.solved_nodes if node.solve_time]),
                            sum([1 for node in self.solved_nodes if not node.solve_time])]]

                save_data(self.data_bounds, self.data_subproblems, d_cover)
                break

            else:
                print(f'Upper bound: {self.upper_bound}, Lower bound: {max(self.lower_bounds)}, Optimality Gap: {self.optimality_gap}')

            ################################# Save data, not needed for algorithm only to save solution#################
            d_cover = [[len(self.found_feasibility_covers), len(self.found_node_covers),
                        sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                        sum([1 for node in self.solved_nodes if node.solve_time]),
                        sum([1 for node in self.solved_nodes if not node.solve_time])]]

            save_data(self.data_bounds, self.data_subproblems, d_cover)
            self.data_bounds = list()
            self.data_subproblems = list()
            data_CG.append([0, 0, sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                            sum([1 for node in self.solved_nodes if node.solve_time]),
                            sum([1 for node in self.solved_nodes if not node.solve_time])])

            filename = Path.cwd() / 'results' / f'CG_time_{self.seed}-{len(self.graph.nodes)}-{len(self.demand)}-{self.theta}-{self.rho}_new_CG.csv'  # linux
            try:
                with open(filename, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(data_CG)

            except FileNotFoundError:
                with open(filename, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(data_CG)

        filename = Path.cwd() / 'results' / f'Problems_BB_{instance_name}.csv'  # linux
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([[len(self.solved_nodes), num_CG, len(self.found_feasibility_covers)]])

        filename = Path.cwd() / 'results' / f'Design_{instance_name}.csv'  # linux
        if len(self.solved_nodes) > 0:
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['compulsory stops', 'node cover', 'feasibility cover', 'route edges', 'design cost'])
                writer.writerows([[list(self.compulsory_stops), list(self.solved_nodes[0].node_cover), list(self.solved_nodes[0].feasibility_cover), list(self.solved_nodes[0].route_edges), self.solved_nodes[0].design_cost]])
    ##################################### Solve only column generation end #############################################

    ##################################### Solve hybrid with local search start #########################################
    def explore_in_CG(self):
        found_feasibility_covers_copy = list(self.found_feasibility_covers)

        self.explore(new_node_covers_min=5, new_smaller_node_covers=5)
        new_covers = self.found_feasibility_covers - set(found_feasibility_covers_copy)
        for cover in new_covers:
            self.rmp.add_new_cover(cover)

        return new_covers

    def local_search_in_CG(self, node, new_feasibility_covers=10, option=None):
        found_feasibility_covers_copy = list(self.found_feasibility_covers)

        print(f'Local search, new covers to find: {new_feasibility_covers}')
        for i in range(new_feasibility_covers):
            self.find_new_request_cover_with_node(node, new_feasibility_covers, option)

        new_covers = self.found_feasibility_covers - set(found_feasibility_covers_copy)
        for cover in new_covers:
            self.rmp.add_new_cover(cover)

        return new_covers

    def solve_CG_with_local_search(self, covers_per_iter=5):
        global instance_name
        self.start_time = time.time()
        # Initialize Node class
        self.initialization()

        # initialize RMP and Pricing problem
        self.initilization_CG()

        self.needed_improvement = 0.0015

        added_covers = [self.D]

        instance_name = f'{self.seed}-{len(self.graph.nodes)}-{len(self.demand)}-{self.theta}-{self.rho}_hybrid'

        # Find random feasibility covers through explore before solving the first rmp
        self.explore_in_CG()

        num_CG = 0
        while time.time() - self.start_time < self.max_computation_time:
            num_CG += 1
            data_CG = []
            print('Next iteration')
            for i in range(covers_per_iter):

                t = time.time()
                self.rmp.solve_lp()
                t_rmp = time.time() - t

                self.update_pricing_solved_covers()
                self.update_pricing_duals()

                t = time.time()
                self.solve_pricing()
                t_pricing = time.time() - t

                # Terminate for loop if no new cover can be found in pricing problem through branch constraints
                if self.pricing.m.status == 3 or self.pricing.m.status == 6:
                    break
                # Lower bound can be higher than upper bound if we forbid too many feasibility covers
                try:
                    self.lower_bounds.append(min(self.rmp.m.ObjBound + self.pricing.m.objVal, self.upper_bound))
                except AttributeError:
                    print('Here')

                if self.rmp.m.ObjBound + self.pricing.m.objVal == max(self.lower_bounds):
                    self.data_bounds.append(
                        [time.time() - self.start_time, self.upper_bound, self.rmp.m.ObjBound + self.pricing.m.objVal])
                    d_cover = [[len(self.found_feasibility_covers), len(self.found_node_covers),
                                sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                                sum([1 for node in self.solved_nodes if node.solve_time]),
                                sum([1 for node in self.solved_nodes if not node.solve_time])]]

                    save_data(self.data_bounds, self.data_subproblems, d_cover)
                    self.data_bounds = list()
                    self.data_subproblems = list()

                # Terminate the for loop if no new cover can be found
                if self.pricing.m.objVal > -1e-4:
                    print('Pricing found no new cover')
                    self.lower_bounds.append(min(self.rmp.m.ObjBound, self.upper_bound))

                    # special case in which all reqeusts have to be fulfilled
                    if len(self.solved_nodes) > 0 and self.solved_nodes[0].feasibility_cover == self.D:
                        self.lower_bounds.append(self.solved_nodes[0].cost)
                    break

                new_cover = self.pricing.get_cover()
                nodecover = mc.get_nodes_from_cover(new_cover)
                self.found_feasibility_covers.add(new_cover)
                self.found_node_covers.add(nodecover)

                self.rmp.add_new_cover(new_cover)

                new_node = Node(None, nodecover, new_cover, self.reachable_arcs)
                new_node.get_bounds()
                if new_node.lower_bound < self.upper_bound:
                    new_found_covers_local = self.local_search_in_CG(new_node)
                    added_covers.extend(new_found_covers_local)

                new_found_covers_explore = self.explore_in_CG()
                added_covers.extend(new_found_covers_explore)

                data_CG.append([t_pricing, t_rmp, 0, 0, 0])

                # Lower bound can be higher than upper bound if we forbid too many feasibility covers
                self.lower_bounds.append(min(self.rmp.m.ObjBound + self.pricing.m.ObjBound, self.upper_bound))

                if self.rmp.m.ObjBound + self.pricing.m.ObjBound == max(self.lower_bounds):
                    self.data_bounds.append([time.time() - self.start_time, self.upper_bound, self.rmp.m.objVal + self.pricing.m.objVal])
                    d_cover = [[len(self.found_feasibility_covers), len(self.found_node_covers),
                                sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                                sum([1 for node in self.solved_nodes if node.solve_time]),
                                sum([1 for node in self.solved_nodes if not node.solve_time])]]

                    save_data(self.data_bounds, self.data_subproblems, d_cover)
                    self.data_bounds = list()
                    self.data_subproblems = list()

            if time.time() - self.start_time < self.max_computation_time:
                self.rmp.solve_lp()

                # Add all covers in the basis of the RMP to the queue (if they have not been added so far)
                relevant_covers = [cover for cover in self.rmp.chi if self.rmp.chi[cover].X > 0]

                for cover in relevant_covers:

                    if cover not in added_covers:
                        new_node = Node(None, mc.get_nodes_from_cover(cover), cover, self.reachable_arcs)
                        new_node.get_bounds()
                        self.try_to_add_node_to_queue(new_node)
                        added_covers.append(cover)

                        # Make sure the every cover is at most once in the basis of the RMP
                        self.rmp.solved_cover_constr[cover] = self.rmp.m.addConstr(self.rmp.chi[cover] == 0)

                # If no new promising feasibility covers can be found and all candidates have been solved, the optimal solution has been found
                if self.queue.empty():
                    self.lower_bounds.append(self.solved_nodes[0].cost)
                    self.optimality_gap = self.upper_bound / max(self.lower_bounds) - 1
                    self.data_bounds.append(
                        [time.time() - self.start_time, self.upper_bound, max(self.lower_bounds)])

                    d_cover = [[len(self.found_feasibility_covers), len(self.found_node_covers),
                                sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                                sum([1 for node in self.solved_nodes if node.solve_time]),
                                sum([1 for node in self.solved_nodes if not node.solve_time])]]

                    save_data(self.data_bounds, self.data_subproblems, d_cover)
                    break

                if time.time() - self.start_time < self.max_computation_time:
                    self.solve_nodes_from_queue()

            # Terminate the algorithm if the desired optimality gap is reached
            self.optimality_gap = self.upper_bound / max(self.lower_bounds) - 1
            if self.optimality_gap <= self.goal_optimality_gap:
                break

            else:
                print(f'Upper bound: {self.upper_bound}, Lower bound: {max(self.lower_bounds)}, Optimality Gap: {self.optimality_gap}')

            ################################# Save data ########################################
            d_cover = [[len(self.found_feasibility_covers), len(self.found_node_covers),
                        sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                        sum([1 for node in self.solved_nodes if node.solve_time]),
                        sum([1 for node in self.solved_nodes if not node.solve_time])]]

            save_data(self.data_bounds, self.data_subproblems, d_cover)
            self.data_bounds = list()
            self.data_subproblems = list()
            data_CG.append([0, 0, sum([node.solve_time for node in self.solved_nodes if node.solve_time]),
                            sum([1 for node in self.solved_nodes if node.solve_time]),
                            sum([1 for node in self.solved_nodes if not node.solve_time])])

            filename = Path.cwd() / 'results' / f'CG_time_{instance_name}.csv'  # linux
            try:
                with open(filename, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(data_CG)

            except FileNotFoundError:
                with open(filename, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerows(data_CG)

        filename = Path.cwd() / 'results' / f'Problems_BB_{instance_name}.csv'  # linux
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([[len(self.solved_nodes), num_CG, len(self.found_feasibility_covers)]])

        filename = Path.cwd() / 'results' / f'Design_{instance_name}.csv'  # linux
        if len(self.solved_nodes) > 0:
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['compulsory stops', 'node cover', 'feasibility cover', 'route edges'])
                writer.writerows([[list(self.compulsory_stops), list(self.solved_nodes[0].node_cover),
                                   list(self.solved_nodes[0].feasibility_cover),
                                   list(self.solved_nodes[0].route_edges)]])
        return
