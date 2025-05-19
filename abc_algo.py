from utils import is_acyclic, exceeds_parent_limit, calculate_fitness

from concurrent.futures import ThreadPoolExecutor
from pgmpy.models import BayesianNetwork
from pgmpy.base import DAG

import numpy as np

import random
import copy
import logging
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class CandidateSolution:
    """Represents a candidate Bayesian Network structure."""
    def __init__(self, edges, variables, score):
        self.edges = edges
        self.fitness = calculate_fitness(edges, variables, score)
        self.model = None
        if self.fitness != float('inf'):
            self.model = BayesianNetwork()
            self.model.add_nodes_from(variables)
            self.model.add_edges_from(edges)

    def get_fitness(self):
        return self.fitness

    def copy(self):
        return copy.deepcopy(self)

class ABC_BN:
    """Artificial Bee Colony optimizer for Bayesian Network structure learning."""
    def __init__(
        self,
        data,
        variables,
        score,
        pop_size=20,
        limit=50,
        num_iters=50,
        num_workers=4,
        seed=42
    ):
        self.data = data
        self.variables = variables
        self.score = score
        self.pop_size = pop_size
        self.limit = limit
        self.num_iters = num_iters
        self.num_workers = num_workers
        self.seed = seed
        self.edge_prob = 0.05
        self.max_stagnation = 50
        random.seed(seed)
        np.random.seed(seed)

        # Initialize population
        self.population = self.generate_initial_population()
        self.trial = [0] * pop_size

    def generate_initial_population(self):
        pop = []
        attempts = 0
        while len(pop) < self.pop_size:
            edges = self.random_dag()
            if is_acyclic(edges, self.variables) and not exceeds_parent_limit(edges, self.variables):
                candidate = CandidateSolution(edges, self.variables, self.score)
                if candidate.get_fitness() != float('inf'):
                    pop.append(candidate)
            attempts += 1
            if attempts % 10 == 0:
                logging.info(f"Population size: {len(pop)} / {self.pop_size} after {attempts} attempts")
        return pop

    def random_dag(self):
        dag = []
        nodes = self.variables.copy()
        random.shuffle(nodes)
        # initial random edges based on topological shuffle
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if random.random() < self.edge_prob:
                    dag.append((u, v))
        # random flips to diversify
        flips = max(1, len(dag) // 5)
        for _ in range(flips):
            if dag and random.random() < 0.5:
                dag.remove(random.choice(dag))
            else:
                u, v = random.sample(nodes, 2)
                if (u, v) not in dag and (v, u) not in dag and nodes.index(u) < nodes.index(v):
                    dag.append((u, v))
        return dag

    def get_all_fitness_values(self):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            fitnesses = list(executor.map(lambda sol: sol.get_fitness(), self.population))
        return fitnesses

    def obtain_new_solution(self, original, partner):
        new_edges = original.edges.copy()
        if not partner.edges:
            return False, None
        edge = random.choice(partner.edges)
        # flip edge
        if edge in new_edges:
            new_edges.remove(edge)
        else:
            new_edges.append(edge)
        # validate
        if not is_acyclic(new_edges, self.variables) or exceeds_parent_limit(new_edges, self.variables):
            return False, None
        candidate = CandidateSolution(new_edges, self.variables, self.score)
        if candidate.get_fitness() < original.get_fitness():
            return True, candidate
        return False, None

    def deploy_employed(self):
        for i in range(self.pop_size):
            idxs = list(range(self.pop_size))
            idxs.remove(i)
            partner = self.population[random.choice(idxs)]
            success, new_sol = self.obtain_new_solution(self.population[i], partner)
            if success:
                self.population[i] = new_sol
                self.trial[i] = 0
            else:
                self.trial[i] += 1

    def deploy_onlookers(self):
        fitness_vals = self.get_all_fitness_values()
        max_f, min_f = max(fitness_vals), min(fitness_vals)
        scale = max_f - min_f
        if scale == 0:
            probs = [1/self.pop_size]*self.pop_size
        else:
            probs = [1 - ((f - min_f)/scale) for f in fitness_vals]
            total = sum(probs)
            probs = [p/total for p in probs]
        i = 0; count = 0
        while count < self.pop_size:
            if random.random() < probs[i]:
                idxs = list(range(self.pop_size)); idxs.remove(i)
                partner = self.population[random.choice(idxs)]
                success, new_sol = self.obtain_new_solution(self.population[i], partner)
                if success:
                    self.population[i] = new_sol
                    self.trial[i] = 0
                else:
                    self.trial[i] += 1
                count += 1
            i = (i+1) % self.pop_size

    def deploy_scouts(self, global_best):
        for i in range(self.pop_size):
            if self.trial[i] >= self.limit:
                for _ in range(50):
                    new_edges = self.random_dag()
                    if is_acyclic(new_edges, self.variables) and not exceeds_parent_limit(new_edges, self.variables):
                        candidate = CandidateSolution(new_edges, self.variables, self.score)
                        if candidate.get_fitness() != float('inf'):
                            self.population[i] = candidate
                            self.trial[i] = 0
                            break

    def population_diversity(self):
        sets = [set(sol.edges) for sol in self.population]
        n = len(sets)
        if n < 2:
            return 0.0
        total = 0.0; count = 0
        for i in range(n):
            for j in range(i+1, n):
                ui = sets[i] | sets[j]
                inter = sets[i] & sets[j]
                jacc = 1 - len(inter)/len(ui) if ui else 0
                total += jacc; count += 1
        return total/count

    def run(self):
        """Executes the ABC algorithm and returns metrics, best model, and history."""
        fitness_evolution = []
        avg_fitness = []
        diversity = []
        best_edges_history = []
        start = time.time()

        # initial best
        global_best = min(self.population, key=lambda s: s.get_fitness()).copy()
        best_score = global_best.get_fitness()
        stagnation = 0

        for it in range(self.num_iters):
            self.deploy_employed()
            self.deploy_onlookers()
            self.deploy_scouts(global_best)

            current = min(self.population, key=lambda s: s.get_fitness())
            current_score = current.get_fitness()
            fitness_evolution.append(current_score)

            values = self.get_all_fitness_values()
            avg_fitness.append(np.mean(values))
            diversity.append(self.population_diversity())

            if current_score < best_score:
                best_score = current_score
                global_best = current.copy()
                stagnation = 0
            else:
                stagnation += 1

            # elitism
            worst_idx = max(range(self.pop_size), key=lambda i: self.population[i].get_fitness())
            if self.population[worst_idx].get_fitness() > best_score:
                self.population[worst_idx] = global_best.copy()
                self.trial[worst_idx] = 0

            # record history
            best_edges_history.append(list(global_best.edges))

            if stagnation >= self.max_stagnation:
                logging.info(f"Stopping early at iteration {it} due to stagnation.")
                break

        elapsed = time.time() - start
        logging.info(f"Best BIC Score: {-best_score}")
        logging.info(f"Edges in best model: {len(global_best.edges)}")
        return fitness_evolution, avg_fitness, diversity, global_best, best_edges_history
