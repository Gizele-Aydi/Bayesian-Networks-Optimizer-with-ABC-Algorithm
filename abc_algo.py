from utils import is_acyclic, exceeds_parent_limit, calculate_fitness

from concurrent.futures import ThreadPoolExecutor
from pgmpy.models import BayesianNetwork

import numpy as np
import networkx as nx
import random
import copy
import logging
import time
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


class CandidateSolution:
    #Represents a candidate Bayesian Network structure.
    def __init__(self, edges, variables, score, fitness=None):
        self.edges = edges
        self.variables = variables
        self.score_func = score
        if fitness is None:
            self.fitness = calculate_fitness(edges, variables, score)
        else:
            self.fitness = fitness

        self.model = None
        if self.fitness != float('inf'):
            self.model = BayesianNetwork()
            self.model.add_nodes_from(variables)
            self.model.add_edges_from(edges)

    def get_fitness(self):
        return self.fitness

    def copy(self):
        return copy.deepcopy(self)

    def update_edges(self, new_edges):
        """Update edges and recalculate fitness if the new graph is valid."""
        if is_acyclic(new_edges, self.variables) and not exceeds_parent_limit(new_edges, self.variables):
            self.edges = new_edges
            self.fitness = calculate_fitness(new_edges, self.variables, self.score_func)

            if self.fitness != float('inf'):
                self.model = BayesianNetwork()
                self.model.add_nodes_from(self.variables)
                self.model.add_edges_from(new_edges)
                return True
        return False


class ABC_BN:
    """Enhanced Artificial Bee Colony optimizer for Bayesian Network structure learning."""

    def __init__(
            self,
            data,
            variables,
            score,
            pop_size=50,
            limit=30,
            num_iters=100,
            num_workers=8,
            edge_density=0.1,
            seed=42,
            adaptive_neighborhood=True,
            elite_count=3,
            local_search_iters=5
    ):
        self.data = data
        self.variables = variables
        self.score = score
        self.pop_size = pop_size
        self.limit = limit
        self.num_iters = num_iters
        self.num_workers = num_workers
        self.edge_density = edge_density
        self.seed = seed
        self.adaptive_neighborhood = adaptive_neighborhood
        self.elite_count = elite_count
        self.local_search_iters = local_search_iters

        # Advanced parameters
        self.min_diversity = 0.3
        self.tabu_list_size = 10
        self.max_stagnation = 25
        self.adaptive_mutation_rate = 0.1
        self.scout_ratio = 0.2
        self.min_edges = max(1, len(variables) // 2)
        self.initial_temp = 100
        self.temp_decay = 0.98

        # Initialize random generators
        random.seed(seed)
        np.random.seed(seed)

        # Data structures for algorithm
        self.tabu_list = []
        self.current_temp = self.initial_temp

        # Initialize population with improved strategies
        self.population = self.generate_initial_population()
        self.trial = [0] * pop_size
        self.best_fitness_history = []

    def generate_initial_population(self):
        """Create initial population with multiple strategies for diversity."""
        pop = []

        # Strategy 1: Empty graph (minimal assumption)
        empty_edges = []
        empty_candidate = CandidateSolution(empty_edges, self.variables, self.score)
        if empty_candidate.get_fitness() != float('inf'):
            pop.append(empty_candidate)

        # Strategy 2: Naive Bayes structure (one root causes all)
        for root in self.variables:
            edges = [(root, var) for var in self.variables if var != root]
            if is_acyclic(edges, self.variables) and not exceeds_parent_limit(edges, self.variables):
                candidate = CandidateSolution(edges, self.variables, self.score)
                if candidate.get_fitness() != float('inf'):
                    pop.append(candidate)
                    break

        # Strategy 3: Tree structures using Chow-Liu algorithm approximation
        tree_edges = self.generate_tree_structure()
        if tree_edges and is_acyclic(tree_edges, self.variables):
            tree_candidate = CandidateSolution(tree_edges, self.variables, self.score)
            if tree_candidate.get_fitness() != float('inf'):
                pop.append(tree_candidate)

        # Strategy 4: Random DAGs with varying densities
        densities = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        for density in densities:
            edges = self.random_dag(density=density)
            if is_acyclic(edges, self.variables) and not exceeds_parent_limit(edges, self.variables):
                candidate = CandidateSolution(edges, self.variables, self.score)
                if candidate.get_fitness() != float('inf'):
                    pop.append(candidate)

        # Fill remaining population with random DAGs
        attempts = 0
        while len(pop) < self.pop_size:
            density = random.uniform(0.05, 0.3)
            edges = self.random_dag(density=density)
            if is_acyclic(edges, self.variables) and not exceeds_parent_limit(edges, self.variables):
                candidate = CandidateSolution(edges, self.variables, self.score)
                if candidate.get_fitness() != float('inf'):
                    # Only add if not too similar to existing solutions
                    if self.is_diverse_enough(candidate, pop):
                        pop.append(candidate)
            attempts += 1
            if attempts % 20 == 0:
                logging.info(f"Population size: {len(pop)} / {self.pop_size} after {attempts} attempts")
            if attempts > 500:  # Safety limit
                break

        # If we still don't have enough solutions, duplicate and mutate existing ones
        while len(pop) < self.pop_size and pop:
            parent = random.choice(pop)
            child = parent.copy()
            new_edges = self.mutate_solution(child.edges)
            if is_acyclic(new_edges, self.variables) and not exceeds_parent_limit(new_edges, self.variables):
                child_candidate = CandidateSolution(new_edges, self.variables, self.score)
                if child_candidate.get_fitness() != float('inf'):
                    pop.append(child_candidate)

        # If we somehow still don't have a full population, pad with the best solution
        if pop and len(pop) < self.pop_size:
            best_sol = min(pop, key=lambda x: x.get_fitness())
            while len(pop) < self.pop_size:
                pop.append(best_sol.copy())

        return pop

    def is_diverse_enough(self, new_candidate, population, threshold=0.3):
        """Check if a new candidate is diverse enough compared to the population."""
        if not population:
            return True

        new_edges_set = set(new_candidate.edges)

        for sol in population:
            existing_edges_set = set(sol.edges)

            # Calculate Jaccard similarity
            union = len(new_edges_set.union(existing_edges_set))
            if union == 0:  # Both sets are empty
                similarity = 1.0
            else:
                intersection = len(new_edges_set.intersection(existing_edges_set))
                similarity = intersection / union

            if similarity > (1 - threshold):
                return False

        return True

    def generate_tree_structure(self):
        """Generate a tree structure as an approximation to Chow-Liu tree."""
        # Create a complete graph with random weights to simulate mutual information
        G = nx.Graph()
        for i, var1 in enumerate(self.variables):
            for var2 in self.variables[i + 1:]:
                # Random weight - in a real implementation, this would be mutual information
                G.add_edge(var1, var2, weight=random.uniform(0, 1))

        # Find minimum spanning tree
        try:
            mst = nx.minimum_spanning_tree(G)

            # Choose a random root
            root = random.choice(list(mst.nodes()))

            # Direct edges away from root to create a DAG
            directed_edges = []
            visited = {node: False for node in mst.nodes()}

            def direct_edges(node):
                visited[node] = True
                for neighbor in mst.neighbors(node):
                    if not visited[neighbor]:
                        directed_edges.append((node, neighbor))
                        direct_edges(neighbor)

            direct_edges(root)
            return directed_edges
        except Exception as e:
            logging.warning(f"Error in tree generation: {str(e)}")
            return []

    def random_dag(self, density=None):
        """Generate a random DAG with improved topological ordering approach."""
        if density is None:
            density = self.edge_density

        # Use topological ordering to ensure acyclicity
        nodes = self.variables.copy()
        random.shuffle(nodes)
        edges = []

        # For each node, consider adding edges from nodes earlier in the ordering
        for i, v in enumerate(nodes):
            # Available parents are nodes earlier in the ordering
            available_parents = nodes[:i]

            if not available_parents:
                continue

            # Decide how many parents to add (constrained by max_parents)
            max_possible = min(len(available_parents), 3)
            num_parents = min(max_possible, math.ceil(density * len(available_parents)))

            if num_parents > 0:
                # Randomly select parents
                parents = random.sample(available_parents, num_parents)
                for u in parents:
                    edges.append((u, v))

        # Sometimes add extra edges with low probability
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if random.random() < density / 3:  # Lower probability for extra edges
                    edges.append((u, v))

        return edges

    def mutate_solution(self, edges, mutation_strength=1.0):
        """Apply multiple mutation operations to create a neighbor solution."""
        new_edges = edges.copy()
        num_operations = max(1, int(random.expovariate(1 / mutation_strength)))

        for _ in range(num_operations):
            operation = random.choices(
                ['add', 'remove', 'reverse'],
                weights=[0.4, 0.3, 0.3],
                k=1
            )[0]

            if operation == 'add':
                # Add a random edge
                all_possible_edges = [(u, v) for u in self.variables for v in self.variables
                                      if u != v and (u, v) not in new_edges]
                if all_possible_edges:
                    new_edge = random.choice(all_possible_edges)
                    new_edges.append(new_edge)

            elif operation == 'remove' and new_edges:
                # Remove a random edge
                edge_to_remove = random.choice(new_edges)
                new_edges.remove(edge_to_remove)

            elif operation == 'reverse' and new_edges:
                # Reverse a random edge
                edge_to_reverse = random.choice(new_edges)
                u, v = edge_to_reverse
                new_edges.remove(edge_to_reverse)
                reversed_edge = (v, u)
                if reversed_edge not in new_edges:
                    new_edges.append(reversed_edge)

        return new_edges

    def get_all_fitness_values(self):
        """Calculate fitness values for all solutions in parallel."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            fitnesses = list(executor.map(lambda sol: sol.get_fitness(), self.population))
        return fitnesses

    def get_neighborhood(self, solution, partner=None):
        """Generate neighborhood solutions using adaptive strategies."""
        mutation_strength = 1.0

        # Increase mutation strength if solution is in trial mode
        index = self.population.index(solution) if solution in self.population else -1
        if index >= 0 and self.trial[index] > self.limit / 2:
            mutation_strength = 1.5 + (self.trial[index] - self.limit / 2) / (self.limit / 2) * 0.5

        # Base solution to modify
        base_edges = solution.edges.copy()

        # If partner is provided, incorporate some of its structure
        if partner:
            # Learn from partner by adopting some of its edges
            adopt_rate = random.uniform(0.1, 0.3)
            partner_edges = set(partner.edges)
            solution_edges = set(base_edges)

            # Edges in partner but not in solution
            diff_edges = partner_edges - solution_edges
            if diff_edges:
                # Adopt some edges from partner
                num_to_adopt = max(1, int(adopt_rate * len(diff_edges)))
                edges_to_add = random.sample(list(diff_edges), min(num_to_adopt, len(diff_edges)))
                for edge in edges_to_add:
                    base_edges.append(edge)

        # Apply mutations to create a neighbor
        return self.mutate_solution(base_edges, mutation_strength)

    def simulated_annealing_acceptance(self, current_fitness, new_fitness):
        """Accept worse solutions with a probability based on temperature."""
        if new_fitness <= current_fitness:  # Better solution (lower is better for BIC)
            return True
        else:
            # Calculate acceptance probability
            delta = abs(new_fitness - current_fitness)
            probability = math.exp(-delta / self.current_temp)
            return random.random() < probability

    def local_search(self, solution):
        """Apply local search to improve a solution."""
        best_sol = solution.copy()
        best_fitness = best_sol.get_fitness()

        for _ in range(self.local_search_iters):
            # Generate a neighbor solution
            new_edges = self.get_neighborhood(best_sol)

            # Check if it's a valid DAG
            if is_acyclic(new_edges, self.variables) and not exceeds_parent_limit(new_edges, self.variables):
                new_fitness = calculate_fitness(new_edges, self.variables, self.score)

                # Accept if better or with some probability if worse
                if new_fitness != float('inf') and new_fitness < best_fitness:
                    best_sol = CandidateSolution(new_edges, self.variables, self.score, new_fitness)
                    best_fitness = new_fitness

        return best_sol

    def deploy_scouts(self):
        """Deploy scout bees to discover new regions of the search space."""
        num_scouts = int(self.scout_ratio * self.pop_size)
        scout_indices = []

        # Select solutions with highest trial counts
        trial_indices = sorted(range(len(self.trial)), key=lambda i: self.trial[i], reverse=True)
        scout_indices = trial_indices[:num_scouts]

        # For each scout, generate a new solution
        for idx in scout_indices:
            # Introduce completely new solutions with varying strategies
            strategy = random.choice(['random', 'tree', 'sparse', 'hybrid'])

            if strategy == 'random':
                density = random.uniform(0.05, 0.3)
                new_edges = self.random_dag(density=density)
            elif strategy == 'tree':
                new_edges = self.generate_tree_structure()
            elif strategy == 'sparse':
                # Create a very sparse graph
                density = random.uniform(0.01, 0.1)
                new_edges = self.random_dag(density=density)
            else:  # hybrid
                # Start with current solution but apply major mutations
                base_solution = self.population[idx]
                new_edges = self.mutate_solution(base_solution.edges, mutation_strength=3.0)

            # Validate new solution
            if is_acyclic(new_edges, self.variables) and not exceeds_parent_limit(new_edges, self.variables):
                new_fitness = calculate_fitness(new_edges, self.variables, self.score)
                if new_fitness != float('inf'):
                    self.population[idx] = CandidateSolution(new_edges, self.variables, self.score, new_fitness)
                    self.trial[idx] = 0

    def recombination(self, solution1, solution2):
        """Create a new solution by recombining two parent solutions."""
        edges1 = set(solution1.edges)
        edges2 = set(solution2.edges)

        # Common edges from both parents
        common_edges = edges1.intersection(edges2)

        # Unique edges from each parent
        unique_edges1 = edges1 - common_edges
        unique_edges2 = edges2 - common_edges

        # Start with common edges
        new_edges = list(common_edges)

        # Add some unique edges from each parent
        p1_rate = random.uniform(0.4, 0.6)
        p2_rate = random.uniform(0.4, 0.6)

        for edge in unique_edges1:
            if random.random() < p1_rate:
                new_edges.append(edge)

        for edge in unique_edges2:
            if random.random() < p2_rate:
                new_edges.append(edge)

        return new_edges

    def deploy_employed(self):
        """Send employed bees to search near current solutions."""
        for i in range(self.pop_size):
            current_solution = self.population[i]

            # Select a partner solution for information exchange
            idxs = list(range(self.pop_size))
            idxs.remove(i)
            partner_idx = random.choice(idxs)
            partner = self.population[partner_idx]

            # Get neighborhood solution
            new_edges = self.get_neighborhood(current_solution, partner)

            # Validate and evaluate the new solution
            if is_acyclic(new_edges, self.variables) and not exceeds_parent_limit(new_edges, self.variables):
                new_fitness = calculate_fitness(new_edges, self.variables, self.score)

                if new_fitness != float('inf'):
                    # Use simulated annealing acceptance criterion
                    current_fitness = current_solution.get_fitness()

                    if self.simulated_annealing_acceptance(current_fitness, new_fitness):
                        self.population[i] = CandidateSolution(new_edges, self.variables, self.score, new_fitness)
                        self.trial[i] = 0
                    else:
                        self.trial[i] += 1
                else:
                    self.trial[i] += 1
            else:
                self.trial[i] += 1

    def calculate_probabilities(self, fitness_values):
        """Calculate selection probabilities for onlooker bees with rank-based approach."""
        # Sort solutions by fitness (ascending for BIC)
        sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])

        # Use rank-based selection probability
        n = len(sorted_indices)
        ranks = [0] * n
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = n - i  # Best solution gets highest rank

        # Apply non-linear scaling to increase selection pressure
        scaled_ranks = [r ** 1.5 for r in ranks]

        # Normalize to probabilities
        total = sum(scaled_ranks)
        if total > 0:
            probs = [r / total for r in scaled_ranks]
        else:
            probs = [1 / n] * n

        return probs

    def deploy_onlookers(self):
        """Send onlooker bees to promising solutions using a rank-based approach."""
        fitness_vals = [sol.get_fitness() for sol in self.population]
        probs = self.calculate_probabilities(fitness_vals)

        i = 0
        count = 0

        while count < self.pop_size:
            if random.random() < probs[i]:
                current_solution = self.population[i]

                # Apply recombination with probability
                if random.random() < 0.3:  # 30% chance for recombination
                    # Select another solution for recombination
                    idxs = list(range(self.pop_size))
                    idxs.remove(i)
                    partner_idx = random.choices(idxs, weights=[probs[j] for j in idxs if j != i], k=1)[0]
                    partner = self.population[partner_idx]

                    # Create recombined solution
                    new_edges = self.recombination(current_solution, partner)
                else:
                    # Regular neighborhood search
                    # Select partner (prefer better solutions)
                    idxs = list(range(self.pop_size))
                    idxs.remove(i)
                    weights = [probs[j] for j in idxs]
                    partner_idx = random.choices(idxs, weights=weights, k=1)[0]
                    partner = self.population[partner_idx]

                    # Generate new solution
                    new_edges = self.get_neighborhood(current_solution, partner)

                # Validate and evaluate
                if is_acyclic(new_edges, self.variables) and not exceeds_parent_limit(new_edges, self.variables):
                    new_fitness = calculate_fitness(new_edges, self.variables, self.score)

                    if new_fitness != float('inf'):
                        current_fitness = current_solution.get_fitness()

                        # Accept better solution or with probability for worse
                        if new_fitness < current_fitness:
                            self.population[i] = CandidateSolution(new_edges, self.variables, self.score, new_fitness)
                            self.trial[i] = 0
                        else:
                            # Small chance to accept even slightly worse solutions to escape local optima
                            if random.random() < 0.05:  # 5% chance
                                self.population[i] = CandidateSolution(new_edges, self.variables, self.score,
                                                                       new_fitness)
                                self.trial[i] = 0
                            else:
                                self.trial[i] += 1
                    else:
                        self.trial[i] += 1
                else:
                    self.trial[i] += 1

                count += 1

            i = (i + 1) % self.pop_size

    def population_diversity(self):
        """Calculate population diversity using average Jaccard distance."""
        sets = [set(sol.edges) for sol in self.population]
        n = len(sets)
        if n < 2:
            return 0.0

        total = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                union_size = len(sets[i] | sets[j])
                if union_size > 0:
                    inter_size = len(sets[i] & sets[j])
                    jacc = 1 - inter_size / union_size  # Jaccard distance
                    total += jacc
                    count += 1
                else:
                    total += 0  # Both empty sets
                    count += 1

        return total / count if count > 0 else 0.0

    def run(self):
        """Executes the enhanced ABC algorithm and returns metrics, best model, and history."""
        fitness_evolution = []
        avg_fitness = []
        diversity = []
        best_edges_history = []
        start = time.time()

        # Initial best solution
        global_best = min(self.population, key=lambda s: s.get_fitness()).copy()
        best_score = global_best.get_fitness()
        stagnation = 0

        for it in range(self.num_iters):
            # Apply local search to the best solution
            if it % 5 == 0:  # Every 5 iterations
                improved_best = self.local_search(global_best)
                if improved_best.get_fitness() < best_score:
                    global_best = improved_best
                    best_score = global_best.get_fitness()
                    stagnation = 0

            # Deploy bees
            self.deploy_employed()
            self.deploy_onlookers()

            # Check for abandoned solutions and deploy scouts
            self.deploy_scouts()

            # Update temperature for simulated annealing
            self.current_temp *= self.temp_decay

            # Evaluate current best
            current_best = min(self.population, key=lambda s: s.get_fitness())
            current_score = current_best.get_fitness()
            fitness_evolution.append(current_score)

            # Track metrics
            fitness_values = [sol.get_fitness() for sol in self.population]
            avg_fitness.append(np.mean(fitness_values))
            pop_diversity = self.population_diversity()
            diversity.append(pop_diversity)

            # Update global best
            if current_score < best_score:
                global_best = current_best.copy()
                best_score = current_score
                stagnation = 0
                logging.info(f"Iteration {it}: New best score: {best_score}")
            else:
                stagnation += 1

            # Elitism - ensure best solution stays in population
            worst_idx = max(range(self.pop_size), key=lambda i: self.population[i].get_fitness())
            if self.population[worst_idx].get_fitness() > best_score:
                self.population[worst_idx] = global_best.copy()
                self.trial[worst_idx] = 0

            # Record history
            best_edges_history.append(list(global_best.edges))

            # Log every 10 iterations
            if it % 10 == 0:
                logging.info(
                    f"Iteration {it}: Best score: {best_score}, Avg score: {np.mean(fitness_values):.2f}, Diversity: {pop_diversity:.2f}")

            # Diversity maintenance
            if pop_diversity < self.min_diversity and it % 5 == 0:
                logging.info(f"Low diversity detected ({pop_diversity:.2f}), injecting new solutions")
                # Replace 20% of worst solutions with new random ones
                num_to_replace = max(1, int(self.pop_size * 0.2))
                worst_indices = sorted(range(self.pop_size), key=lambda i: -self.population[i].get_fitness())[
                                :num_to_replace]

                for idx in worst_indices:
                    density = random.uniform(0.05, 0.3)
                    new_edges = self.random_dag(density=density)
                    if is_acyclic(new_edges, self.variables) and not exceeds_parent_limit(new_edges, self.variables):
                        new_fitness = calculate_fitness(new_edges, self.variables, self.score)
                        if new_fitness != float('inf'):
                            self.population[idx] = CandidateSolution(new_edges, self.variables, self.score, new_fitness)
                            self.trial[idx] = 0

            # Early stopping if stagnation persists
            if stagnation >= self.max_stagnation:
                logging.info(f"Stopping early at iteration {it} due to stagnation.")
                break

        elapsed = time.time() - start
        logging.info(f"Best BIC Score: {-best_score}")
        logging.info(f"Edges in best model: {len(global_best.edges)}")
        logging.info(f"Time taken: {elapsed:.2f} seconds")

        return fitness_evolution, avg_fitness, diversity, global_best, best_edges_history