import networkx as nx
import random
import copy
import numpy as np


class GeneticAlgorithmBN:
    def __init__(self, population_size, generations, mutation_rate, scoring_function, max_parents=3):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.scoring_function = scoring_function
        self.max_parents = max_parents  # Max parents per node constraint
        self.stagnation_limit = 10  # Number of generations without improvement before taking action
        self.stagnation_count = 0

    def _random_dag(self, nodes):
        """Generate a random DAG with constraints on parent count"""
        G = nx.DiGraph()
        G.add_nodes_from(nodes)

        # Use a topological ordering to ensure acyclicity
        node_order = list(nodes)
        random.shuffle(node_order)

        for i in range(len(node_order)):
            # Only consider nodes that come later in the ordering as potential children
            # This guarantees acyclicity
            possible_children = node_order[i + 1:]

            # Add some random edges
            for child in possible_children:
                if random.random() < 0.2:
                    # Check parent limit constraint before adding edge
                    if G.in_degree(child) < self.max_parents:
                        G.add_edge(node_order[i], child)

        return G

    def _is_valid_dag_with_constraints(self, graph):
        """Check if graph is a valid DAG and satisfies parent limit constraint"""
        if not nx.is_directed_acyclic_graph(graph):
            return False

        # Check max parent constraint
        for node in graph.nodes():
            if graph.in_degree(node) > self.max_parents:
                return False

        return True

    def _crossover(self, parent1, parent2):
        """Improved crossover that ensures valid DAG creation"""
        # Start with a copy of parent1
        child = parent1.copy()
        edges_to_try = list(parent2.edges())
        random.shuffle(edges_to_try)  # Randomize edge order for better exploration

        for u, v in edges_to_try:
            if not child.has_edge(u, v):
                # Try to add the edge
                child.add_edge(u, v)

                # Revert if it creates an invalid structure
                if not self._is_valid_dag_with_constraints(child):
                    child.remove_edge(u, v)

        return child

    def _mutate(self, graph):
        """Enhanced mutation with multiple possible operations"""
        if random.random() >= self.mutation_rate:
            return graph

        mutated = graph.copy()
        nodes = list(graph.nodes())

        # Choose mutation type:
        # 1. Add edge
        # 2. Remove edge
        # 3. Reverse edge
        mutation_type = random.choice([1, 2, 3])

        if mutation_type == 1 and len(nodes) >= 2:  # Add edge
            attempts = 0
            while attempts < 10:  # Limit attempts to avoid infinite loops
                u, v = random.sample(nodes, 2)
                if not mutated.has_edge(u, v):
                    mutated.add_edge(u, v)
                    if self._is_valid_dag_with_constraints(mutated):
                        return mutated
                    mutated.remove_edge(u, v)
                attempts += 1

        elif mutation_type == 2 and mutated.number_of_edges() > 0:  # Remove edge
            if mutated.number_of_edges() > 0:
                edge = random.choice(list(mutated.edges()))
                mutated.remove_edge(*edge)
                # Removing an edge will maintain DAG property
                return mutated

        elif mutation_type == 3 and mutated.number_of_edges() > 0:  # Reverse edge
            attempts = 0
            while attempts < 10:
                edge = random.choice(list(mutated.edges()))
                u, v = edge
                mutated.remove_edge(u, v)
                mutated.add_edge(v, u)

                if self._is_valid_dag_with_constraints(mutated):
                    return mutated

                # Revert if invalid
                mutated.remove_edge(v, u)
                mutated.add_edge(u, v)
                attempts += 1

        return graph  # Return original if no valid mutation found

    def _compute_selection_weights(self, scores):
        """Improved weight calculation for selection that handles negative scores properly"""
        min_score = min(scores)

        # Create a positive offset and apply scaling to improve selection pressure
        if min_score < 0:
            # For negative scores like BIC, make them positive with larger magnitude = better
            adjusted_scores = [-(score - min_score) + 1 for score in scores]
        else:
            adjusted_scores = [score + 1 for score in scores]

        # Apply scaling to increase selection pressure
        mean = np.mean(adjusted_scores)
        if mean > 0:
            adjusted_scores = [s ** 2 / mean for s in adjusted_scores]

        return adjusted_scores

    def _inject_diversity(self, population, percentage=0.2):
        """Inject fresh individuals to increase diversity"""
        nodes = list(population[0].nodes())
        num_to_replace = max(1, int(self.population_size * percentage))

        # Generate new random individuals
        for i in range(num_to_replace):
            # Replace the worst individuals
            if i < len(population):
                population[-(i + 1)] = self._random_dag(nodes)

        return population

    def run(self, nodes):
        # Initialize population with valid DAGs
        population = [self._random_dag(nodes) for _ in range(self.population_size)]
        scores = [self.scoring_function(g) for g in population]

        # Keep track of the best solution found
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_graph = copy.deepcopy(population[best_idx])
        best_score = scores[best_idx]

        prev_best_score = float('-inf')

        for gen in range(self.generations):
            # Update best solution if better found
            current_best_idx = max(range(len(scores)), key=lambda i: scores[i])
            current_best_score = scores[current_best_idx]

            if current_best_score > best_score:
                best_score = current_best_score
                best_graph = copy.deepcopy(population[current_best_idx])
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1

            # Track progress for debugging
            if gen % 10 == 0:
                avg_edges = np.mean([g.number_of_edges() for g in population])
                print(f"Generation {gen}: Best score = {best_score}, Avg edges = {avg_edges:.1f}")

            # Calculate selection weights properly
            selection_weights = self._compute_selection_weights(scores)

            # Create new population
            new_population = []

            # Elitism: Always keep the best individual
            new_population.append(copy.deepcopy(best_graph))

            # Create the rest of the population through selection, crossover, mutation
            while len(new_population) < self.population_size:
                # Tournament selection (more robust than weighted random)
                parent1 = self._tournament_selection(population, scores)
                parent2 = self._tournament_selection(population, scores)

                # Crossover and mutate
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)

                # Only add if it's a valid DAG
                if self._is_valid_dag_with_constraints(child) and child.number_of_edges() > 0:
                    new_population.append(child)

            # Handle stagnation by injecting diversity if needed
            if self.stagnation_count >= self.stagnation_limit:
                print(f"Stagnation detected at generation {gen}, injecting diversity")
                new_population = self._inject_diversity(new_population)
                self.stagnation_count = 0

            # Update population and calculate new scores
            population = new_population
            scores = [self.scoring_function(g) for g in population]

            # Early stopping if no improvement for many generations
            if gen > 20 and self.stagnation_count > self.stagnation_limit * 2:
                print(f"Early stopping at generation {gen} due to prolonged stagnation")
                break

        # Return the best solution found during the entire run
        return best_graph, best_score

    def _tournament_selection(self, population, scores, tournament_size=3):
        """Tournament selection - more robust than weighted random selection"""
        indices = random.sample(range(len(population)), min(tournament_size, len(population)))
        winner_idx = max(indices, key=lambda i: scores[i])
        return population[winner_idx]