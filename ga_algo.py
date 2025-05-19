import networkx as nx
import random
import copy
import numpy as np


class GeneticAlgorithmBN:
    def __init__(self, population_size, generations, mutation_rate, scoring_function):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.scoring_function = scoring_function

    def _random_dag(self, nodes):
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if random.random() < 0.2:
                    G.add_edge(nodes[i], nodes[j])
        return G if nx.is_directed_acyclic_graph(G) else self._random_dag(nodes)

    def _crossover(self, parent1, parent2):
        child = parent1.copy()
        for u, v in parent2.edges():
            if random.random() < 0.5 and not child.has_edge(u, v):
                child.add_edge(u, v)
                if not nx.is_directed_acyclic_graph(child):
                    child.remove_edge(u, v)
        return child

    def _mutate(self, graph):
        if random.random() < self.mutation_rate:
            nodes = list(graph.nodes())
            u, v = random.choice(nodes), random.choice(nodes)
            if u != v:
                if graph.has_edge(u, v):
                    graph.remove_edge(u, v)
                else:
                    graph.add_edge(u, v)
                    if not nx.is_directed_acyclic_graph(graph):
                        graph.remove_edge(u, v)
        return graph

    def run(self, nodes):
        population = [self._random_dag(nodes) for _ in range(self.population_size)]
        scores = [self.scoring_function(g) for g in population]

        best_graph = population[0]
        best_score = scores[0]

        for gen in range(self.generations):
            # Find the best score in the current population
            for i, score in enumerate(scores):
                if score > best_score:
                    best_score = score
                    best_graph = copy.deepcopy(population[i])

            # Convert scores to selection probabilities (handling negative scores)
            selection_weights = self._compute_selection_weights(scores)

            # Create new population through selection, crossover, and mutation
            new_population = []
            new_scores = []

            # Elitism: keep the best individual
            best_idx = scores.index(max(scores))
            new_population.append(copy.deepcopy(population[best_idx]))
            new_scores.append(scores[best_idx])

            # Fill the rest of the population
            while len(new_population) < self.population_size:
                if np.sum(selection_weights) <= 0:
                    # If all weights are non-positive, select randomly
                    parent_indices = random.sample(range(len(population)), 2)
                else:
                    # Select parents based on weights
                    parent_indices = random.choices(
                        range(len(population)),
                        weights=selection_weights,
                        k=2
                    )

                parent1 = population[parent_indices[0]]
                parent2 = population[parent_indices[1]]

                child = self._crossover(parent1, parent2)
                child = self._mutate(child)

                child_score = self.scoring_function(child)
                new_population.append(child)
                new_scores.append(child_score)

            population = new_population
            scores = new_scores

        best_idx = scores.index(max(scores))
        return population[best_idx], scores[best_idx]

    def _compute_selection_weights(self, scores):
        """Convert scores to non-negative weights for selection."""
        min_score = min(scores)

        # If all scores are negative, shift them to make the minimum score 1
        if min_score <= 0:
            adjusted_scores = [score - min_score + 1 for score in scores]
        else:
            adjusted_scores = scores.copy()

        # Check if all scores are equal (would result in uniform selection)
        if all(score == adjusted_scores[0] for score in adjusted_scores):
            return [1] * len(adjusted_scores)

        return adjusted_scores