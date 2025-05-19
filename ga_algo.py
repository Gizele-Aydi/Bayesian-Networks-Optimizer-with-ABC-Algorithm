import networkx as nx
import random
import copy

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

        for _ in range(self.generations):
            selected = random.choices(population, weights=scores, k=2)
            child = self._crossover(selected[0], selected[1])
            child = self._mutate(child)

            worst_idx = scores.index(min(scores))
            population[worst_idx] = child
            scores[worst_idx] = self.scoring_function(child)

        best_idx = scores.index(max(scores))
        return population[best_idx], scores[best_idx]

