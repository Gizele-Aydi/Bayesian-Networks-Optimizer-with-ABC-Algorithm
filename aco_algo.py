import networkx as nx
import numpy as np
import random


class AntColonyBN:
    def __init__(self, num_ants, iterations, alpha, beta, evaporation_rate, scoring_function):
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.scoring_function = scoring_function

    def _initialize_pheromones(self, nodes):
        return {(i, j): 1.0 for i in nodes for j in nodes if i != j}

    def _construct_solution(self, pheromones, nodes):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        edges = list(pheromones.keys())
        random.shuffle(edges)

        for u, v in edges:
            if (random.random() < pheromones[(u, v)] / (1 + pheromones[(u, v)])):
                graph.add_edge(u, v)
                if not nx.is_directed_acyclic_graph(graph):
                    graph.remove_edge(u, v)
        return graph

    def run(self, nodes):
        pheromones = self._initialize_pheromones(nodes)
        best_graph = None
        best_score = float('-inf')

        for _ in range(self.iterations):
            all_graphs = []
            all_scores = []

            for _ in range(self.num_ants):
                g = self._construct_solution(pheromones, nodes)
                score = self.scoring_function(g)
                all_graphs.append(g)
                all_scores.append(score)

                if score > best_score:
                    best_score = score
                    best_graph = g

            for (u, v) in pheromones:
                pheromones[(u, v)] *= (1 - self.evaporation_rate)

            for i in range(len(all_graphs)):
                g = all_graphs[i]
                score = all_scores[i]
                for u, v in g.edges():
                    pheromones[(u, v)] += score / 100

        return best_graph, best_score
