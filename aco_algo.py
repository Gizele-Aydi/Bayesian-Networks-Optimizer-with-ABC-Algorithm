import networkx as nx
import numpy as np
import random


class AntColonyBN:
    def __init__(self, num_ants, iterations, alpha=1.0, beta=2.0, evaporation_rate=0.1,
                 scoring_function=None, max_parents=3, max_children=None, q0=0.1):
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta  # Heuristic information importance
        self.evaporation_rate = evaporation_rate
        self.scoring_function = scoring_function
        self.max_parents = max_parents  # Maximum parents per node
        self.max_children = max_children  # Maximum children per node (None = unlimited)
        self.q0 = q0  # Probability of exploitation vs exploration
        self.stagnation_limit = 10
        self.stagnation_count = 0
        self.min_edges = 1  # Minimum edges to consider a graph valid

    def _initialize_pheromones(self, nodes):
        """Initialize pheromone matrix with a small positive value"""
        return {(i, j): 1.0 for i in nodes for j in nodes if i != j}

    def _initialize_heuristic(self, nodes):
        """Initialize edge desirability heuristic"""
        # By default, all edges are equally desirable
        return {(i, j): 1.0 for i in nodes for j in nodes if i != j}

    def _valid_edges_for_node(self, graph, target_node, nodes):
        """Find valid edges that could be added to target_node without creating cycles"""
        valid_edges = []

        # If node already has max parents, no more edges can be added
        if graph.in_degree(target_node) >= self.max_parents:
            return valid_edges

        for source in nodes:
            if source != target_node and not graph.has_edge(source, target_node):
                # Check if adding this edge would create a cycle
                test_graph = graph.copy()
                test_graph.add_edge(source, target_node)

                # Check if adding this edge would exceed max children for source
                if self.max_children is not None and test_graph.out_degree(source) > self.max_children:
                    continue

                if nx.is_directed_acyclic_graph(test_graph):
                    valid_edges.append((source, target_node))

        return valid_edges

    def _construct_solution(self, pheromones, heuristic, nodes):
        """Construct a DAG solution using pheromone information and heuristic guidance"""
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)

        # Create a list of nodes in random order to avoid bias
        node_order = list(nodes)
        random.shuffle(node_order)

        # For each node, decide on potential parent nodes
        for target_node in node_order:
            # Get valid edges that could be added to this node
            valid_edges = self._valid_edges_for_node(graph, target_node, nodes)

            if not valid_edges:
                continue

            # Calculate selection probabilities based on pheromone and heuristic values
            edge_weights = {}
            total_weight = 0

            for source, target in valid_edges:
                weight = (pheromones[(source, target)] ** self.alpha) * (heuristic[(source, target)] ** self.beta)
                edge_weights[(source, target)] = weight
                total_weight += weight

            if total_weight == 0:
                continue  # No valid edges with non-zero weight

            # Normalize weights to probabilities
            probabilities = {edge: weight / total_weight for edge, weight in edge_weights.items()}

            # Select edges to add using roulette wheel selection with exploitation/exploration balance
            # Implement a pseudo-ACS (Ant Colony System) approach
            for edge in valid_edges:
                if random.random() < self.q0:
                    # Exploitation: select the edge with highest probability
                    if edge == max(probabilities, key=probabilities.get):
                        # Check max_children constraint
                        source, target = edge
                        if self.max_children is None or graph.out_degree(source) < self.max_children:
                            graph.add_edge(*edge)
                else:
                    # Exploration: probabilistic selection based on pheromone
                    if random.random() < probabilities[edge]:
                        # Check max_children constraint
                        source, target = edge
                        if self.max_children is None or graph.out_degree(source) < self.max_children:
                            graph.add_edge(*edge)

                # Stop if we've reached max parents
                if graph.in_degree(target_node) >= self.max_parents:
                    break

        return graph

    def _calculate_local_pheromone_update(self, edge, base_value=0.1):
        """Local pheromone update rule to encourage exploration"""
        return base_value

    def _calculate_global_pheromone_update(self, score, min_score, max_score):
        """Convert score to pheromone update amount with proper scaling"""
        # Scale score to a value between 0.1 and 1.0
        if max_score == min_score:
            return 0.5  # Default value if all scores are the same

        normalized_score = (score - min_score) / (max_score - min_score)
        # For negative scores like BIC, invert the normalization
        if min_score < 0:
            normalized_score = 1 - normalized_score

        # Scale to a reasonable range for pheromone updates
        return 0.1 + 0.9 * normalized_score

    def _apply_edge_penalty(self, graph, score, penalty_factor=0.01):
        """Apply penalties for too few or too many edges"""
        edge_count = graph.number_of_edges()
        node_count = graph.number_of_nodes()

        # Penalize graphs with very few edges
        if edge_count < self.min_edges:
            return score - (penalty_factor * abs(score) * (self.min_edges - edge_count))

        # Penalize overly dense graphs
        max_expected_edges = node_count * self.max_parents / 2
        if edge_count > max_expected_edges:
            return score - (penalty_factor * abs(score) * (edge_count - max_expected_edges))

        return score

    def run(self, nodes):
        """Run the ACO algorithm to find the best Bayesian Network structure"""
        pheromones = self._initialize_pheromones(nodes)
        heuristic = self._initialize_heuristic(nodes)

        best_graph = None
        best_score = float('-inf')

        # Track scores for stagnation detection
        previous_best_scores = []

        for iteration in range(self.iterations):
            all_graphs = []
            all_scores = []

            # Each ant constructs a solution
            for ant in range(self.num_ants):
                graph = self._construct_solution(pheromones, heuristic, nodes)
                score = self.scoring_function(graph)

                # Apply edge penalty if needed
                score = self._apply_edge_penalty(graph, score)

                all_graphs.append(graph)
                all_scores.append(score)

                # Update best solution if better
                if score > best_score:
                    best_score = score
                    best_graph = graph.copy()
                    self.stagnation_count = 0

            # Detect stagnation
            previous_best_scores.append(best_score)
            if len(previous_best_scores) > self.stagnation_limit:
                previous_best_scores.pop(0)
                if len(set(previous_best_scores)) <= 1:
                    self.stagnation_count += 1
                else:
                    self.stagnation_count = 0

            # Calculate min and max scores for normalization
            min_score = min(all_scores)
            max_score = max(all_scores)

            # Global pheromone evaporation
            for edge in pheromones:
                pheromones[edge] *= (1 - self.evaporation_rate)

            # Pheromone updates based on solutions
            # Elite strategy: only the best ants deposit pheromones
            num_elite = max(1, self.num_ants // 4)
            elite_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)[:num_elite]

            for idx in elite_indices:
                graph = all_graphs[idx]
                score = all_scores[idx]

                # Skip invalid or extremely poor solutions
                if graph.number_of_edges() < self.min_edges:
                    continue

                pheromone_amount = self._calculate_global_pheromone_update(score, min_score, max_score)

                for u, v in graph.edges():
                    pheromones[(u, v)] += pheromone_amount

            # Handle stagnation by adding random pheromones
            if self.stagnation_count >= self.stagnation_limit:
                print(f"Stagnation detected at iteration {iteration}, adding random pheromones")
                for edge in random.sample(list(pheromones.keys()), len(pheromones) // 4):
                    pheromones[edge] += random.uniform(0.5, 1.0)
                self.stagnation_count = 0

            # Status update every few iterations
            if iteration % 5 == 0:
                avg_edges = np.mean([g.number_of_edges() for g in all_graphs])
                print(f"Iteration {iteration}: Best score = {best_score}, Avg edges = {avg_edges:.1f}")

        # Ensure we have a valid result
        if best_graph is None or best_graph.number_of_edges() < self.min_edges:
            print("Warning: No valid solution found, creating fallback solution")
            best_graph = nx.DiGraph()
            best_graph.add_nodes_from(nodes)
            best_score = self.scoring_function(best_graph)

        return best_graph, best_score