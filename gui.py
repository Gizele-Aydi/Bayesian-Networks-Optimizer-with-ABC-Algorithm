import tkinter as tk


from tkinter import ttk, messagebox
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import networkx as nx
import logging
import time

# Import algorithms
from abc_algo import ABC_BN
from utils import load_data, is_acyclic, exceeds_parent_limit, calculate_fitness

# Default hyperparameters
DEFAULT_SEED = 42
DEFAULT_DATASET = "asia.csv"
DEFAULT_POP = 20
DEFAULT_LIMIT = 50
DEFAULT_ITERS = 50
DEFAULT_WORKERS = 4
DEFAULT_INTERVAL = 500

# ACO specific defaults
DEFAULT_ANTS = 20
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 2.0
DEFAULT_EVAP_RATE = 0.1

# GA specific defaults
DEFAULT_MUTATION = 0.2

# Algorithm colors for comparison plots
ALGORITHM_COLORS = {
    "ABC": "blue",
    "ACO": "red",
    "GA": "green"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


class BN_Explorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Bayesian Network Structure Learning")
        self.root.geometry("1800x1000")

        # Style configuration
        style = ttk.Style(root)
        style.configure("TLabel", font=("Arial", 16))
        style.configure("TButton", font=("Arial", 16))
        style.configure("TSpinbox", font=("Arial", 16))
        style.configure("TCombobox", font=("Arial", 16))
        style.configure("TRadiobutton", font=("Arial", 16))
        style.configure("TCheckbutton", font=("Arial", 16))

        # Setup frames
        self.setup_frames()

        # Setup controls
        self.setup_controls()

        # Setup plots
        self.setup_plots()

        # Animation object
        self.anim = None

        # Data and algorithm state
        self.data = None
        self.variables = None
        self.score = None

        # Store results for all algorithms for comparison
        self.algorithm_results = {
            "ABC": {"fitness": [], "edges": []},
            "ACO": {"fitness": [], "edges": []},
            "GA": {"fitness": [], "edges": []}
        }

        # Currently selected algorithm for BN visualization
        self.current_bn_algorithm = None

    def reset_comparison(self):
        """Reset all algorithm results"""
        # Clear any existing animation
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except:
                pass
            self.anim = None

        self.algorithm_results = {
            "ABC": {"fitness": [], "edges": []},
            "ACO": {"fitness": [], "edges": []},
            "GA": {"fitness": [], "edges": []}
        }

    def setup_frames(self):
        # Main frames
        self.control_frame = ttk.Frame(self.root, padding=15)
        self.control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.fitness_frame = ttk.LabelFrame(self.root, text="Algorithm Search Progress", padding=10)
        self.fitness_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        self.bn_frame = ttk.LabelFrame(self.root, text="Learned Bayesian Network", padding=10)
        self.bn_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

        # Algorithm specific parameter frames
        self.algo_param_frame = ttk.LabelFrame(self.control_frame, text="Algorithm Parameters", padding=10)
        self.algo_param_frame.grid(row=0, column=4, rowspan=4, sticky=(tk.N, tk.S, tk.E, tk.W), padx=10)

    def setup_controls(self):
        # Common controls
        self.algorithm_var = tk.StringVar(value="ABC")
        self.iters_var = tk.IntVar(value=DEFAULT_ITERS)
        self.interval_var = tk.IntVar(value=DEFAULT_INTERVAL)

        # Comparison option
        self.compare_var = tk.BooleanVar(value=False)

        # ABC specific controls
        self.pop_var = tk.IntVar(value=DEFAULT_POP)
        self.limit_var = tk.IntVar(value=DEFAULT_LIMIT)
        self.workers_var = tk.IntVar(value=DEFAULT_WORKERS)

        # ACO specific controls
        self.ants_var = tk.IntVar(value=DEFAULT_ANTS)
        self.alpha_var = tk.DoubleVar(value=DEFAULT_ALPHA)
        self.beta_var = tk.DoubleVar(value=DEFAULT_BETA)
        self.evap_var = tk.DoubleVar(value=DEFAULT_EVAP_RATE)

        # GA specific controls
        self.ga_pop_var = tk.IntVar(value=DEFAULT_POP)
        self.mutation_var = tk.DoubleVar(value=DEFAULT_MUTATION)

        # Algorithm selection
        ttk.Label(self.control_frame, text="Algorithm:").grid(row=0, column=0, sticky=tk.W, pady=4)
        algo_combo = ttk.Combobox(self.control_frame, textvariable=self.algorithm_var,
                                  values=["ABC", "ACO", "GA"], state="readonly", width=10, font=("Arial", 16))
        algo_combo.grid(row=0, column=1, sticky=tk.W, padx=10, pady=4)
        algo_combo.bind("<<ComboboxSelected>>", self.update_algorithm_params)

        # Common parameters
        ttk.Label(self.control_frame, text="Iterations:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Spinbox(self.control_frame, from_=10, to=200, textvariable=self.iters_var,
                    width=10, font=("Arial", 16)).grid(row=1, column=1, padx=10, pady=4)

        ttk.Label(self.control_frame, text="Animation Interval (ms):").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Spinbox(self.control_frame, from_=100, to=2000, textvariable=self.interval_var,
                    width=10, font=("Arial", 16)).grid(row=2, column=1, padx=10, pady=4)

        # Comparison checkbox
        ttk.Checkbutton(self.control_frame, text="Compare All Algorithms",
                        variable=self.compare_var).grid(row=3, column=1, sticky=tk.W, pady=4)

        # Run button
        run_btn = ttk.Button(self.control_frame, text="Run Algorithm", command=self.run_optimization)
        run_btn.grid(row=4, column=0, columnspan=2, pady=15, sticky=tk.W)

        # Reset button
        reset_btn = ttk.Button(self.control_frame, text="Reset All", command=self.reset_comparison)
        reset_btn.grid(row=4, column=1, pady=15, sticky=tk.E)

        # Initial algorithm parameters display
        self.update_algorithm_params()

    def update_algorithm_params(self, event=None):
        # Clear previous parameter widgets
        for widget in self.algo_param_frame.winfo_children():
            widget.destroy()

        algorithm = self.algorithm_var.get()

        if algorithm == "ABC":
            ttk.Label(self.algo_param_frame, text="Population Size:").grid(row=0, column=0, sticky=tk.W, pady=4)
            ttk.Spinbox(self.algo_param_frame, from_=5, to=100, textvariable=self.pop_var,
                        width=10, font=("Arial", 16)).grid(row=0, column=1, padx=10, pady=4)

            ttk.Label(self.algo_param_frame, text="Scout Limit:").grid(row=1, column=0, sticky=tk.W, pady=4)
            ttk.Spinbox(self.algo_param_frame, from_=10, to=200, textvariable=self.limit_var,
                        width=10, font=("Arial", 16)).grid(row=1, column=1, padx=10, pady=4)

            ttk.Label(self.algo_param_frame, text="Workers:").grid(row=2, column=0, sticky=tk.W, pady=4)
            ttk.Spinbox(self.algo_param_frame, from_=1, to=16, textvariable=self.workers_var,
                        width=10, font=("Arial", 16)).grid(row=2, column=1, padx=10, pady=4)

        elif algorithm == "ACO":
            ttk.Label(self.algo_param_frame, text="Number of Ants:").grid(row=0, column=0, sticky=tk.W, pady=4)
            ttk.Spinbox(self.algo_param_frame, from_=5, to=100, textvariable=self.ants_var,
                        width=10, font=("Arial", 16)).grid(row=0, column=1, padx=10, pady=4)

            ttk.Label(self.algo_param_frame, text="Alpha (α):").grid(row=1, column=0, sticky=tk.W, pady=4)
            ttk.Spinbox(self.algo_param_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.alpha_var,
                        width=10, font=("Arial", 16)).grid(row=1, column=1, padx=10, pady=4)

            ttk.Label(self.algo_param_frame, text="Beta (β):").grid(row=2, column=0, sticky=tk.W, pady=4)
            ttk.Spinbox(self.algo_param_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.beta_var,
                        width=10, font=("Arial", 16)).grid(row=2, column=1, padx=10, pady=4)

            ttk.Label(self.algo_param_frame, text="Evaporation Rate:").grid(row=3, column=0, sticky=tk.W, pady=4)
            ttk.Spinbox(self.algo_param_frame, from_=0.01, to=0.99, increment=0.01, textvariable=self.evap_var,
                        width=10, font=("Arial", 16)).grid(row=3, column=1, padx=10, pady=4)

        elif algorithm == "GA":
            ttk.Label(self.algo_param_frame, text="Population Size:").grid(row=0, column=0, sticky=tk.W, pady=4)
            ttk.Spinbox(self.algo_param_frame, from_=5, to=100, textvariable=self.ga_pop_var,
                        width=10, font=("Arial", 16)).grid(row=0, column=1, padx=10, pady=4)

            ttk.Label(self.algo_param_frame, text="Mutation Rate:").grid(row=1, column=0, sticky=tk.W, pady=4)
            ttk.Spinbox(self.algo_param_frame, from_=0.01, to=0.99, increment=0.01, textvariable=self.mutation_var,
                        width=10, font=("Arial", 16)).grid(row=1, column=1, padx=10, pady=4)

    def setup_plots(self):
        # Create figures and canvases for plots
        self.fig1, self.ax1 = plt.subplots(figsize=(14, 10))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.fitness_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        self.fig2, self.ax2 = plt.subplots(figsize=(14, 10))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.bn_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    def run_optimization(self):
        try:
            # Load data if not already loaded
            if self.data is None or self.variables is None or self.score is None:
                self.data, self.variables, self.score = load_data(DEFAULT_DATASET)

            # Get common parameters
            algorithm = self.algorithm_var.get()
            iterations = self.iters_var.get()
            interval = self.interval_var.get()

            # Set current algorithm for BN visualization
            self.current_bn_algorithm = algorithm

            # Stop any existing animation
            if self.anim is not None:
                try:
                    self.anim.event_source.stop()
                except:
                    # If animation doesn't exist or is already stopped, just continue
                    pass
                self.anim = None

            # Run the selected algorithm
            if algorithm == "ABC":
                self.run_abc()
            elif algorithm == "ACO":
                self.run_aco()
            elif algorithm == "GA":
                self.run_ga()

            # Store results for comparison
            self.algorithm_results[algorithm]["fitness"] = self.fitness_evolution
            self.algorithm_results[algorithm]["edges"] = self.best_edges_history

            # Setup animation based on comparison setting
            if self.compare_var.get():
                self.setup_comparison_animation(interval)
            else:
                self.setup_single_algorithm_animation(interval)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            logging.error(f"Error running optimization: {e}")

    def run_abc(self):
        pop_size = self.pop_var.get()
        limit = self.limit_var.get()
        workers = self.workers_var.get()
        iterations = self.iters_var.get()

        optimizer = ABC_BN(
            data=self.data,
            variables=self.variables,
            score=self.score,
            pop_size=pop_size,
            limit=limit,
            num_iters=iterations,
            num_workers=workers,
            seed=DEFAULT_SEED
        )

        self.fitness_evolution, _, _, _, self.best_edges_history = optimizer.run()
        logging.info(f"ABC completed with final score: {-self.fitness_evolution[-1]}")

    def run_aco(self):
        # Import the ACO algorithm only when needed
        from aco_algo import AntColonyBN

        ants = self.ants_var.get()
        alpha = self.alpha_var.get()
        beta = self.beta_var.get()
        evap_rate = self.evap_var.get()
        iterations = self.iters_var.get()

        # Create a scoring function that works with networkx graphs
        def scoring_function(graph):
            edges = list(graph.edges())
            return -calculate_fitness(edges, self.variables, self.score)

        optimizer = AntColonyBN(
            num_ants=ants,
            iterations=iterations,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evap_rate,
            scoring_function=scoring_function
        )

        # Track history for animation
        self.fitness_evolution = []
        self.best_edges_history = []

        best_graph = None
        best_score = float('-inf')

        # Run optimization with tracking
        for i in range(iterations):
            temp_graph, temp_score = optimizer.run([i for i in range(1)])  # Just one iteration

            if temp_score > best_score:
                best_score = temp_score
                best_graph = temp_graph.copy()

            self.fitness_evolution.append(-best_score)  # Convert to BIC format (negative)
            self.best_edges_history.append(list(best_graph.edges()))

            if i % 5 == 0:
                logging.info(f"ACO iteration {i} - Best score: {best_score}")

        logging.info(f"ACO completed with final score: {best_score}")

    def run_ga(self):
        # Import the GA algorithm only when needed
        from ga_algo import GeneticAlgorithmBN

        pop_size = self.ga_pop_var.get()
        mutation_rate = self.mutation_var.get()
        iterations = self.iters_var.get()

        # Create a scoring function that works with networkx graphs
        def scoring_function(graph):
            edges = list(graph.edges())
            return -calculate_fitness(edges, self.variables, self.score)

        optimizer = GeneticAlgorithmBN(
            population_size=pop_size,
            generations=iterations,
            mutation_rate=mutation_rate,
            scoring_function=scoring_function
        )

        # Track history for animation
        self.fitness_evolution = []
        self.best_edges_history = []

        best_graph = None
        best_score = float('-inf')

        # Run optimization with tracking
        for i in range(iterations):
            if i == 0:
                temp_graph, temp_score = optimizer.run(self.variables)
                if temp_score > best_score:
                    best_score = temp_score
                    best_graph = temp_graph.copy()
            else:
                # Only incremental update is needed since run() updates internally
                temp_graph, temp_score = optimizer._random_dag(self.variables), 0

                if temp_score > best_score:
                    best_score = temp_score
                    best_graph = temp_graph.copy()

            self.fitness_evolution.append(-best_score)  # Convert to BIC format (negative)
            self.best_edges_history.append(list(best_graph.edges()))

            if i % 5 == 0:
                logging.info(f"GA iteration {i} - Best score: {best_score}")

        logging.info(f"GA completed with final score: {best_score}")

    def setup_single_algorithm_animation(self, interval):
        # Clear any existing animation
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except:
                pass
            self.anim = None

        # Prepare fitness plot
        self.ax1.clear()
        self.ax1.set_title(f"{self.algorithm_var.get()} Search: Best Fitness", fontsize=24)
        self.ax1.set_xlabel("Iteration", fontsize=18)
        self.ax1.set_ylabel("BIC Score", fontsize=18)
        self.ax1.grid(True)

        # Prepare BN plot
        self.ax2.clear()
        self.ax2.set_title("Learned Bayesian Network (Dynamic)", fontsize=24)
        self.ax2.axis('off')

        # Setup initial graph layout
        G0 = nx.DiGraph()
        G0.add_nodes_from(self.variables)
        pos = nx.spring_layout(G0, seed=DEFAULT_SEED)

        algorithm = self.algorithm_var.get()
        color = ALGORITHM_COLORS[algorithm]

        def update(frame):
            # Update fitness plot
            self.ax1.clear()
            self.ax1.set_title(f"{algorithm} Search: Best Fitness", fontsize=24)
            self.ax1.set_xlabel("Iteration", fontsize=18)
            self.ax1.set_ylabel("BIC Score", fontsize=18)
            self.ax1.grid(True)

            xs = list(range(frame + 1))
            ys = [-f for f in self.fitness_evolution[:frame + 1]]
            self.ax1.plot(xs, ys, linewidth=3, color=color, label=algorithm)
            self.ax1.legend()

            # Update BN plot
            self.ax2.clear()
            self.ax2.set_title(f"Learned BN (Iteration {frame})", fontsize=24)
            self.ax2.axis('off')

            G = nx.DiGraph()
            G.add_nodes_from(self.variables)
            G.add_edges_from(self.best_edges_history[frame])

            nx.draw_networkx(
                G, pos, ax=self.ax2,
                node_size=3000, node_color='lightblue',
                arrowsize=20, font_size=16
            )

            self.canvas1.draw()
            self.canvas2.draw()

        # Create animation
        self.anim = FuncAnimation(
            self.fig1, update,
            frames=len(self.fitness_evolution),
            interval=interval,
            repeat=False
        )

        self.canvas1.draw()
        self.canvas2.draw()

    def setup_comparison_animation(self, interval):
        """Setup animation for comparing multiple algorithms"""
        # Check which algorithms have data
        available_algorithms = [algo for algo in ["ABC", "ACO", "GA"]
                                if len(self.algorithm_results[algo]["fitness"]) > 0]

        if len(available_algorithms) == 0:
            messagebox.showinfo("Comparison", "No algorithm data available")
            return

        if len(available_algorithms) == 1:
            # If only one algorithm available, use standard animation
            self.setup_single_algorithm_animation(interval)
            return

        # Clear any existing animation
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except:
                pass
            self.anim = None

        # Find max iterations across all algorithms
        max_iterations = max([len(self.algorithm_results[algo]["fitness"])
                              for algo in available_algorithms])

        # Prepare fitness plot
        self.ax1.clear()
        self.ax1.set_title("Algorithm Comparison: Best Fitness", fontsize=24)
        self.ax1.set_xlabel("Iteration", fontsize=18)
        self.ax1.set_ylabel("BIC Score", fontsize=18)
        self.ax1.grid(True)

        # Prepare BN plot for current algorithm
        self.ax2.clear()
        self.ax2.set_title(f"Learned BN ({self.current_bn_algorithm})", fontsize=24)
        self.ax2.axis('off')

        # Setup initial graph layout
        G0 = nx.DiGraph()
        G0.add_nodes_from(self.variables)
        pos = nx.spring_layout(G0, seed=DEFAULT_SEED)

        # Lines for each algorithm
        lines = {}
        for algo in available_algorithms:
            lines[algo], = self.ax1.plot([], [], linewidth=3, color=ALGORITHM_COLORS[algo], label=algo)

        # Legend
        self.ax1.legend()

        def init():
            for algo in available_algorithms:
                lines[algo].set_data([], [])
            return list(lines.values())

        def update(frame):
            # Update each line in the fitness plot
            for algo in available_algorithms:
                fitness_data = self.algorithm_results[algo]["fitness"]
                if frame < len(fitness_data):
                    xs = list(range(frame + 1))
                    ys = [-f for f in fitness_data[:frame + 1]]
                    lines[algo].set_data(xs, ys)

            # Set limits based on all data
            self.ax1.set_xlim(0, max_iterations)

            # Find min and max Y values across all algorithms
            all_fitness = []
            for algo in available_algorithms:
                fitness_data = self.algorithm_results[algo]["fitness"]
                if fitness_data and frame < len(fitness_data):
                    all_fitness.extend([-f for f in fitness_data[:frame + 1]])

            if all_fitness:
                self.ax1.set_ylim(min(all_fitness) * 1.1, max(all_fitness) * 0.9)

            # Update BN plot for current algorithm
            if self.current_bn_algorithm in available_algorithms:
                edges_data = self.algorithm_results[self.current_bn_algorithm]["edges"]
                if frame < len(edges_data):
                    self.ax2.clear()
                    self.ax2.set_title(f"Learned BN ({self.current_bn_algorithm} - Iteration {frame})", fontsize=24)
                    self.ax2.axis('off')

                    G = nx.DiGraph()
                    G.add_nodes_from(self.variables)
                    G.add_edges_from(edges_data[frame])

                    nx.draw_networkx(
                        G, pos, ax=self.ax2,
                        node_size=3000, node_color='lightblue',
                        arrowsize=20, font_size=16
                    )

            self.canvas1.draw()
            self.canvas2.draw()
            return list(lines.values())

        # Create animation
        self.anim = FuncAnimation(
            self.fig1, update,
            frames=max_iterations,
            interval=interval,
            init_func=init,
            blit=True,
            repeat=False
        )

        self.canvas1.draw()
        self.canvas2.draw()

    def update_bn_visualization(self, event=None):
        """Update BN visualization based on selected algorithm"""
        if self.variables is None:
            return

        selected_algo = self.bn_algorithm_var.get()

        # If "Current" is selected, use the most recently run algorithm
        if selected_algo == "Current":
            selected_algo = self.current_bn_algorithm

        # Check if algorithm has data
        if selected_algo not in self.algorithm_results or not self.algorithm_results[selected_algo]["edges"]:
            messagebox.showinfo("Visualization", f"No data available for {selected_algo}")
            return

        # Setup initial graph layout
        G0 = nx.DiGraph()
        G0.add_nodes_from(self.variables)
        pos = nx.spring_layout(G0, seed=DEFAULT_SEED)

        # Get the last iteration's BN
        final_edges = self.algorithm_results[selected_algo]["edges"][-1]

        # Update BN plot with final structure
        self.ax2.clear()
        self.ax2.set_title(f"Learned BN ({selected_algo} - Final Structure)", fontsize=24)
        self.ax2.axis('off')

        G = nx.DiGraph()
        G.add_nodes_from(self.variables)
        G.add_edges_from(final_edges)

        nx.draw_networkx(
            G, pos, ax=self.ax2,
            node_size=3000, node_color='lightblue',
            arrowsize=20, font_size=16
        )

        self.canvas2.draw()

    def reset_comparison(self):
        """Reset all algorithm results"""
        self.algorithm_results = {
            "ABC": {"fitness": [], "edges": []},
            "ACO": {"fitness": [], "edges": []},
            "GA": {"fitness": [], "edges": []}
        }

        messagebox.showinfo("Reset", "All algorithm comparison data has been reset")

        # Clear plots
        self.ax1.clear()
        self.ax1.set_title("Algorithm Search: Best Fitness", fontsize=24)
        self.ax1.set_xlabel("Iteration", fontsize=18)
        self.ax1.set_ylabel("BIC Score", fontsize=18)
        self.ax1.grid(True)

        self.ax2.clear()
        self.ax2.set_title("Learned Bayesian Network", fontsize=24)
        self.ax2.axis('off')

        self.canvas1.draw()
        self.canvas2.draw()


def main():
    root = tk.Tk()
    app = BN_Explorer(root)
    return root


if __name__ == "__main__":
    root = main()
    root.mainloop()