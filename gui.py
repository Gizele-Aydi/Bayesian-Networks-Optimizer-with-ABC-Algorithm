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
        self.best_edges_history = []
        self.fitness_evolution = []

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

        # Run button
        run_btn = ttk.Button(self.control_frame, text="Run Algorithm", command=self.run_optimization)
        run_btn.grid(row=3, column=0, columnspan=2, pady=15, sticky=tk.W)

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

            # Run the selected algorithm
            if algorithm == "ABC":
                self.run_abc()
            elif algorithm == "ACO":
                self.run_aco()
            elif algorithm == "GA":
                self.run_ga()

            # Setup animation
            self.setup_animation(interval)

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

    def setup_animation(self, interval):
        # Clear any existing animation
        if self.anim is not None:
            self.anim.event_source.stop()

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

        def update(frame):
            # Update fitness plot
            self.ax1.clear()
            self.ax1.set_title(f"{self.algorithm_var.get()} Search: Best Fitness", fontsize=24)
            self.ax1.set_xlabel("Iteration", fontsize=18)
            self.ax1.set_ylabel("BIC Score", fontsize=18)
            self.ax1.grid(True)

            xs = list(range(frame + 1))
            ys = [-f for f in self.fitness_evolution[:frame + 1]]
            self.ax1.plot(xs, ys, linewidth=3)

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


def main():
    root = tk.Tk()
    app = BN_Explorer(root)
    return root


if __name__ == "__main__":
    root = main()
    root.mainloop()