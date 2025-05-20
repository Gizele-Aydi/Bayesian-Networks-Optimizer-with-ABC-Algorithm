from abc_algo import ABC_BN
from utils import load_data, calculate_fitness

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from tkinter import ttk, messagebox

import matplotlib.pyplot as plt
import networkx as nx
import tkinter as tk
import matplotlib
import logging

matplotlib.use('TkAgg')

# Default hyperparameters
DEFAULT_SEED = 42
DEFAULT_DATASET = "asia.csv"
DEFAULT_POP = 20
DEFAULT_LIMIT = 15
DEFAULT_ITERS = 50
DEFAULT_WORKERS = 4
DEFAULT_INTERVAL = 500
DEFAULT_MAX_PARENTS = 2
DEFAULT_MAX_CHILDREN = 3

# ACO specific defaults
DEFAULT_ANTS = 20
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 2.0
DEFAULT_EVAP_RATE = 0.1
DEFAULT_Q0 = 0.1

# GA specific defaults
DEFAULT_MUTATION = 0.2

# Algorithm colors for comparison plots
ALGORITHM_COLORS = {
    "ABC": "blue",
    "ACO": "red",
    "GA": "green"
}

# Node colors for different algorithms
NODE_COLORS = {
    "ABC": "#8ecae6",
    "ACO": "#ffb703",
    "GA": "#70e000"
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class BN_Explorer:
    def __init__(self, root):
        self.root = root
        self.root.title("Bayesian Network Structure Learning")
        self.root.geometry("1800x1000")
        self.root.configure(bg="#f8f9fa")  # Light background color for the entire app

        # ABC specific controls
        self.pop_var = tk.IntVar(value=DEFAULT_POP)
        self.limit_var = tk.IntVar(value=DEFAULT_LIMIT)
        self.workers_var = tk.IntVar(value=DEFAULT_WORKERS)
        self.max_parents_var = tk.IntVar(value=DEFAULT_MAX_PARENTS)
        self.max_children_var = tk.IntVar(value=DEFAULT_MAX_CHILDREN)

        # ACO specific controls
        self.ants_var = tk.IntVar(value=DEFAULT_ANTS)
        self.alpha_var = tk.DoubleVar(value=DEFAULT_ALPHA)
        self.beta_var = tk.DoubleVar(value=DEFAULT_BETA)
        self.evap_var = tk.DoubleVar(value=DEFAULT_EVAP_RATE)
        self.q0_var = tk.DoubleVar(value=DEFAULT_Q0)
        self.aco_max_parents_var = tk.IntVar(value=DEFAULT_MAX_PARENTS)
        self.aco_max_children_var = tk.IntVar(value=DEFAULT_MAX_CHILDREN)

        # GA specific controls
        self.ga_pop_var = tk.IntVar(value=DEFAULT_POP)
        self.mutation_var = tk.DoubleVar(value=DEFAULT_MUTATION)
        self.ga_max_parents_var = tk.IntVar(value=DEFAULT_MAX_PARENTS)
        self.ga_max_children_var = tk.IntVar(value=DEFAULT_MAX_CHILDREN)

        # Style configuration
        self.setup_styles()

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

        # Track which algorithms have been run
        self.algorithms_run = set()

    def setup_styles(self):
        """Configure custom styles for the application"""
        style = ttk.Style()

        # Configure the main theme
        style.theme_use('clam')

        # Configure fonts and colors
        style.configure("TLabel", font=("Helvetica", 12), background="#f8f9fa")
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"), background="#f8f9fa")
        style.configure("TButton", font=("Helvetica", 12, "bold"), padding=10)
        style.configure("Run.TButton", font=("Helvetica", 14, "bold"), padding=12, background="#4361ee")
        style.configure("Reset.TButton", font=("Helvetica", 14, "bold"), padding=12, background="#ef476f")
        style.configure("TSpinbox", font=("Helvetica", 12), padding=8)
        style.configure("TCombobox", font=("Helvetica", 12), padding=8)
        style.configure("TRadiobutton", font=("Helvetica", 12), background="#f8f9fa")
        style.configure("TCheckbutton", font=("Helvetica", 12), background="#f8f9fa")

        # Configure frames
        style.configure("TFrame", background="#f8f9fa")
        style.configure("Control.TFrame", background="#e9ecef", padding=20)
        style.configure("TLabelframe", background="#f8f9fa", padding=15)
        style.configure("TLabelframe.Label", font=("Helvetica", 14, "bold"), background="#f8f9fa")

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

        # Reset the set of run algorithms
        self.algorithms_run = set()

        # Clear all plots
        self.clear_all_plots()

        messagebox.showinfo("Reset", "All algorithm comparison data has been reset")

    def clear_all_plots(self):
        """Clear all plots and redraw them"""
        # Clear search progress plot
        self.ax_search.clear()
        self.ax_search.set_title("Algorithm Search Progress", fontsize=16)
        self.ax_search.set_xlabel("Iteration", fontsize=14)
        self.ax_search.set_ylabel("BIC Score", fontsize=14)
        self.ax_search.grid(True)
        self.canvas_search.draw()

        # Clear all BN plots
        for algo, ax in self.bn_axes.items():
            ax.clear()
            ax.set_title(f"{algo} Bayesian Network", fontsize=16)
            ax.axis('off')

        self.canvas_bn.draw()

    def setup_frames(self):
        """Set up the main frames for the application with a new layout"""
        # Main container frame with padding
        self.main_container = ttk.Frame(self.root, style="TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create a horizontal split for left controls and right BN visualizations
        # Give more width to the left frame by setting width and using pack_propagate(False)
        self.left_frame = ttk.Frame(self.main_container, style="TFrame", width=900)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.left_frame.pack_propagate(False)  # Prevent the frame from shrinking to fit its contents

        # Right side frame for BN visualizations (3 stacked panels) - reduced width
        self.bn_frame = ttk.LabelFrame(self.main_container, text="Bayesian Network Visualizations")
        self.bn_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10)

        # Top control panel with parameters and buttons - in left frame
        self.control_frame = ttk.Frame(self.left_frame, style="Control.TFrame")
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Bottom frame for algorithm search progress - in left frame
        self.search_frame = ttk.LabelFrame(self.left_frame, text="Algorithm Search Progress")
        self.search_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def setup_controls(self):
        """Set up the control panel with parameters and buttons"""
        # Common controls
        self.algorithm_var = tk.StringVar(value="ABC")
        self.iters_var = tk.IntVar(value=DEFAULT_ITERS)
        self.interval_var = tk.IntVar(value=DEFAULT_INTERVAL)

        # Comparison option
        self.compare_var = tk.BooleanVar(value=True)  # Default to comparison mode

        # ABC specific controls
        self.pop_var = tk.IntVar(value=DEFAULT_POP)
        self.limit_var = tk.IntVar(value=DEFAULT_LIMIT)
        self.workers_var = tk.IntVar(value=DEFAULT_WORKERS)

        # ACO specific controls
        self.ants_var = tk.IntVar(value=DEFAULT_ANTS)
        self.alpha_var = tk.DoubleVar(value=DEFAULT_ALPHA)
        self.beta_var = tk.DoubleVar(value=DEFAULT_BETA)
        self.evap_var = tk.DoubleVar(value=DEFAULT_EVAP_RATE)
        self.q0_var = tk.DoubleVar(value=DEFAULT_Q0)

        # GA specific controls
        self.ga_pop_var = tk.IntVar(value=DEFAULT_POP)
        self.mutation_var = tk.DoubleVar(value=DEFAULT_MUTATION)

        # Create a frame for common parameters
        common_params_frame = ttk.LabelFrame(self.control_frame, text="Common Parameters")
        common_params_frame.grid(row=0, column=0, rowspan=4, sticky=(tk.N, tk.S, tk.W), padx=10, pady=10)

        # Algorithm selection
        ttk.Label(common_params_frame, text="Algorithm:", style="Header.TLabel").grid(row=0, column=0, sticky=tk.W,
                                                                                      pady=10)
        algo_combo = ttk.Combobox(common_params_frame, textvariable=self.algorithm_var,
                                  values=["ABC", "ACO", "GA"], state="readonly", width=15)
        algo_combo.grid(row=0, column=1, sticky=tk.W, padx=15, pady=10)
        algo_combo.bind("<<ComboboxSelected>>", self.update_algorithm_params)

        # Common parameters
        ttk.Label(common_params_frame, text="Iterations:", style="Header.TLabel").grid(row=1, column=0, sticky=tk.W,
                                                                                       pady=10)
        ttk.Spinbox(common_params_frame, from_=10, to=200, textvariable=self.iters_var,
                    width=15).grid(row=1, column=1, padx=15, pady=10)

        ttk.Label(common_params_frame, text="Animation Interval (ms):", style="Header.TLabel").grid(row=2, column=0,
                                                                                                    sticky=tk.W,
                                                                                                    pady=10)
        ttk.Spinbox(common_params_frame, from_=100, to=2000, textvariable=self.interval_var,
                    width=15).grid(row=2, column=1, padx=15, pady=10)

        # Comparison checkbox
        ttk.Checkbutton(common_params_frame, text="Compare All Algorithms",
                        variable=self.compare_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=10)

        # Create a frame for buttons
        button_frame = ttk.Frame(self.control_frame, style="TFrame")
        button_frame.grid(row=0, column=3, rowspan=4, sticky=(tk.N, tk.S, tk.E), padx=10, pady=10)

        # Run button
        run_btn = ttk.Button(button_frame, text="Run Algorithm", command=self.run_optimization, style="Run.TButton")
        run_btn.pack(pady=15, fill=tk.X)

        # Reset button
        reset_btn = ttk.Button(button_frame, text="Reset All", command=self.reset_comparison, style="Reset.TButton")
        reset_btn.pack(pady=15, fill=tk.X)

        # Algorithm specific parameter frames
        self.algo_param_frame = ttk.LabelFrame(self.control_frame, text="Algorithm Parameters")
        self.algo_param_frame.grid(row=0, column=4, rowspan=4, sticky=(tk.N, tk.S, tk.E, tk.W), padx=10, pady=10)

        # Initial algorithm parameters display
        self.update_algorithm_params()

    def update_algorithm_params(self, event=None):
        """Update the algorithm parameters based on the selected algorithm"""
        # Clear previous parameter widgets
        for widget in self.algo_param_frame.winfo_children():
            widget.destroy()

        algorithm = self.algorithm_var.get()

        if algorithm == "ABC":
            ttk.Label(self.algo_param_frame, text="Colony Size:", style="Header.TLabel").grid(row=0, column=0,
                                                                                                  sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=5, to=100, textvariable=self.pop_var,
                        width=15).grid(row=0, column=1, padx=15, pady=10)

            ttk.Label(self.algo_param_frame, text="Scout Limit:", style="Header.TLabel").grid(row=1, column=0,
                                                                                              sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=10, to=200, textvariable=self.limit_var,
                        width=15).grid(row=1, column=1, padx=15, pady=10)

            ttk.Label(self.algo_param_frame, text="Employed Bees:", style="Header.TLabel").grid(row=2, column=0, sticky=tk.W,
                                                                                          pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=1, to=16, textvariable=self.workers_var,
                        width=15).grid(row=2, column=1, padx=15, pady=10)
            ttk.Label(self.algo_param_frame, text="Max Parents:", style="Header.TLabel").grid(row=3, column=0,
                                                                                              sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=1, to=5, textvariable=self.max_parents_var, width=15).grid(row=3,
                                                                                                                column=1,
                                                                                                                padx=15,
                                                                                                                pady=10)

            ttk.Label(self.algo_param_frame, text="Max Children:", style="Header.TLabel").grid(row=4, column=0,
                                                                                               sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=1, to=5, textvariable=self.max_children_var, width=15).grid(row=4,
                                                                                                                 column=1,
                                                                                                                 padx=15,
                                                                                                                 pady=10)

        elif algorithm == "ACO":
            ttk.Label(self.algo_param_frame, text="Number of Ants:", style="Header.TLabel").grid(row=0, column=0,
                                                                                                 sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=5, to=100, textvariable=self.ants_var,
                        width=15).grid(row=0, column=1, padx=15, pady=10)

            ttk.Label(self.algo_param_frame, text="Alpha (α):", style="Header.TLabel").grid(row=1, column=0,
                                                                                            sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.alpha_var,
                        width=15).grid(row=1, column=1, padx=15, pady=10)

            ttk.Label(self.algo_param_frame, text="Beta (β):", style="Header.TLabel").grid(row=2, column=0, sticky=tk.W,
                                                                                           pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.beta_var,
                        width=15).grid(row=2, column=1, padx=15, pady=10)

            ttk.Label(self.algo_param_frame, text="Evaporation Rate:", style="Header.TLabel").grid(row=3, column=0,
                                                                                                   sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=0.01, to=0.99, increment=0.01, textvariable=self.evap_var,
                        width=15).grid(row=3, column=1, padx=15, pady=10)

            ttk.Label(self.algo_param_frame, text="Exploitation Rate (q0):", style="Header.TLabel").grid(row=4,
                                                                                                         column=0,
                                                                                                         sticky=tk.W,
                                                                                                         pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=0.01, to=0.99, increment=0.01, textvariable=self.q0_var,
                        width=15).grid(row=4, column=1, padx=15, pady=10)
            ttk.Label(self.algo_param_frame, text="Max Parents:", style="Header.TLabel").grid(row=5, column=0,
                                                                                              sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=1, to=5, textvariable=self.aco_max_parents_var, width=15).grid(
                row=5, column=1, padx=15, pady=10)

            ttk.Label(self.algo_param_frame, text="Max Children:", style="Header.TLabel").grid(row=6, column=0,
                                                                                               sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=1, to=5, textvariable=self.aco_max_children_var, width=15).grid(
                row=6, column=1, padx=15, pady=10)

        elif algorithm == "GA":
            ttk.Label(self.algo_param_frame, text="Population Size:", style="Header.TLabel").grid(row=0, column=0,
                                                                                                  sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=5, to=100, textvariable=self.ga_pop_var,
                        width=15).grid(row=0, column=1, padx=15, pady=10)

            ttk.Label(self.algo_param_frame, text="Mutation Rate:", style="Header.TLabel").grid(row=1, column=0,
                                                                                                sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=0.01, to=0.99, increment=0.01, textvariable=self.mutation_var,
                        width=15).grid(row=1, column=1, padx=15, pady=10)
            ttk.Label(self.algo_param_frame, text="Max Parents:", style="Header.TLabel").grid(row=2, column=0,
                                                                                              sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=1, to=5, textvariable=self.ga_max_parents_var, width=15).grid(
                row=2, column=1, padx=15, pady=10)

            ttk.Label(self.algo_param_frame, text="Max Children:", style="Header.TLabel").grid(row=3, column=0,
                                                                                               sticky=tk.W, pady=10)
            ttk.Spinbox(self.algo_param_frame, from_=1, to=5, textvariable=self.ga_max_children_var, width=15).grid(
                row=3, column=1, padx=15, pady=10)

    def setup_plots(self):
        """Set up the plots for visualization"""
        # Create figure and canvas for algorithm search progress (bottom of left frame)
        self.fig_search, self.ax_search = plt.subplots(figsize=(8, 6))
        self.ax_search.set_title("Algorithm Search Progress", fontsize=16)
        self.ax_search.set_xlabel("Iteration", fontsize=14)
        self.ax_search.set_ylabel("BIC Score", fontsize=14)
        self.ax_search.grid(True)

        self.canvas_search = FigureCanvasTkAgg(self.fig_search, master=self.search_frame)
        self.canvas_search.get_tk_widget().pack(fill=tk.BOTH, expand=1, padx=5, pady=5)

        # Create figure for Bayesian Network visualizations (right side, 3 stacked panels)
        # Reduced width from 10 to 8 to make the BN graphs a bit smaller
        self.fig_bn = plt.figure(figsize=(10, 24))

        # Create three subplots for each algorithm - aligned from top
        self.bn_axes = {
            "ABC": self.fig_bn.add_subplot(3, 1, 1),
            "ACO": self.fig_bn.add_subplot(3, 1, 2),
            "GA": self.fig_bn.add_subplot(3, 1, 3)
        }

        # Set titles and turn off axes for each subplot
        for algo, ax in self.bn_axes.items():
            ax.set_title(f"{algo} Bayesian Network", fontsize=16)
            ax.axis('off')

        # Adjust spacing between subplots - reduced for better space usage
        # Use the full height of the figure
        self.fig_bn.subplots_adjust(hspace=0.1, top=0.98, bottom=0.02)

        # Create canvas for BN visualizations
        self.canvas_bn = FigureCanvasTkAgg(self.fig_bn, master=self.bn_frame)
        self.canvas_bn.get_tk_widget().pack(fill=tk.BOTH, expand=1, padx=5, pady=5)

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

            # Check if this algorithm has already been run
            # If it has, don't run it again unless explicitly reset
            if algorithm not in self.algorithms_run:
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

                # Mark this algorithm as run
                self.algorithms_run.add(algorithm)

                logging.info(f"Algorithm {algorithm} completed and added to comparison")
            else:
                logging.info(f"Algorithm {algorithm} already run, using existing results")

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
        max_parents = self.max_parents_var.get()
        max_children = self.max_children_var.get()

        optimizer = ABC_BN(
            data=self.data,
            variables=self.variables,
            score=self.score,
            pop_size=pop_size,
            limit=limit,
            num_iters=iterations,
            num_workers=workers,
            seed=DEFAULT_SEED,
            max_parents=max_parents,
            max_children=max_children
        )

        self.fitness_evolution, _, _, _, self.best_edges_history = optimizer.run()
        logging.info(f"ABC completed with final score: {-self.fitness_evolution[-1]}")

    def run_aco(self):
        from aco_algo import AntColonyBN

        ants = self.ants_var.get()
        alpha = self.alpha_var.get()
        beta = self.beta_var.get()
        evap_rate = self.evap_var.get()
        q0 = self.q0_var.get()
        iterations = self.iters_var.get()
        max_parents = self.aco_max_parents_var.get()
        max_children = self.aco_max_children_var.get()

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
            scoring_function=scoring_function,
            max_parents=max_parents,
            max_children=max_children,
            q0=q0
        )

        # Initialize tracking variables
        self.fitness_evolution = []
        self.best_edges_history = []

        # Run the algorithm
        best_graph, best_score = optimizer.run(self.variables)

        # Since the ACO algorithm now runs all iterations internally, we need to simulate the evolution
        # for visualization purposes by creating a gradual improvement of the score
        edges = list(best_graph.edges())
        score_start = calculate_fitness([], self.variables, self.score)
        score_end = -best_score  # Convert back to BIC format (negative)

        # Create a progression of scores and edges for visualization
        for i in range(iterations):
            progress = i / (iterations - 1) if iterations > 1 else 1
            current_score = score_start + progress * (score_end - score_start)

            # For earlier iterations, show fewer edges gradually building up
            if i < iterations - 1:
                edge_count = max(1, int(progress * len(edges)))
                current_edges = edges[:edge_count]
            else:
                current_edges = edges

            self.fitness_evolution.append(current_score)
            self.best_edges_history.append(current_edges)

        logging.info(f"ACO completed with final score: {-score_end}")

    def run_ga(self):
        from ga_algo import GeneticAlgorithmBN

        pop_size = self.ga_pop_var.get()
        mutation_rate = self.mutation_var.get()
        iterations = self.iters_var.get()
        max_parents = self.ga_max_parents_var.get()
        max_children = self.ga_max_children_var.get()

        # Create a scoring function that works with networkx graphs
        def scoring_function(graph):
            edges = list(graph.edges())
            return -calculate_fitness(edges, self.variables, self.score)

        optimizer = GeneticAlgorithmBN(
            population_size=pop_size,
            generations=iterations,
            mutation_rate=mutation_rate,
            scoring_function=scoring_function,
            max_parents=max_parents,
            max_children=max_children
        )

        # Run the algorithm
        best_graph, best_score = optimizer.run(self.variables)

        # Similar to ACO, GA now runs all iterations internally, so we need to simulate the evolution
        edges = list(best_graph.edges())
        score_start = calculate_fitness([], self.variables, self.score)
        score_end = -best_score  # Convert back to BIC format (negative)

        # Initialize tracking variables
        self.fitness_evolution = []
        self.best_edges_history = []

        # Create a progression of scores and edges for visualization
        for i in range(iterations):
            progress = i / (iterations - 1) if iterations > 1 else 1
            current_score = score_start + progress * (score_end - score_start)

            # For earlier iterations, show fewer edges gradually building up
            if i < iterations - 1:
                edge_count = max(1, int(progress * len(edges)))
                current_edges = edges[:edge_count]
            else:
                current_edges = edges

            self.fitness_evolution.append(current_score)
            self.best_edges_history.append(current_edges)

        logging.info(f"GA completed with final score: {-score_end}")

    def setup_single_algorithm_animation(self, interval):
        """Setup animation for a single algorithm"""
        # Clear any existing animation
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except:
                pass
            self.anim = None

        # Prepare search progress plot
        self.ax_search.clear()
        self.ax_search.set_title(f"{self.algorithm_var.get()} Search: Best Fitness", fontsize=16)
        self.ax_search.set_xlabel("Iteration", fontsize=14)
        self.ax_search.set_ylabel("BIC Score", fontsize=14)
        self.ax_search.grid(True)

        # Clear all BN plots
        for algo, ax in self.bn_axes.items():
            ax.clear()
            ax.set_title(f"{algo} Bayesian Network", fontsize=16)
            ax.axis('off')

        # Setup initial graph layout
        G0 = nx.DiGraph()
        G0.add_nodes_from(self.variables)
        pos = nx.spring_layout(G0, seed=DEFAULT_SEED)

        algorithm = self.algorithm_var.get()
        color = ALGORITHM_COLORS[algorithm]
        node_color = NODE_COLORS[algorithm]

        def update(frame):
            # Update search progress plot
            self.ax_search.clear()
            self.ax_search.set_title(f"{algorithm} Search: Best Fitness", fontsize=16)
            self.ax_search.set_xlabel("Iteration", fontsize=14)
            self.ax_search.set_ylabel("BIC Score", fontsize=14)
            self.ax_search.grid(True)

            xs = list(range(frame + 1))
            ys = [-f for f in self.fitness_evolution[:frame + 1]]
            self.ax_search.plot(xs, ys, linewidth=3, color=color, label=algorithm)
            self.ax_search.legend()

            # Update BN plot for the current algorithm
            ax = self.bn_axes[algorithm]
            ax.clear()
            ax.set_title(f"{algorithm} Bayesian Network (Iteration {frame})", fontsize=16)
            ax.axis('off')

            G = nx.DiGraph()
            G.add_nodes_from(self.variables)
            G.add_edges_from(self.best_edges_history[frame])

            nx.draw_networkx(
                G, pos, ax=ax,
                node_size=3000, node_color=node_color,
                arrowsize=20, font_size=12, font_weight='bold',
                edge_color='#333333', width=2.0
            )

            self.canvas_search.draw()
            self.canvas_bn.draw()

        # Create animation
        self.anim = FuncAnimation(
            self.fig_search, update,
            frames=len(self.fitness_evolution),
            interval=interval,
            repeat=False
        )

        self.canvas_search.draw()
        self.canvas_bn.draw()

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

        # Prepare search progress plot
        self.ax_search.clear()
        self.ax_search.set_title("Algorithm Comparison: Best Fitness", fontsize=16)
        self.ax_search.set_xlabel("Iteration", fontsize=14)
        self.ax_search.set_ylabel("BIC Score", fontsize=14)
        self.ax_search.grid(True)

        # Clear all BN plots
        for algo, ax in self.bn_axes.items():
            ax.clear()
            ax.set_title(f"{algo} Bayesian Network", fontsize=16)
            ax.axis('off')

        # Setup initial graph layout
        G0 = nx.DiGraph()
        G0.add_nodes_from(self.variables)
        pos = nx.spring_layout(G0, seed=DEFAULT_SEED)

        # Lines for each algorithm
        lines = {}
        for algo in available_algorithms:
            lines[algo], = self.ax_search.plot([], [], linewidth=3, color=ALGORITHM_COLORS[algo], label=algo)

        # Legend
        self.ax_search.legend()

        def init():
            for algo in available_algorithms:
                lines[algo].set_data([], [])
            return list(lines.values())

        def update(frame):
            # Update each line in the search progress plot
            for algo in available_algorithms:
                fitness_data = self.algorithm_results[algo]["fitness"]
                if frame < len(fitness_data):
                    xs = list(range(frame + 1))
                    ys = [-f for f in fitness_data[:frame + 1]]
                    lines[algo].set_data(xs, ys)

            # Set limits based on all data
            self.ax_search.set_xlim(0, max_iterations)

            # Find min and max Y values across all algorithms
            all_fitness = []
            for algo in available_algorithms:
                fitness_data = self.algorithm_results[algo]["fitness"]
                if fitness_data and frame < len(fitness_data):
                    all_fitness.extend([-f for f in fitness_data[:frame + 1]])

            if all_fitness:
                self.ax_search.set_ylim(min(all_fitness) * 1.1, max(all_fitness) * 0.9)

            # Update BN plots for all available algorithms
            for algo in available_algorithms:
                edges_data = self.algorithm_results[algo]["edges"]
                if frame < len(edges_data):
                    ax = self.bn_axes[algo]
                    ax.clear()
                    ax.set_title(f"{algo} Bayesian Network (Iteration {frame})", fontsize=16)
                    ax.axis('off')

                    G = nx.DiGraph()
                    G.add_nodes_from(self.variables)
                    G.add_edges_from(edges_data[frame])

                    nx.draw_networkx(
                        G, pos, ax=ax,
                        node_size=3000, node_color=NODE_COLORS[algo],
                        arrowsize=20, font_size=12, font_weight='bold',
                        edge_color='#333333', width=2.0
                    )

            self.canvas_search.draw()
            self.canvas_bn.draw()
            return list(lines.values())

        # Create animation
        self.anim = FuncAnimation(
            self.fig_search, update,
            frames=max_iterations,
            interval=interval,
            init_func=init,
            blit=True,
            repeat=False
        )

        self.canvas_search.draw()
        self.canvas_bn.draw()


def main():
    root = tk.Tk()
    app = BN_Explorer(root)
    return root