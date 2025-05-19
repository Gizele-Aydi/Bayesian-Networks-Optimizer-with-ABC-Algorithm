from abc_algo import ABC_BN
from utils import load_data

from tkinter import ttk, messagebox
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import networkx as nx
import tkinter as tk
import logging
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')

# === Default Hyperparameters ===
DEFAULT_SEED     = 42
DEFAULT_POP      = 20
DEFAULT_LIMIT    = 50
DEFAULT_ITERS    = 50
DEFAULT_WORKERS  = 8
DEFAULT_INTERVAL = 500

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def init_gui():
    root = tk.Tk()
    root.title("ABC-BN Interactive Explorer")
    root.geometry("1800x1000")  # wider to support side-by-side plots

    # Style adjustments for visibility
    style = ttk.Style(root)
    style.configure("TLabel",   font=("Arial", 30))
    style.configure("TButton",  font=("Arial", 30))
    style.configure("TSpinbox", font=("Arial", 30))

    # Frames
    control_frame = ttk.Frame(root, padding=15)
    control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

    fitness_frame = ttk.LabelFrame(root, text="ABC Search", padding=10)
    fitness_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

    bn_frame = ttk.LabelFrame(root, text="Learned Bayesian Network", padding=10)
    bn_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=10)

    # Control variables
    pop_var      = tk.IntVar(value=DEFAULT_POP)
    limit_var    = tk.IntVar(value=DEFAULT_LIMIT)
    iters_var    = tk.IntVar(value=DEFAULT_ITERS)
    workers_var  = tk.IntVar(value=DEFAULT_WORKERS)
    interval_var = tk.IntVar(value=DEFAULT_INTERVAL)

    def add_control(label_text, var, frm, to, row):
        ttk.Label(control_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=4)
        sb = ttk.Spinbox(control_frame, from_=frm, to=to, textvariable=var, width=10, font=("Arial", 30))
        sb.grid(row=row, column=1, padx=10, pady=4)

    add_control("Population Size:", pop_var, 5,    100, 0)
    add_control("Scout Limit:",    limit_var, 10,   200, 1)
    add_control("Iterations:",     iters_var, 10,   200, 2)
    add_control("Workers:",        workers_var, 1,  16,  3)
    add_control("Interval (ms):",  interval_var, 100, 2000, 4)

    # Create figures and canvases with larger size for side-by-side display
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    canvas1 = FigureCanvasTkAgg(fig1, master=fitness_frame)
    canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    fig2, ax2 = plt.subplots(figsize=(14, 10))
    canvas2 = FigureCanvasTkAgg(fig2, master=bn_frame)
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    anim = None

    def run_optimization():
        nonlocal anim
        try:
            # Read parameters
            pop      = pop_var.get()
            limit    = limit_var.get()
            iters    = iters_var.get()
            workers  = workers_var.get()
            interval = interval_var.get()
            seed     = DEFAULT_SEED

            data, variables, score = load_data("asia.csv")
            optimizer = ABC_BN(
                data=data,
                variables=variables,
                score=score,
                pop_size=pop,
                limit=limit,
                num_iters=iters,
                num_workers=workers,
                seed=seed
            )
            fitness_evol, _, _, _, edges_hist = optimizer.run()

            # Prepare fitness plot
            ax1.clear()
            ax1.set_title("ABC Search: Best Fitness", fontsize=32)
            ax1.set_xlabel("Iteration", fontsize=22)
            ax1.set_ylabel("BIC Score", fontsize=22)
            ax1.grid(True)

            # Prepare BN plot
            ax2.clear()
            ax2.set_title("Learned Bayesian Network (Dynamic)", fontsize=32)
            ax2.axis('off')

            G0 = nx.DiGraph()
            G0.add_nodes_from(variables)
            pos = nx.spring_layout(G0, seed=seed)

            def update(frame):
                ax1.clear()
                ax1.set_title("ABC Search: Best Fitness", fontsize=24)
                ax1.set_xlabel("Iteration", fontsize=22)
                ax1.set_ylabel("BIC Score", fontsize=22)
                ax1.grid(True)
                xs = list(range(frame + 1))
                ys = [-f for f in fitness_evol[:frame + 1]]
                ax1.plot(xs, ys, linewidth=4)

                ax2.clear()
                ax2.set_title(f"Learned BN (Iteration {frame})", fontsize=24)
                ax2.axis('off')
                G = nx.DiGraph()
                G.add_nodes_from(variables)
                G.add_edges_from(edges_hist[frame])
                nx.draw_networkx(
                    G, pos, ax=ax2,
                    node_size=3000, node_color='lightblue',
                    arrowsize=20, font_size=16
                )

                canvas1.draw()
                canvas2.draw()

            anim = FuncAnimation(
                fig1, update,
                frames=len(fitness_evol),
                interval=interval,
                repeat=False
            )
            canvas1.draw()
            canvas2.draw()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    run_btn = ttk.Button(control_frame, text="Run", command=run_optimization)
    run_btn.grid(row=5, column=0, columnspan=2, pady=15)

    return root
