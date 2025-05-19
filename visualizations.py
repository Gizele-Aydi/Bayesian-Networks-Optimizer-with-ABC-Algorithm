import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation, PillowWriter

from IPython.display import Image, display

def animate_abc_and_bn(
    fitness_evolution,
    best_edges_history,
    variables,
    interval=500,
    seed=42,
    save_gif_path="abc_bn_search.gif"
):
    """
    Top: best fitness curve.
    Bottom: evolving Bayesian network.

    Saves a GIF and displays it inline (in IDE/Jupyter).
    """
    # --- Static Best Fitness plot ---
    plt.figure(figsize=(8, 4))
    plt.plot([-f for f in fitness_evolution], label="Best BIC Score")
    plt.xlabel("Iteration")
    plt.ylabel("BIC Score")
    plt.title("Evolution of Best BIC Score")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Static BN plot ---
    G_final = nx.DiGraph()
    G_final.add_nodes_from(variables)
    G_final.add_edges_from(best_edges_history[-1])
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_final, seed=seed, k=2.5)  # More spacing
    nx.draw_networkx(G_final, pos, node_size=500, node_color='skyblue', arrowsize=12, font_size=10)
    plt.title("Learned Bayesian Network (Final)")
    plt.axis('off')
    plt.show()

    # --- Animation Setup ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    plt.tight_layout(pad=4)

    ax1.set_title("ABC Search: Best Fitness Over Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("BIC Score")
    line_best, = ax1.plot([], [], label="Best BIC Score")
    ax1.legend()
    ax1.grid(True)

    # Consistent layout for all BN graphs
    G0 = nx.DiGraph()
    G0.add_nodes_from(variables)
    pos = nx.spring_layout(G0, seed=seed, k=2.5)  # More spacing

    def init():
        line_best.set_data([], [])
        ax2.clear()
        ax2.set_title("Learned BN (Iteration 0)")
        ax2.axis('off')
        nx.draw_networkx_nodes(G0, pos, ax=ax2, node_size=500, node_color='skyblue')
        return line_best,

    def update(frame):
        xs = list(range(frame + 1))
        line_best.set_data(xs, [-f for f in fitness_evolution[:frame + 1]])
        ax1.set_xlim(0, len(fitness_evolution))
        ax1.set_ylim(min([-f for f in fitness_evolution]) * 1.1,
                     max([-f for f in fitness_evolution]) * 0.9)

        ax2.clear()
        ax2.set_title(f"Learned BN (Iteration {frame})")
        ax2.axis('off')
        G = nx.DiGraph()
        G.add_nodes_from(variables)
        G.add_edges_from(best_edges_history[frame])
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=500, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, ax=ax2, arrowsize=12, alpha=0.7)
        nx.draw_networkx_labels(G, pos, ax=ax2, font_size=10)
        return line_best,

    anim = FuncAnimation(fig, update, frames=len(fitness_evolution), init_func=init,
                         blit=False, interval=interval, repeat=False)

    try:
        pw = PillowWriter(fps=1000 / interval)
        anim.save(save_gif_path, writer=pw)
        print(f"Animation GIF saved to {save_gif_path}")
        display(Image(filename=save_gif_path))
    except Exception as e:
        print(f"Could not save/display GIF: {e}")

    plt.show()
    return anim


