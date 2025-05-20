import pandas as pd
from pgmpy.estimators import BicScore
from pgmpy.base import DAG


def load_data(path):
    data = pd.read_csv(path)
    variables = data.columns.tolist()
    score = BicScore(data)
    return data, variables, score


def is_acyclic(edges, variables):
    try:
        temp = DAG()
        temp.add_nodes_from(variables)
        temp.add_edges_from(edges)
        return True
    except Exception:
        return False


def exceeds_parent_limit(edges, variables, max_parents=10, max_children=None):
    """
    Check if any node exceeds the maximum allowed parents or children.

    Parameters:
    -----------
    edges : list of tuples
        List of edge tuples (parent, child)
    variables : list
        List of all node variables
    max_parents : int, default=10
        Maximum number of parents allowed for any node
    max_children : int or None, default=None
        Maximum number of children allowed for any node. If None, no limit is applied.

    Returns:
    --------
    bool
        True if any constraints are exceeded, False otherwise
    """
    # Check parent constraint
    parent_count = {var: 0 for var in variables}

    # Check children constraint only if specified
    child_count = {var: 0 for var in variables} if max_children is not None else None

    for parent, child in edges:
        parent_count[child] += 1

        # Check if parent limit exceeded
        if parent_count[child] > max_parents:
            return True

        # Check children limit if specified
        if child_count is not None:
            child_count[parent] += 1
            if child_count[parent] > max_children:
                return True

    return False


def calculate_fitness(edges, variables, score):
    from pgmpy.models import BayesianNetwork
    try:
        model = BayesianNetwork()
        model.add_nodes_from(variables)
        model.add_edges_from(edges)
        return -score.score(model)
    except Exception:
        return float('inf')