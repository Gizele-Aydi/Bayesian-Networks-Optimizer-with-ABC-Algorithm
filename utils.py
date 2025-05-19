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


def exceeds_parent_limit(edges, variables, max_parents=10):
    parent_count = {var: 0 for var in variables}
    for u, v in edges:
        parent_count[v] += 1
        if parent_count[v] > max_parents:
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
