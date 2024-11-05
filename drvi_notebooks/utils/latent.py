import numpy as np
from scipy.cluster import hierarchy


def find_optimal_var_ordering(node_activity, metric='euclidean'):
    X = node_activity.T
    if metric.endswith('+'):
        metric = metric[:-1]
        X = np.abs(X)
    Z = hierarchy.linkage(X, metric=metric)
    return hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, X))


def set_optimal_ordering(embed, key_added='optimal_var_order', **kwargs):
    optimal_ordering = find_optimal_var_ordering(embed.X, **kwargs)
    embed.uns['optimal_var_order'] = optimal_ordering
    embed.var[key_added] = np.vectorize({o: i for i, o in enumerate(optimal_ordering)}.get)(np.arange(len(optimal_ordering)))
