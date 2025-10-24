from pgmpy.models import MarkovNetwork
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools

var = ['A1', 'A2', 'A3', 'A4', 'A5']
edges = [('A1', 'A2'), ('A1', 'A3'), ('A2', 'A4'), ('A2', 'A5'), ('A3', 'A4'), ('A4', 'A5')]
model = MarkovNetwork(edges)
pos = nx.circular_layout(model)
nx.draw(model, pos, with_labels=True, node_size=800, font_size=12)
plt.show()
cliques = list(nx.find_cliques(model))
print("Clicele sunt: ", cliques)


def function(clique, values):
    ind = [int(v[1]) for v in clique]
    vals = [values[v] for v in clique]
    s = sum(i * a for i, a in zip(ind, vals))
    return np.exp(s)


alv = list(itertools.product([-1, 1], repeat=5))
best_state = None
best_prod = -1

for av in alv:
    val = dict(zip(var, av))
    prod = np.prod([function(clique, val) for clique in cliques])
    if prod > best_prod:
        best_state = val
        best_prod = prod
print("Cea mai buna configuratie este:")
print(best_state)
print(best_prod)
