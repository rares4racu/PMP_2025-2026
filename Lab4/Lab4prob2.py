import numpy as np
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
from pgmpy.models import MarkovNetwork

lambda_reg = 4
np.random.seed(42)
original = np.random.choice([0, 1], size=(5, 5))
print("Original:\n", original)
noisy = original.copy()
noise = int(0.1 * 5 * 5)
noise_pixels = np.random.choice(25, noise, replace=False)
for n in noise_pixels:
    i, j = divmod(n, 5)
    noisy[i, j] = 1 - noisy[i, j]
print("Noisy:\n", noisy)
model = MarkovNetwork()
nodes = [(i, j) for i in range(5) for j in range(5)]
model.add_nodes_from(nodes)


def neighbours(i, j):
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni = i + di
        nj = j + dj
        if 0 <= ni < 5 and 0 <= nj < 5:
            yield ni, nj


edges = []
for i in range(5):
    for j in range(5):
        for ni, nj in neighbours(i, j):
            if ((ni, nj), (i, j)) not in edges and ((i, j), (ni, nj)) not in edges:
                edges.append(((i, j), (ni, nj)))

model.add_edges_from(edges)

values = [0, 1]
factors = []
for node in nodes:
    y = noisy[node]
    potential_values = np.array([np.exp(-lambda_reg * (x - y) ** 2) for x in values])
    factor = DiscreteFactor(variables=[node], cardinality=[2], values=potential_values)
    model.add_factors(factor)

for edge in edges:
    n1, n2 = edge
    potential_values = np.zeros((2, 2))
    for i, xi in enumerate(values):
        for j, xj in enumerate(values):
            potential_values[i, j] = np.exp(-(xi - xj) ** 2)
    factor = DiscreteFactor(variables=[n1, n2], cardinality=[2, 2], values=potential_values)
    model.add_factors(factor)

bp = BeliefPropagation(model)
map_query = bp.map_query(variables=nodes)
clean_image = np.zeros(shape=(5, 5), dtype=int)
for node, val in map_query.items():
    clean_image[node[0], node[1]] = val
print("Clean:\n", clean_image)
