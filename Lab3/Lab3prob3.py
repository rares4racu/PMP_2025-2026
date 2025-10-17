import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from math import comb
from pgmpy.inference import VariableElimination
import networkx as nx

np.random.seed(42)
sim = 10000
p0 = 0
p1 = 0
for i in range(sim):
    start = np.random.choice([0, 1])
    n = np.random.randint(1, 7)
    if start == 0:
        ph = 4 / 7
    else:
        ph = 1 / 2
    m = np.random.binomial(2 * n, ph)
    if n >= m:
        if start == 0:
            p0 += 1
        else:
            p1 += 1
    else:
        if start == 0:
            p1 += 1
        else:
            p0 += 1
prob_p0 = p0 / sim
prob_p1 = p1 / sim
print("The probability of p0 to win is: ", prob_p0)
print("The probability of p1 to win is: ", prob_p1)

game_model = DiscreteBayesianNetwork([('S', 'N'), ('S', 'M'), ('N', 'M')])
pos = nx.circular_layout(game_model)
nx.draw(game_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)

CPD_S = TabularCPD(variable='S', variable_card=2, values=[[0.5], [0.5]])
CPD_N = TabularCPD(variable='N', variable_card=6, values=[[1 / 6] * 2] * 6, evidence=['S'], evidence_card=[2])
vdie = np.arange(1, 7)
start = [0, 1]
pcoin = [1 / 2, 4 / 7]
ph1 = []
for s in start:
    p = pcoin[s]
    for die in vdie:
        n = 2 * die
        h1 = comb(n, 1) * (p ** 1) * ((1 - p) ** (n - 1))
        ph1.append(h1)

CPD_M = TabularCPD(variable='M', variable_card=2, values=[[1 - p for p in ph1], ph1], evidence=['S', 'N'],
                   evidence_card=[2, 6])
game_model.add_cpds(CPD_S, CPD_N, CPD_M)
# print(game_model.check_model())

infer = VariableElimination(game_model)
posterior_p = infer.query(['S'], evidence={'M': 1})
print("Posterior probability distribution:")
print(posterior_p)
