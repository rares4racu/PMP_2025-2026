import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

np.random.seed(42)
model = DiscreteBayesianNetwork([('O', 'H'), ('O', 'W'), ('H', 'R'), ('W', 'R'), ('H', 'E'), ('R', 'C')])
pos = nx.circular_layout(model)
nx.draw(model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.show()

CPD_O = TabularCPD(variable='O', variable_card=2, values=[[0.3], [0.7]])
CPD_H = TabularCPD(variable='H', variable_card=2, values=[[0.9, 0.2], [0.1, 0.8]], evidence=['O'], evidence_card=[2])
CPD_W = TabularCPD(variable='W', variable_card=2, values=[[0.1, 0.6], [0.9, 0.4]], evidence=['O'], evidence_card=[2])
CPD_R = TabularCPD(variable='R', variable_card=2, values=[[0.6, 0.9, 0.3, 0.5], [0.4, 0.1, 0.7, 0.5]],
                   evidence=['H', 'W'], evidence_card=[2, 2])
CPD_E = TabularCPD(variable='E', variable_card=2, values=[[0.8, 0.2], [0.2, 0.8]], evidence=['H'], evidence_card=[2])
CPD_C = TabularCPD(variable='C', variable_card=2, values=[[0.85, 0.40], [0.15, 0.60]], evidence=['R'],
                   evidence_card=[2])
model.add_cpds(CPD_O, CPD_H, CPD_W, CPD_R, CPD_E, CPD_C)
infer = VariableElimination(model)
posterior_h = infer.query(['H'], evidence={'C': 0})
print(posterior_h)
# P(H = yes | C = comfortable) = 0.4823
posterior_e = infer.query(['E'], evidence={'C': 0})
print(posterior_e)
# P(E = high | C = comfortable) = 0.4894
posterior_hw = infer.query(['H', 'W'], evidence={'C': 0})
print(posterior_hw)

print(model.local_independencies(['W']))
# Nu se regaseste (W ⟂ E | H).
print(model.local_independencies(['O', 'C', 'R']))
# Regasim (C ⟂ O | R) deci da O este independent de C cand este dat R
