from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx

email_model = DiscreteBayesianNetwork([('S', 'O'), ('S', 'L'), ('S', 'M'), ('L', 'M')])
pos = nx.circular_layout(email_model)
nx.draw(email_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)

CPD_S = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]])
CPD_O = TabularCPD(variable='O', variable_card=2, values=[[0.9, 0.3], [0.1, 0.7]], evidence=['S'], evidence_card=[2])
CPD_L = TabularCPD(variable='L', variable_card=2, values=[[0.7, 0.2], [0.3, 0.8]], evidence=['S'], evidence_card=[2])
CPD_M = TabularCPD(variable='M', variable_card=2, values=[[0.8, 0.4, 0.5, 0.1], [0.2, 0.6, 0.5, 0.9]],
                   evidence=['S', 'L'], evidence_card=[2, 2])

email_model.add_cpds(CPD_S, CPD_O, CPD_L, CPD_M)
# print(email_model.get_cpds())
# print(email_model.check_model())
print("Independencies in the network:")
print(email_model.local_independencies(['S', 'O', 'L', 'M']))

infer = VariableElimination(email_model)
posterior_p = infer.query(['S'], evidence={'O': 1, 'L': 1, 'M': 1})
print("Posterior probability distribution:")
print(posterior_p)
