from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx

urn_model = DiscreteBayesianNetwork([('Z', 'R'), ('Z', 'A'), ('Z', 'N')])
pos = nx.circular_layout(urn_model)
nx.draw(urn_model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)

CPD_Z = TabularCPD(variable='Z', variable_card=3, values=[[1 / 6], [2 / 6], [3 / 6]])
CPD_R = TabularCPD(variable='R', variable_card=2, values=[[6 / 10, 7 / 10, 7 / 10], [4 / 10, 3 / 10, 3 / 10]],
                   evidence=['Z'], evidence_card=[3])
CPD_A = TabularCPD(variable='A', variable_card=2, values=[[6 / 10, 5 / 10, 6 / 10], [4 / 10, 5 / 10, 4 / 10]],
                   evidence=['Z'], evidence_card=[3])
CPD_N = TabularCPD(variable='N', variable_card=2, values=[[8 / 10, 8 / 10, 7 / 10], [2 / 10, 2 / 10, 3 / 10]],
                   evidence=['Z'], evidence_card=[3])

urn_model.add_cpds(CPD_Z, CPD_R, CPD_A, CPD_N)
# print(urn_model.get_cpds())
# print(urn_model.check_model())

infer = VariableElimination(urn_model)
posterior_p = infer.query(['R'])
print("Posterior probability distribution:")
print(posterior_p)
