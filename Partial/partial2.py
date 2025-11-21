import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import networkx as nx

states = ['Walking', 'Running', 'Resting']

model = hmm.CategoricalHMM(n_components=3, init_params="")
model.startprob_ = np.array([0.4, 0.3, 0.3])
model.transmat_ = np.array([[0.6, 0.3, 0.1],
                            [0.2, 0.7, 0.1],
                            [0.3, 0.2, 0.5]])
model.emissionprob_ = np.array([[0.1, 0.7, 0.2],
                                [0.05, 0.25, 0.7],
                                [0.8, 0.15, 0.05]])

observations = ['Medium', 'High', 'Low']
obs_seq = np.array([1, 2, 0]).reshape(-1, 1)
obs_log = model.score(obs_seq)
obs_prob = np.exp(obs_log)
print(f"Probabilitatea secvenței observate: {obs_prob:.10f}")

viterbi, hs = model.decode(obs_seq, algorithm="viterbi")
hsl = [states[s] for s in hs]
print("\nSecvența cea mai probabilă:")
print(hsl)
viterbi_prob = np.exp(viterbi)
print(f"\nProbabilitatea secvenței Viterbi: {viterbi_prob:.10f}")
# Viterbi ar fi mai eficient decat sa facem brute-force
