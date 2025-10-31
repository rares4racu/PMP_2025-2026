import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import networkx as nx

states = ['Difficult', 'Medium', 'Easy']

model = hmm.CategoricalHMM(n_components=3, init_params="")
model.startprob_ = np.array([1 / 3, 1 / 3, 1 / 3])
model.transmat_ = np.array([[0.0, 0.5, 0.5],
                            [0.5, 0.25, 0.25],
                            [0.5, 0.25, 0.25]])
model.emissionprob_ = np.array([[0.1, 0.2, 0.4, 0.3],
                                [0.15, 0.25, 0.5, 0.1],
                                [0.2, 0.3, 0.4, 0.1]])

G = nx.DiGraph()
for i, si in enumerate(states):
    for j, sj in enumerate(states):
        if model.transmat_[i, j] > 0:
            G.add_edge(si, sj, weight=model.transmat_[i, j])
pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2500, node_color='lightblue')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{d['weight']:.2f}" for i, j, d in G.edges(data=True)})
# plt.show()

obs = ['FB', 'FB', 'S', 'B', 'B', 'S', 'B', 'B', 'NS', 'B', 'B']
obs_seq = np.array([0, 0, 2, 1, 1, 2, 1, 1, 3, 1, 1]).reshape(-1, 1)
obs_log = model.score(obs_seq)
obs_prob = np.exp(obs_log)
print(f"Probabilitatea secvenței observate: {obs_prob:.10f}")

viterbi, hs = model.decode(obs_seq, algorithm="viterbi")
hsl = [states[s] for s in hs]
print("\nSecvența cea mai probabilă de dificultăți:")
print(hsl)
viterbi_prob = np.exp(viterbi)
print(f"\nProbabilitatea secvenței Viterbi: {viterbi_prob:.10f}")

sp = {'Difficult': 1 / 3, 'Medium': 1 / 3, 'Easy': 1 / 3}
tp = {'Difficult': {'Difficult': 0, 'Medium': 0.5, 'Easy': 0.5},
      'Medium': {'Difficult': 0.5, 'Medium': 0.25, 'Easy': 0.25},
      'Easy': {'Difficult': 0.5, 'Medium': 0.25, 'Easy': 0.25}}
ep = {'Difficult': {'FB': 0.1, 'B': 0.2, 'S': 0.4, 'NS': 0.3},
      'Medium': {'FB': 0.15, 'B': 0.25, 'S': 0.5, 'NS': 0.1},
      'Easy': {'FB': 0.2, 'B': 0.3, 'S': 0.4, 'NS': 0.1}}


def viterbi(states, obs, sp, tp, ep):
    d = [{}]
    path = {}
    for s in states:
        d[0][s] = sp[s] * ep[s][obs[0]]
        path[s] = [s]
    for t in range(1, len(obs)):
        d.append({})
        newpath = {}

        for s0 in states:
            (p, s) = max(
                [(d[t - 1][s1] * tp[s1][s0] * ep[s0][obs[t]], s1) for s1 in states]
            )
            d[t][s0] = p
            newpath[s0] = path[s] + [s0]

        path = newpath

    (p, s) = max([(d[-1][s], s) for s in states])
    return p, path[s]

(probs, state) = viterbi(states, obs, sp, tp, ep)
print(f"\nProbabilitatea secvenței Viterbi (varianta care nu foloseste libraria): {probs:.10f}")
print(state)
