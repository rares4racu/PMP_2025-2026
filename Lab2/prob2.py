import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
sim = 1000
lam = [1, 2, 5, 10]

# Distributiile Poisson fixe
X1 = np.random.poisson(lam[0], sim)
X2 = np.random.poisson(lam[1], sim)
X3 = np.random.poisson(lam[2], sim)
X4 = np.random.poisson(lam[3], sim)

# Distributia Poisson cu lambda random
lam_random = np.random.choice(lam, sim)
X5 = np.array([np.random.poisson(lam=l) for l in lam_random])

# Distributia Poisson cu lambda=5 cu probabilitatea mai mare
lam_5ml = np.random.choice(lam, sim, p=[0.2, 0.2, 0.4, 0.2])
X6 = np.array([np.random.poisson(lam=l) for l in lam_5ml])

# Plot-uri
fig, axes = plt.subplots(2, 3, figsize=(10, 10))

axes[0, 0].hist(X1, bins='auto', color='blue', edgecolor='black')
axes[0, 0].set_title('lambda=1')

axes[0, 1].hist(X2, bins='auto', color='red', edgecolor='black')
axes[0, 1].set_title('lambda=2')

axes[0, 2].hist(X3, bins='auto', color='green', edgecolor='black')
axes[0, 2].set_title('lambda=5')

axes[1, 0].hist(X4, bins='auto', color='yellow', edgecolor='black')
axes[1, 0].set_title('lambda=10')

axes[1, 1].hist(X5, bins='auto', color='violet', edgecolor='black')
axes[1, 1].set_title('Random lambda')

axes[1, 2].hist(X6, bins='auto', color='orange', edgecolor='black')
axes[1, 2].set_title('lambda=5 is more likely')

plt.tight_layout()
plt.show()

# Forma distributiei cu lambda random nu are o forma constanta in comparatie cu cele fixe.
# In realizarea modelarilor proceselor din viata reala, incertitudinea parametrilor creste dificultatea realizarii unui astfel de model.