import arviz as az
from scipy.stats import gamma
import matplotlib.pyplot as plt

n = 10
calls = 180

alpha = 1
beta = 0

alpha_post = alpha + calls
beta_post = beta + n

mean = alpha_post / beta_post
mode = (alpha_post - 1) / beta_post

print(f"Media posterioară: {mean:.2f}")
print(f"Modul posteriorului: {mode:.2f}")

posterior_samples = gamma.rvs(a=alpha_post, scale=1 / beta_post, size=100_000)

hdi_94 = az.hdi(posterior_samples, hdi_prob=0.94)
print(f"94% HDI: {hdi_94}")
az.plot_posterior(posterior_samples, hdi_prob=0.94)
plt.show()
