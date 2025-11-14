import pymc as pm
import numpy as np
import arviz as az

data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])
mu0 = np.mean(data)
sigma0 = np.std(data, ddof=1)

def main():
    with pm.Model() as first_model:
        mu1 = pm.Normal("mu1", mu=mu0, sigma=10)
        sigma1 = pm.HalfNormal("sigma1", sigma=10)
        x1 = pm.Normal("x1", mu=mu1, sigma=sigma1, observed=data)

        first_trace = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)

    first_summary = az.summary(first_trace, hdi_prob=0.95)
    print(first_summary)

    print(f"mu bayesian: {first_summary.loc['mu1', 'mean']:.3f} mu sample: {mu0:.3f}")
    print(f"sigma bayesian: {first_summary.loc['sigma1', 'mean']:.3f} sigma sample: {sigma0:.3f}")

    with pm.Model() as second_model:
        mu2 = pm.Normal("mu2", mu=50, sigma=1)
        sigma2 = pm.HalfNormal("sigma2", sigma=10)
        x2 = pm.Normal("x2", mu=mu2, sigma=sigma2, observed=data)

        second_trace = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)

    second_summary = az.summary(second_trace, hdi_prob=0.95)
    print(second_summary)

    print(f"mu (first): {first_summary.loc['mu1', 'mean']:.3f} mu (second): {second_summary.loc['mu2', 'mean']:.3f} mu (sample): {mu0:.3f}")
    print(f"sigma (first): {first_summary.loc['sigma1', 'mean']:.2f} sigma (second): {second_summary.loc['sigma2', 'mean']:.3f} sigma (sample): {sigma0:.2f}")

if __name__ == "__main__":
    main()
