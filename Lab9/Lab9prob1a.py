import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

y_val = [0, 5, 10]
theta_val = [0.2, 0.5]
posterior_val = []
fig, axes = plt.subplots(len(y_val), len(theta_val), figsize=(15, 10))
def main():
    for i,Yv in enumerate(y_val):
        for j,theta in enumerate(theta_val):
            with pm.Model() as model:
                n = pm.Poisson("n", mu=10)
                y = pm.Binomial("y", n=n, p=theta,observed=Yv)
                trace = pm.sample(1000,tune=1000)
                ax = axes[i,j]
                az.plot_posterior(trace, ax=ax)
                ax.set_title(f"Y={Yv}, theta={theta}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


