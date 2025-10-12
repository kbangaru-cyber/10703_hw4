import numpy as np
import matplotlib.pyplot as plt

# ---------- CMA-ES (minimal) ----------
class CMAES:
    def __init__(self, dim, mu=None, cov=None, pop_size=100, elite_frac=0.1, eps=0.25, seed=0):
        self.dim = dim
        self.mu = np.zeros(dim) if mu is None else np.array(mu, dtype=float)
        self.cov = (100.0 * np.eye(dim)) if cov is None else np.array(cov, dtype=float)
        self.pop_size = int(pop_size)
        self.elite_size = max(1, int(np.ceil(elite_frac * self.pop_size)))
        self.eps = float(eps)
        self.rng = np.random.RandomState(seed)

    def ask(self):
        # x_i ~ N(mu, Sigma)
        return self.rng.multivariate_normal(self.mu, self.cov, size=self.pop_size)

    def tell(self, X, fitness):
        # Sort by fitness (maximize)
        idx = np.argsort(-fitness)
        elites = X[idx[:self.elite_size]]

        # mu_{t+1} = mean of elites
        self.mu = elites.mean(axis=0)

        # Sigma_{t+1} = empirical cov of elites + eps*I
        if self.elite_size > 1:
            C = elites - self.mu
            cov = (C.T @ C) / (self.elite_size - 1)
        else:
            cov = np.zeros((self.dim, self.dim))
        self.cov = cov + self.eps * np.eye(self.dim)

# ---------- Problem 1.1 objective ----------
def f_simple(x, x_star=np.array([65.0, 49.0])):
    x = np.atleast_2d(x)
    # f(x) = -||x - x*||^2 (to maximize)
    return -np.sum((x - x_star) ** 2, axis=1)

# ---------- Run CMA-ES on 1.1 ----------
def run_p1_1(iters=30, seed=0):
    cma = CMAES(dim=2, mu=np.zeros(2), cov=100.0*np.eye(2),
                pop_size=100, elite_frac=0.1, eps=0.25, seed=seed)
    mus = [cma.mu.copy()]
    for _ in range(iters):
        X = cma.ask()                 # sample population
        fit = f_simple(X)             # evaluate fitness
        cma.tell(X, fit)              # update mu and Sigma
        mus.append(cma.mu.copy())
    return np.stack(mus)

mus = run_p1_1(iters=30, seed=0)
print("Final mean:", mus[-1])

# ---------- Plot mean path ----------
plt.figure(figsize=(6,6))
plt.plot(mus[:,0], mus[:,1], marker="o", linewidth=2)
plt.scatter([0],[0], marker="x", s=120, label="init μ₀")
plt.scatter([65],[49], marker="*", s=180, label="optimum x*=(65,49)")
plt.scatter([mus[-1,0]],[mus[-1,1]], marker="s", s=120, label="final μ_T")
plt.xlabel("μ[0]"); plt.ylabel("μ[1]")
plt.title("CMA-ES mean path on f(x) = -||x - x*||²")
plt.legend(); plt.grid(True, linestyle=":")
plt.show()
