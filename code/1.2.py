import numpy as np
import gym

# If you keep gym 0.26 + NumPy 2.0, patch np.bool8 once:
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

def make_cartpole_v0():
    env = gym.make("CartPole-v0")  # v0: 200-step limit
    spec = getattr(env, "spec", None)
    print("Env spec id:", spec.id if spec else "unknown")
    print("Max episode steps:", getattr(env, "_max_episode_steps", "unknown"))
    return env

def make_stochastic_linear_policy(x, rng=None):
    """ Bernoulli policy: P(LEFT|s) = sigmoid(s·w + b); sample action each step. """
    x = np.asarray(x, float)
    w, b = x[:4], x[4]
    if rng is None:
        rng = np.random.RandomState(0)
    def policy(s):
        z = float(np.dot(s, w) + b)
        p_left = 1.0 / (1.0 + np.exp(-z))   # sigmoid
        return 0 if rng.rand() < p_left else 1  # 0=LEFT, 1=RIGHT
    return policy

def run_episode(env, policy, seed=None):
    out = env.reset(seed=seed)
    obs = out[0] if isinstance(out, tuple) else out
    total = 0.0
    while True:
        a = policy(obs)
        step = env.step(a)
        if len(step) == 5:  # gymnasium-style, just in case
            obs, r, terminated, truncated, _ = step
            done = terminated or truncated
        else:               # classic gym 0.26
            obs, r, done, _ = step
        total += r
        if done:
            return float(total)

def eval_policy_average_return(x, n_episodes=1000, seed_base=0):
    env = make_cartpole_v0()
    # Use a different RNG stream for each episode to keep Bernoulli sampling independent
    returns = []
    for i in range(n_episodes):
        rng = np.random.RandomState(seed_base + i)
        pi = make_stochastic_linear_policy(x, rng=rng)
        returns.append(run_episode(env, pi, seed=seed_base + i))
    env.close()
    return float(np.mean(returns))

# Sanity checks (first should be ~15.6 on v0):
for x in [
    (-1, -1, -1, -1, -1),
    ( 1,  0,  1,  0,  1),
    ( 0,  1,  2,  3,  4),
]:
    avg = eval_policy_average_return(x, n_episodes=1000, seed_base=0)
    print(f"{x} -> {avg:.1f}")
