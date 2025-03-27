import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import gymnasium as gym
import numpy as np

# Check available devices
print(f"Available devices: {jax.devices()}")

# ---------------------------
# Define the Policy Network
# ---------------------------
class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Two hidden layers with tanh activations
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        # Output logits for each action
        logits = nn.Dense(self.action_dim)(x)
        return logits

# ---------------------------
# Action Sampling Function
# ---------------------------
def sample_action(params, state, key):
    # Compute logits for the given state
    logits = PolicyNetwork(action_dim).apply(params, state)
    # Sample an action from the categorical distribution defined by logits
    action = jax.random.categorical(key, logits)
    # Compute log probability of the sampled action
    log_prob = jax.nn.log_softmax(logits)[action]
    return int(action), float(log_prob)

# ---------------------------
# PPO Policy-Only Loss Function
# ---------------------------
def ppo_policy_loss(params, old_log_probs, states, actions, returns, epsilon=0.2):
    # Compute new log-probabilities for the batch of states
    logits = jax.vmap(lambda s: PolicyNetwork(action_dim).apply(params, s))(states)
    log_probs = jax.nn.log_softmax(logits)
    new_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()
    # Compute probability ratio between new and old policies
    ratio = jnp.exp(new_log_probs - old_log_probs)

    # Compute "advantages" as returns normalized by subtracting the mean
    # NOTE: funny enough this is basically what GRPO is; instead, the baseline for GRPO is the mean of many more 
    # trajectories' returns. See https://yugeten.github.io/posts/2025/01/ppogrpo/.
    advantages = returns - jnp.mean(returns) 

    # Clipped surrogate objective (PPO objective)
    surrogate1 = ratio * advantages
    surrogate2 = jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
    loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))
    return loss

# ---------------------------
# Environment and Initialization
# ---------------------------
env = gym.make("CartPole-v1")
action_dim = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Initialize the policy network parameters.
rng = jax.random.PRNGKey(0)
dummy_obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
policy = PolicyNetwork(action_dim)
params = policy.init(rng, dummy_obs)

# Setup optimizer.
optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(params)

# PPO hyperparameters.
num_steps = 128  # number of steps per trajectory collection
num_epochs = 4   # number of update epochs per trajectory
gamma = 0.99     # discount factor

# ---------------------------
# Training Loop
# ---------------------------
for episode in range(1000):
    # --- Trajectory Collection ---
    obs, _ = env.reset()
    obs = jnp.array(obs, dtype=jnp.float32)
    trajectory = []  # will store tuples: (obs, action, reward, old_log_prob, done)

    for t in range(num_steps):
        rng, subkey = jax.random.split(rng)
        action, log_prob = sample_action(params, obs, subkey)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((obs, action, reward, log_prob, done))
        obs = jnp.array(next_obs, dtype=jnp.float32)
        if done:
            obs, _ = env.reset()
            obs = jnp.array(obs, dtype=jnp.float32)

    # Process trajectory into arrays.
    states = jnp.stack([s for (s, a, r, lp, d) in trajectory])
    actions = jnp.array([a for (s, a, r, lp, d) in trajectory], dtype=jnp.int32)
    rewards = np.array([r for (s, a, r, lp, d) in trajectory], dtype=np.float32)
    old_log_probs = jnp.array([lp for (s, a, r, lp, d) in trajectory], dtype=jnp.float32)
    dones = np.array([d for (s, a, r, lp, d) in trajectory], dtype=np.float32)

    # --- Compute Returns ---
    # Use a simple Monte Carlo return with discounting.
    returns = np.zeros_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G * (1 - dones[t])
        returns[t] = G
    returns = jnp.array(returns, dtype=jnp.float32)

    # --- PPO Update ---
    for _ in range(num_epochs):
        grads = jax.grad(ppo_policy_loss)(params, old_log_probs, states, actions, returns)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    # --- Evaluation Phase ---
    # Run a full episode until termination to evaluate performance.
    eval_obs, _ = env.reset()
    eval_obs = jnp.array(eval_obs, dtype=jnp.float32)
    eval_reward_sum = 0.0
    done_flag = False
    eval_steps = 0
    max_eval_steps = 1000  # optionally cap evaluation steps
    while not done_flag and eval_steps < max_eval_steps:
        rng, subkey = jax.random.split(rng)
        action, _ = sample_action(params, eval_obs, subkey)
        eval_obs, reward, terminated, truncated, _ = env.step(action)
        eval_reward_sum += reward
        done_flag = terminated or truncated
        eval_obs = jnp.array(eval_obs, dtype=jnp.float32)
        eval_steps += 1

    if episode % 50 == 0:
        print(f"Episode {episode}: Eval Return = {eval_reward_sum}")
