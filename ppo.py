import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import gymnasium as gym
import numpy as np

# Check if JAX is using GPU
print(f"Available devices: {jax.devices()}")

# ---------------------------
# Define Actor-Critic Network
# ---------------------------
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Shared layers
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        # Actor head: output logits over actions
        logits = nn.Dense(self.action_dim)(x)
        # Critic head: scalar value prediction
        value = nn.Dense(1)(x)
        return logits, jnp.squeeze(value)

# ---------------------------
# Helper functions
# ---------------------------
def sample_action(params, state, key):
    logits, value = ActorCritic(action_dim).apply(params, state)
    action = jax.random.categorical(key, logits)
    log_prob = jax.nn.log_softmax(logits)[action]
    return int(action), float(log_prob), float(value)

def ppo_loss(params, old_log_probs, states, actions, advantages, targets, epsilon=0.2, vf_coef=0.5, ent_coef=0.01):
    # NOTE: This recomputation is redundant in the first epoch; thereafter the policy has changed so the logits will be 
    # different from the original ones computed in the trajectory collection phase.
    logits, values = jax.vmap(lambda s: ActorCritic(action_dim).apply(params, s))(states)
    log_probs = jax.nn.log_softmax(logits)
    new_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()
    ratio = jnp.exp(new_log_probs - old_log_probs)
    surrogate1 = ratio * advantages
    surrogate2 = jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
    actor_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))
    critic_loss = jnp.mean((targets - values) ** 2)
    entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * log_probs, axis=1))
    total_loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy
    return total_loss

# ---------------------------
# Environment and Initialization
# ---------------------------
env = gym.make("CartPole-v1")
action_dim = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Create the ActorCritic model instance and initialize parameters.
model = ActorCritic(action_dim)
rng = jax.random.PRNGKey(0)
dummy_obs = jnp.zeros((obs_dim,), dtype=jnp.float32)
params = model.init(rng, dummy_obs)

# Create optimizer and initialize optimizer state.
optimizer = optax.adam(learning_rate=3e-4)
opt_state = optimizer.init(params)

# PPO hyperparameters
num_steps = 128      # number of steps to collect per update
num_epochs = 4       # number of epochs per update
gamma = 0.99         # discount factor
lam = 0.95           # GAE lambda

# ---------------------------
# Training Loop
# ---------------------------
for episode in range(1000):
    # --- Trajectory Collection ---
    obs, _ = env.reset()
    obs = jnp.array(obs, dtype=jnp.float32)
    trajectory = []  # stores tuples: (obs, action, reward, log_prob, value, done)

    for t in range(num_steps):
        rng, subkey = jax.random.split(rng)
        action, log_prob, value = sample_action(params, obs, subkey)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((obs, action, reward, log_prob, value, done))
        obs = jnp.array(next_obs, dtype=jnp.float32)
        if done:
            obs, _ = env.reset()
            obs = jnp.array(obs, dtype=jnp.float32)

    # Process trajectory into arrays.
    states = jnp.stack([s for (s, a, r, lp, v, d) in trajectory])
    actions = jnp.array([a for (s, a, r, lp, v, d) in trajectory], dtype=jnp.int32)
    rewards = np.array([r for (s, a, r, lp, v, d) in trajectory], dtype=np.float32)
    old_log_probs = jnp.array([lp for (s, a, r, lp, v, d) in trajectory], dtype=jnp.float32)
    values = np.array([v for (s, a, r, lp, v, d) in trajectory], dtype=np.float32)
    dones = np.array([d for (s, a, r, lp, v, d) in trajectory], dtype=np.float32)

    # Compute returns and advantages using Generalized Advantage Estimation (GAE)
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    gae = 0
    next_value = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]
    advantages = jnp.array(advantages, dtype=jnp.float32)
    returns = jnp.array(returns, dtype=jnp.float32)

    # --- PPO Update ---
    for _ in range(num_epochs):
        grads = jax.grad(ppo_loss)(params, old_log_probs, states, actions, advantages, returns)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    # --- Evaluation Phase ---
    # Run a full episode until termination to evaluate performance.
    eval_obs, _ = env.reset()
    eval_obs = jnp.array(eval_obs, dtype=jnp.float32)
    eval_reward_sum = 0.0
    done_flag = False
    # Optionally, limit evaluation to a maximum number of steps (e.g. 1000)
    eval_steps = 0
    max_eval_steps = 1000
    while not done_flag and eval_steps < max_eval_steps:
        rng, subkey = jax.random.split(rng)
        action, _, _ = sample_action(params, eval_obs, subkey)
        eval_obs, reward, terminated, truncated, _ = env.step(action)
        eval_reward_sum += reward
        done_flag = terminated or truncated
        eval_obs = jnp.array(eval_obs, dtype=jnp.float32)
        eval_steps += 1

    if episode % 50 == 0:
        print(f"Episode {episode}: Eval Return = {eval_reward_sum}")
