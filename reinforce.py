import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import gymnasium as gym
import numpy as np

# Policy Network (MLP)
class PolicyNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(64)(state)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.action_dim)(x)
        return logits  # Unnormalized action logits

# Sample action from policy
def sample_action(params, state, key):
    logits = PolicyNetwork(action_dim).apply(params, state)
    action_probs = jax.nn.softmax(logits)
    action = jax.random.categorical(key, logits)
    return int(action), action_probs[action]  # Convert action to Python int

# Compute REINFORCE loss
def reinforce_loss(params, states, actions, returns):
    logits = jax.vmap(lambda s: PolicyNetwork(action_dim).apply(params, s))(states)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()
    return -jnp.mean(action_log_probs * returns)  # Policy gradient loss

# Initialize environment
env = gym.make("CartPole-v1")
action_dim = env.action_space.n
obs_dim = env.observation_space.shape[0]
policy = PolicyNetwork(action_dim)

# Initialize JAX parameters
rng = jax.random.PRNGKey(0)
dummy_state = jnp.zeros((obs_dim,), dtype=jnp.float32)
params = policy.init(rng, dummy_state)
optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(params)

# Training loop
for episode in range(1000):
    state = jnp.array(env.reset()[0], dtype=jnp.float32)  # Ensure NumPy array
    states, actions, rewards = [], [], []
    done = False
    while not done:
        rng, subkey = jax.random.split(rng)
        action, _ = sample_action(params, state, subkey)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Gym v26 compatibility

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = jnp.array(next_state, dtype=jnp.float32)  # Convert to NumPy array

    # Compute returns (discounted sum of rewards)
    returns = np.zeros(len(rewards))
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + 0.99 * G
        returns[t] = G
    returns = jnp.array(returns, dtype=jnp.float32)

    # Convert lists to JAX arrays
    states = jnp.stack(states)
    actions = jnp.array(actions, dtype=jnp.int32)

    # Compute gradients & update policy
    grads = jax.grad(reinforce_loss)(params, states, actions, returns)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Logging
    if episode % 50 == 0:
        print(f"Episode {episode}: Return {sum(rewards)}")
