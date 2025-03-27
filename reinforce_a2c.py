import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import gymnasium as gym
import numpy as np

# Check if JAX is using GPU
print(f"Available devices: {jax.devices()}")

# Policy (Actor) Network
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

# Value (Critic) Network
class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, state):
        x = nn.Dense(64)(state)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        value = nn.Dense(1)(x)
        return value.squeeze()  # Scalar output

# Sample action from policy
def sample_action(actor_params, state, key):
    logits = PolicyNetwork(action_dim).apply(actor_params, state)
    action = jax.random.categorical(key, logits)
    return int(action)  # Convert action to Python int

# Compute loss for actor and critic
def loss_fn(params, states, actions, returns):
    actor_params, critic_params = params
    
    # Policy logits and value predictions
    logits = jax.vmap(lambda s: PolicyNetwork(action_dim).apply(actor_params, s))(states)
    values = jax.vmap(lambda s: ValueNetwork().apply(critic_params, s))(states)

    # Compute log probabilities of actions
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze()

    # Compute advantage (A_t = R_t - V(s_t))
    advantages = returns - values

    # Losses
    policy_loss = -jnp.mean(action_log_probs * advantages)
    value_loss = jnp.mean((returns - values) ** 2)  # MSE loss for critic

    return policy_loss + value_loss  # Total loss

# Initialize environment
env = gym.make("CartPole-v1")
action_dim = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Initialize networks
actor = PolicyNetwork(action_dim)
critic = ValueNetwork()
rng = jax.random.PRNGKey(0)
dummy_state = jnp.zeros((obs_dim,), dtype=jnp.float32)

actor_params = actor.init(rng, dummy_state)
critic_params = critic.init(rng, dummy_state)
params = (actor_params, critic_params)

# Optimizers
optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(params)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    state = jnp.array(state, dtype=jnp.float32)
    
    states, actions, rewards = [], [], []
    done = False
    while not done:
        rng, subkey = jax.random.split(rng)
        action = sample_action(actor_params, state, subkey)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Gymnasium compatibility

        # Store experience
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = jnp.array(next_state, dtype=jnp.float32)

    # Compute returns (discounted rewards)
    returns = np.zeros(len(rewards))
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + 0.99 * G
        returns[t] = G
    returns = jnp.array(returns, dtype=jnp.float32)

    # Convert to JAX arrays
    states = jnp.stack(states)
    actions = jnp.array(actions, dtype=jnp.int32)

    # Compute gradients & update actor-critic
    grads = jax.grad(loss_fn)(params, states, actions, returns)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Logging
    if episode % 50 == 0:
        print(f"Episode {episode}: Return {sum(rewards)}")
