"""
DDQN Agent for Crypto Trading (Phase 2)

Double Deep Q-Network implementation with discrete action space for crypto trading.
Actions are discretized into bins for each asset independently.

Action Space:
- For each asset: 10 discrete actions
  - Bins: [-100%, -80%, -60%, -40%, -20%, 0%, 20%, 40%, 60%, 80%, 100%]
  - Negative = sell, Positive = buy, 0 = hold
- Total action space: 10^N where N = number of assets

For 7 assets: 10^7 = 10M combinations (too large for discrete Q-learning)

Alternative Architecture (used here):
- Sequential action selection: Pick one asset at a time
- Action space per step: N_assets * N_bins = 7 * 10 = 70 actions
- More tractable for Q-learning

Note: This is a simplified DDQN for comparison with PPO in Phase 2.
      For production, PPO's continuous action space is more suitable.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

from config_two_phase import PHASE2


class QNetwork(nn.Module):
    """
    Q-Network for DDQN.

    Maps state to Q-values for each discrete action.
    """

    def __init__(self, state_dim, action_dim, net_dim=1024):
        """
        Initialize Q-Network.

        Args:
            state_dim: State dimension
            action_dim: Number of discrete actions
            net_dim: Hidden layer dimension
        """
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, net_dim)
        self.fc2 = nn.Linear(net_dim, net_dim)
        self.fc3 = nn.Linear(net_dim, net_dim // 2)
        self.fc4 = nn.Linear(net_dim // 2, action_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        """
        Forward pass.

        Args:
            state: State tensor (batch_size, state_dim)

        Returns:
            Q-values tensor (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values


class ReplayBuffer:
    """
    Experience replay buffer for DDQN.
    """

    def __init__(self, capacity=100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample batch from buffer.

        Args:
            batch_size: Batch size

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    """
    DDQN Agent for crypto trading.

    Uses simplified action space: select one asset to trade at a time.
    Action encoding: asset_idx * n_bins + bin_idx
    """

    def __init__(
        self,
        state_dim,
        n_assets,
        n_bins=10,
        net_dim=1024,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000,
        device='cuda'
    ):
        """
        Initialize DDQN agent.

        Args:
            state_dim: State dimension
            n_assets: Number of assets
            n_bins: Number of discretization bins
            net_dim: Q-network hidden dimension
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            device: 'cuda' or 'cpu'
        """
        self.state_dim = state_dim
        self.n_assets = n_assets
        self.n_bins = n_bins
        self.action_dim = n_assets * n_bins  # Total discrete actions

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Q-networks
        self.q_network = QNetwork(state_dim, self.action_dim, net_dim).to(self.device)
        self.target_network = QNetwork(state_dim, self.action_dim, net_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training stats
        self.steps = 0
        self.episodes = 0

        # Action bins (percentage of available cash/holdings)
        self.action_bins = np.linspace(-1.0, 1.0, n_bins)  # -1 = sell all, +1 = buy max

    def select_action(self, state, epsilon=None):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            epsilon: Exploration rate (uses self.epsilon if None)

        Returns:
            Selected action index
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def decode_action(self, action_idx):
        """
        Decode discrete action to asset index and bin index.

        Args:
            action_idx: Discrete action index

        Returns:
            Tuple of (asset_idx, bin_idx)
        """
        asset_idx = action_idx // self.n_bins
        bin_idx = action_idx % self.n_bins
        return asset_idx, bin_idx

    def action_to_continuous(self, action_idx):
        """
        Convert discrete action to continuous action vector.

        Args:
            action_idx: Discrete action index

        Returns:
            Continuous action vector for environment
        """
        asset_idx, bin_idx = self.decode_action(action_idx)
        action_value = self.action_bins[bin_idx]

        # Create action vector (all zeros except for selected asset)
        continuous_action = np.zeros(self.n_assets, dtype=np.float32)
        continuous_action[asset_idx] = action_value

        return continuous_action

    def train_step(self):
        """
        Perform one training step.

        Returns:
            Loss value or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, filepath):
        """Save agent checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
        }, filepath)

    def load(self, filepath):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']


# =============================================================================
# Training and Evaluation Utilities
# =============================================================================

def train_ddqn_agent(env, agent, n_episodes=1000, max_steps_per_episode=1000):
    """
    Train DDQN agent on environment.

    Args:
        env: Trading environment
        agent: DDQN agent
        n_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode

    Returns:
        Training statistics
    """
    episode_rewards = []
    episode_losses = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []

        for step in range(max_steps_per_episode):
            # Select action
            action_idx = agent.select_action(state)
            continuous_action = agent.action_to_continuous(action_idx)

            # Take step in environment
            next_state, reward, done, info = env.step(continuous_action)

            # Store experience
            agent.replay_buffer.push(state, action_idx, reward, next_state, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.episodes += 1
        episode_rewards.append(episode_reward)
        episode_losses.append(np.mean(episode_loss) if episode_loss else 0.0)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss = np.mean(episode_losses[-10:])
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Avg Reward={avg_reward:.4f}, "
                  f"Avg Loss={avg_loss:.6f}, "
                  f"Epsilon={agent.epsilon:.4f}")

    return {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
    }


def evaluate_ddqn_agent(env, agent, n_episodes=10):
    """
    Evaluate DDQN agent (no exploration).

    Args:
        env: Trading environment
        agent: DDQN agent
        n_episodes: Number of evaluation episodes

    Returns:
        Evaluation statistics
    """
    episode_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        done = False
        while not done:
            # Greedy action (no exploration)
            action_idx = agent.select_action(state, epsilon=0.0)
            continuous_action = agent.action_to_continuous(action_idx)

            next_state, reward, done, info = env.step(continuous_action)

            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'episode_rewards': episode_rewards,
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("DDQN Agent Testing")
    print("=" * 60)

    # Mock parameters
    state_dim = 91  # Phase 2 enhanced state
    n_assets = 7
    n_bins = PHASE2.DDQN_ACTION_BINS

    print(f"State dimension: {state_dim}")
    print(f"Number of assets: {n_assets}")
    print(f"Action bins: {n_bins}")
    print(f"Total discrete actions: {n_assets * n_bins}")

    # Create agent
    print("\nCreating DDQN agent...")
    agent = DDQNAgent(
        state_dim=state_dim,
        n_assets=n_assets,
        n_bins=n_bins,
        net_dim=1024,
        device='cuda'
    )
    print(f"✓ Agent created on device: {agent.device}")

    # Test action selection
    print("\nTesting action selection...")
    mock_state = np.random.randn(state_dim).astype(np.float32)

    # Exploration
    action_explore = agent.select_action(mock_state, epsilon=1.0)
    print(f"  Exploration action: {action_explore}")
    asset_idx, bin_idx = agent.decode_action(action_explore)
    print(f"    Asset: {asset_idx}, Bin: {bin_idx}, Value: {agent.action_bins[bin_idx]:.2f}")

    # Exploitation
    action_exploit = agent.select_action(mock_state, epsilon=0.0)
    print(f"  Exploitation action: {action_exploit}")
    asset_idx, bin_idx = agent.decode_action(action_exploit)
    print(f"    Asset: {asset_idx}, Bin: {bin_idx}, Value: {agent.action_bins[bin_idx]:.2f}")

    # Test action conversion
    continuous_action = agent.action_to_continuous(action_exploit)
    print(f"  Continuous action: {continuous_action}")

    # Test replay buffer
    print("\nTesting replay buffer...")
    for i in range(10):
        next_state = np.random.randn(state_dim).astype(np.float32)
        reward = np.random.randn()
        done = i == 9
        agent.replay_buffer.push(mock_state, action_explore, reward, next_state, done)
        mock_state = next_state

    print(f"  Buffer size: {len(agent.replay_buffer)}")

    # Test training step (need more samples)
    print("\nAdding more samples for training...")
    for i in range(agent.batch_size):
        state = np.random.randn(state_dim).astype(np.float32)
        action = random.randint(0, agent.action_dim - 1)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = False
        agent.replay_buffer.push(state, action, reward, next_state, done)

    print(f"  Buffer size: {len(agent.replay_buffer)}")

    print("\nTesting training step...")
    loss = agent.train_step()
    print(f"  Loss: {loss:.6f}")
    print(f"  Steps: {agent.steps}")
    print(f"  Epsilon: {agent.epsilon:.4f}")

    print("\n✓ All DDQN agent tests passed")
