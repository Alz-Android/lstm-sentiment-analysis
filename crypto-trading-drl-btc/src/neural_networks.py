"""
CNN-LSTM neural network for DRL trading agent.
Implements actor and critic networks as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class CNNLSTMActor(nn.Module):
    """
    Actor network: CNN-LSTM architecture that outputs action probabilities.
    Based on the paper's actor network design.
    """

    def __init__(self, input_shape=(100, 4), action_dim=3):
        """
        Initialize the actor network.

        Args:
            input_shape (tuple): Shape of input (timesteps, features)
            action_dim (int): Number of actions (3 for Buy/Hold/Sell)
        """
        super(CNNLSTMActor, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=input_shape[1],  # 4 features
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # LSTM layers for temporal processing
        # After conv1d + pool1d: (100, 32) -> (50, 32)
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, action_dim)

        logger.info(f"Actor network initialized with input shape {input_shape}")

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, timesteps, features)

        Returns:
            torch.Tensor: Action probabilities
        """
        batch_size = x.size(0)

        # CNN feature extraction
        # Input: (batch_size, timesteps, features) -> (batch_size, features, timesteps)
        x = x.permute(0, 2, 1)

        # Conv1D: (batch_size, 4, 100) -> (batch_size, 32, 100)
        x = F.relu(self.conv1(x))

        # MaxPool1D: (batch_size, 32, 100) -> (batch_size, 32, 50)
        x = self.pool1(x)

        # Permute back for LSTM: (batch_size, 32, 50) -> (batch_size, 50, 32)
        x = x.permute(0, 2, 1)

        # LSTM: (batch_size, 50, 32) -> (batch_size, 50, 32)
        lstm_out, _ = self.lstm(x)

        # Take the last timestep output: (batch_size, 32)
        x = lstm_out[:, -1, :]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        action_logits = self.fc2(x)

        # Convert to probabilities
        action_probs = F.softmax(action_logits, dim=-1)

        return action_probs

    def get_action(self, state, deterministic=False):
        """
        Get action from the actor network.

        Args:
            state (np.array): Current state
            deterministic (bool): If True, return argmax action

        Returns:
            tuple: (action, log_prob, value)
        """
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            action_probs = self.forward(state)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1).item()
            log_prob = torch.log(action_probs.squeeze(0)[action]).item()
        else:
            # Sample from distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
            log_prob = action_dist.log_prob(torch.tensor(action)).item()

        return action, log_prob, action_probs.squeeze(0).numpy()


class CNNLSTMCritic(nn.Module):
    """
    Critic network: CNN-LSTM architecture that outputs state values.
    Based on the paper's critic network design.
    """

    def __init__(self, input_shape=(100, 4)):
        """
        Initialize the critic network.

        Args:
            input_shape (tuple): Shape of input (timesteps, features)
        """
        super(CNNLSTMCritic, self).__init__()

        self.input_shape = input_shape

        # Same CNN architecture as actor
        self.conv1 = nn.Conv1d(
            in_channels=input_shape[1],
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        # Fully connected layers
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)  # Single value output

        logger.info(f"Critic network initialized with input shape {input_shape}")

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, timesteps, features)

        Returns:
            torch.Tensor: State value
        """
        batch_size = x.size(0)

        # Same CNN processing as actor
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.permute(0, 2, 1)

        # LSTM processing
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        value = self.fc2(x)

        return value


class PPOAgent:
    """
    PPO Agent that combines actor and critic networks.
    Implements the PPO algorithm for training.
    """

    def __init__(self, input_shape=(100, 4), action_dim=3, lr=3e-4, device='cpu'):
        """
        Initialize the PPO agent.

        Args:
            input_shape (tuple): Shape of input state
            action_dim (int): Number of actions
            lr (float): Learning rate
            device (str): Device to run on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.action_dim = action_dim

        # Create actor and critic networks
        self.actor = CNNLSTMActor(input_shape, action_dim).to(self.device)
        self.critic = CNNLSTMCritic(input_shape).to(self.device)

        # Create old networks for PPO (they don't get updated during training)
        self.actor_old = CNNLSTMActor(input_shape, action_dim).to(self.device)
        self.critic_old = CNNLSTMCritic(input_shape).to(self.device)

        # Copy parameters to old networks
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01
        
        # Memory for experience replay
        self.memory = []
        self.batch_size = 64

        logger.info("PPO Agent initialized")

    def get_action(self, state, deterministic=False):
        """
        Get action from current policy.

        Args:
            state (np.array): Current state

        Returns:
            tuple: (action, log_prob, value)
        """
        return self.actor.get_action(state, deterministic)
    
    def act(self, state, deterministic=False):
        """
        Get action from current policy (training script interface).

        Args:
            state (np.array): Current state
            deterministic (bool): If True, return argmax action

        Returns:
            tuple: (action, log_prob)
        """
        action, log_prob, _ = self.actor.get_action(state, deterministic)
        return action, log_prob
    
    def store_transition(self, state, action, log_prob, reward, done):
        """
        Store transition in memory for later training.

        Args:
            state: Current state
            action: Action taken
            log_prob: Log probability of action
            reward: Reward received
            done: Whether episode is done
        """
        self.memory.append({
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'done': done
        })
        
        # Keep memory size manageable
        if len(self.memory) > 10000:
            self.memory = self.memory[-5000:]
    
    def save_model(self, filepath):
        """
        Save the trained model.

        Args:
            filepath (str): Path to save the model
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': {
                'input_shape': self.actor.input_shape,
                'action_dim': self.action_dim,
                'device': str(self.device)
            }
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model.

        Args:
            filepath (str): Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        if 'actor_optimizer_state_dict' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        if 'critic_optimizer_state_dict' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
        # Update old networks
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
        
        logger.info(f"Model loaded from {filepath}")

    def evaluate(self, states, actions):
        """
        Evaluate states and actions using old networks.

        Args:
            states (torch.Tensor): Batch of states
            actions (torch.Tensor): Batch of actions

        Returns:
            tuple: (log_probs, state_values, entropy)
        """
        # Get action probabilities from old actor
        action_probs = self.actor_old(states)

        # Get state values from old critic
        state_values = self.critic_old(states)

        # Create action distribution
        action_dist = torch.distributions.Categorical(action_probs)

        # Get log probabilities for the actions taken
        log_probs = action_dist.log_prob(actions)

        # Get entropy
        entropy = action_dist.entropy()

        return log_probs, state_values.squeeze(), entropy

    def update(self, trajectories=None):
        """
        Update the actor and critic networks using PPO.
        If no trajectories provided, use stored memory.

        Args:
            trajectories (dict, optional): Dictionary containing trajectory data
        """
        if trajectories is None:
            # Use stored memory - simplified update for compatibility
            if len(self.memory) < self.batch_size:
                return
            
            # Simple policy gradient update with stored memory
            batch = self.memory[-self.batch_size:]
            
            states = []
            actions = []
            rewards = []
            log_probs = []
            
            for transition in batch:
                states.append(transition['state'])
                actions.append(transition['action'])
                rewards.append(transition['reward'])
                log_probs.append(transition['log_prob'])
            
            # Convert to tensors
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            old_log_probs = torch.FloatTensor(log_probs).to(self.device)
            
            # Calculate simple advantages (rewards)
            advantages = rewards
            returns = rewards
            
        else:
            # Use provided trajectories
            states = torch.FloatTensor(trajectories['states']).to(self.device)
            actions = torch.LongTensor(trajectories['actions']).to(self.device)
            rewards = torch.FloatTensor(trajectories['rewards']).to(self.device)
            old_log_probs = torch.FloatTensor(trajectories['log_probs']).to(self.device)
            advantages = torch.FloatTensor(trajectories['advantages']).to(self.device)
            returns = torch.FloatTensor(trajectories['returns']).to(self.device)

        # PPO update
        actor_loss = torch.tensor(0.0)
        critic_loss = torch.tensor(0.0)
        
        for _ in range(3):  # Reduced epochs for stability
            # Evaluate current policy
            log_probs, state_values, entropy = self.evaluate(states, actions)

            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            if trajectories is None:
                critic_loss = F.mse_loss(state_values, advantages)  # Simple case
            else:
                critic_loss = F.mse_loss(state_values, returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            total_loss = actor_loss + self.value_coeff * critic_loss + self.entropy_coeff * entropy_loss

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # Update old networks
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        logger.info(f"PPO update completed - Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")


# Example usage and testing
if __name__ == "__main__":
    import sys
    import os

    # Add src to path
    sys.path.append(os.path.dirname(__file__))

    def test_neural_networks():
        """Test the neural networks"""
        print("Testing CNN-LSTM Neural Networks...")

        # Create a simple test state
        test_state = np.random.randn(100, 4).astype(np.float32)

        # Initialize networks
        actor = CNNLSTMActor()
        critic = CNNLSTMCritic()

        # Test forward pass
        state_tensor = torch.FloatTensor(test_state).unsqueeze(0)

        actor_output = actor(state_tensor)
        critic_output = critic(state_tensor)

        print(f"Test state shape: {test_state.shape}")
        print(f"Actor output shape: {actor_output.shape}")
        print(f"Actor output (probabilities): {actor_output.detach().numpy()}")
        print(f"Critic output shape: {critic_output.shape}")
        print(f"Critic output (value): {critic_output.item():.4f}")

        # Test PPO agent
        agent = PPOAgent()
        action, log_prob, probs = agent.get_action(test_state)

        print(f"\nPPO Agent test:")
        print(f"Selected action: {action}")
        print(f"Action probabilities: {probs}")
        print(f"Log probability: {log_prob}")

        print("\nNeural networks test completed successfully! ✅")

    test_neural_networks()