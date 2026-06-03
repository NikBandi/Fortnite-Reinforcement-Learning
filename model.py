import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time


from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        # Two heads
        self.policy_head = nn.Linear(64, 5)  # 5 actions
        self.value_head  = nn.Linear(64, 1)  # scalar V(s)

    def forward(self, x):
        features = self.shared(x)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values

    def get_action(self, obs):
        # obs: numpy array or tensor of shape (5,)
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, values

    def evaluate(self, obs, actions):
        # obs: (batch, 5), actions: (batch,)
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            print('Model loaded from', file_name)
        else:
            print('No saved model found, starting fresh')

class ModelTrainer:

    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.optimizer = optim.Adam(model.parameters(), lr=3e-4)
        
        # PPO hyperparameters
        self.n_steps = 1024        # steps to collect per rollout
        self.n_epochs = 10         # update epochs per rollout
        self.batch_size = 64       # minibatch size
        self.gamma = 0.99          # discount factor
        self.gae_lambda = 0.95     # GAE smoothing
        self.clip_epsilon = 0.2    # PPO clip range
        self.c1 = 0.5              # value loss coefficient
        self.c2 = 0.03             # entropy coefficient
    
    def collect_rollout(self):
        """
        1. Initialize empty lists for obs, actions, rewards, log_probs, values, dones
        2. Get initial obs from env (or use last obs if continuing)
        3. Loop n_steps times:
        a. Call model.get_action(obs) → action, log_prob, values
        b. Call env.step(action) → next_obs, reward, done, truncated, info
        c. Append everything to your lists
        d. If done or truncated: reset env, get new obs
        e. Else: obs = next_obs
        4. Convert all lists to tensors
        5. Return everything as a dictionary or tuple
        
        """

        obs_list      = []
        action_list   = []
        reward_list   = []
        log_prob_list = []
        value_list    = []
        done_list     = []
        trunc_list      = []    
        next_value_list = []

        obs, _ = self.env.reset()
        

        for _ in range(self.n_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_tensor)

            next_obs, reward, done, truncated, _ = self.env.step(action)

            with torch.no_grad():
                next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
                _, next_val = self.model.forward(next_obs_tensor)
                next_value_list.append(next_val.squeeze(-1).detach())

            obs_list.append(obs_tensor)
            action_list.append(action)
            reward_list.append(reward)
            log_prob_list.append(log_prob)
            value_list.append(value.squeeze(-1))
            done_list.append(done)          # true terminal — guard eliminated
            trunc_list.append(truncated)  # truncated — max steps reached

            
            if done or truncated:
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        # Convert to tensors
        obs_t      = torch.stack(obs_list)
        actions_t  = torch.tensor(action_list,   dtype=torch.long)
        rewards_t  = torch.tensor(reward_list,   dtype=torch.float32)
        log_probs_t= torch.stack([lp.detach() for lp in log_prob_list])
        values_t   = torch.stack([v.detach()  for v  in value_list])
        dones_t    = torch.tensor(done_list,     dtype=torch.float32)
        truncs_t = torch.tensor(trunc_list, dtype=torch.float32)
        next_values_t = torch.stack(next_value_list)
    

        return obs_t, actions_t, rewards_t, log_probs_t, values_t, next_values_t, dones_t, truncs_t
    
    def compute_gae(self, rewards, values, next_values, dones):

        """
        1. Get bootstrap value for state after last step
        2. Initialize advantage = 0
        3. Loop backwards from n_steps-1 to 0:
        a. If done at step t: next_value = 0, reset advantage = 0
        b. Else: next_value = values[t+1] (or bootstrap for last step)
        c. delta = rewards[t] + gamma * next_value - values[t]
        d. advantage = delta + gamma * lambda * advantage
        e. Store advantage[t]
        4. Returns = advantages + values
        5. Normalize advantages (zero mean, unit std)
        6. Return advantages, returns
        """

        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(self.n_steps)):
            next_val = next_values[t].item()
            next_non_terminal = 1.0 - dones[t].item()

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t].item()
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            advantages[t] = last_advantage

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, obs, actions, old_log_probs, returns, advantages, old_values):
        """
        1. Generate random indices for the full rollout
        2. For n_epochs:
        a. Shuffle the indices
        b. For each minibatch slice:
            i.   Get minibatch of obs, actions, old_log_probs, advantages, returns
            ii.  Run model.evaluate(obs, actions) → new_log_probs, values, entropy
            iii. Compute ratio = exp(new_log_probs - old_log_probs)
            iv.  Compute clipped policy loss
            v.   Compute value loss
            vi.  Compute entropy bonus
            vii. Combine into total loss
            viii.Zero gradients, backprop, step optimizer
        3. Return mean total loss for logging
        """

        total_loss_sum = 0
        n_updates = 0

        indices = np.arange(self.n_steps)

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, self.n_steps, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                # Get minibatch
                obs_batch        = obs[batch_idx]
                actions_batch    = actions[batch_idx]
                old_log_probs_batch = old_log_probs[batch_idx]
                advantages_batch = advantages[batch_idx]
                returns_batch    = returns[batch_idx]

                # Forward pass
                new_log_probs, values, entropy = self.model.evaluate(obs_batch, actions_batch)

                # Policy loss
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surrogate1 = ratio * advantages_batch
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # Value loss
                old_values_batch = old_values[batch_idx]
                values_clipped = old_values_batch + torch.clamp(values - old_values_batch, -self.clip_epsilon, self.clip_epsilon)
                value_loss = torch.max(F.mse_loss(values, returns_batch), F.mse_loss(values_clipped, returns_batch))

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss_sum += loss.item()
                n_updates += 1

        return total_loss_sum / n_updates
    
    def train(self, total_steps):
        """
        1. Initialize TensorBoard SummaryWriter
        2. Initialize step counter, episode reward tracker
        3. While steps_collected < total_steps:
        a. collect_rollout() → get rollout data + last_obs
        b. compute_gae() → get advantages + returns
        c. update() → get mean loss
        d. Log everything to TensorBoard
        e. Every 10 rollouts: save model checkpoint
        f. Print progress to console
        g. Increment steps_collected by n_steps
        """

        writer = SummaryWriter(log_dir="runs/ppo_fortnite")

        steps_collected = 0
        rollout_count = 0

        print("Starting PPO training...")

        while steps_collected < total_steps:
            # Step 1: Collect rollout
            obs, actions, rewards, log_probs, values, next_values, dones, truncs = self.collect_rollout()

            # Step 2: Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)

            # Step 3: Update network
            mean_loss = self.update(obs, actions, log_probs, advantages, returns, values)

            # Step 4: Logging
            steps_collected += self.n_steps
            rollout_count += 1

            mean_reward = rewards.mean().item()
            n_episodes = max(dones.sum().item() + truncs.sum().item(), 1)
            mean_ep_len = self.n_steps / n_episodes

            writer.add_scalar("train/mean_reward",   mean_reward,   steps_collected)
            writer.add_scalar("train/mean_loss",      mean_loss,     steps_collected)
            writer.add_scalar("train/mean_ep_length", mean_ep_len,   steps_collected)

            print(f"Rollout {rollout_count:4d} | Steps: {steps_collected:7d} | "
                f"Mean Reward: {mean_reward:7.4f} | Loss: {mean_loss:.4f}")

            # Step 5: Save checkpoint every 10 rollouts

            # in train():
            if rollout_count % 5 == 0:
                self.model.save(f"checkpoint_{steps_collected}_{int(time.time())}.pth")
                print(f"Checkpoint saved at step {steps_collected}")

        writer.close()
        self.model.save("final_model.pth")
        print("Training complete.")
        
        pass