# dqn_cartpole.py
import math
import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def select_action(q_net, state, epsilon, action_space, device):
    if random.random() < epsilon:
        return action_space.sample()
    with torch.no_grad():
        q = q_net(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
        return int(q.argmax(dim=1).item())


def compute_td_targets(target_net, rewards, next_states, dones, gamma, device):
    with torch.no_grad():
        next_q = target_net(torch.as_tensor(next_states, dtype=torch.float32, device=device))
        max_next_q = next_q.max(dim=1).values
        targets = torch.as_tensor(rewards, dtype=torch.float32, device=device) + \
                  gamma * (1.0 - torch.as_tensor(dones, dtype=torch.float32, device=device)) * max_next_q
    return targets


def train_step(q_net, target_net, optimizer, batch, gamma, device):
    states = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch.action, dtype=torch.long, device=device)
    rewards = np.array(batch.reward, dtype=np.float32)
    next_states = np.array(batch.next_state, dtype=np.float32)
    dones = np.array(batch.done, dtype=np.float32)

    # Q(s,a)
    q_values = q_net(states).gather(1, actions.view(-1, 1)).squeeze(1)
    # r + gamma * max_a' Q_target(s', a') * (1-done)
    targets = compute_td_targets(target_net, rewards, next_states, dones, gamma, device)

    loss = nn.functional.mse_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def main():
    env = gym.make("CartPole-v1")  # set render_mode="human" to visualize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=50_000)

    # Hyperparameters
    gamma = 0.99
    batch_size = 64
    start_training_after = 1_000  # warmup steps before updates
    update_every = 4  # gradient step frequency
    target_sync_every = 1_000  # steps to copy weights to target
    max_steps = 1_000_000

    # Epsilon-greedy schedule: from 1.0 -> 0.05
    eps_start, eps_end, eps_decay = 1.0, 0.05, 30_000  # decay over ~30k steps

    episode_rewards = []
    state, _ = env.reset(seed=0)
    total_steps = 0
    episode_return = 0.0
    best_avg100 = -float("inf")

    for step in range(1, max_steps + 1):
        epsilon = eps_end + (eps_start - eps_end) * math.exp(-1.0 * total_steps / eps_decay)

        action = select_action(q_net, state, epsilon, env.action_space, device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        episode_return += reward
        total_steps += 1

        # Learn
        if len(buffer) >= start_training_after and step % update_every == 0:
            batch = buffer.sample(batch_size)
            loss = train_step(q_net, target_net, optimizer, batch, gamma, device)

        # Sync target net
        if step % target_sync_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        # End of episode
        if done:
            episode_rewards.append(episode_return)
            avg100 = np.mean(episode_rewards[-100:])
            print(f"Steps={total_steps:6d}  EpReturn={episode_return:6.1f}  "
                  f"Avg100={avg100:6.1f}  Eps={epsilon:0.3f}")
            if avg100 > best_avg100:
                best_avg100 = avg100
                # lightweight checkpoint
                torch.save(q_net.state_dict(), "dqn_cartpole.pt")
            state, _ = env.reset()
            episode_return = 0.0

            # Consider "solved" when Avg100 >= 475 (CartPole-v1 max is 500)
            if avg100 >= 475.0 and len(episode_rewards) >= 100:
                print("Solved! Saved to dqn_cartpole.pt")
                break

    env.close()

    # Quick evaluation run (no exploration)
    def evaluate(n_episodes=10):
        env_eval = gym.make("CartPole-v1")
        returns = []
        q_net.eval()
        with torch.no_grad():
            for _ in range(n_episodes):
                s, _ = env_eval.reset()
                done = False
                ep_ret = 0.0
                while not done:
                    q = q_net(torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0))
                    a = int(q.argmax(dim=1).item())
                    s, r, term, trunc, _ = env_eval.step(a)
                    done = term or trunc
                    ep_ret += r
                returns.append(ep_ret)
        env_eval.close()
        print(f"Eval avg return over {n_episodes} episodes: {np.mean(returns):.1f}")

    evaluate()


if __name__ == "__main__":
    main()
