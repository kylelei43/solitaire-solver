import random
from collections import namedtuple

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib.common.wrappers import ActionMasker
from torch import nn, optim

from solitaire_rl_env import KlondikeSolitaireEnv

ACTION_N = 562

Transition = namedtuple("Transition", "obs action reward next_obs done mask next_mask")

def flatten_obs(obs: dict) -> np.ndarray:
    """Flatten observation dict into a 1D array."""
    parts = [
        obs["tableau_cards"].reshape(-1).astype(np.float32),
        obs["tableau_faceup"].reshape(-1).astype(np.float32),
        obs["foundations"].reshape(-1).astype(np.float32),
        obs["waste_top"].reshape(-1).astype(np.float32),
        obs["stock_count"].reshape(-1).astype(np.float32),
        obs["steps"].reshape(-1).astype(np.float32),
    ]
    x = np.concatenate(parts, axis=0)
    # map paddings -1 (no card) to a distinct value
    x[x == -1.0] = -2.0
    return x

def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor, probabilistic: bool = False) -> torch.Tensor:
    # q_values: (B, A), mask: (B, A) with True=legal
    neg_inf = torch.finfo(q_values.dtype).min
    masked_q = q_values.masked_fill(~mask, neg_inf)
    # If a row has no legal action, argmax on all -inf returns 0; you should avoid such states or treat them as terminal
    # Instead of use argmax, choice by probability
    if probabilistic:
        probs = torch.softmax(masked_q, dim=1)
        action = torch.multinomial(probs, num_samples=1).squeeze(1)
        return action
    else:
        return masked_q.argmax(dim=1)

def masked_max(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.finfo(q_values.dtype).min
    masked_q = q_values.masked_fill(~mask, neg_inf)
    # Replace all -inf rows (no legal actions) with 0 for stability
    row_all_illegal = (~mask).all(dim=1)
    max_vals, _ = masked_q.max(dim=1)
    max_vals[row_all_illegal] = 0.0
    return max_vals

class QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = ACTION_N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            # nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, A)

class DuelingQ(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fe = nn.Sequential(nn.Linear(obs_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.V  = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.A  = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_dim))
    def forward(self, x):
        h = self.fe(x)
        v = self.V(h)
        a = self.A(h)
        return v + (a - a.mean(dim=1, keepdim=True))

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buf = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, *args):
        max_prio = self.priorities.max() if self.buf else 1.0
        if len(self.buf) < self.capacity:
            self.buf.append(Transition(*args))
        else:
            self.buf[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buf) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buf), batch_size, p=probs)
        samples = [self.buf[idx] for idx in indices]
        weights = (len(self.buf) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        batch = Transition(*zip(*samples))
        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities

    def __len__(self):
        return len(self.buf)

def select_action(qnet, obs_vec, mask_np, eps, device):
    """Epsilon-greedy action selection."""
    # mask_np: boolean numpy array (A,)
    legal_idxs = np.flatnonzero(mask_np)
    if len(legal_idxs) == 0:
        raise ValueError("No legal actions")
    if random.random() < eps:
        return int(random.choice(legal_idxs))

    with torch.no_grad():
        x = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)  # (1, D)
        q = qnet(x) # (1, A)
        mask = torch.tensor(mask_np, dtype=torch.bool, device=device).unsqueeze(0)  # (1, A)
        a = masked_argmax(q, mask).item()
        return int(a)

def dqn_train_step(batch, weights, qnet, target_qnet, optimizer, gamma, device, double_dqn: bool = True):
    obs = torch.tensor(np.array(batch.obs), dtype=torch.float32, device=device)
    act = torch.tensor(np.array(batch.action), dtype=torch.long,    device=device)
    rew = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=device)
    nxt = torch.tensor(np.array(batch.next_obs), dtype=torch.float32, device=device)
    done = torch.tensor(np.array(batch.done), dtype=torch.float32,   device=device)
    mask = torch.tensor(np.array(batch.mask), dtype=torch.bool,      device=device)
    next_mask = torch.tensor(np.array(batch.next_mask), dtype=torch.bool, device=device)

    # Q(s,a)
    q_all = qnet(obs)  # (B, A)
    q_sa = q_all.gather(1, act.view(-1, 1)).squeeze(1)  # (B,)

    with torch.no_grad():
        if double_dqn:
            # online argmax under mask, evaluate with target net
            q_next_online = qnet(nxt)  # (B, A)
            a_next = masked_argmax(q_next_online, next_mask)  # (B,)
            q_next_target = target_qnet(nxt)  # (B, A)
            q_next_max = q_next_target.gather(1, a_next.view(-1, 1)).squeeze(1)  # (B,)
        else:
            q_next_target = target_qnet(nxt)  # (B, A)
            q_next_max = masked_max(q_next_target, next_mask)

        target = rew + gamma * (1.0 - done) * q_next_max  # (B,)

    td_error = q_sa - target
    loss = (weights * nn.functional.smooth_l1_loss(q_sa, target, reduction="none")).mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
    optimizer.step()
    return loss.item(), td_error.detach().cpu().numpy()

def mask_fn(env):
    return env._action_mask()

def make_env(seed=0) -> gym.Env:
    env = KlondikeSolitaireEnv(seed=seed)
    # ActionMasker ensures info['action_mask'] is always present for us to store
    return ActionMasker(env, mask_fn)

def train(max_steps: int = 500_000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(0)

    # Get obs_dim
    obs, info = env.reset()
    obs_dim = flatten_obs(obs).shape[0]

    qnet = DuelingQ(obs_dim, ACTION_N).to(device)
    target_qnet = DuelingQ(obs_dim, ACTION_N).to(device)
    target_qnet.load_state_dict(qnet.state_dict())
    optimizer = optim.Adam(qnet.parameters(), lr=3e-4)

    buffer = PrioritizedReplayBuffer(capacity=20_000)

    epsilon_start, epsilon_end, epsilon_decay_steps = 1.0, 0.05, 100_000

    def beta_by_step(step, beta_start=0.4, beta_frames=100_000):
        t = min(1.0, step / beta_frames)
        return beta_start + t * (1.0 - beta_start)


    def epsilon(iters):  # linear decay
        t = min(1.0, iters / epsilon_decay_steps)
        return epsilon_start + t * (epsilon_end - epsilon_start)

    @torch.no_grad()
    def soft_update(target, online, tau=0.005):
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)

    gamma = 0.99
    batch_size = 512
    warmup = 20_000
    target_evaluate_interval = 5000  # steps
    train_every = 10  # learn every N env steps
    gradient_steps = 5  # G updates per learn call

    obs, info = env.reset()
    obs_vec = flatten_obs(obs)
    mask = info["action_mask"]
    global_step = 0
    total_reward = 0.0
    all_rewards = []
    all_wins = []

    while global_step < max_steps:
        # --- act ---
        a = select_action(qnet, obs_vec, mask, eps=epsilon(global_step), device=device)
        next_obs, r, terminated, truncated, next_info = env.step(a)
        done = terminated or truncated

        # Store transition (with masks!)
        next_obs_vec = flatten_obs(next_obs)
        next_mask = next_info["action_mask"]
        buffer.push(obs_vec, a, float(r), next_obs_vec, float(done), mask.copy(), next_mask.copy())

        obs_vec, mask = next_obs_vec, next_mask
        global_step += 1
        total_reward += r

        if done:
            obs, info = env.reset()
            obs_vec = flatten_obs(obs)
            mask = info["action_mask"]

            final_foundation_sum = next_info.get("foundation_sum", 0)
            win = final_foundation_sum == 52
            all_wins.append(win)
            all_rewards.append(total_reward)

            # if len(all_rewards) >= 100:
            #     print(f"Episode done, total reward={total_reward:.1f}, foundation_sum={final_foundation_sum}, "
            #           f"avg100={np.mean(all_rewards[-100:]):.1f}, win_rate={np.mean(all_wins[-100:]):.2f}")
            # else:
            #     print(f"Episode done, total reward={total_reward:.1f}, foundation_sum={final_foundation_sum}")
            total_reward = 0.0

        # --- learn ---
        if len(buffer) >= warmup and (global_step % train_every == 0):
            losses = []
            for _ in range(gradient_steps):
                batch, indices, weights = buffer.sample(batch_size, beta=beta_by_step(global_step))
                weights = weights.to(device)
                loss, td_err = dqn_train_step(batch, weights, qnet, target_qnet, optimizer, gamma, device, double_dqn=True)
                buffer.update_priorities(indices, np.abs(td_err) + 1e-6)
                losses.append(loss)

            if global_step % 1000 == 0:
                print(f"Step {global_step}, loss={np.mean(losses):.3f}")

        # --- soft target update ---
        soft_update(target_qnet, qnet, tau=0.005)

        # --- target sync ---
        if global_step % target_evaluate_interval == 0:
            evaluate(target_qnet, n_episodes=100, seed=global_step, auto_play=True)

    return target_qnet

def evaluate(model, n_episodes=100, seed=1, auto_play=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wins, sums, rewards = 0, [], []
    for ep in range(n_episodes):
        env = make_env(seed + ep)
        obs, info = env.reset()
        obs_vec = flatten_obs(obs)
        mask = info["action_mask"]
        done = False
        reward = 0.0
        while not done:
            with torch.no_grad():
                x = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
                q = model(x)[0]
                mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                a = masked_argmax(q.unsqueeze(0), mask_t).item()
            obs, r, terminated, truncated, info = env.step(a)
            obs_vec = flatten_obs(obs)
            mask = info["action_mask"]
            done = terminated or truncated
            reward += r

            if not auto_play:
                print(env.render())
                input("Press Enter to continue...")


        sums.append(info.get("foundation_sum", 0))
        wins += (sum(env.unwrapped.foundations) == 52)
        rewards.append(reward)
    print(f"Eval: win_rate={wins / n_episodes:.2f}, avg_foundation_sum={np.mean(sums):.1f}, "
          f"avg_reward={np.mean(rewards):.1f}")


TRAINING = True
if __name__ == "__main__":
    if TRAINING:
        model = train()
        torch.save(model.state_dict(), "dqn_solitaire.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(0)
    # Get obs_dim
    obs, info = env.reset()
    obs_dim = flatten_obs(obs).shape[0]
    model = DuelingQ(obs_dim, ACTION_N).to(device)
    model.load_state_dict(torch.load("dqn_solitaire.pt", map_location=device))
    evaluate(model, auto_play=True)
