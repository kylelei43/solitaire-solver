import numpy as np
from sb3_contrib.common.wrappers import ActionMasker

from solitaire_rl_env import KlondikeSolitaireEnv

def mask_fn(env):
    return env._action_mask()

def eval_random(n_episodes=1000, seed=0):
    wins, sums = 0, []
    for ep in range(n_episodes):
        env = ActionMasker(KlondikeSolitaireEnv(seed=seed+ep), mask_fn)
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            legal = np.flatnonzero(info["action_mask"])
            a = int(np.random.choice(legal)) if len(legal) else 0
            obs, reward, done, truncated, info = env.step(a)
        sums.append(info.get("foundation_sum", 0))
        wins += (sum(env.unwrapped.foundations) == 52)
        print(f"Episode {ep} done, foundation_sum={info.get('foundation_sum', 0)}")
    print({"win_rate": wins/n_episodes, "avg_foundation_sum": float(np.mean(sums))})

if __name__ == "__main__":
    eval_random(1000)