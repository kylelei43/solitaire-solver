"""
Klondike Solitaire (Turn-1, unlimited redeals) — Gymnasium Environment

This file provides:
  1) A functional Gymnasium environment `KlondikeSolitaireEnv` with action masking.
  2) A training scaffold using sb3-contrib's MaskablePPO.

Design notes:
- Variant: Klondike, draw 1 (Turn-1), unlimited redeals. Moving from foundation back to tableau is disabled (keeps state simpler).
- Observations: padded tableau representation + stock/waste/foundation summary + face-up mask.
- Actions: fixed discrete set with masking.
    * 0                          : draw from stock to waste
    * 1                          : move waste -> foundation (if legal)
    * 2..8 (7 actions)           : move waste -> tableau[k]
    * 9..15 (7 actions)          : move tableau[i] top -> foundation
    * 16..(16+546-1) (546 actions): move tableau[i] run of depth d (1..13) -> tableau[j], i!=j
- Rewards: sparse + shaped (tune as needed):
    * +100 on win
    * +1.0 per card moved to foundation
    * +0.5 when you flip a face-down card
    * +0.2 when placing a King into an empty tableau column
    * -0.01 per step (time penalty)
    * -0.05 for illegal action (no state change)

Dependencies:
- gymnasium>=0.29
- numpy
- sb3-contrib (for MaskablePPO) and stable-baselines3 (optional training scaffold)

Usage:
>>> import gymnasium as gym
>>> from solitaire_rl_env import KlondikeSolitaireEnv
>>> env = KlondikeSolitaireEnv(seed=42)
>>> obs, info = env.reset()
>>> mask = info["action_mask"]
>>> obs, reward, terminated, truncated, info = env.step(env.sample_legal_action(info))

Training (optional): see `if __name__ == "__main__":` at the bottom.
"""
from __future__ import annotations

import numpy as np

from solitaire_rl_env import KlondikeSolitaireEnv

# -------------------------
# Optional: SB3-Contrib MaskablePPO Training Scaffold
# -------------------------
if __name__ == "__main__":
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as e:
        print("Install sb3-contrib and stable-baselines3 to run training:")
        print("  pip install stable-baselines3 sb3-contrib gymnasium")
        raise


    def mask_fn(env: KlondikeSolitaireEnv) -> np.ndarray:
        return env._action_mask()


    def make_env(seed=0):
        def _thunk():
            env = KlondikeSolitaireEnv(seed=seed)
            env = ActionMasker(env, mask_fn)
            return env

        return _thunk

        # Vectorize (optional)


    env = DummyVecEnv([make_env(0)])

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.995,
        ent_coef=0.01,
        learning_rate=3e-4,
        tensorboard_log="./tb_solitaire/",
    )

    print("Starting training...")
    model.learn(total_timesteps=200_000)
    model.save("ppo_klondike_masked.zip")
    print("Saved model to ppo_klondike_masked.zip")

