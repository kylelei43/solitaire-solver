import argparse

from solitaire_rl_env import KlondikeSolitaireEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained policy in the console")
    parser.add_argument("--model-path", default="ppo_klondike_masked.zip", help="Path to MaskablePPO model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the environment")
    parser.add_argument("--auto", action="store_true", help="Run policy automatically without pause")
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
    except Exception as exc:  # pragma: no cover - import error handling
        raise SystemExit("sb3-contrib is required to load MaskablePPO models") from exc

    env = KlondikeSolitaireEnv(seed=args.seed)
    model = MaskablePPO.load(args.model_path)

    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0.0

    while not (done or truncated):
        mask = info.get("action_mask")
        action, _ = model.predict(obs, action_masks=mask)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        print(env.render())
        print(f"step={env.steps} reward={reward:.2f} total={total_reward:.2f}")
        if not args.auto:
            input("Press Enter to continue...")

    env.close()
    print(f"Finished after {env.steps} steps, total reward {total_reward:.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()