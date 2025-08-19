import torch
import gymnasium as gym
from dqn_cartpole import QNetwork   # <- from your training script

if __name__ == "__main__":
    # Same architecture and device as training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", render_mode="human")  # "human" shows live window

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load network
    q_net = QNetwork(state_dim, action_dim).to(device)
    q_net.load_state_dict(torch.load("dqn_cartpole.pt", map_location=device))
    q_net.eval()

    for episode in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                q_values = q_net(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(q_values.argmax(dim=1).item())

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1}: return = {total_reward}")

    env.close()
