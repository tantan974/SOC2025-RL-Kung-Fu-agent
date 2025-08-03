import gymnasium as gym
import ale_py            # Required to register Atari envs
import cv2
import numpy as np
from collections import deque
from memory import ReplayMemory
from agent import DQNAgent

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(resized, dtype=np.uint8)

def stack_frames(stacked_frames, state, is_new_episode):
    if is_new_episode:
        stacked_frames.clear()
        for _ in range(4):
            stacked_frames.append(state)
    else:
        stacked_frames.append(state)
    stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

def main():
    print("Starting training...")

    env = gym.make('ALE/KungFuMaster-v5', render_mode='rgb_array')
    action_size = env.action_space.n

    agent = DQNAgent(action_size)
    memory = ReplayMemory(100000)
    stacked_frames = deque(maxlen=4)

    num_episodes = 150
    batch_size = 32

    for episode in range(1, num_episodes + 1):
        print(f"--- Starting episode {episode} ---")
        obs, info = env.reset()
        state = preprocess_frame(obs)
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_state = preprocess_frame(obs)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            memory.push((state, action, reward, next_state, done))
            state = next_state

            agent.train(memory, batch_size)

        # Every 5 episodes, update target network and save weights
        if episode % 5 == 0:
            agent.update_target_model()
            agent.model.save_weights('trained_agent.weights.h5')
            print(f"Saved model weights at episode {episode}")
            print("Target network updated.")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon -= 1/num_episodes
        else:
            agent.epsilon = agent.epsilon_min

        print(f"Episode {episode}/{num_episodes} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.3f}")

    env.close()
    print("Training finished!")

if __name__ == "__main__":
    main()