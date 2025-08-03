import gymnasium as gym
import ale_py
import numpy as np
import cv2
from collections import deque
from agent import DQNAgent

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(resized, dtype=np.uint8)

def stack_frames(stacked_frames, state, new_episode):
    if new_episode:
        stacked_frames.clear()
        for _ in range(4):
            stacked_frames.append(state)
    else:
        stacked_frames.append(state)
    return np.stack(stacked_frames, axis=2), stacked_frames

def watch_gameplay(num_episodes=3):
    env = gym.make('ALE/KungFuMaster-v5', render_mode='rgb_array')
    action_size = env.action_space.n
    agent = DQNAgent(action_size)

    agent.model.load_weights('trained_agent.weights.h5')
    agent.epsilon = 0.0

    stacked_frames = deque(maxlen=4)

    for episode in range(num_episodes):
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

            cv2.imshow('Agent Gameplay', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(30) & 0xFF == ord('q'):
                done = True
                break

            next_state = preprocess_frame(obs)
            state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        print(f"Episode {episode+1} ended with total reward: {total_reward}")

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    watch_gameplay()