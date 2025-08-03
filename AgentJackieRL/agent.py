from helper import KungFu
import numpy as np

class DQNAgent:
    def __init__(self, action_size, learn_rate=0.00025):
        self.possible_actions = list(range(action_size))  # Must be set before KungFu()
        self.learn_rate = learn_rate

        # Bind the external KungFu method to this instance so it acts like a method
        self.KungFu = KungFu.__get__(self)

        # Now build the model and target_network by calling bound method
        self.model = self.KungFu()
        self.target_model = self.KungFu()

        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.possible_actions)
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        return int(np.argmax(q_values))

    def train(self, memory, batch_size):
        if len(memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        q_values = self.model.predict(states, verbose=0)
        q_next = self.target_model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(q_next[i])

        self.model.train_on_batch(states, q_values)
