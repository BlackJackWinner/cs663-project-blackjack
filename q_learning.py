import random


class QLAgent():
    def __init__(self, env, epsilon, learning_rate, gamma, num_episodes_to_train=30000):
        self.env = env
        self.valid_actions = list(range(self.env.action_space.n))

        # params
        self.Q = dict()  # Q[(state)[(action, q-value)], state is the observation from env
        self.epsilon = epsilon  # Random exploration factor
        self.lr = learning_rate  # aka. alpha, used to adjust step size
        self.gamma = gamma  # discount factor, used to balance immediate and future reward
        self.num_episodes_to_train_left = num_episodes_to_train

    def get_q_value(self, state, action):
        return self.Q[state][action]

    def max_q_value(self, state):
        if state not in self.Q:
            self.Q[state] = dict((action, 0.0) for action in self.valid_actions)
        return max(self.Q[state].values())

    def choose_action(self, state):
        if self.epsilon > random.random():
            if state not in self.Q:
                self.Q[state] = dict((action, 0.0) for action in self.valid_actions)
            max_q = self.max_q_value(state)
            for keys, values in self.Q[state].items():
                if values == max_q:
                    return keys
        else:
            return random.choice(self.valid_actions)

    def learn(self, state, action, reward, next_state):
        self.Q[state][action] = self.Q[state][action] + \
                                self.lr * (reward + self.gamma * self.max_q_value(next_state) - self.Q[state][action])
        return self.Q[state][action]

