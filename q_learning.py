import numpy as np


class QLAgent():
    def __init__(self, env, learning_rate, gamma, train_amount, epsilon=1.):
        self.env = env
        self.valid_actions = list(range(self.env.action_space.n))

        # params
        self.Q = dict()  # Q[(state)[(action, q-value)], state is the observation from env
        self.epsilon = epsilon  # exploration rate [0.01 ~ 1], probability of taking the best action
        self.lr = learning_rate  # aka. alpha, used to adjust step size
        self.gamma = gamma  # discount factor, when closer to 1 more sensitive to reward
        self.train_amount = train_amount
        self.done_train_amount = 0

    def update_epsilon(self):
        rest = self.train_amount - self.done_train_amount
        if rest > 0.8 * self.train_amount:
            self.epsilon = 0.9
        elif rest > 0.6 * self.train_amount:
            self.epsilon = 0.8
        elif rest > 0.5 * self.train_amount:
            self.epsilon = 0.6
        elif rest > 0.4 * self.train_amount:
            self.epsilon = 0.5
        elif rest > 0.2 * self.train_amount:
            self.epsilon = 0.2
        elif rest >= 0:
            self.epsilon = 0.01
        else:
            self.epsilon = 0

        self.done_train_amount += 1

    def get_q_value(self, state, action):
        return self.Q[state][action]

    def max_q_value(self, state):
        # if the state is not listed in the Q table
        if state not in self.Q:
            self.Q[state] = dict((action, 0.0) for action in self.valid_actions)
        return max(self.Q[state].values())

    def choose_action(self, state):
        # if the state is not listed in the Q table
        if state not in self.Q:
            self.Q[state] = dict((action, 0.0) for action in self.valid_actions)
        # epsilon greedy
        # when random number > epsilon, select the best action at each state (aka. "exploitation")
        if np.random.rand() > self.epsilon:
            max_q = self.max_q_value(state)
            for keys, values in self.Q[state].items():
                if values == max_q:
                    return keys
        else:
            # do exploration by randomly choose action
            return np.random.choice(self.valid_actions)

    def learn(self, state, action, reward, next_state):
        # Bellman Equation
        self.Q[state][action] = self.Q[state][action] + \
                                self.lr * (reward + self.gamma * self.max_q_value(next_state) - self.Q[state][action])
        self.update_epsilon()
        return self.Q[state][action]
