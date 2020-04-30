#!/usr/bin/env python
# coding: utf-8

# In[3]:


import gym
from gym import error, spaces, utils
from gym.utils import seeding


def cmp(a, b):
    return int((a > b)) - int((a < b))

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return np_random.choice(deck)


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
            return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)

def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]

class Blackjack3DecksEnv(gym.Env):
    def __init__(self, ply_deck=False, dl_deck=False, game_round=False, has_df=False, natural=False):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32),
            spaces.Discrete(11),
            spaces.Discrete(2)))
        self.ply_deck = ply_deck
        self.dl_deck = dl_deck
        self.game_round = game_round
        self.has_df = has_df
        self.natural = natural
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def draw_kaggle_dl(self):
        return self.dl_deck[self.game_round][self.dealer.size]
    
    def draw_kaggle_ply(self):
        return self.ply_deck[self.game_round][self.player.size]
    
    def step(self, action):
        assert self.action_space.contains(action)
        if action:  # hit: add a card to players hand and return
            if(self.has_df):
                self.dealer.append(draw_kaggle_ply())
            else:
                self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        else:  
            done = True
            while sum_hand(self.dealer) < 17:
                if(self.has_df):
                    self.dealer.append(draw_kaggle_dl())
                else:
                    self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        if(self.has_df):
            self.dealer = draw_hand(draw_kaggle_dl())
            self.player = draw_hand(draw_kaggle_ply())
        else:
            self.dealer = draw_hand(self.np_random)
            self.player = draw_hand(self.np_random)
        return self._get_obs()


# In[ ]:




