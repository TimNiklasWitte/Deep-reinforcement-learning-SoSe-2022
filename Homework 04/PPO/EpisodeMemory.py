import numpy as np

class EpisodeMemory:
    def __init__(self, episode_len, input_dim, num_actions):

        self.episode_len = episode_len
        self.input_dim = input_dim
        self.num_actions = num_actions

        self.reset()
        
    
    def reset(self):
        self.states = np.zeros(shape=(self.episode_len, *self.input_dim))
        self.values = np.zeros(shape=(self.episode_len, ))

        self.actions = np.zeros(shape=(self.episode_len, self.num_actions))
     
        self.log_probs = np.zeros(shape=(self.episode_len, self.num_actions))
        
        self.rewards = np.zeros(shape=(self.episode_len, ))
        self.advantages = np.zeros(shape=(self.episode_len, ))

        self.next_states = np.zeros(shape=(self.episode_len, *self.input_dim))
        
        self.idx = 0
    
    def store(self, state, value, action, log_prob, reward, advantage, next_state):
        idx = self.idx

        self.states[idx] = state 
        self.values[idx] = value

        self.actions[idx] = action 
        self.log_probs[idx] = log_prob

        self.rewards[idx] = reward
        self.advantages[idx] = advantage

        self.next_states[idx] = next_state 

        self.idx += 1