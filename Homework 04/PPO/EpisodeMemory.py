import numpy as np

class EpisodeMemory:
    def __init__(self, episode_len, input_dim, num_actions):

        self.episode_len = episode_len
        self.input_dim = input_dim
        self.num_actions = num_actions

        self.reset()
        
    
    def reset(self):
        self.states = np.zeros(shape=(self.episode_len, *self.input_dim))
        self.values_target = np.zeros(shape=(self.episode_len, ))

        self.actions = np.zeros(shape=(self.episode_len, self.num_actions))
     
        self.log_probs = np.zeros(shape=(self.episode_len, self.num_actions), dtype=np.float32)

        self.advantages = np.zeros(shape=(self.episode_len, ))

    
        self.idx = 0
    
    def store(self, state, value_target, action, log_prob, advantage):
        idx = self.idx

        self.states[idx] = state 
        self.values_target[idx] = value_target

        self.actions[idx] = action 
        self.log_probs[idx] = log_prob

        self.advantages[idx] = advantage

        self.idx += 1