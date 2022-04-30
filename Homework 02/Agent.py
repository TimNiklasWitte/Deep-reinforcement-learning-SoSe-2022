from EpsilonGreedyStrategy import *
import numpy as np

class Agent:

    def __init__(self, state_shape, num_actions):
        self.strategy = EpsilonGreedyStrategy(start=1.0, end=0.05, decay=0.99)
        self.q_table = np.zeros(shape=(*state_shape, num_actions))

        self.num_actions = num_actions

    def choose_action(self, state, isValidAction):
        validActionFound = False 

        while not validActionFound:
            
            # Exploration
            if np.random.random() < self.strategy.get_exploration_rate():
                action_idx = np.random.randint(0, self.num_actions)

            # Exploitation
            else:
                q_values = self.q_table[state[0], state[1], :]
                action_idx = np.argmax(q_values)
                   
            validActionFound, _ = isValidAction(action_idx)
        
        return action_idx