from GridWorldGym import *
from EpsilonGreedyStrategy import *
import numpy as np

import matplotlib.pyplot as plt

def main():
    gym = GridWorldGym()
    actions = gym.Actions
    
    strategy = EpsilonGreedyStrategy(start=1.0, end=0.05, decay=0.99)

    n_sarsa = 5
    num_episodes = 5000
    alpha = 0.01
    gamma = 0.99

    q_table = np.zeros(shape=(*gym.GRID_SHAPE, gym.NUM_ACTIONS))

    for i in range(num_episodes):
       
        done = False

        state = gym.reset()
      
        while not done:
            
            # Choose an action
            validActionFound = False 
            while not validActionFound:

                # Exploration
                if np.random.random() < strategy.get_exploration_rate():
                    best_action_idx = np.random.randint(0, gym.NUM_ACTIONS)

                # Exploitation
                else:
                    q_values = q_table[state[0], state[1], :]
                    best_action_idx = np.argmax(q_values)
                
                best_action = mapIntToAction(best_action_idx, actions)
                    
                validActionFound, _ = gym.isValidAction(best_action) 

            # discounted_reward = 0
            # for _ in range(n_sarsa):
            #     state = gym.state_transistion_function(state, action)
               
            next_state, reward, isTerminal = gym.step(best_action)

            q_next = q_table[next_state[0],next_state[1], :]
            next_best_action_idx = np.argmax(q_next)

            predict = q_table[state[0], state[1], best_action_idx]
            target = reward + gamma*q_table[next_state[0], next_state[1], next_best_action_idx]

            q_table[state[0], state[1], best_action_idx] = predict + alpha *(target - predict)

            state = next_state
            done = isTerminal
        
    
        
        strategy.reduce_epsilon()

    print(strategy.get_exploration_rate())
    
    gym.visualize()

    fig, axs = plt.subplots(nrows=2, ncols=2)

    cnt = 0
    for i in range(2):
        for j in range(2):
            img = axs[i, j].imshow(q_table[:, :,cnt])
            plt.colorbar(img, ax=axs[i, j])
            axs[i, j].set_axis_off()
            cnt += 1
 
    axs[0, 0].set_title("TOP")
    axs[0, 1].set_title("DOWN")
    axs[1, 0].set_title("RIGHT")
    axs[1, 1].set_title("LEFT")

    plt.tight_layout()
    plt.show()


def mapIntToAction(actionID, actions):
    if actionID == 0:
        return actions.TOP
    elif actionID == 1:
        return actions.DOWN
    elif actionID == 2:
        return actions.RIGHT
    else:
        return actions.LEFT 

    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")