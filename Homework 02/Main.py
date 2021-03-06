from GridWorldGym import *
from Agent import *

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    # ------------------------------------------------
    # Argsparse stuff: Set parameter etc.
    # ------------------------------------------------

    parser = argparse.ArgumentParser(description="Present the MC or n-step SARSA algorithm. After the episodes are done the Q-Table is plotted: Each Q-value is presented.")
    parser.add_argument("--n", help="Set the n in n step SARSA (default = 1).", required=False, default = 1, type=int)
    parser.add_argument("--MC", help="Use MC Control with Exploring Starts instead of SARSA.", required=False, default = False, action='store_true')
    
    parser.add_argument("--plotInterval", help=r"Set X. Each X times the current Q-Table will be plotted and stored in ./Plots/{n_sarsa}-step_SARSA/episode_{num_episode}. The directory will be created if it does not exists. (default = 0 -> create no plots).", required=False, default=0, type=int)
    parser.add_argument("--episodes", help="Set the number of episodes (default = 100).", required=False, default=100, type=int)
    parser.add_argument("--lastPlotName", help="File name of the last plot.", required=False, default="")

    parser.add_argument("--alpha", help="Set the learning rate (default = 0.01).", required=False, default=0.01, type=float)
    parser.add_argument("--gamma", help="Set the discount factor (default = 0.99).", required=False, default=0.99, type=float)
    args = parser.parse_args()

    n_sarsa = args.n
    isMC = args.MC

    num_episodes = args.episodes 
    alpha = args.alpha
    gamma = args.gamma

    # ------------------------------------------------
    # Gym and Agent Init
    # ------------------------------------------------

    gym = GridWorldGym()
    agent = Agent(gym.GRID_SHAPE, gym.NUM_ACTIONS)

    gym.visualize()
    
    for num_episode in range(num_episodes):
        print(f"Run episode {num_episode}")

        done = False
        state = gym.reset()

        # MC Control with exploring starts
        if isMC:
            x = np.random.randint(0,5)
            y = np.random.randint(0,5)
            gym.state = (x,y)

        while not done: # no terminal state reached
            
            discounted_rewards = 0
            current_state = gym.state
            current_taken_actionID = None 

            next_state = None 

            # ------------------------------------------------
            # Reward chain: r_1 + ??^1 * r_2 + ??^2 * r_2 + ...
            # ------------------------------------------------
            k = 0
            while k < n_sarsa or isMC: # isMC = True -> Run until terminal state is reached
          
                best_action_idx = agent.choose_action(gym.state, gym.isValidAction)

                # Do not forget the first taken action for the prediction (see a) )
                if k == 0:
                    current_taken_actionID = best_action_idx
                    
            
                next_state, reward, isTerminal = gym.step(best_action_idx)
                state = next_state
                done = isTerminal

                discounted_rewards += gamma**k * reward

                if done:
                    break
              
                k += 1
            
            # ------------------------------------------------
            # Q part: ... + ??^n_sarsa * Q(s',a') 
            # 
            # note a' is selected also as the "best" action 
            # based on the Q-Table
            # ------------------------------------------------
            
            # a) need first taken action 
            predict = agent.q_table[current_state[0], current_state[1], current_taken_actionID]

            # already done -> take discounted rewards as target
            if done: # isMC -> this branch is taken
                target = discounted_rewards
            else:
                best_action_idx = agent.choose_action(state, gym.isValidAction)
                target = discounted_rewards + gamma**k *agent.q_table[state[0], state[1], best_action_idx]
            
            agent.q_table[current_state[0], current_state[1], current_taken_actionID] = predict + alpha *(target - predict)
               
        agent.strategy.reduce_epsilon()

        # ------------------------------------------------
        # Plot each X episodes the Q-Table
        # ------------------------------------------------
        isPeriodicPlot = (args.plotInterval != 0 and num_episode % args.plotInterval == 0)
        isLastPlot = num_episode == args.episodes - 1
        if isPeriodicPlot or isLastPlot:
            
            fig, axs = plt.subplots(nrows=2, ncols=2)

            cnt = 0
            for i in range(2):
                for j in range(2):
                    img = axs[i, j].imshow(agent.q_table[:, :,cnt])
                    plt.colorbar(img, ax=axs[i, j])
                    axs[i, j].set_axis_off()
                    cnt += 1
        
            axs[0, 0].set_title("TOP")
            axs[0, 1].set_title("DOWN")
            axs[1, 0].set_title("RIGHT")
            axs[1, 1].set_title("LEFT")

            if isMC:
                plt.suptitle(f"Q-values: MC Control with Exploring Starts - {num_episode}th episode")
            else:
                plt.suptitle(f"Q-values: {n_sarsa}-step SARSA - {num_episode}th episode")

            fig.set_size_inches(10, 10)
            plt.tight_layout()
            

            if isLastPlot and args.lastPlotName != "": # last plot shall be saved
                plt.savefig(args.lastPlotName)

            # Save periodically the plot: all args.plot times
            if args.plotInterval != 0:
                
                # Create dir if not exists
                if isMC:
                    path = f"./Plots/MC_Control"
                else:
                    path = f"./Plots/{n_sarsa}-step_SARSA"

                if not os.path.exists(path):
                    os.makedirs(path)
             
                plt.savefig(f"{path}/episode_{num_episode}")
            else:
                plt.show()         

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")