import sys

from matplotlib.pyplot import step
sys.path.append("../")

import gym
import imageio
import numpy as np
from DQN import *


class dummy_context_mgr():
    """
    A null object required for a conditional with statement
    """
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

def main():

    env = gym.make('LunarLander-v2')

    # Load model
    q_net = DQN(num_actions=env.action_space.n)

    q_net.build((1, *env.observation_space.shape))  # need a batch size

    for episode in range(0, 700, 100):

        q_net.load_weights(f"../saved_models/trainied_weights_epoch_{episode}")

  
    
        state = env.reset()

    
        gif_path = f"./GIFs/episode_{episode}.gif"

        done = False
        with imageio.get_writer(gif_path, mode='I', fps=55) if gif_path != "" else dummy_context_mgr() as gif_writer:
            while not done:
                # Add batch dim
                state = np.expand_dims(state, axis=0)
                # Predict best action
                target = q_net(state)

                target = target[0]  # Remove batch dim
                best_action = np.argmax(target)

                # Execute best action
                state, reward, done, _ = env.step(best_action)

                if gif_path != "":
                    img = env.render(mode='rgb_array')
                    gif_writer.append_data(img)




if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")