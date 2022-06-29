import gym
import numpy as np
from numpy.random import default_rng

from PolicyNet import *

import tqdm

import matplotlib.pyplot as plt


def preprocess(state):

    state = np.cast['float32'](state)
    state = (state / 128.) - 1
    return state


def make_env():
    return gym.make("CarRacing-v1")

def main():

    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    batch_size = 16
    episode_len = 500
    gamma = 0.99
        
    env = gym.vector.make("CarRacing-v1", num_envs=batch_size)

    policy_net = PolicyNet()
    policy_net.build(env.observation_space.shape)
    policy_net.summary()


    for num_episode in range(250):

        buff_states = np.zeros(shape=(episode_len, batch_size, *env.observation_space.shape[1:]), dtype=np.float32)
        buff_rewards = np.zeros(shape=(episode_len, batch_size), dtype=np.float32)

        states = env.reset()
        for idx in tqdm.tqdm(range(episode_len), position=0, leave=True):
            states = preprocess(states)
            actions, _ = policy_net(states)
            actions = actions.numpy()
         
            next_states, rewards, dones , _ = env.step(actions)

            buff_states[idx] = states
            buff_rewards[idx] = rewards

            if np.any(dones):
                next_states = env.reset()

            states = next_states
        
        # Evaluation: Consider only first env
        rewards = buff_rewards[:, 0]
        score = np.sum(rewards)
        avg_rewards = np.mean(rewards)
        
        print(f"  Episode: {num_episode}")
        print(f"    Score: {round(score,2)}")
        print(f"Avg Score: {round(avg_rewards, 2)}")
        print("------------------------") 

        with train_summary_writer.as_default():
            tf.summary.scalar(f"Average reward", avg_rewards, step=num_episode)
            tf.summary.scalar(f"Score", score, step=num_episode)


        for start_idx in range(episode_len):

            g_t = np.zeros(shape=(batch_size,), dtype=np.float32)
            for i in range(episode_len - start_idx): #range(len(rewards[j:])):
                rewards = buff_rewards[start_idx + i, :]
                g_t += (gamma**i) * rewards

            states = buff_states[start_idx]

            policy_net.train_step(states, g_t)
   


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
