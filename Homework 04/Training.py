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

def main():

    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    num_envs = 15
    episode_len = 1000
    gamma = 0.99
    batch_size = 32

    
    env = gym.vector.make("CarRacing-v0", num_envs=num_envs)
    env.close()
    policy_net = PolicyNet()
    policy_net.build(env.observation_space.shape)
    policy_net.summary()

    for num_episode in range(250):
        
        env = gym.vector.make("CarRacing-v0", num_envs=num_envs)

        #
        # Sampling
        #
 
        buff_states = np.zeros(shape=(episode_len, num_envs, *env.observation_space.shape[1:]), dtype=np.uint8)
        buff_rewards = np.zeros(shape=(episode_len, num_envs), dtype=np.float32)

        states = env.reset()
        for idx in tqdm.tqdm(range(episode_len), position=0, leave=True):

            states = preprocess(states)
            action = policy_net(states)

            action = action.numpy() # remove batch dim and convert to numpy
            action[:,0] = 2*action[:,0] - 1
            
            next_states, rewards, dones , _ = env.step(action)

            buff_states[idx] = states
            buff_rewards[idx] = rewards

            if np.any(dones):
                states = env.reset()

            states = next_states
        
        env.close()
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


        # calc return
        buff_returns = np.zeros(shape=(episode_len, num_envs), dtype=np.float32)

        for start_idx in range(episode_len):

            g_t = np.zeros(shape=(num_envs,), dtype=np.float32)
            for i in range(episode_len - start_idx): #range(len(rewards[j:])):
                rewards = buff_rewards[start_idx + i, :]
                g_t += (gamma**i) * rewards

            buff_returns[start_idx] = g_t

        # Merge two axes: episode_len and num_envs 
        buff_size =  episode_len * num_envs 
        buff_states = np.reshape(buff_states, newshape=(buff_size, *env.observation_space.shape[1:]) )
        buff_returns = np.reshape(buff_returns, newshape=(buff_size,))


        #
        # Training
        #

        # sample batch
        rng = default_rng()
        batch_indices = rng.choice(buff_size, size=batch_size, replace=False)

        sampled_states = buff_states[batch_indices, ...]
        sampled_states = preprocess(sampled_states)

        sampled_returns = buff_returns[batch_indices]

        policy_net.train_step(sampled_states, sampled_returns)
    
   


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
