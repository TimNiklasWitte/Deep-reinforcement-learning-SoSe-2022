import gym
import numpy as np

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

    # large batch size + small learning rate help against noise (MC estimate!!!)
    batch_size = 64
    episode_len = 50
    gamma = 0.99
        
    env = gym.vector.make("CarRacing-v1", num_envs=batch_size)

    policy_net = PolicyNet()
    policy_net.build(env.observation_space.shape)
    policy_net.summary()


    for num_episode in range(1000):

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
        
        # Evaluation
        env_test = gym.make("CarRacing-v1")
        state = env_test.reset()

        rewards = []
        for test_run_step in range(episode_len):
            state = preprocess(state)
            state = np.expand_dims(state, axis=0)
            action, _ = policy_net(state)
            action = action.numpy()
            action = action[0]
                
            next_state, reward, done , _ = env_test.step(action)

            state = next_state
            rewards.append(reward)
            if done:
                break 

        score = np.sum(rewards)
        avg_rewards = np.mean(rewards)
        
        print(f"  Episode: {num_episode}")
        print(f"    Score: {round(score,2)}")
        print(f"Avg Reward: {round(avg_rewards, 2)}")
        print("------------------------") 

        with train_summary_writer.as_default():
            tf.summary.scalar(f"Average reward", avg_rewards, step=num_episode)
            tf.summary.scalar(f"Score", score, step=num_episode)



        buff_returns = np.zeros(shape=(episode_len, batch_size), dtype=np.float32)
        for start_idx in range(episode_len):

            g_t = np.zeros(shape=(batch_size,), dtype=np.float32)
            for i in range(episode_len - start_idx): #range(len(rewards[j:])):
                rewards = buff_rewards[start_idx + i, :]
                g_t += (gamma**i) * rewards

            buff_returns[start_idx] = g_t


        policy_net.train_step(buff_states, buff_returns)

        if num_episode % 10 == 0:
            policy_net.save_weights(f"./saved_models/trained_weights_episode_{num_episode}", save_format="tf")

   


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")