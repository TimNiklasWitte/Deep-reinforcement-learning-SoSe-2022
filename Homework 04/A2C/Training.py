import gym
import numpy as np

from PolicyNet import *
from ValueNet import *

import tqdm

def preprocess(state):

    state = np.cast['float32'](state)
    state = (state / 128.) - 1
    return state


def main():

    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    batch_size = 32
    episode_len = 50
    gamma = 0.99
        
    env = gym.vector.make("CarRacing-v1", num_envs=batch_size)

    policy_net = PolicyNet()
    policy_net.build(env.observation_space.shape)
    policy_net.summary()

    value_net = ValueNet()
    value_net.build(env.observation_space.shape)
    value_net.summary()

  
    states = env.reset()
    for i in range(5000):
        
        states = preprocess(states)
        actions, _ = policy_net(states)

        actions = actions.numpy()
        
        next_states, rewards, dones , _ = env.step(actions)

        #
        # Train policy net
        #

        # A(s,a) = Q(s,a) - V(s) 
        #        = r_t + gamma*V(s_(t+1)) - V(s_t)

        next_states = preprocess(next_states)
        rewards = np.expand_dims(rewards, axis=-1)
        advantages = rewards + gamma*value_net(next_states) - value_net(states)

        policy_net.train_step(states, advantages)

        # 
        # Train value net
        #
        values_target = rewards + gamma*value_net(next_states)
        loss = value_net.train_step(states, values_target)
     

        states = next_states
        if np.any(dones):
            print("reset")
            states = env.reset()


        #
        # Log
        #

        avg_reward = np.mean(rewards[0, :])
        loss = loss.numpy()
        # print(f" Iteration: {i}")
        # print(f"Avg reward: {avg_reward}")
        # print(f"      Loss: {loss}")
        # print("------------------------")


        with train_summary_writer.as_default():
            tf.summary.scalar(f"Average reward", avg_reward, step=i)
            tf.summary.scalar(f"Loss", loss, step=i)

    return 

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