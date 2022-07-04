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
    gamma = 0.99
        
    env = gym.vector.make("CarRacing-v1", num_envs=batch_size)

    policy_net = PolicyNet()
    policy_net.build(env.observation_space.shape)
    policy_net.summary()

    value_net = ValueNet()
    value_net.build(env.observation_space.shape)
    value_net.summary()

  
    states = env.reset()
    for i in range(100000):
        
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

        if i % 1000:
            policy_net.save_weights(f"./saved_models/policy_net/trained_weights_{i}", save_format="tf")
            value_net.save_weights(f"./saved_models/value_net/trained_weights_{i}", save_format="tf")
   


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")