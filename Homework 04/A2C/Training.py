import gym
import numpy as np

from PolicyNet import *
from ValueNet import *

def preprocess(state):

    state = np.cast['float32'](state)
    state = (state / 128.) - 1
    return state


def main():

    # Logging
    file_path = "test_logs/test"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    gamma = 0.99
    num_episodes = 1000
    episode_len = 50
    episodes_until_saveNet = 25

    env = gym.make("CarRacing-v1")

    policy_net = PolicyNet()
    policy_net.build((1, *env.observation_space.shape))
    policy_net.summary()

    value_net = ValueNet()
    value_net.build((1, *env.observation_space.shape))
    value_net.summary()


    for num_episode in range(num_episodes):

        states = np.zeros(shape=(episode_len, *env.observation_space.shape))
        next_states = np.zeros(shape=(episode_len, *env.observation_space.shape))
        rewards = np.zeros(shape=(episode_len,))

        state = env.reset()
        for step_num in range(episode_len):
            state = preprocess(state)
            state = np.expand_dims(state, axis=0)
            action, _ = policy_net(state)
            action = action.numpy()
            action = action[0] # remove batch dim

            next_state, reward, done , _ = env.step(action)

            states[step_num] = state
            next_states[step_num] = next_state
            rewards[step_num] = reward

            state = next_state
            if done:
                state = env.reset()
    

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


        # Evalation
        env_test = gym.make("CarRacing-v1")
        state = env_test.reset()

        rewards = []
        score = 0
        for test_run_step in range(episode_len):
            state = preprocess(state)
            state = np.expand_dims(state, axis=0)
            action, _ = policy_net(state)
            action = action.numpy()
            action = action[0]
                
            next_state, reward, done , _ = env_test.step(action)

            state = next_state
            rewards.append(reward)
            score += reward
            if done:
                break 
            
        env_test.close()
        avg_reward = np.mean(rewards)

            
        print(f" Iteration: {num_episode}")
        print(f"     Score: {score}")
        print(f"Avg reward: {avg_reward}")
        print(f"      Loss: {loss}")
        print("------------------------")


        #
        # Log
        #
        loss = loss.numpy()
        with train_summary_writer.as_default():
            tf.summary.scalar(f"Average reward", avg_reward, step=num_episode)
            tf.summary.scalar(f"Loss", loss, step=num_episode)

        if num_episode % episodes_until_saveNet == 0:
            policy_net.save_weights(f"./saved_models/policy_net/trained_weights_{num_episode}", save_format="tf")
            value_net.save_weights(f"./saved_models/value_net/trained_weights_{num_episode}", save_format="tf")
   


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")